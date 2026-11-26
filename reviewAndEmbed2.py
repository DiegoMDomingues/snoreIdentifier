import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import sounddevice as sd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy import signal

from energy import Energy
from threshold import Threshold
from event import Event

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ========= CONFIG =========

# Pasta raiz com subpastas por paciente
# Estrutura esperada:
#   PATIENTS_ROOT/
#     paciente_001/
#       amostra1.m4a
#       amostra2.m4a
#       ...
#     paciente_002/
#       ...
PATIENTS_ROOT = r"C:\Users\diego\Documents\Python\snore2026\M4A_audios"

# Pasta onde os CSVs por paciente serão salvos
OUTPUT_DIR = r"C:\Users\diego\Documents\Python\snore2026\M4A_classification"

# Extensões de áudio aceitas
AUDIO_EXTENSIONS = (".m4a", ".wav", ".mp3", ".flac")

# Lista de pacientes prioritários (IDs das subpastas em PATIENTS_ROOT).
# Se populada, o script processa SÓ esses pacientes, na ordem definida.
# Se vazia, processa todos os pacientes normalmente.
# Exemplo: PRIORITY_PATIENTS = ["1234567AB", "9876543XY"]
PRIORITY_PATIENTS = []

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"

TARGET_SAMPLE_RATE = 8000   # Hz — exigido por Energy/Threshold/Event
# Pad usado para REPRODUÇÃO (você ouvir o contexto em volta do evento)
PAD_BEFORE_SEC     = 1.0
PAD_AFTER_SEC      = 1.0

# Pad usado para o EMBEDDING / classificação (foca no evento, sem contexto excessivo)
EMBED_PAD_BEFORE_SEC = 0.15
EMBED_PAD_AFTER_SEC  = 0.15
MAX_RESULTS        = 5
CLASS_SCORE_THRESHOLD = 0.0

# Limites de tempo dentro do bloco de 80s
# m4a independentes: 0.0 e 78.0
# Noite completa com overlap: 10.0 e 70.0
EVENT_T_START = 0.0
EVENT_T_END   = 78.0

# Normaliza o volume na reprodução
AUTO_NORMALIZE_PLAYBACK = True

# ── Predição do seu modelo (em vez dos top-5 do YAMNet) ────────────────────
# Se True, treina um modelo rápido (LogisticRegression + embedding) usando
# os dados já classificados e mostra a probabilidade de snore para cada
# evento durante a revisão, ao invés dos top-5 labels genéricos do YAMNet.
SHOW_MODEL_PREDICTION = True

# Pasta com os CSVs já classificados (reviewed_*.csv) para treinar o modelo
REVIEWED_CSV_FOLDER = OUTPUT_DIR

# ========= MODELO PREDITIVO (SEU CLASSIFICADOR) =========

def train_quick_model(reviewed_folder):
    """
    Treina rapidamente uma LogisticRegression usando todos os eventos já
    classificados (só embedding, sem vizinhança — modelo "base").
    Usado para dar uma prévia da predição durante a revisão de novos eventos.
    Retorna None se não houver dados suficientes.
    """
    import glob as _glob

    csv_files = _glob.glob(os.path.join(reviewed_folder, "reviewed_*.csv"))
    if not csv_files:
        print("Nenhum CSV classificado encontrado — predição do modelo desativada.")
        return None

    dfs = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "manual_label" not in df.columns:
            continue
        df["manual_label"] = df["manual_label"].astype(str).str.strip().str.lower()
        df = df[df["manual_label"].isin(["snore", "not_snore"])].copy()
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        if not emb_cols or df.empty:
            continue
        dfs.append(df)

    if not dfs:
        print("Sem dados suficientes para treinar o modelo de prévia.")
        return None

    df_all = pd.concat(dfs, ignore_index=True)
    emb_cols = [c for c in df_all.columns if c.startswith("emb_")]

    X = df_all[emb_cols].apply(pd.to_numeric, errors="coerce")
    y = df_all["manual_label"].map({"not_snore": 0, "snore": 1}).astype(int)

    if y.nunique() < 2:
        print("Dados insuficientes (só uma classe) — predição do modelo desativada.")
        return None

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        )),
    ])
    model.fit(X, y)

    print(f"Modelo de prévia treinado: {len(df_all)} eventos "
          f"({int((y==1).sum())} snore / {int((y==0).sum())} not_snore)\n")

    return model, emb_cols


def predict_with_quick_model(quick_model_bundle, mean_embedding):
    """Retorna a probabilidade de snore prevista pelo modelo de prévia."""
    if quick_model_bundle is None or mean_embedding is None:
        return None

    model, emb_cols = quick_model_bundle
    row = {col: float(mean_embedding[i]) for i, col in enumerate(emb_cols)
           if i < len(mean_embedding)}
    X_event = pd.DataFrame([row])[emb_cols]

    try:
        prob = float(model.predict_proba(X_event)[0, 1])
        return prob
    except Exception:
        return None


# ========= MODELO =========

def load_yamnet():
    print("Carregando YAMNet... (pode demorar alguns minutos na primeira vez)")
    import sys
    sys.stdout.flush()
    model = hub.load(YAMNET_HANDLE)
    print("Modelo baixado, carregando class map...")
    sys.stdout.flush()
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    class_names = list(pd.read_csv(class_map_path)["display_name"])
    print(f"YAMNet carregado. {len(class_names)} classes.\n")
    sys.stdout.flush()
    return model, class_names

# ========= ÁUDIO =========

def load_audio(file_path):
    import sys
    ext = os.path.splitext(file_path)[1].lower()
    print(f"    [load_audio] extensao detectada: {ext}"); sys.stdout.flush()

    if ext == ".wav":
        import soundfile as sf
        print("    [load_audio] lendo com soundfile..."); sys.stdout.flush()
        data_float, sr_native = sf.read(file_path, always_2d=False)
        print(f"    [load_audio] lido. sr_native={sr_native}, len={len(data_float)}"); sys.stdout.flush()
        if data_float.ndim == 2:
            data_float = data_float.mean(axis=1)
        data_float = data_float.astype(np.float32)
        if sr_native != TARGET_SAMPLE_RATE:
            print("    [load_audio] resampling com librosa..."); sys.stdout.flush()
            data_float = librosa.resample(data_float, orig_sr=sr_native, target_sr=TARGET_SAMPLE_RATE)
        data_int16 = (np.clip(data_float, -1.0, 1.0) * 32768).astype(np.int16)
    else:
        print("    [load_audio] importando pydub..."); sys.stdout.flush()
        from pydub import AudioSegment
        print("    [load_audio] chamando AudioSegment.from_file (ffmpeg)..."); sys.stdout.flush()
        audio = AudioSegment.from_file(file_path)
        print(f"    [load_audio] ffmpeg concluido. duracao_ms={len(audio)}"); sys.stdout.flush()
        audio = audio.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE)
        print("    [load_audio] convertendo para array numpy..."); sys.stdout.flush()
        data_int16 = np.array(audio.get_array_of_samples(), dtype=np.int16)
        print(f"    [load_audio] array pronto. len={len(data_int16)}"); sys.stdout.flush()

    return TARGET_SAMPLE_RATE, data_int16

def normalize_audio(audio):
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    else:
        audio = audio.astype(np.float32)
    return np.clip(audio, -1.0, 1.0)

def normalize_for_playback(audio_float):
    if AUTO_NORMALIZE_PLAYBACK:
        peak = np.max(np.abs(audio_float))
        if peak > 0:
            audio_float = audio_float / peak * 0.9
    return audio_float.astype(np.float32)

def resample_to_16k(audio, sample_rate):
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate == 16000:
        return tf.convert_to_tensor(audio, dtype=tf.float32)
    num_samples = int(round(len(audio) * 16000 / sample_rate))
    if num_samples <= 0:
        return tf.convert_to_tensor(np.array([], dtype=np.float32), dtype=tf.float32)
    audio_resampled = signal.resample(audio, num_samples)
    return tf.convert_to_tensor(audio_resampled.astype(np.float32), dtype=tf.float32)

def extract_event_audio(block, sample_rate, event_start_sec, event_end_sec,
                        pad_before=0.15, pad_after=0.15):
    start_sample = int(max(0, (event_start_sec - pad_before) * sample_rate))
    end_sample   = int(min(len(block), (event_end_sec + pad_after) * sample_rate))
    if end_sample <= start_sample:
        return None
    return block[start_sample:end_sample]

# ========= YAMNET =========

def embed_event(model, class_names, event_audio, sample_rate):
    if event_audio is None or len(event_audio) == 0:
        return [], None, 0

    event_audio_f = normalize_audio(event_audio)

    # YAMNet precisa de pelo menos ~0.025s (400 amostras a 16kHz) para processar.
    # Eventos muito curtos geram só 1 frame de embedding — ok para o nosso caso,
    # já que fazemos mean pooling dos frames de qualquer forma.
    min_samples_8k = int(0.05 * sample_rate)  # 50ms mínimo
    if len(event_audio_f) < min_samples_8k:
        return [], None, 0

    waveform = resample_to_16k(event_audio_f, sample_rate)

    if tf.size(waveform).numpy() == 0:
        return [], None, 0

    scores, embeddings, _ = model(waveform)
    scores_np     = scores.numpy()
    embeddings_np = embeddings.numpy()

    if scores_np.size == 0 or embeddings_np.size == 0:
        return [], None, 0

    mean_scores    = scores_np.mean(axis=0)
    mean_embedding = embeddings_np.mean(axis=0).astype(np.float32)
    top_indices    = np.argsort(mean_scores)[::-1][:MAX_RESULTS]

    predictions = []
    for idx in top_indices:
        score = float(mean_scores[idx])
        if score >= CLASS_SCORE_THRESHOLD:
            predictions.append({
                "label": class_names[idx],
                "score": score,
                "index": int(idx),
            })

    return predictions, mean_embedding, embeddings_np.shape[0]

def add_embedding_to_row(row, embedding, prefix="emb"):
    if embedding is None:
        return row
    for i, value in enumerate(embedding):
        row[f"{prefix}_{i:04d}"] = float(value)
    return row

# ========= DETECÇÃO DE EVENTOS =========

def detect_events_in_file(data, sample_rate):
    """
    Como os m4a têm exatamente 80s e são independentes,
    processamos como um único bloco sem overlap.
    """
    import sys, time

    energy_obj    = Energy()
    threshold_obj = Threshold()
    event_obj     = Event()

    print(f"    [detect_events] tamanho do array de energia (vazio): {len(energy_obj.energy)}")
    sys.stdout.flush()

    t0 = time.time()
    print("    [detect_events] chamando energy_obj.calculate()...")
    sys.stdout.flush()
    energy_obj.calculate(data)
    print(f"    [detect_events] energy.calculate concluido em {time.time()-t0:.2f}s")
    sys.stdout.flush()

    t0 = time.time()
    print("    [detect_events] chamando threshold_obj.calculate_threshold()...")
    sys.stdout.flush()
    threshold_value = threshold_obj.calculate_threshold(energy_obj.energy)
    print(f"    [detect_events] threshold concluido em {time.time()-t0:.2f}s")
    sys.stdout.flush()

    t0 = time.time()
    print("    [detect_events] chamando event_obj.find_events()...")
    sys.stdout.flush()
    events = event_obj.find_events(
        energy_obj.energy,
        threshold_value,
        t_start=EVENT_T_START,
        t_end=EVENT_T_END,
    )
    print(f"    [detect_events] find_events concluido em {time.time()-t0:.2f}s")
    sys.stdout.flush()

    return events, threshold_value, energy_obj.energy

# ========= ÍNDICE DE ARQUIVOS COMPLETOS =========

def get_index_path(csv_path):
    """Arquivo de índice que registra quais m4a já foram 100% classificados."""
    return csv_path.replace(".csv", "_done.txt")

def load_done_files(csv_path):
    """Retorna set com nomes de arquivos já completamente classificados."""
    index_path = get_index_path(csv_path)
    if not os.path.exists(index_path):
        return set()
    with open(index_path, "r") as f:
        return set(line.strip() for line in f if line.strip())

def mark_file_as_done(csv_path, audio_file):
    """Registra que um arquivo foi 100% classificado."""
    index_path = get_index_path(csv_path)
    with open(index_path, "a") as f:
        f.write(audio_file + "\n")


# ========= CSV =========

def get_output_csv(patient_id):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, f"reviewed_{patient_id}.csv")

def load_existing_csv(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

def save_csv(rows, csv_path):
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

def build_reviewed_set(existing_df, audio_file, tol=0.05):
    """
    Constrói um set de (start_rounded, end_rounded) já classificados
    para um dado arquivo. Muito mais rápido que varrer o DataFrame a cada evento.
    """
    if existing_df.empty:
        return set()
    subset = existing_df[existing_df["source_file"] == audio_file]
    reviewed = set()
    for _, row in subset.iterrows():
        key = (round(row["global_start_sec"] / tol), round(row["global_end_sec"] / tol))
        reviewed.add(key)
    return reviewed


def already_reviewed(reviewed_set, event_start, event_end, tol=0.05):
    """Verifica se esse evento já foi classificado usando o set pré-construído."""
    key = (round(event_start / tol), round(event_end / tol))
    return key in reviewed_set

# ========= PLOT =========

def plot_event(energy_array, threshold_value, ev, sample_rate, audio_file):
    from snoreConstants import SnoreConstants

    dt = SnoreConstants.WINDOW_LENGTH_SECS * SnoreConstants.OVERLAPPING
    n  = len(energy_array)
    time_axis = [i * dt for i in range(n)]

    fig, ax = plt.subplots(figsize=(13, 3))

    ax.plot(time_axis, energy_array, color="#2563EB", linewidth=1.2, label="Energia")

    ax.axhline(y=threshold_value, color="#16A34A", linewidth=1.5,
               linestyle="--", label=f"Threshold ({threshold_value:.3f})")

    ax.axvspan(ev.eventStart, ev.eventEnd, alpha=0.35, color="#F97316", label="Evento")

    # Marca o pad de escuta (1s antes e depois)
    ax.axvspan(
        max(0, ev.eventStart - PAD_BEFORE_SEC),
        min(n * dt, ev.eventEnd + PAD_AFTER_SEC),
        alpha=0.10, color="#F97316"
    )

    ax.set_xlabel("Tempo (s)", fontsize=10)
    ax.set_ylabel("Energia (log)", fontsize=10)
    ax.set_title(f"{audio_file}  |  {ev.eventStart:.2f}s → {ev.eventEnd:.2f}s", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(time_axis[0], time_axis[-1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.pause(0.1)
    plt.show(block=False)


def close_plot():
    plt.close("all")


# ========= REVIEW INTERATIVO =========

def build_predictions_text(preds):
    lines = []
    for i, p in enumerate(preds, 1):
        lines.append(f"  {i}. {p['label']} ({p['score']:.3f})")
    return "\n".join(lines) if lines else "  (sem predições)"

def build_model_prediction_text(prob_snore):
    """Formata a predição do modelo de prévia com uma barra visual simples."""
    if prob_snore is None:
        return "  (modelo de prévia indisponível)"

    label = "SNORE" if prob_snore >= 0.5 else "NOT_SNORE"
    bar_len = 20
    filled  = int(round(prob_snore * bar_len))
    bar     = "█" * filled + "░" * (bar_len - filled)

    confidence_flag = ""
    if 0.35 <= prob_snore <= 0.65:
        confidence_flag = "  ⚠ incerto"

    return f"  {bar}  {prob_snore:.1%} snore  →  {label}{confidence_flag}"

def play_audio_and_get_input(event_audio, sample_rate, prompt):
    """
    Toca o áudio em background e coleta input do usuário em paralelo.
    Se o usuário digitar antes do áudio terminar, para a reprodução imediatamente.
    Retorna o comando digitado.
    """
    import threading

    audio_f = normalize_audio(event_audio)
    audio_f = normalize_for_playback(audio_f)

    sd.play(audio_f, sample_rate)

    result = [None]

    def get_input():
        result[0] = input(prompt).strip()

    t = threading.Thread(target=get_input, daemon=True)
    t.start()

    while sd.get_stream().active and result[0] is None:
        t.join(timeout=0.05)

    sd.stop()
    t.join()

    return result[0] if result[0] is not None else ""




def play_audio(event_audio, sample_rate):
    """ Reprodução simples sem interação (usada no modo lote). """
    audio_f = normalize_audio(event_audio)
    audio_f = normalize_for_playback(audio_f)
    sd.play(audio_f, sample_rate)
    sd.wait()

def ask_label(event_num, total, audio_file, ev, preds, event_audio, sample_rate,
              energy_array=None, threshold_value=None, model_prob=None):
    """
    Exibe info do evento, toca o áudio e coleta o label do usuário.
    Retorna: 'snore', 'not_snore', 'skip', 'quit'
    """
    first = True
    while True:
        print("\n" + "=" * 65)
        print(f"Arquivo : {audio_file}")
        print(f"Evento  : {event_num}/{total}  |  "
              f"{ev.eventStart:.2f}s → {ev.eventEnd:.2f}s  "
              f"({ev.eventEnd - ev.eventStart:.2f}s)")

        if SHOW_MODEL_PREDICTION:
            print(f"Seu modelo:\n{build_model_prediction_text(model_prob)}")
        else:
            print(f"YAMNet top-{MAX_RESULTS}:\n{build_predictions_text(preds)}")

        # Mostra o gráfico só na primeira vez ou ao repetir
        if energy_array is not None and threshold_value is not None:
            plot_event(energy_array, threshold_value, ev, sample_rate, audio_file)

        prompt = (
            "\n[s] snore  [n] not_snore  [r] repetir  [p] pular  [q] sair\n"
            "[S] todos restantes = snore  [N] todos restantes = not_snore\n"
            "Reproduzindo... > "
        )
        cmd_raw = play_audio_and_get_input(event_audio, sample_rate, prompt).strip()
        cmd = cmd_raw.lower()

        if cmd_raw == "S":
            close_plot()
            return "snore_all"
        elif cmd_raw == "N":
            close_plot()
            return "not_snore_all"
        elif cmd == "r":
            close_plot()
            continue
        elif cmd == "s":
            close_plot()
            return "snore"
        elif cmd == "n":
            close_plot()
            return "not_snore"
        elif cmd == "p":
            close_plot()
            return "skip"
        elif cmd == "q":
            close_plot()
            return "quit"
        else:
            print("Comando inválido. Use s / n / r / p / q / S / N.")

# ========= PROCESSAMENTO POR PACIENTE =========

def process_patient(patient_folder, yamnet_model, class_names, quick_model_bundle=None):
    patient_id  = os.path.basename(patient_folder)
    csv_path    = get_output_csv(patient_id)
    existing_df = load_existing_csv(csv_path)
    rows        = [] if existing_df.empty else existing_df.to_dict("records")
    done_files  = load_done_files(csv_path)

    audio_files = sorted([
        f for f in glob.glob(os.path.join(patient_folder, "*"))
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    ])

    if not audio_files:
        print(f"Nenhum áudio encontrado em: {patient_folder}")
        return "continue"

    print(f"\n{'='*65}")
    print(f"Paciente: {patient_id}  |  {len(audio_files)} arquivo(s)")
    print(f"CSV de saída: {csv_path}")
    if not existing_df.empty:
        print(f"Retomando — {len(existing_df)} evento(s) já classificados.")

    for audio_path in audio_files:
        audio_file = os.path.basename(audio_path)

        # Pula imediatamente se o arquivo já foi 100% classificado numa sessão anterior
        if audio_file in done_files:
            print(f"  [pulado] {audio_file} — já classificado completamente.")
            continue

        print(f"\n--- {audio_file} ---")

        try:
            print(f"  Carregando áudio: {audio_file}...")
            import sys; sys.stdout.flush()
            sample_rate, data = load_audio(audio_path)
            print(f"  Áudio carregado: {len(data)/sample_rate:.1f}s")
            sys.stdout.flush()
        except Exception as e:
            print(f"Erro ao carregar {audio_file}: {e}")
            continue

        print("  Detectando eventos...")
        sys.stdout.flush()
        events, threshold_value, energy_array = detect_events_in_file(data, sample_rate)

        print(f"  Threshold: {threshold_value:.3f}  |  Eventos detectados: {len(events)}")

        # Verificação rápida antes de montar pending:
        # se o número de já classificados >= eventos detectados, pula sem nem montar a lista
        if not existing_df.empty:
            n_already = (existing_df["source_file"] == audio_file).sum()
            if n_already >= len(events):
                print(f"  Todos os {len(events)} evento(s) já classificados. Pulando.")
                mark_file_as_done(csv_path, audio_file)
                continue

        if not events:
            print("Nenhum evento detectado neste arquivo.")
            continue

        # Filtra eventos já classificados
        reviewed_set = build_reviewed_set(existing_df, audio_file)
        pending = [
            ev for ev in events
            if not already_reviewed(reviewed_set, ev.eventStart, ev.eventEnd)
        ]

        print(f"Pendentes para classificar: {len(pending)} / {len(events)}")

        bulk_label = None  # quando setado, classifica os restantes sem interação

        for idx, ev in enumerate(pending, 1):
            # Áudio para REPRODUÇÃO (com pad maior, contexto pra você ouvir melhor)
            event_audio = extract_event_audio(
                block=data,
                sample_rate=sample_rate,
                event_start_sec=ev.eventStart,
                event_end_sec=ev.eventEnd,
                pad_before=PAD_BEFORE_SEC,
                pad_after=PAD_AFTER_SEC,
            )

            # Áudio para EMBEDDING (pad pequeno, focado no evento)
            embed_audio = extract_event_audio(
                block=data,
                sample_rate=sample_rate,
                event_start_sec=ev.eventStart,
                event_end_sec=ev.eventEnd,
                pad_before=EMBED_PAD_BEFORE_SEC,
                pad_after=EMBED_PAD_AFTER_SEC,
            )

            # Evento sem áudio válido (em qualquer um dos dois cortes)
            if (event_audio is None or len(event_audio) == 0 or
                    embed_audio is None or len(embed_audio) == 0):
                if bulk_label is None:
                    print(f"\n[{idx}/{len(pending)}] Evento sem áudio válido "
                          f"({ev.eventStart:.2f}s → {ev.eventEnd:.2f}s)")
                    cmd = input("Pular? [s para pular / q para sair] > ").strip().lower()
                    if cmd == "q":
                        save_csv(rows, csv_path)
                        return "quit"
                continue

            # Extrai embeddings YAMNet (necessário mesmo no modo em lote)
            preds, mean_embedding, n_frames = embed_event(
                yamnet_model, class_names, embed_audio, sample_rate
            )

            model_prob = predict_with_quick_model(quick_model_bundle, mean_embedding)

            if bulk_label is not None:
                # Modo em lote: classifica sem tocar áudio nem perguntar
                result = bulk_label
                print(f"  [lote] Evento {idx}/{len(pending)} "
                      f"({ev.eventStart:.2f}s → {ev.eventEnd:.2f}s) → {result}")
            else:
                # Review interativo normal
                result = ask_label(
                    event_num=idx,
                    total=len(pending),
                    audio_file=audio_file,
                    ev=ev,
                    preds=preds,
                    event_audio=event_audio,
                    sample_rate=sample_rate,
                    energy_array=energy_array,
                    threshold_value=threshold_value,
                    model_prob=model_prob,
                )

                if result == "quit":
                    save_csv(rows, csv_path)
                    print(f"\nSaindo. Progresso salvo em: {csv_path}")
                    return "quit"

                if result == "skip":
                    continue

                if result == "snore_all":
                    bulk_label = "snore"
                    result = "snore"
                    print(f"\n  >>> Modo lote ativado: todos os eventos restantes "
                          f"deste arquivo serão marcados como 'snore'.\n")
                elif result == "not_snore_all":
                    bulk_label = "not_snore"
                    result = "not_snore"
                    print(f"\n  >>> Modo lote ativado: todos os eventos restantes "
                          f"deste arquivo serão marcados como 'not_snore'.\n")

            # Monta linha do CSV
            row = {
                "patient_id":       patient_id,
                "source_file":      audio_file,
                "global_start_sec": float(ev.eventStart),
                "global_end_sec":   float(ev.eventEnd),
                "duration_sec":     float(ev.eventEnd - ev.eventStart),
                "corr":             float(getattr(ev, "corr", np.nan)),
                "yamnet_frames":    int(n_frames),
                "manual_label":     result,
                "review_timestamp": pd.Timestamp.now().isoformat(),
            }

            for i in range(MAX_RESULTS):
                if i < len(preds):
                    row[f"top{i+1}_label"] = preds[i]["label"]
                    row[f"top{i+1}_score"] = preds[i]["score"]
                    row[f"top{i+1}_index"] = preds[i]["index"]
                else:
                    row[f"top{i+1}_label"] = ""
                    row[f"top{i+1}_score"] = np.nan
                    row[f"top{i+1}_index"] = np.nan

            row = add_embedding_to_row(row, mean_embedding, prefix="emb")
            rows.append(row)

            # Salva após cada label para não perder progresso
            save_csv(rows, csv_path)

        if bulk_label is not None:
            print(f"\n  Resumo do modo lote em '{audio_file}': "
                  f"todos os eventos restantes marcados como '{bulk_label}'.")

        # Marca o arquivo como completamente classificado
        if len(pending) > 0:  # só marca se havia algo pendente (ou seja, acabou de terminar)
            mark_file_as_done(csv_path, audio_file)
            done_files.add(audio_file)

    print(f"\nPaciente {patient_id} concluído. CSV: {csv_path}")
    return "continue"

# ========= MAIN =========

def main():
    yamnet_model, class_names = load_yamnet()

    quick_model_bundle = None
    if SHOW_MODEL_PREDICTION:
        print("Treinando modelo de prévia com dados já classificados...")
        quick_model_bundle = train_quick_model(REVIEWED_CSV_FOLDER)

    if PRIORITY_PATIENTS:
        # Usa só os pacientes da lista, na ordem definida
        patient_folders = []
        for pid in PRIORITY_PATIENTS:
            folder = os.path.join(PATIENTS_ROOT, pid)
            if os.path.isdir(folder):
                patient_folders.append(folder)
            else:
                print(f"AVISO: pasta não encontrada para paciente '{pid}': {folder}")
        print(f"{len(patient_folders)} paciente(s) na lista prioritária.")
    else:
        # Comportamento normal: todos os pacientes
        patient_folders = sorted([
            d for d in glob.glob(os.path.join(PATIENTS_ROOT, "*"))
            if os.path.isdir(d)
        ])
        print(f"{len(patient_folders)} paciente(s) encontrado(s).")

    if not patient_folders:
        print(f"Nenhuma pasta de paciente encontrada em: {PATIENTS_ROOT}")
        return

    for folder in patient_folders:
        result = process_patient(folder, yamnet_model, class_names, quick_model_bundle)
        if result == "quit":
            print("\nSessão encerrada pelo usuário.")
            break

    print("\nProcessamento finalizado.")


if __name__ == "__main__":
    main()