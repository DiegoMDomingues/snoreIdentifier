"""
suggestPatients.py

Analisa pacientes ainda não classificados e sugere quais priorizar
para ouvir, baseado na estimativa de proporção de eventos not_snore.

Fluxo:
  1. Carrega todos os CSVs já classificados em M4A_classification
  2. Retreina a LogisticRegression com esses dados
  3. Para cada paciente sem CSV ainda:
     - Carrega os m4a
     - Detecta eventos (Energy/Threshold/Event)
     - Extrai embeddings YAMNet
     - Aplica o modelo para estimar snore/not_snore
  4. Ordena pacientes por proporção estimada de not_snore (decrescente)
  5. Exibe o ranking
"""

import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa

from scipy import signal
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from energy import Energy
from threshold import Threshold
from event import Event

# ========= CONFIG =========

PATIENTS_ROOT        = r"C:\Users\diego\Documents\Python\snore2026\M4A_audios"
REVIEWED_CSV_FOLDER  = r"C:\Users\diego\Documents\Python\snore2026\M4A_classification"

AUDIO_EXTENSIONS     = (".m4a", ".wav", ".mp3", ".flac")

YAMNET_HANDLE        = "https://tfhub.dev/google/yamnet/1"
TARGET_SAMPLE_RATE   = 8000
EMBED_PAD_BEFORE_SEC = 0.15
EMBED_PAD_AFTER_SEC  = 0.15
EVENT_T_START        = 0.0
EVENT_T_END          = 78.0
MAX_RESULTS          = 5
SCORE_THRESHOLD      = 0.0

LABEL_COLUMN         = "manual_label"
GROUP_COLUMN         = "patient_id"
RANDOM_STATE         = 42

# Número mínimo de eventos para considerar a estimativa confiável
MIN_EVENTS_FOR_ESTIMATE = 5

# ========= MODELO =========

def load_yamnet():
    print("Carregando YAMNet...")
    model = hub.load(YAMNET_HANDLE)
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    class_names = list(pd.read_csv(class_map_path)["display_name"])
    print(f"YAMNet carregado.\n")
    return model, class_names


def train_model(reviewed_folder):
    """Carrega todos os CSVs classificados e treina a LogisticRegression."""
    csv_files = sorted(glob.glob(os.path.join(reviewed_folder, "reviewed_*.csv")))

    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV classificado encontrado em: {reviewed_folder}")

    dfs = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if LABEL_COLUMN not in df.columns:
            continue
        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip().str.lower()
        df = df[df[LABEL_COLUMN].isin(["snore", "not_snore"])].copy()
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        if not emb_cols:
            continue
        dfs.append(df)

    if not dfs:
        raise ValueError("Nenhum dado válido encontrado para treinar.")

    df_all = pd.concat(dfs, ignore_index=True)
    emb_cols = [c for c in df_all.columns if c.startswith("emb_")]

    df_all["target"] = df_all[LABEL_COLUMN].map({"not_snore": 0, "snore": 1})
    X = df_all[emb_cols].apply(pd.to_numeric, errors="coerce")
    y = df_all["target"].astype(int)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
    model.fit(X, y)

    n_patients = df_all[GROUP_COLUMN].nunique() if GROUP_COLUMN in df_all.columns else "?"
    print(f"Modelo treinado com {len(df_all)} eventos de {n_patients} pacientes.")
    print(f"  Snore: {(y==1).sum()}  |  Not_snore: {(y==0).sum()}\n")

    return model, emb_cols

# ========= ÁUDIO =========

def load_audio(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".wav":
        import soundfile as sf
        data, sr_native = sf.read(file_path, always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if sr_native != TARGET_SAMPLE_RATE:
            data = librosa.resample(data, orig_sr=sr_native, target_sr=TARGET_SAMPLE_RATE)
        return (data * 32768).astype(np.int16)
    else:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE)
        return np.array(audio.get_array_of_samples(), dtype=np.int16)


def normalize_audio(audio):
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def resample_to_16k(audio, sample_rate):
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate == 16000:
        return tf.convert_to_tensor(audio, dtype=tf.float32)
    num_samples = int(round(len(audio) * 16000 / sample_rate))
    if num_samples <= 0:
        return tf.convert_to_tensor(np.array([], dtype=np.float32))
    return tf.convert_to_tensor(signal.resample(audio, num_samples).astype(np.float32))


def extract_event_audio(block, sample_rate, start_sec, end_sec,
                        pad_before=0.15, pad_after=0.15):
    start = int(max(0, (start_sec - pad_before) * sample_rate))
    end   = int(min(len(block), (end_sec + pad_after) * sample_rate))
    if end <= start:
        return None
    return block[start:end]

# ========= EMBEDDING =========

def embed_event(yamnet_model, class_names, event_audio, sample_rate):
    audio_f = normalize_audio(event_audio)
    if len(audio_f) < int(0.05 * sample_rate):
        return None

    waveform = resample_to_16k(audio_f, sample_rate)
    if tf.size(waveform).numpy() == 0:
        return None

    scores, embeddings, _ = yamnet_model(waveform)
    embeddings_np = embeddings.numpy()
    if embeddings_np.size == 0:
        return None

    return embeddings_np.mean(axis=0).astype(np.float32)

# ========= DETECÇÃO DE EVENTOS =========

def detect_events(data, sample_rate):
    energy_obj    = Energy()
    threshold_obj = Threshold()
    event_obj     = Event()

    energy_obj.calculate(data)
    threshold_value = threshold_obj.calculate_threshold(energy_obj.energy)
    events = event_obj.find_events(
        energy_obj.energy, threshold_value,
        t_start=EVENT_T_START, t_end=EVENT_T_END,
    )
    return events

# ========= ANÁLISE POR PACIENTE =========

def analyze_patient(patient_folder, yamnet_model, class_names, clf_model, emb_cols):
    """
    Processa todos os m4a de um paciente e estima a proporção de not_snore.
    Retorna dict com estatísticas.
    """
    patient_id  = os.path.basename(patient_folder)
    audio_files = sorted([
        f for f in glob.glob(os.path.join(patient_folder, "*"))
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    ])

    if not audio_files:
        return None

    total_events   = 0
    not_snore_count = 0
    snore_count    = 0
    low_conf_count = 0  # eventos com probabilidade entre 0.3 e 0.7 (incertos)

    for audio_path in audio_files:
        try:
            data = load_audio(audio_path)
        except Exception as e:
            print(f"    Erro ao carregar {os.path.basename(audio_path)}: {e}")
            continue

        events = detect_events(data, TARGET_SAMPLE_RATE)

        for ev in events:
            embed_audio = extract_event_audio(
                data, TARGET_SAMPLE_RATE,
                ev.eventStart, ev.eventEnd,
                EMBED_PAD_BEFORE_SEC, EMBED_PAD_AFTER_SEC,
            )
            if embed_audio is None or len(embed_audio) == 0:
                continue

            embedding = embed_event(yamnet_model, class_names, embed_audio, TARGET_SAMPLE_RATE)
            if embedding is None:
                continue

            # Monta o vetor de features na mesma ordem que o modelo espera
            row_dict = {col: float(embedding[i]) for i, col in enumerate(emb_cols) if i < len(embedding)}
            X_event = pd.DataFrame([row_dict])[emb_cols]

            prob_snore = float(clf_model.predict_proba(X_event)[0, 1])
            total_events += 1

            if prob_snore < 0.5:
                not_snore_count += 1
            else:
                snore_count += 1

            if 0.3 <= prob_snore <= 0.7:
                low_conf_count += 1

    if total_events == 0:
        return None

    pct_not_snore = not_snore_count / total_events * 100

    return {
        "patient_id":       patient_id,
        "total_events":     total_events,
        "est_snore":        snore_count,
        "est_not_snore":    not_snore_count,
        "pct_not_snore":    round(pct_not_snore, 1),
        "low_conf_events":  low_conf_count,
        "n_files":          len(audio_files),
        "confiavel":        total_events >= MIN_EVENTS_FOR_ESTIMATE,
    }

# ========= MAIN =========

def main():
    # Descobre quais pacientes já foram classificados
    reviewed_csvs = glob.glob(os.path.join(REVIEWED_CSV_FOLDER, "reviewed_*.csv"))
    reviewed_ids  = set()
    for csv_path in reviewed_csvs:
        pid = os.path.basename(csv_path).replace("reviewed_", "").replace(".csv", "")
        reviewed_ids.add(pid)

    # Lista pacientes sem CSV ainda
    all_folders = sorted([
        d for d in glob.glob(os.path.join(PATIENTS_ROOT, "*"))
        if os.path.isdir(d)
    ])
    unreviewed_folders = [
        d for d in all_folders
        if os.path.basename(d) not in reviewed_ids
    ]

    print(f"Pacientes já classificados:    {len(reviewed_ids)}")
    print(f"Pacientes ainda não revisados: {len(unreviewed_folders)}")

    if not unreviewed_folders:
        print("\nTodos os pacientes já foram classificados!")
        return

    # Treina modelo com dados existentes
    print("\n--- Treinando modelo com dados classificados ---")
    clf_model, emb_cols = train_model(REVIEWED_CSV_FOLDER)

    # Carrega YAMNet
    yamnet_model, class_names = load_yamnet()

    # Analisa cada paciente não revisado
    print(f"--- Analisando {len(unreviewed_folders)} paciente(s) não revisado(s) ---\n")
    results = []

    for i, folder in enumerate(unreviewed_folders):
        patient_id = os.path.basename(folder)
        print(f"[{i+1}/{len(unreviewed_folders)}] {patient_id}...", end=" ", flush=True)

        result = analyze_patient(folder, yamnet_model, class_names, clf_model, emb_cols)

        if result is None:
            print("sem eventos detectados.")
            continue

        print(f"{result['total_events']} eventos | "
              f"~{result['pct_not_snore']:.1f}% not_snore")
        results.append(result)

    if not results:
        print("\nNenhum resultado obtido.")
        return

    # Ordena por % not_snore decrescente
    df_results = pd.DataFrame(results).sort_values("pct_not_snore", ascending=False)

    print("\n" + "=" * 70)
    print("SUGESTÃO DE PACIENTES — ordenados por % estimada de not_snore")
    print("(priorize os primeiros para equilibrar o dataset)")
    print("=" * 70)

    for _, row in df_results.iterrows():
        confiavel_str = "" if row["confiavel"] else " ⚠ poucos eventos"
        print(
            f"\n  Paciente : {row['patient_id']}"
            f"\n  Arquivos : {row['n_files']} m4a"
            f"\n  Eventos  : {row['total_events']}  "
            f"(~{row['est_snore']} snore | ~{row['est_not_snore']} not_snore)"
            f"\n  % not_snore estimada : {row['pct_not_snore']:.1f}%{confiavel_str}"
            f"\n  Eventos incertos (30-70%): {row['low_conf_events']}"
        )

    print("\n" + "=" * 70)
    print(f"Top 5 pacientes recomendados:")
    for _, row in df_results.head(5).iterrows():
        print(f"  {row['patient_id']:20s}  ~{row['pct_not_snore']:.1f}% not_snore  "
              f"({row['total_events']} eventos)")

    # Salva resultado
    output_path = os.path.join(REVIEWED_CSV_FOLDER, "patient_suggestions.csv")
    df_results.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nRanking salvo em: {output_path}")


if __name__ == "__main__":
    main()