"""
extractFrameEmbeddings.py

Reextrai os embeddings YAMNet de cada evento já classificado, mas SEM fazer
a média dos frames — mantém cada frame individual (tipicamente 1-6 frames
de ~1s por evento, com overlap).

Motivação: quando um evento captura mais de um "som" (ex: um galo cantando
seguido de um ronco), a média dos frames mistura as duas coisas e pode
confundir o classificador. Mantendo os frames separados, é possível treinar
um classificador no nível do frame e agregar as PROBABILIDADES na hora da
decisão (ex: pega o frame com maior probabilidade de snore), em vez de
agregar os embeddings antes de classificar.

Usa os timestamps já confirmados em reviewed_<patient_id>.csv — não roda
o detector de eventos de novo, só reprocessa o áudio já classificado.

Salva por paciente:
  frames_<patient_id>.npz   → embeddings de todos os frames (float32)
  frames_meta_<patient_id>.csv → metadados (event_index, source_file,
                                   start, end, label, n_frames)
"""

import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa

from scipy import signal

# ========= CONFIG =========

PATIENTS_ROOT        = r"C:\Users\diego\Documents\Python\snore2026\M4A_audios"
REVIEWED_CSV_FOLDER  = r"C:\Users\diego\Documents\Python\snore2026\M4A_classification"
FRAMES_OUTPUT_DIR    = r"C:\Users\diego\Documents\Python\snore2026\M4A_classification_v5_frames"

AUDIO_EXTENSIONS = (".m4a", ".wav", ".mp3", ".flac")

YAMNET_HANDLE      = "https://tfhub.dev/google/yamnet/1"
TARGET_SAMPLE_RATE = 8000

# Mesmo pad usado no embed_audio do reviewAndEmbed2.py, para manter consistência
EMBED_PAD_BEFORE_SEC = 0.15
EMBED_PAD_AFTER_SEC  = 0.15

# ========= MODELO =========

def load_yamnet():
    print("Carregando YAMNet...")
    model = hub.load(YAMNET_HANDLE)
    print("YAMNet carregado.\n")
    return model

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


def extract_event_segment(data, sr, start_sec, end_sec, pad_before, pad_after):
    start = int(max(0, (start_sec - pad_before) * sr))
    end   = int(min(len(data), (end_sec + pad_after) * sr))
    if end <= start:
        return None
    return data[start:end]

# ========= EXTRAÇÃO POR FRAME =========

def extract_frame_embeddings(model, event_audio, sample_rate):
    """Retorna array (n_frames, 1024) com o embedding de CADA frame, sem média."""
    if event_audio is None or len(event_audio) == 0:
        return None

    audio_f = normalize_audio(event_audio)
    if len(audio_f) < int(0.05 * sample_rate):
        return None

    waveform = resample_to_16k(audio_f, sample_rate)
    if tf.size(waveform).numpy() == 0:
        return None

    _, embeddings, _ = model(waveform)
    embeddings_np = embeddings.numpy()

    if embeddings_np.size == 0:
        return None

    return embeddings_np.astype(np.float32)  # shape (n_frames, 1024)

# ========= PROCESSAMENTO POR PACIENTE =========

def get_frames_paths(patient_id):
    os.makedirs(FRAMES_OUTPUT_DIR, exist_ok=True)
    npz_path  = os.path.join(FRAMES_OUTPUT_DIR, f"frames_{patient_id}.npz")
    meta_path = os.path.join(FRAMES_OUTPUT_DIR, f"frames_meta_{patient_id}.csv")
    return npz_path, meta_path


def process_patient(patient_id, reviewed_df, yamnet_model):
    npz_path, meta_path = get_frames_paths(patient_id)

    if os.path.exists(npz_path) and os.path.exists(meta_path):
        existing_meta = pd.read_csv(meta_path)
        if len(existing_meta) >= len(reviewed_df):
            print(f"  Já processado ({len(existing_meta)} eventos). Pulando.")
            return

    patient_folder = os.path.join(PATIENTS_ROOT, patient_id)
    if not os.path.isdir(patient_folder):
        print(f"  AVISO: pasta não encontrada: {patient_folder}")
        return

    all_embeddings = []   # lista de arrays (n_frames_i, 1024)
    meta_rows = []

    for source_file, group in reviewed_df.groupby("source_file"):
        audio_path = os.path.join(patient_folder, source_file)
        if not os.path.exists(audio_path):
            base = os.path.splitext(source_file)[0]
            found = False
            for ext in AUDIO_EXTENSIONS:
                candidate = os.path.join(patient_folder, base + ext)
                if os.path.exists(candidate):
                    audio_path = candidate
                    found = True
                    break
            if not found:
                print(f"  AVISO: áudio não encontrado para {source_file}")
                continue

        try:
            data = load_audio(audio_path)
        except Exception as e:
            print(f"  Erro ao carregar {source_file}: {e}")
            continue

        for _, row in group.iterrows():
            start_sec = float(row["global_start_sec"])
            end_sec   = float(row["global_end_sec"])
            label     = row["manual_label"]

            segment = extract_event_segment(
                data, TARGET_SAMPLE_RATE, start_sec, end_sec,
                EMBED_PAD_BEFORE_SEC, EMBED_PAD_AFTER_SEC
            )

            frame_embs = extract_frame_embeddings(yamnet_model, segment, TARGET_SAMPLE_RATE)

            if frame_embs is None:
                continue

            event_index = len(meta_rows)
            all_embeddings.append(frame_embs)
            meta_rows.append({
                "event_index":      event_index,
                "patient_id":       patient_id,
                "source_file":      source_file,
                "global_start_sec": start_sec,
                "global_end_sec":   end_sec,
                "manual_label":     label,
                "n_frames":         frame_embs.shape[0],
            })

        print(f"    {source_file}: {len(group)} evento(s) processados")

    if not meta_rows:
        print(f"  Nenhum evento processado para {patient_id}.")
        return

    # Concatena tudo num único array + índice de qual evento cada frame pertence
    stacked = np.concatenate(all_embeddings, axis=0)  # (total_frames, 1024)
    frame_event_index = np.concatenate([
        np.full(emb.shape[0], meta_rows[i]["event_index"], dtype=np.int32)
        for i, emb in enumerate(all_embeddings)
    ])

    np.savez_compressed(npz_path,
                        embeddings=stacked,
                        frame_event_index=frame_event_index)
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False, encoding="utf-8-sig")

    print(f"  Salvo: {len(meta_rows)} eventos, {stacked.shape[0]} frames totais")
    print(f"    → {os.path.basename(npz_path)}")
    print(f"    → {os.path.basename(meta_path)}")

# ========= MAIN =========

def main():
    yamnet_model = load_yamnet()

    csv_files = sorted(glob.glob(os.path.join(REVIEWED_CSV_FOLDER, "reviewed_*.csv")))
    if not csv_files:
        print(f"Nenhum CSV encontrado em: {REVIEWED_CSV_FOLDER}")
        return

    print(f"{len(csv_files)} paciente(s) para processar.\n")

    for csv_path in csv_files:
        file_name  = os.path.basename(csv_path)
        patient_id = file_name.replace("reviewed_", "").replace(".csv", "")

        print(f"{'='*60}")
        print(f"Paciente: {patient_id}")

        df = pd.read_csv(csv_path)
        if "manual_label" in df.columns:
            df = df[df["manual_label"].isin(["snore", "not_snore"])].copy()

        if df.empty:
            print("  Sem eventos classificados ainda. Pulando.")
            continue

        process_patient(patient_id, df, yamnet_model)

    print("\nExtração de frames concluída.")


if __name__ == "__main__":
    main()