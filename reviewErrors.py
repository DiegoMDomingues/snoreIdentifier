import os
import numpy as np
import pandas as pd
import sounddevice as sd

from pydub import AudioSegment

# ========= CONFIG =========

# CSV gerado pelo trainNN.py com as predições do conjunto de teste
TEST_PREDICTIONS_CSV = r"C:\Users\diego\Documents\Python\snore2026\M4A_classification\test_predictions_logreg.csv"

# Pasta com os CSVs de labels originais (reviewed_*.csv) — para corrigir labels
REVIEWED_CSV_FOLDER = r"C:\Users\diego\Documents\Python\snore2026\M4A_classification"

# Pasta raiz com subpastas por paciente (onde estão os m4a originais)
PATIENTS_ROOT = r"C:\Users\diego\Documents\Python\snore2026\M4A_audios"

# Pad de escuta (segundos antes e depois do evento)
PAD_BEFORE_SEC = 1.0
PAD_AFTER_SEC  = 1.0

# Taxa de amostragem para reprodução
PLAYBACK_SAMPLE_RATE = 22050

# Filtro: quais erros ouvir
# "all"          → todos os erros
# "fp"           → só falsos positivos (not_snore classificado como snore)
# "fn"           → só falsos negativos (snore classificado como not_snore)
ERROR_FILTER = "all"

# ========= AUDIO =========

def load_event_audio(m4a_path, start_sec, end_sec, pad_before=1.0, pad_after=1.0):
    audio = AudioSegment.from_file(m4a_path)
    audio = audio.set_channels(1)

    start_ms = max(0, int((start_sec - pad_before) * 1000))
    end_ms   = min(len(audio), int((end_sec + pad_after) * 1000))

    segment = audio[start_ms:end_ms]
    segment = segment.set_frame_rate(PLAYBACK_SAMPLE_RATE)

    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = samples / peak * 0.9

    return samples


def play_audio(samples):
    sd.play(samples, PLAYBACK_SAMPLE_RATE)
    sd.wait()


def play_audio_nonblocking(samples):
    sd.play(samples, PLAYBACK_SAMPLE_RATE)


# ========= HELPERS =========

def find_m4a(patient_id, source_file):
    """Localiza o m4a original na pasta do paciente."""
    patient_folder = os.path.join(PATIENTS_ROOT, patient_id)
    if not os.path.isdir(patient_folder):
        return None
    candidate = os.path.join(patient_folder, source_file)
    if os.path.exists(candidate):
        return candidate
    # Tenta sem extensão / com variações
    base = os.path.splitext(source_file)[0]
    for ext in (".m4a", ".wav", ".mp3", ".flac"):
        candidate = os.path.join(patient_folder, base + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def error_type(y_true, y_pred):
    if y_true == 1 and y_pred == 0:
        return "FN"  # Falso Negativo: era ronco, disse not_snore
    if y_true == 0 and y_pred == 1:
        return "FP"  # Falso Positivo: não era ronco, disse snore
    return None


def label_name(val):
    return "snore" if val == 1 else "not_snore"


# ========= CORREÇÃO DE LABEL =========

def correct_label(patient_id, source_file, start_sec, end_sec, new_label, tol=0.05):
    """
    Corrige o manual_label no CSV reviewed_<patient_id>.csv correspondente.
    Retorna True se encontrou e corrigiu, False se não encontrou o evento.
    """
    csv_path = os.path.join(REVIEWED_CSV_FOLDER, f"reviewed_{patient_id}.csv")
    if not os.path.exists(csv_path):
        print(f"AVISO: CSV não encontrado: {csv_path}")
        return False

    df = pd.read_csv(csv_path)

    mask = (
        (df["source_file"] == source_file) &
        ((df["global_start_sec"] - start_sec).abs() <= tol) &
        ((df["global_end_sec"]   - end_sec).abs()   <= tol)
    )

    if mask.sum() == 0:
        print("AVISO: evento não encontrado no CSV de labels.")
        return False

    old_label = df.loc[mask, "manual_label"].values[0]
    df.loc[mask, "manual_label"] = new_label
    df.loc[mask, "review_timestamp"] = pd.Timestamp.now().isoformat()
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"  Label corrigido: {old_label} → {new_label}  (salvo em {os.path.basename(csv_path)})")
    return True


# ========= MAIN =========

def main():
    print("Carregando predições...")
    df = pd.read_csv(TEST_PREDICTIONS_CSV)

    # Filtra só os erros
    errors = df[df["y_true"] != df["y_pred"]].copy()
    errors["error_type"] = errors.apply(
        lambda r: error_type(int(r["y_true"]), int(r["y_pred"])), axis=1
    )

    if ERROR_FILTER == "fp":
        errors = errors[errors["error_type"] == "FP"]
    elif ERROR_FILTER == "fn":
        errors = errors[errors["error_type"] == "FN"]

    errors = errors.sort_values("snore_probability", ascending=False).reset_index(drop=True)

    total = len(errors)
    fn_count = (errors["error_type"] == "FN").sum()
    fp_count = (errors["error_type"] == "FP").sum()

    print(f"\nTotal de erros: {total}  |  FN: {fn_count}  |  FP: {fp_count}")
    print("FN = era ronco, modelo disse not_snore")
    print("FP = não era ronco, modelo disse snore")

    for idx, row in errors.iterrows():
        patient_id  = str(row["patient_id"])
        source_file = str(row["source_file"])
        start_sec   = float(row["global_start_sec"])
        end_sec     = float(row["global_end_sec"])
        duration    = float(row["duration_sec"])
        y_true      = int(row["y_true"])
        y_pred      = int(row["y_pred"])
        prob        = float(row["snore_probability"])
        err_type    = row["error_type"]

        m4a_path = find_m4a(patient_id, source_file)

        print("\n" + "=" * 65)
        print(f"Erro {idx + 1}/{total}  [{err_type}]")
        print(f"Paciente   : {patient_id}")
        print(f"Arquivo    : {source_file}")
        print(f"Evento     : {start_sec:.2f}s → {end_sec:.2f}s  ({duration:.2f}s)")
        print(f"Label real : {label_name(y_true)}")
        print(f"Predito    : {label_name(y_pred)}  (prob snore: {prob:.3f})")

        if m4a_path is None:
            print("AVISO: arquivo m4a não encontrado. Pulando reprodução.")
            cmd = input("\n[p] próximo  [q] sair\n> ").strip().lower()
            if cmd == "q":
                break
            continue

        while True:
            try:
                samples = load_event_audio(m4a_path, start_sec, end_sec,
                                           PAD_BEFORE_SEC, PAD_AFTER_SEC)
            except Exception as e:
                print(f"Erro ao carregar áudio: {e}")
                break

            correct_hint = "not_snore" if y_true == 1 else "snore"
            prompt = (
                f"\n[r] repetir  [p] próximo  [q] sair\n"
                f"[c] corrigir label para '{correct_hint}' (concordar com o modelo)\n"
                "Reproduzindo... > "
            )
            sd.play(samples, PLAYBACK_SAMPLE_RATE)

            import threading
            result = [None]
            def get_input():
                result[0] = input(prompt).strip().lower()
            t = threading.Thread(target=get_input, daemon=True)
            t.start()
            while sd.get_stream().active and result[0] is None:
                t.join(timeout=0.05)
            sd.stop()
            t.join()

            cmd = result[0] if result[0] else ""

            if cmd == "r":
                continue
            elif cmd == "c":
                # Corrige o label para o valor que o modelo previu
                correct_label(patient_id, source_file, start_sec, end_sec, correct_hint)
                break
            elif cmd == "q":
                print("\nSaindo.")
                return
            else:
                break  # próximo evento

    print(f"\nFim dos {total} eventos com erro.")


if __name__ == "__main__":
    main()
