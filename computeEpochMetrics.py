"""
computeEpochMetrics.py

Calcula sensibilidade/especificidade por ÉPOCA (janela de tempo fixa),
em vez de por evento — metodologia padrão usada em publicações de detecção
de ronco (ex: janelas de 5s ou épocas de 30s cobrindo a gravação inteira).

Por que isso muda o resultado:
  A avaliação por EVENTO (o que já fazíamos) só julga os candidatos que
  sobreviveram ao detector de energia/threshold. Trechos de silêncio óbvio
  que o detector corretamente NUNCA transformou em candidato não entram
  na conta — mesmo sendo, na prática, acertos do sistema como um todo.

  A avaliação por ÉPOCA divide cada clipe de 80s em janelas fixas (ex: 5s)
  e verifica, em cada janela, se havia ronco de verdade e se o pipeline
  completo (detector + classificador) disse que havia. Uma época sem
  NENHUM evento detectado já conta como acerto (verdadeiro-negativo),
  creditando o detector — exatamente como a literatura da área reporta.

Pré-requisitos e limitações importantes:
  - Só usa arquivos m4a 100% revisados (marcados como "done" pelo
    reviewAndEmbed2.py), porque só neles temos certeza de que TODOS os
    candidatos do detector já foram classificados. Arquivos parcialmente
    revisados são ignorados para não contaminar a métrica.
  - Assume que uma época sem nenhum evento detectado é realmente livre de
    ronco. Isso é uma aproximação padrão na literatura, mas vale fazer um
    spot-check ocasional (ouvir algumas dessas épocas "vazias" para
    confirmar).
  - Avalia só nos PACIENTES DE TESTE (TEST_PATIENTS), com o modelo
    treinado nos demais — mantém a validação honesta (sem vazamento).

Requer trainNN_main.py no mesmo diretório (reaproveita as funções de
carregamento e construção de features).
"""

import os
import numpy as np
import pandas as pd

from trainNN_main import (
    load_reviewed_data, load_frame_data, merge_main_with_frames,
    build_morph_features, build_feature_vector, build_models,
    REVIEWED_CSV_FOLDER, FRAMES_DIR, GROUP_COLUMN, SNORE_THRESHOLD,
)

# ========= CONFIG =========

# Pacientes usados como conjunto de teste (obrigatório, não pode ficar vazio)
TEST_PATIENTS = ["3219524FU", "6914928XR", "9258818LC", "0299380UD"]

# Duração da época em segundos (5s é o padrão mais comum na literatura)
EPOCH_SEC = 5.0

# Duração de cada clipe m4a
CLIP_DURATION_SEC = 80.0

MODEL_NAME = "XGBoost"  # qual modelo usar para essa avaliação

# ========= ARQUIVOS "DONE" =========

def get_done_source_files(patient_id, reviewed_folder):
    """Lê o índice de arquivos 100% revisados gerado pelo reviewAndEmbed2.py."""
    index_path = os.path.join(reviewed_folder, f"reviewed_{patient_id}_done.txt")
    if not os.path.exists(index_path):
        return set()
    with open(index_path, "r") as f:
        return set(line.strip() for line in f if line.strip())

# ========= CONSTRUÇÃO DE ÉPOCAS =========

def build_epochs(clip_duration, epoch_sec):
    """Retorna lista de (t0, t1) cobrindo o clipe inteiro."""
    epochs = []
    t0 = 0.0
    while t0 < clip_duration:
        t1 = min(t0 + epoch_sec, clip_duration)
        epochs.append((t0, t1))
        t0 = t1
    return epochs


def event_overlaps_epoch(event_start, event_end, epoch_start, epoch_end):
    return event_start < epoch_end and event_end > epoch_start


def compute_epoch_labels(events_df, clip_duration, epoch_sec):
    """
    events_df: DataFrame com colunas global_start_sec, global_end_sec,
               manual_label (ground truth), y_pred (predição do modelo).
               Pode estar vazio (clipe sem nenhum evento detectado).

    Retorna lista de (ground_truth, predicted) por época, 0/1.
    """
    epochs = build_epochs(clip_duration, epoch_sec)
    results = []

    for t0, t1 in epochs:
        gt = 0
        pred = 0
        for _, ev in events_df.iterrows():
            if event_overlaps_epoch(ev["global_start_sec"], ev["global_end_sec"], t0, t1):
                if ev["manual_label"] == "snore":
                    gt = 1
                if ev["y_pred"] == 1:
                    pred = 1
        results.append((gt, pred))

    return results

# ========= MÉTRICAS =========

def compute_metrics(pairs):
    """pairs: lista de (ground_truth, predicted) 0/1."""
    tp = sum(1 for gt, pred in pairs if gt == 1 and pred == 1)
    tn = sum(1 for gt, pred in pairs if gt == 0 and pred == 0)
    fp = sum(1 for gt, pred in pairs if gt == 0 and pred == 1)
    fn = sum(1 for gt, pred in pairs if gt == 1 and pred == 0)

    total = tp + tn + fp + fn
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    esp  = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    acc  = (tp + tn) / total if total > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1   = (2 * prec * sens / (prec + sens)
           if (prec + sens) > 0 and not np.isnan(prec) and not np.isnan(sens) else float("nan"))

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn, "Total": total,
        "Sensibilidade": sens, "Especificidade": esp, "Acurácia": acc,
        "Precisão": prec, "F1": f1,
    }

# ========= MAIN =========

def main():
    if not TEST_PATIENTS:
        raise ValueError("TEST_PATIENTS não pode ficar vazio para essa análise.")

    print("Carregando dados...")
    df_main = load_reviewed_data(REVIEWED_CSV_FOLDER)
    emb_cols = [c for c in df_main.columns if c.startswith("emb_")]

    df_frames_meta, all_frame_embeddings, all_frame_event_ids = load_frame_data(FRAMES_DIR)
    df_main = merge_main_with_frames(df_main, df_frames_meta)

    X_morph, morph_cols = build_morph_features(df_main)
    y = df_main["target"].astype(int)

    test_pats = [str(p) for p in TEST_PATIENTS]
    test_mask = df_main[GROUP_COLUMN].isin(test_pats)

    df_tr = df_main[~test_mask].reset_index(drop=True)
    df_te = df_main[test_mask].reset_index(drop=True)
    y_tr  = y[~test_mask.values].reset_index(drop=True)
    y_te  = y[test_mask.values].reset_index(drop=True)

    print(f"Treino: {sorted(df_tr[GROUP_COLUMN].unique())}")
    print(f"Teste : {sorted(df_te[GROUP_COLUMN].unique())}")

    print(f"\nTreinando modelo ({MODEL_NAME})...")
    X_train, X_test = build_feature_vector(
        df_tr, df_te, y_tr, emb_cols, X_morph, all_frame_embeddings, all_frame_event_ids
    )
    model = build_models()[MODEL_NAME]
    model.fit(X_train, y_tr)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= SNORE_THRESHOLD).astype(int)

    df_te = df_te.copy()
    df_te["y_pred"] = y_pred

    # ── Métrica de evento (baseline, o que já fazíamos) ─────────────────────
    event_pairs = list(zip(y_te.values.tolist(), y_pred.tolist()))
    event_metrics = compute_metrics(event_pairs)

    print(f"\n{'='*70}")
    print("MÉTRICA POR EVENTO (classificador isolado — o que já reportávamos)")
    print(f"{'='*70}")
    for k, v in event_metrics.items():
        print(f"  {k:15s}: {v:.4f}" if isinstance(v, float) else f"  {k:15s}: {v}")

    # ── Métrica de época (credita o detector) ───────────────────────────────
    print(f"\nCalculando métricas por época ({EPOCH_SEC:.0f}s)...")

    all_epoch_pairs = []
    n_files_used = 0
    n_files_skipped = 0

    for patient_id in test_pats:
        done_files = get_done_source_files(patient_id, REVIEWED_CSV_FOLDER)
        patient_events = df_te[df_te[GROUP_COLUMN] == patient_id]

        source_files_with_events = set(patient_events["source_file"].unique())

        # Usa apenas arquivos marcados como 100% revisados
        for source_file in done_files:
            events_this_file = patient_events[patient_events["source_file"] == source_file]
            pairs = compute_epoch_labels(events_this_file, CLIP_DURATION_SEC, EPOCH_SEC)
            all_epoch_pairs.extend(pairs)
            n_files_used += 1

        n_skipped_this_patient = len(source_files_with_events - done_files)
        n_files_skipped += n_skipped_this_patient

    print(f"Arquivos m4a usados (100% revisados): {n_files_used}")
    print(f"Arquivos ignorados (parcialmente revisados): {n_files_skipped}")
    print(f"Total de épocas avaliadas: {len(all_epoch_pairs)}")

    epoch_metrics = compute_metrics(all_epoch_pairs)

    print(f"\n{'='*70}")
    print(f"MÉTRICA POR ÉPOCA DE {EPOCH_SEC:.0f}s (pipeline completo: detector + classificador)")
    print(f"{'='*70}")
    for k, v in epoch_metrics.items():
        print(f"  {k:15s}: {v:.4f}" if isinstance(v, float) else f"  {k:15s}: {v}")

    print(f"\n{'='*70}")
    print("COMPARAÇÃO")
    print(f"{'='*70}")
    print(f"  Especificidade — por evento: {event_metrics['Especificidade']:.4f}  "
          f"vs  por época: {epoch_metrics['Especificidade']:.4f}")
    print(f"  Sensibilidade  — por evento: {event_metrics['Sensibilidade']:.4f}  "
          f"vs  por época: {epoch_metrics['Sensibilidade']:.4f}")
    print(f"  F1             — por evento: {event_metrics['F1']:.4f}  "
          f"vs  por época: {epoch_metrics['F1']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()