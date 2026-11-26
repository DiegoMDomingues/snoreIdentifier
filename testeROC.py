"""
trainNNv2.py

Versão aprimorada do trainNN.py que adiciona ao embedding YAMNet (1024 dims):
  - Features morfológicas: corr, energyAnt, energyPos, duration_sec
  - Features de vizinhança: probabilidade média de snore dos N eventos
    anteriores do mesmo arquivo (usando predições do modelo base,
    garantindo consistência com produção)

Fluxo em dois estágios:
  Etapa 1: treina modelo base (só embedding YAMNet)
  Etapa 2: usa modelo base para gerar probabilidades de todos os eventos,
            constrói features de vizinhança, treina modelo final com
            embedding + morfologia + vizinhança
"""

import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost não instalado — pulando. Para instalar: pip install xgboost")

# =========================
# CONFIG
# =========================

REVIEWED_CSV_FOLDER = r"C:\Users\diego\Documents\Python\snore2026\M4A_classification"

LABEL_COLUMN  = "manual_label"
GROUP_COLUMN  = "patient_id"

RANDOM_STATE  = 42
TEST_SIZE     = 0.25
N_SPLITS      = 5

SNORE_THRESHOLD = 0.50

# Features morfológicas a incluir (devem existir no CSV)
MORPH_FEATURES = ["duration_sec", "corr", "yamnet_frames"]

# Janela de vizinhança temporal: segundos antes e depois do evento atual
NEIGHBORHOOD_PAST_SEC   = 10.0   # segundos para trás
NEIGHBORHOOD_FUTURE_SEC = 10.0   # segundos para frente

# Lista de pacientes para teste manual (vazia = GroupShuffleSplit automático)
TEST_PATIENTS = []

# =========================
# CARREGAMENTO DE DADOS
# =========================

def get_csv_files(folder_path):
    pattern = os.path.join(folder_path, "reviewed_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em: {folder_path}")
    return files


def load_single_csv(csv_path):
    file_name = os.path.basename(csv_path)
    df = pd.read_csv(csv_path)

    if "patient_id" not in df.columns:
        fallback_id = file_name.replace("reviewed_", "").replace(".csv", "")
        df["patient_id"] = fallback_id

    df["patient_id"] = df["patient_id"].astype(str)

    if "source_file" not in df.columns:
        df["source_file"] = file_name

    return df


def load_and_prepare_data(folder_path):
    csv_files = get_csv_files(folder_path)
    dfs = []

    for csv_path in csv_files:
        file_name = os.path.basename(csv_path)
        df = load_single_csv(csv_path)

        if LABEL_COLUMN not in df.columns:
            continue

        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip().str.lower()
        df = df[df[LABEL_COLUMN].isin(["snore", "not_snore"])].copy()

        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        if not emb_cols:
            continue

        print(f"Loaded: {file_name} | rows: {len(df)}")
        dfs.append(df)

    if not dfs:
        raise ValueError("Nenhum CSV válido encontrado.")

    df_all = pd.concat(dfs, ignore_index=True)

    # Ordena por paciente e arquivo para garantir ordem temporal correta
    df_all = df_all.sort_values(
        [GROUP_COLUMN, "source_file", "global_start_sec"]
    ).reset_index(drop=True)

    df_all["target"] = df_all[LABEL_COLUMN].map({"not_snore": 0, "snore": 1})

    emb_cols = [c for c in df_all.columns if c.startswith("emb_")]

    return df_all, emb_cols

# =========================
# CONSTRUÇÃO DE FEATURES
# =========================

def build_morph_features(df):
    """Extrai features morfológicas disponíveis no CSV."""
    available = [f for f in MORPH_FEATURES if f in df.columns]
    if not available:
        print("AVISO: nenhuma feature morfológica encontrada no CSV.")
        return pd.DataFrame(index=df.index), []

    X_morph = df[available].apply(pd.to_numeric, errors="coerce")
    print(f"Features morfológicas usadas: {available}")
    return X_morph, available


def build_neighborhood_features(df, base_model, emb_cols,
                                past_sec=10.0, future_sec=10.0):
    """
    Etapa 2: usa o modelo base para predizer probabilidade de snore
    de todos os eventos e constrói features de vizinhança temporal bidirecional.

    Para cada evento, considera todos os eventos dentro de uma janela
    de tempo (não de índice) ao redor do evento atual:
      - neigh_past_mean_prob:     média das prob. de eventos nos últimos past_sec
      - neigh_past_snore_count:   quantos eventos nos últimos past_sec eram snore
      - neigh_past_n:             quantos eventos existem nos últimos past_sec
      - neigh_future_mean_prob:   média das prob. de eventos nos próximos future_sec
      - neigh_future_snore_count: quantos eventos nos próximos future_sec eram snore
      - neigh_future_n:           quantos eventos existem nos próximos future_sec
      - neigh_interval_mean:      intervalo médio entre eventos vizinhos (passado)
      - neigh_interval_std:       desvio padrão dos intervalos (passado)

    Usar eventos futuros é válido pois o processamento é feito pós-exame.
    Vizinhança nunca cruza arquivos diferentes.
    """
    print(f"\nGerando features de vizinhança temporal "
          f"(past={past_sec}s, future={future_sec}s)...")

    X_emb = df[emb_cols].apply(pd.to_numeric, errors="coerce")
    all_probs = base_model.predict_proba(X_emb)[:, 1]
    df = df.copy().reset_index(drop=True)  # garante índices 0..n-1
    df["_base_prob"] = all_probs

    n = len(df)
    neigh_past_mean    = np.full(n, 0.5)
    neigh_past_count   = np.zeros(n)
    neigh_past_n       = np.zeros(n)
    neigh_future_mean  = np.full(n, 0.5)
    neigh_future_count = np.zeros(n)
    neigh_future_n     = np.zeros(n)
    neigh_interval_mean = np.zeros(n)
    neigh_interval_std  = np.zeros(n)

    # Processa por arquivo (garante que vizinhança não cruza arquivos diferentes)
    for (patient_id, source_file), group in df.groupby(
        [GROUP_COLUMN, "source_file"], sort=False
    ):
        # Usa posições locais no array (índice do DataFrame já é 0..n-1 após reset)
        positions = group.index.tolist()
        probs   = group["_base_prob"].values
        starts  = group["global_start_sec"].values

        for local_pos, arr_pos in enumerate(positions):
            t_current = starts[local_pos]

            # Eventos no passado dentro da janela temporal
            past_mask   = (starts < t_current) & (starts >= t_current - past_sec)
            past_probs  = probs[past_mask]
            past_starts = starts[past_mask]

            if len(past_probs) > 0:
                neigh_past_mean[arr_pos]  = float(np.mean(past_probs))
                neigh_past_count[arr_pos] = float(np.sum(past_probs > 0.5))
                neigh_past_n[arr_pos]     = float(len(past_probs))
                if len(past_starts) >= 2:
                    intervals = np.diff(past_starts)
                    neigh_interval_mean[arr_pos] = float(np.mean(intervals))
                    neigh_interval_std[arr_pos]  = float(np.std(intervals))

            # Eventos no futuro dentro da janela temporal
            future_mask  = (starts > t_current) & (starts <= t_current + future_sec)
            future_probs = probs[future_mask]

            if len(future_probs) > 0:
                neigh_future_mean[arr_pos]  = float(np.mean(future_probs))
                neigh_future_count[arr_pos] = float(np.sum(future_probs > 0.5))
                neigh_future_n[arr_pos]     = float(len(future_probs))

    neigh_cols = [
        "neigh_past_mean_prob",
        "neigh_past_snore_count",
        "neigh_past_n",
        "neigh_future_mean_prob",
        "neigh_future_snore_count",
        "neigh_future_n",
        "neigh_interval_mean",
        "neigh_interval_std",
    ]

    X_neigh = pd.DataFrame({
        "neigh_past_mean_prob":    neigh_past_mean,
        "neigh_past_snore_count":  neigh_past_count,
        "neigh_past_n":            neigh_past_n,
        "neigh_future_mean_prob":  neigh_future_mean,
        "neigh_future_snore_count":neigh_future_count,
        "neigh_future_n":          neigh_future_n,
        "neigh_interval_mean":     neigh_interval_mean,
        "neigh_interval_std":      neigh_interval_std,
    })  # índice já é 0..n-1 após reset_index

    print(f"Features de vizinhança geradas: {neigh_cols}")
    return X_neigh, neigh_cols

# =========================
# MODELOS
# =========================

def build_base_model():
    """Modelo base: só embedding YAMNet (igual ao trainNN.py original)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def build_models():
    """Modelos finais com todas as features."""
    class_weight = "balanced"
    models = {}

    models["LogisticRegression"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        )),
    ])

    models["MLP"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     MLPClassifier(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            max_iter=500,
            random_state=RANDOM_STATE,
        )),
    ])

    models["SVM"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     SVC(
            kernel="rbf",
            class_weight=class_weight,
            probability=True,
            random_state=RANDOM_STATE,
        )),
    ])

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=1.0,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                verbosity=0,
            )),
        ])

    return models

# =========================
# ANÁLISE DE THRESHOLD
# =========================

def analyze_threshold(model_name, y_true, y_proba,
                      thresholds=None):
    """
    Testa múltiplos thresholds e mostra sensibilidade, especificidade e F1
    para cada um. Identifica o ponto de equilíbrio (sens ≈ esp).
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)

    print(f"\n  Análise de threshold — {model_name}")
    print(f"  {'Threshold':>10}  {'F1':>7}  {'Sens':>7}  {'Esp':>7}  {'Equil.':>7}")
    print(f"  {'─'*50}")

    best_eq_thresh = None
    best_eq_diff   = float("inf")
    best_f1_thresh = None
    best_f1_val    = 0.0

    rows = []
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            esp  = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1   = f1_score(y_true, y_pred, zero_division=0)
            diff = abs(sens - esp)
            eq_marker = " ◄" if diff < 0.05 else ""

            print(f"  {thr:>10.2f}  {f1:>7.4f}  {sens:>7.4f}  {esp:>7.4f}{eq_marker}")

            if diff < best_eq_diff:
                best_eq_diff   = diff
                best_eq_thresh = thr
            if f1 > best_f1_val:
                best_f1_val    = f1
                best_f1_thresh = thr

            rows.append({"threshold": thr, "F1": f1, "Sens": sens, "Esp": esp})

    print(f"\n  → Melhor equilíbrio (sens≈esp): threshold={best_eq_thresh:.2f}")
    print(f"  → Melhor F1:                    threshold={best_f1_thresh:.2f} (F1={best_f1_val:.4f})")

    # AUC-ROC
    try:
        auc = roc_auc_score(y_true, y_proba)
        print(f"  → AUC-ROC: {auc:.4f}")
    except Exception:
        pass

    return best_eq_thresh, best_f1_thresh


# =========================
# AVALIAÇÃO
# =========================#

def evaluate(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc  = (tn + tp) / (tn + tp + fn + fp)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    esp  = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)

    print(f"\n{'='*70}")
    print(f"MODELO: {model_name}")
    print(f"{'='*70}")
    print(f"F1-score:      {f1:.4f}")
    print(f"Acurácia:      {acc:.4f}")
    print(f"Sensibilidade: {sens:.4f}")
    print(f"Especificidade:{esp:.4f}")
    print(f"Precision:     {prec:.4f}")
    print(f"\nConfusion Matrix [[TN FP] [FN TP]]:\n{cm}")

    return {"F1": f1, "Sensibilidade": sens, "Especificidade": esp,
            "Acurácia": acc, "Precision_snore": prec}

# =========================
# MAIN
# =========================

def main():
    print("Carregando dados...")
    df, emb_cols = load_and_prepare_data(REVIEWED_CSV_FOLDER)

    X_morph, morph_cols = build_morph_features(df)
    y = df["target"].astype(int)

    n_patients = df[GROUP_COLUMN].nunique()
    print(f"\nTotal: {len(df)} eventos | {n_patients} pacientes")
    print(f"Snore: {(y==1).sum()} | Not_snore: {(y==0).sum()}")
    print(f"Embedding: {len(emb_cols)} dims + {len(morph_cols)} morfológicas + 8 vizinhança temporal")

    if TEST_PATIENTS:
        # ── Split manual ──────────────────────────────────────────────────
        test_pats  = [str(p) for p in TEST_PATIENTS]
        test_mask  = df[GROUP_COLUMN].isin(test_pats)
        train_mask = ~test_mask

        train_idx = df.index[train_mask].tolist()
        test_idx  = df.index[test_mask].tolist()

        splits = [(train_idx, test_idx)]
        print(f"\nSplit manual | Treino: {sorted(df.loc[train_mask, GROUP_COLUMN].unique())}")
        print(f"Teste: {sorted(df.loc[test_mask, GROUP_COLUMN].unique())}")

    else:
        # ── Validação cruzada ─────────────────────────────────────────────
        gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE,
                                random_state=RANDOM_STATE)
        splits = list(gss.split(df, y, groups=df[GROUP_COLUMN]))
        print(f"\nValidação cruzada: {N_SPLITS} splits")

    models     = build_models()
    cv_results = {name: {"F1": [], "Sensibilidade": [], "Especificidade": [],
                          "Acurácia": [], "Precision_snore": []}
                  for name in models}

    for split_num, (train_idx, test_idx) in enumerate(splits):
        print(f"\n{'─'*70}")
        print(f"Split {split_num+1}/{len(splits)}")
        train_pats = sorted(df.iloc[train_idx][GROUP_COLUMN].unique()) if isinstance(train_idx[0], int) else sorted(df.loc[train_idx, GROUP_COLUMN].unique())
        test_pats_actual  = sorted(df.iloc[test_idx][GROUP_COLUMN].unique()) if isinstance(test_idx[0], int) else sorted(df.loc[test_idx, GROUP_COLUMN].unique())
        print(f"  Treino: {train_pats}")
        print(f"  Teste : {test_pats_actual}")

        # Subsets por índice
        if isinstance(train_idx[0], int):
            df_tr = df.iloc[train_idx]
            df_te = df.iloc[test_idx]
        else:
            df_tr = df.loc[train_idx]
            df_te = df.loc[test_idx]

        y_tr = y.loc[df_tr.index]
        y_te = y.loc[df_te.index]

        X_emb_tr = df_tr[emb_cols].apply(pd.to_numeric, errors="coerce")
        X_emb_te = df_te[emb_cols].apply(pd.to_numeric, errors="coerce")

        # ── Etapa 1: treina modelo base só no treino ──────────────────────
        print("\n  [Etapa 1] Treinando modelo base (embedding only)...")
        base_model = build_base_model()
        base_model.fit(X_emb_tr, y_tr)

        # ── Etapa 2: gera features de vizinhança para TREINO e TESTE ──────
        # Importante: as probabilidades dos eventos de teste são geradas
        # com o modelo treinado SÓ no treino (sem vazamento)
        print("  [Etapa 2] Gerando features de vizinhança...")

        X_neigh_tr, neigh_cols = build_neighborhood_features(
            df_tr, base_model, emb_cols,
            past_sec=NEIGHBORHOOD_PAST_SEC, future_sec=NEIGHBORHOOD_FUTURE_SEC
        )
        X_neigh_te, _ = build_neighborhood_features(
            df_te, base_model, emb_cols,
            past_sec=NEIGHBORHOOD_PAST_SEC, future_sec=NEIGHBORHOOD_FUTURE_SEC
        )

        # ── Monta vetor final: embedding + morfologia + vizinhança ────────
        X_morph_tr = X_morph.loc[df_tr.index]
        X_morph_te = X_morph.loc[df_te.index]

        # Usa .reset_index(drop=True) para evitar conflitos de índice no concat
        X_train_final = pd.concat([
            X_emb_tr.reset_index(drop=True),
            X_morph_tr.reset_index(drop=True),
            X_neigh_tr.reset_index(drop=True),
        ], axis=1)
        X_test_final = pd.concat([
            X_emb_te.reset_index(drop=True),
            X_morph_te.reset_index(drop=True),
            X_neigh_te.reset_index(drop=True),
        ], axis=1)
        y_tr = y_tr.reset_index(drop=True)
        y_te = y_te.reset_index(drop=True)

        print(f"  Dimensão final: {X_train_final.shape[1]} features")
        print(f"  Treino snore: {(y_tr==1).sum()} | not_snore: {(y_tr==0).sum()}")
        print(f"  Teste  snore: {(y_te==1).sum()} | not_snore: {(y_te==0).sum()}")

        # ── Etapa 3: treina e avalia modelos finais ───────────────────────
        for model_name, model in models.items():
            model.fit(X_train_final, y_tr)
            y_pred = (model.predict_proba(X_test_final)[:,1] >= SNORE_THRESHOLD).astype(int)

            metrics = evaluate(f"{model_name} [Split {split_num+1}]", y_te, y_pred)
            for k, v in metrics.items():
                cv_results[model_name][k].append(v)

            # Análise de threshold no último split (para não repetir 5x)
            if split_num == len(splits) - 1:
                y_proba = model.predict_proba(X_test_final)[:, 1]
                analyze_threshold(model_name, y_te, y_proba)

    # ── Tabela comparativa final ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"COMPARAÇÃO FINAL — média ± desvio padrão ({len(splits)} split(s))")
    print(f"{'='*70}")

    summary_rows = []
    for model_name, metrics in cv_results.items():
        row = {"Modelo": model_name}
        for metric, values in metrics.items():
            arr = np.array(values)
            if len(arr) == 1:
                row[metric] = f"{arr[0]:.4f}"
            else:
                row[metric] = f"{arr.mean():.4f} ± {arr.std():.4f}"
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows).set_index("Modelo")
    print(df_summary.to_string())

    best_f1 = max(cv_results, key=lambda m: np.mean(cv_results[m]["F1"]))
    best_sen = max(cv_results, key=lambda m: np.mean(cv_results[m]["Sensibilidade"]))
    best_esp = max(cv_results, key=lambda m: np.mean(cv_results[m]["Especificidade"]))
    print(f"\n→ Melhor F1 médio:            {best_f1} ({np.mean(cv_results[best_f1]['F1']):.4f})")
    print(f"→ Melhor Sensibilidade média: {best_sen} ({np.mean(cv_results[best_sen]['Sensibilidade']):.4f})")
    print(f"→ Melhor Especificidade média:{best_esp} ({np.mean(cv_results[best_esp]['Especificidade']):.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()