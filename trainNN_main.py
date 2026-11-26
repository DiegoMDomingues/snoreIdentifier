"""
..py

Pipeline principal de treino do classificador de ronco.

Combina:
  - Embedding médio do evento (YAMNet, 1024 dims)
  - Features morfológicas: duration_sec, corr, yamnet_frames
  - Vizinhança temporal (±10s): 8 features baseadas na probabilidade de
    snore dos eventos vizinhos no tempo (mesmo arquivo m4a)
  - Meta-features de frame: frame_prob_mean, frame_prob_max, frame_prob_std
    (extraídas de um classificador treinado no nível do FRAME, não no
    embedding médio — o desvio padrão entre frames captura eventos com
    conteúdo misto, ex: um galo cantando + ronco na mesma janela)

Vetor final: 1024 + 3 + 8 + 3 = 1038 features.

Pré-requisito: rodar extractFrameEmbeddings.py antes (gera os arquivos
frames_*.npz / frames_meta_*.csv em FRAMES_DIR).

Suporta:
  - Split automático por paciente (GroupShuffleSplit, N_SPLITS)
  - Split manual (TEST_PATIENTS)
  - Comparação entre 4 modelos: LogisticRegression, MLP, SVM, XGBoost
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
from sklearn.metrics import confusion_matrix, f1_score, precision_score

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
FRAMES_DIR           = r"C:\Users\diego\Documents\Python\snore2026\M4A_classification_v5_frames"

LABEL_COLUMN  = "manual_label"
GROUP_COLUMN  = "patient_id"

RANDOM_STATE  = 42
TEST_SIZE     = 0.25
N_SPLITS      = 5
SNORE_THRESHOLD = 0.50

MORPH_FEATURES = ["duration_sec", "corr", "yamnet_frames"]

NEIGHBORHOOD_PAST_SEC   = 10.0
NEIGHBORHOOD_FUTURE_SEC = 10.0

MERGE_TOL_SEC = 0.05  # tolerância para casar eventos entre reviewed_*.csv e frames_meta_*.csv

# Lista de pacientes para o conjunto de TESTE.
# Se populada, esses pacientes vão sempre para teste e o restante para treino.
# Se vazia, o GroupShuffleSplit decide automaticamente (N_SPLITS rodadas).
TEST_PATIENTS = []

SAVE_TEST_PREDICTIONS = True
TEST_PREDICTIONS_CSV = r"C:\Users\diego\Documents\Python\snore2026\M4A_classification\test_predictions_v6.csv"

# =========================
# CARREGAMENTO — DADOS PRINCIPAIS
# =========================

def load_reviewed_data(folder_path):
    csv_files = sorted(glob.glob(os.path.join(folder_path, "reviewed_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em: {folder_path}")

    dfs = []
    for csv_path in csv_files:
        file_name = os.path.basename(csv_path)
        df = pd.read_csv(csv_path)

        if "patient_id" not in df.columns:
            df["patient_id"] = file_name.replace("reviewed_", "").replace(".csv", "")
        df["patient_id"] = df["patient_id"].astype(str)

        if "source_file" not in df.columns:
            df["source_file"] = file_name

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
    df_all = df_all.sort_values(
        [GROUP_COLUMN, "source_file", "global_start_sec"]
    ).reset_index(drop=True)

    df_all["target"] = df_all[LABEL_COLUMN].map({"not_snore": 0, "snore": 1})

    return df_all

# =========================
# CARREGAMENTO — DADOS DE FRAME
# =========================

def load_frame_data(frames_dir):
    meta_files = sorted(glob.glob(os.path.join(frames_dir, "frames_meta_*.csv")))
    if not meta_files:
        raise FileNotFoundError(
            f"Nenhum frames_meta_*.csv encontrado em {frames_dir}. "
            "Rode extractFrameEmbeddings.py primeiro."
        )

    all_meta = []
    all_embeddings = []
    offset = 0

    for meta_path in meta_files:
        patient_id = os.path.basename(meta_path).replace("frames_meta_", "").replace(".csv", "")
        npz_path = os.path.join(frames_dir, f"frames_{patient_id}.npz")
        if not os.path.exists(npz_path):
            continue

        meta = pd.read_csv(meta_path).reset_index(drop=True)
        meta["patient_id"] = str(patient_id)

        npz = np.load(npz_path)
        embeddings = npz["embeddings"]
        frame_event_index = npz["frame_event_index"]

        meta["global_event_id"] = meta.index + offset
        frame_global_event_id = frame_event_index + offset

        all_meta.append(meta)
        all_embeddings.append((embeddings, frame_global_event_id))
        offset += len(meta)

    df_frames_meta = pd.concat(all_meta, ignore_index=True)
    stacked_embeddings = np.concatenate([e for e, _ in all_embeddings], axis=0)
    stacked_event_ids  = np.concatenate([idx for _, idx in all_embeddings], axis=0)

    return df_frames_meta, stacked_embeddings, stacked_event_ids

# =========================
# MERGE ENTRE OS DOIS MUNDOS
# =========================

def build_merge_key(df, tol=MERGE_TOL_SEC):
    return list(zip(
        df["patient_id"],
        df["source_file"],
        (df["global_start_sec"] / tol).round().astype(int),
        (df["global_end_sec"] / tol).round().astype(int),
    ))


def merge_main_with_frames(df_main, df_frames_meta):
    """
    Casa os eventos do df_main (reviewed_*.csv) com os do df_frames_meta
    (frames_*.npz), via patient_id + source_file + timestamps.
    Retorna df_main filtrado (só eventos com frame disponível) com uma
    coluna extra 'global_event_id' apontando para o índice no mundo de frames.
    """
    df_main = df_main.copy()
    df_frames_meta = df_frames_meta.copy()

    df_main["_key"] = build_merge_key(df_main)
    df_frames_meta["_key"] = build_merge_key(df_frames_meta)

    key_to_event_id = dict(zip(df_frames_meta["_key"], df_frames_meta["global_event_id"]))

    df_main["global_event_id"] = df_main["_key"].map(key_to_event_id)
    n_before = len(df_main)
    df_main = df_main[df_main["global_event_id"].notna()].copy()
    df_main["global_event_id"] = df_main["global_event_id"].astype(int)
    n_after = len(df_main)

    print(f"Merge reviewed <-> frames: {n_after}/{n_before} eventos com match "
          f"({n_before - n_after} sem dados de frame — rode extractFrameEmbeddings.py "
          f"para incluí-los)")

    df_main = df_main.drop(columns=["_key"]).reset_index(drop=True)
    return df_main

# =========================
# FEATURES
# =========================

def build_morph_features(df):
    available = [f for f in MORPH_FEATURES if f in df.columns]
    return df[available].apply(pd.to_numeric, errors="coerce"), available


def build_frame_base_probs(df_subset, all_frame_embeddings, all_frame_event_ids, frame_model):
    """
    Usa o modelo de frame (já treinado só nos frames de treino) para prever
    a probabilidade de cada frame dos eventos em df_subset, e agrega em 3
    features por evento: mean, max, std.
    """
    event_ids = df_subset["global_event_id"].values
    mask = np.isin(all_frame_event_ids, event_ids)

    frame_embs_subset = all_frame_embeddings[mask]
    frame_eids_subset = all_frame_event_ids[mask]

    frame_probs = frame_model.predict_proba(frame_embs_subset)[:, 1]

    df_fp = pd.DataFrame({"event_id": frame_eids_subset, "prob": frame_probs})
    agg = df_fp.groupby("event_id")["prob"].agg(["mean", "max", "std"])
    agg["std"] = agg["std"].fillna(0.0)  # eventos com 1 frame só não têm desvio
    agg = agg.reindex(event_ids)

    return pd.DataFrame({
        "frame_prob_mean": agg["mean"].values,
        "frame_prob_max":  agg["max"].values,
        "frame_prob_std":  agg["std"].values,
    })


def build_neighborhood_features_from_probs(df, prob_series, past_sec=10.0, future_sec=10.0):
    """Vizinhança temporal bidirecional, usando uma probabilidade já calculada."""
    df = df.copy().reset_index(drop=True)
    n = len(df)
    probs = np.asarray(prob_series)

    neigh_past_mean    = np.full(n, 0.5)
    neigh_past_count   = np.zeros(n)
    neigh_past_n       = np.zeros(n)
    neigh_future_mean  = np.full(n, 0.5)
    neigh_future_count = np.zeros(n)
    neigh_future_n     = np.zeros(n)
    neigh_interval_mean = np.zeros(n)
    neigh_interval_std  = np.zeros(n)

    for (pid, src), group in df.groupby([GROUP_COLUMN, "source_file"], sort=False):
        positions = group.index.tolist()
        starts = group["global_start_sec"].values
        group_probs = probs[positions]

        for local_pos, arr_pos in enumerate(positions):
            t = starts[local_pos]

            past_mask  = (starts < t) & (starts >= t - past_sec)
            past_p = group_probs[past_mask]; past_s = starts[past_mask]
            if len(past_p) > 0:
                neigh_past_mean[arr_pos]  = float(np.mean(past_p))
                neigh_past_count[arr_pos] = float(np.sum(past_p > 0.5))
                neigh_past_n[arr_pos]     = float(len(past_p))
                if len(past_s) >= 2:
                    ivs = np.diff(past_s)
                    neigh_interval_mean[arr_pos] = float(np.mean(ivs))
                    neigh_interval_std[arr_pos]  = float(np.std(ivs))

            future_mask  = (starts > t) & (starts <= t + future_sec)
            future_p = group_probs[future_mask]
            if len(future_p) > 0:
                neigh_future_mean[arr_pos]  = float(np.mean(future_p))
                neigh_future_count[arr_pos] = float(np.sum(future_p > 0.5))
                neigh_future_n[arr_pos]     = float(len(future_p))

    neigh_cols = ["neigh_past_mean_prob", "neigh_past_snore_count", "neigh_past_n",
                  "neigh_future_mean_prob", "neigh_future_snore_count", "neigh_future_n",
                  "neigh_interval_mean", "neigh_interval_std"]

    return pd.DataFrame({
        "neigh_past_mean_prob":    neigh_past_mean,
        "neigh_past_snore_count":  neigh_past_count,
        "neigh_past_n":            neigh_past_n,
        "neigh_future_mean_prob":  neigh_future_mean,
        "neigh_future_snore_count":neigh_future_count,
        "neigh_future_n":          neigh_future_n,
        "neigh_interval_mean":     neigh_interval_mean,
        "neigh_interval_std":      neigh_interval_std,
    }), neigh_cols

# =========================
# MODELOS
# =========================

def build_frame_model():
    """Modelo simples e rápido para o nível de frame (etapa 0)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])


def build_models():
    models = {}
    models["LogisticRegression"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])
    models["MLP"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     MLPClassifier(
            hidden_layer_sizes=(256, 64), activation="relu",
            max_iter=500, random_state=RANDOM_STATE
        )),
    ])
    models["SVM"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     SVC(
            kernel="rbf", class_weight="balanced",
            probability=True, random_state=RANDOM_STATE
        )),
    ])
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0
            )),
        ])
    return models

# =========================
# FEATURE ENGINEERING POR SPLIT
# =========================

def build_feature_vector(df_tr, df_te, y_tr, emb_cols, X_morph,
                         all_frame_embeddings, all_frame_event_ids):
    """
    Executa o pipeline completo de features para um split de treino/teste:
      Etapa 0: treina modelo de frame só no treino
      Etapa 1: agrega probabilidades de frame por evento (meta-features)
      Etapa 2: constrói vizinhança temporal a partir dessas probabilidades
      Etapa 3: monta o vetor final (embedding + morfologia + vizinhança + meta)
    """
    X_emb_tr = df_tr[emb_cols].apply(pd.to_numeric, errors="coerce")
    X_emb_te = df_te[emb_cols].apply(pd.to_numeric, errors="coerce")

    train_event_ids = set(df_tr["global_event_id"].tolist())
    train_frame_mask = np.isin(all_frame_event_ids, list(train_event_ids))
    X_frame_tr = all_frame_embeddings[train_frame_mask]

    event_to_label = dict(zip(df_tr["global_event_id"], y_tr))
    y_frame_tr = np.array([
        event_to_label[eid] for eid in all_frame_event_ids[train_frame_mask]
    ])

    frame_model = build_frame_model()
    frame_model.fit(X_frame_tr, y_frame_tr)

    frame_meta_tr = build_frame_base_probs(df_tr, all_frame_embeddings, all_frame_event_ids, frame_model)
    frame_meta_te = build_frame_base_probs(df_te, all_frame_embeddings, all_frame_event_ids, frame_model)

    X_neigh_tr, neigh_cols = build_neighborhood_features_from_probs(
        df_tr, frame_meta_tr["frame_prob_mean"].values,
        past_sec=NEIGHBORHOOD_PAST_SEC, future_sec=NEIGHBORHOOD_FUTURE_SEC
    )
    X_neigh_te, _ = build_neighborhood_features_from_probs(
        df_te, frame_meta_te["frame_prob_mean"].values,
        past_sec=NEIGHBORHOOD_PAST_SEC, future_sec=NEIGHBORHOOD_FUTURE_SEC
    )

    X_train = pd.concat([
        X_emb_tr.reset_index(drop=True),
        X_morph.loc[df_tr.index].reset_index(drop=True),
        X_neigh_tr.reset_index(drop=True),
        frame_meta_tr.reset_index(drop=True),
    ], axis=1)
    X_test = pd.concat([
        X_emb_te.reset_index(drop=True),
        X_morph.loc[df_te.index].reset_index(drop=True),
        X_neigh_te.reset_index(drop=True),
        frame_meta_te.reset_index(drop=True),
    ], axis=1)

    return X_train, X_test

# =========================
# AVALIAÇÃO
# =========================

def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "F1":              f1_score(y_true, y_pred, zero_division=0),
        "Sensibilidade":   tp / (tp + fn) if (tp + fn) > 0 else 0,
        "Especificidade":  tn / (tn + fp) if (tn + fp) > 0 else 0,
        "Acurácia":        (tn + tp) / (tn + tp + fn + fp),
        "Precision_snore": precision_score(y_true, y_pred, zero_division=0),
    }


def print_summary(cv_results, title):
    print(f"\n{'='*72}")
    print(title)
    print(f"{'='*72}")
    rows = []
    for model_name, metrics in cv_results.items():
        row = {"Modelo": model_name}
        for metric, values in metrics.items():
            arr = np.array(values)
            row[metric] = (f"{arr[0]:.4f}" if len(arr) == 1
                           else f"{arr.mean():.4f} ± {arr.std():.4f}")
        rows.append(row)
    df_summary = pd.DataFrame(rows).set_index("Modelo")
    print(df_summary.to_string())

    best_f1  = max(cv_results, key=lambda m: np.mean(cv_results[m]["F1"]))
    best_sen = max(cv_results, key=lambda m: np.mean(cv_results[m]["Sensibilidade"]))
    best_esp = max(cv_results, key=lambda m: np.mean(cv_results[m]["Especificidade"]))
    print(f"\n→ Melhor F1 médio:            {best_f1} ({np.mean(cv_results[best_f1]['F1']):.4f})")
    print(f"→ Melhor Sensibilidade média: {best_sen} ({np.mean(cv_results[best_sen]['Sensibilidade']):.4f})")
    print(f"→ Melhor Especificidade média:{best_esp} ({np.mean(cv_results[best_esp]['Especificidade']):.4f})")

# =========================
# MAIN
# =========================

def main():
    print("Carregando dados principais (reviewed_*.csv)...")
    df_main = load_reviewed_data(REVIEWED_CSV_FOLDER)
    emb_cols = [c for c in df_main.columns if c.startswith("emb_")]

    print("Carregando dados de frame (frames_*.npz)...")
    df_frames_meta, all_frame_embeddings, all_frame_event_ids = load_frame_data(FRAMES_DIR)

    df_main = merge_main_with_frames(df_main, df_frames_meta)
    y = df_main["target"].astype(int)

    X_morph, morph_cols = build_morph_features(df_main)

    n_patients = df_main[GROUP_COLUMN].nunique()
    print(f"\nTotal: {len(df_main)} eventos | {n_patients} pacientes")
    print(f"Snore: {(y==1).sum()} | Not_snore: {(y==0).sum()}")
    print(f"Features: {len(emb_cols)} embedding + {len(morph_cols)} morfológicas "
          f"+ 8 vizinhança + 3 meta-frame = {len(emb_cols)+len(morph_cols)+8+3}")

    if TEST_PATIENTS:
        test_pats = [str(p) for p in TEST_PATIENTS]
        test_mask = df_main[GROUP_COLUMN].isin(test_pats)
        splits = [(df_main.index[~test_mask].tolist(), df_main.index[test_mask].tolist())]
        print(f"\nSplit manual | Teste: {sorted(df_main.loc[test_mask, GROUP_COLUMN].unique())}")
    else:
        gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        splits = list(gss.split(df_main, y, groups=df_main[GROUP_COLUMN]))
        print(f"\nValidação cruzada: {N_SPLITS} splits")

    models = build_models()
    cv_results = {name: {"F1": [], "Sensibilidade": [], "Especificidade": [],
                          "Acurácia": [], "Precision_snore": []}
                  for name in models}

    last_split_predictions = None  # guarda predições do último split p/ salvar em CSV

    for split_num, (train_idx, test_idx) in enumerate(splits):
        print(f"\n{'─'*72}")
        print(f"Split {split_num+1}/{len(splits)}")

        df_tr = df_main.iloc[train_idx].reset_index(drop=True)
        df_te = df_main.iloc[test_idx].reset_index(drop=True)
        y_tr = y.iloc[train_idx].reset_index(drop=True)
        y_te = y.iloc[test_idx].reset_index(drop=True)

        print(f"  Treino: {sorted(df_tr[GROUP_COLUMN].unique())}")
        print(f"  Teste : {sorted(df_te[GROUP_COLUMN].unique())}")
        print(f"  Treino snore: {(y_tr==1).sum()} | not_snore: {(y_tr==0).sum()}")
        print(f"  Teste  snore: {(y_te==1).sum()} | not_snore: {(y_te==0).sum()}")

        X_train, X_test = build_feature_vector(
            df_tr, df_te, y_tr, emb_cols, X_morph,
            all_frame_embeddings, all_frame_event_ids
        )

        if split_num == 0:
            print(f"  Dimensão final: {X_train.shape[1]} features")

        for model_name, model in models.items():
            model.fit(X_train, y_tr)
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred  = (y_proba >= SNORE_THRESHOLD).astype(int)

            metrics = evaluate(y_te, y_pred)
            for k, v in metrics.items():
                cv_results[model_name][k].append(v)

            print(f"    {model_name:20s} F1={metrics['F1']:.4f}  "
                  f"Sens={metrics['Sensibilidade']:.4f}  "
                  f"Esp={metrics['Especificidade']:.4f}")

            if split_num == len(splits) - 1 and model_name == "XGBoost":
                results_df = df_te.copy()
                results_df["y_true"] = y_te.values
                results_df["y_pred"] = y_pred
                results_df["snore_probability"] = y_proba
                last_split_predictions = results_df

    print_summary(cv_results, f"RESULTADO FINAL ({len(splits)} split(s))")

    if SAVE_TEST_PREDICTIONS and last_split_predictions is not None:
        cols_to_save = [c for c in [
            GROUP_COLUMN, "source_file", "global_start_sec", "global_end_sec",
            "duration_sec", "manual_label", "y_true", "y_pred", "snore_probability"
        ] if c in last_split_predictions.columns]
        last_split_predictions[cols_to_save].to_csv(
            TEST_PREDICTIONS_CSV, index=False, encoding="utf-8-sig"
        )
        print(f"\nPredições do último split (XGBoost) salvas em:\n{TEST_PREDICTIONS_CSV}")

    print("\nDone.")


if __name__ == "__main__":
    main()