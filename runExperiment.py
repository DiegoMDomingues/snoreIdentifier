"""
runExperiment.py
================
Harness de experimentos para melhoria autônoma do classificador de ronco.
Roda UMA configuração (arquivo JSON) com validação cruzada por paciente,
SOMENTE nos pacientes de desenvolvimento, e registra o resultado em
experiments/experiments_log.csv + experiments/results/<exp_id>.json.

Os pacientes do HOLDOUT (experiments/holdout.json) são removidos logo após
o carregamento e nunca entram na busca. Eles só são usados no modo --final,
que é rodado pelo Diego, não pelo loop autônomo.

Uso:
    python runExperiment.py experiments/configs/exp_000_baseline.json
    python runExperiment.py experiments/configs/exp_007.json --repeats 3
    python runExperiment.py --compare exp_000_baseline exp_007
    python runExperiment.py --leaderboard
    python runExperiment.py experiments/configs/<cfg>.json --final --i-am-diego

Formato do JSON (todos os campos são opcionais; ausentes = baseline v6):
{
  "exp_id": "exp_007",
  "description": "XGBoost com scale_pos_weight",
  "features": {"emb": true, "morph": true, "neigh": true, "frame_meta": true},
  "neigh_past_sec": 10.0,
  "neigh_future_sec": 10.0,
  "frame_model_C": 1.0,
  "frame_oof": false,            # vizinhança/meta do TREINO via out-of-fold
  "model": {"type": "xgboost", "params": {"n_estimators": 300}},
  #   type: logreg | svm | mlp | xgboost | ensemble
  #   ensemble: {"type": "ensemble", "members": [{"type": ...}, ...]}
  "calibrate": false,            # CalibratedClassifierCV(isotonic, cv=3)
  "threshold": 0.5,
  "epoch_sec": 5.0
}
"""

import argparse
import copy
import datetime as dt
import glob
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import trainNN_main as T

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(HERE, "experiments")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
CACHE_DIR = os.path.join(EXP_DIR, "_cache")
LOG_CSV = os.path.join(EXP_DIR, "experiments_log.csv")
HOLDOUT_JSON = os.path.join(EXP_DIR, "holdout.json")

CV_SEED = 42
N_SPLITS = 5
TEST_SIZE = 0.25
CLIP_DURATION_SEC = 80.0
PRIMARY_METRIC = "epoch_bal_acc"   # (sens + esp) / 2 por época

DEFAULT_CONFIG = {
    "exp_id": None,
    "description": "",
    "features": {"emb": True, "morph": True, "neigh": True, "frame_meta": True},
    "neigh_past_sec": 10.0,
    "neigh_future_sec": 10.0,
    "frame_model_C": 1.0,
    "frame_oof": False,
    "model": {"type": "xgboost", "params": {}},
    "calibrate": False,
    "threshold": 0.5,
    "epoch_sec": 5.0,
}

SWEEP_THRESHOLDS = [round(x, 2) for x in np.arange(0.20, 0.81, 0.05)]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=HERE, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=HERE, stderr=subprocess.DEVNULL).decode().strip() != ""
        return commit, dirty
    except Exception:  # noqa: BLE001
        return "nogit", True


def load_holdout():
    if not os.path.exists(HOLDOUT_JSON):
        raise FileNotFoundError(
            f"{HOLDOUT_JSON} não existe. Ele define os pacientes trancados para o teste final.")
    with open(HOLDOUT_JSON, encoding="utf-8") as f:
        data = json.load(f)
    pats = [str(p) for p in data.get("holdout_patients", [])]
    if not pats:
        raise ValueError("holdout.json sem pacientes — o protocolo exige um holdout.")
    return pats


# ---------------------------------------------------------------------------
# Dados (com cache local; o cache contém dados de paciente e está no .gitignore)
# ---------------------------------------------------------------------------

def data_signature(reviewed_dir, frames_dir):
    files = sorted(glob.glob(os.path.join(reviewed_dir, "reviewed_*.csv"))
                   + glob.glob(os.path.join(reviewed_dir, "reviewed_*_done.txt"))
                   + glob.glob(os.path.join(frames_dir, "frames_*")))
    h = hashlib.sha1()
    for p in files:
        st = os.stat(p)
        h.update(f"{os.path.basename(p)}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()[:12]


def get_done_files(patient_id, reviewed_dir):
    p = os.path.join(reviewed_dir, f"reviewed_{patient_id}_done.txt")
    if not os.path.exists(p):
        return set()
    with open(p, encoding="utf-8", errors="ignore") as f:
        return {line.strip() for line in f if line.strip()}


def load_all(reviewed_dir, frames_dir):
    os.makedirs(CACHE_DIR, exist_ok=True)
    sig = data_signature(reviewed_dir, frames_dir)
    cache_path = os.path.join(CACHE_DIR, f"data_{sig}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            d = pickle.load(f)
        d["signature"] = sig
        return d

    for old in glob.glob(os.path.join(CACHE_DIR, "data_*.pkl")):
        os.remove(old)

    df_main = T.load_reviewed_data(reviewed_dir)
    df_meta, frame_emb, frame_eid = T.load_frame_data(frames_dir)
    df_main = T.merge_main_with_frames(df_main, df_meta)
    emb_cols = [c for c in df_main.columns if c.startswith("emb_")]
    done = {pid: get_done_files(pid, reviewed_dir) for pid in df_main[T.GROUP_COLUMN].unique()}

    d = {"df": df_main, "emb_cols": emb_cols, "frame_emb": frame_emb.astype(np.float32),
         "frame_eid": frame_eid, "done": done}
    with open(cache_path, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
    d["signature"] = sig
    return d


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

def make_model(spec, calibrate=False):
    t = spec.get("type", "xgboost").lower()
    params = dict(spec.get("params", {}))
    if "hidden_layer_sizes" in params:
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    rs = T.RANDOM_STATE

    if t == "ensemble":
        members = [make_model(m, calibrate) for m in spec["members"]]
        weights = spec.get("weights")
        return SoftVoting(members, weights)

    if t == "logreg":
        clf = LogisticRegression(**{"max_iter": 3000, "class_weight": "balanced",
                                    "random_state": rs, **params})
        steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clf)]
    elif t == "svm":
        clf = SVC(**{"kernel": "rbf", "class_weight": "balanced", "probability": True,
                     "random_state": rs, **params})
        steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clf)]
    elif t == "mlp":
        clf = MLPClassifier(**{"hidden_layer_sizes": (256, 64), "max_iter": 500,
                               "random_state": rs, **params})
        steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clf)]
    elif t == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost não instalado")
        clf = XGBClassifier(**{"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                               "eval_metric": "logloss", "random_state": rs, "verbosity": 0,
                               "n_jobs": -1, **params})
        steps = [("imputer", SimpleImputer(strategy="median")), ("clf", clf)]
    else:
        raise ValueError(f"model.type desconhecido: {t}")

    pipe = Pipeline(steps)
    if calibrate:
        return CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    return pipe


class SoftVoting:
    """Média (ponderada) das probabilidades de vários modelos."""

    def __init__(self, members, weights=None):
        self.members = members
        self.weights = weights

    def fit(self, X, y):
        for m in self.members:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        probs = np.stack([m.predict_proba(X)[:, 1] for m in self.members], axis=1)
        w = np.ones(probs.shape[1]) if self.weights is None else np.asarray(self.weights, float)
        p = probs @ (w / w.sum())
        return np.stack([1 - p, p], axis=1)


def make_frame_model(C):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                   random_state=T.RANDOM_STATE)),
    ])


# ---------------------------------------------------------------------------
# Features por split
# ---------------------------------------------------------------------------

def frame_meta_features(df_sub, frame_emb, frame_eid, frame_model):
    return T.build_frame_base_probs(df_sub, frame_emb, frame_eid, frame_model)


def fit_frame_model(df_tr, y_tr, frame_emb, frame_eid, C):
    ids = df_tr["global_event_id"].values
    mask = np.isin(frame_eid, ids)
    lab = dict(zip(ids, np.asarray(y_tr)))
    y_frames = np.array([lab[e] for e in frame_eid[mask]])
    fm = make_frame_model(C)
    fm.fit(frame_emb[mask], y_frames)
    return fm


def build_features(df_tr, df_te, y_tr, d, cfg):
    fe, fid = d["frame_emb"], d["frame_eid"]
    feats = cfg["features"]
    need_frame = feats.get("neigh") or feats.get("frame_meta")

    blocks_tr, blocks_te = [], []
    if feats.get("emb"):
        blocks_tr.append(df_tr[d["emb_cols"]].apply(pd.to_numeric, errors="coerce").reset_index(drop=True))
        blocks_te.append(df_te[d["emb_cols"]].apply(pd.to_numeric, errors="coerce").reset_index(drop=True))
    if feats.get("morph"):
        cols = [c for c in T.MORPH_FEATURES if c in df_tr.columns]
        blocks_tr.append(df_tr[cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True))
        blocks_te.append(df_te[cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True))

    if need_frame:
        fm_all = fit_frame_model(df_tr, y_tr, fe, fid, cfg["frame_model_C"])
        meta_te = frame_meta_features(df_te, fe, fid, fm_all)

        if cfg.get("frame_oof"):
            # Probabilidades de frame do TREINO fora-da-amostra (por paciente),
            # para que as features de vizinhança/meta do treino tenham a mesma
            # distribuição das do teste (evita stacking otimista).
            meta_tr = pd.DataFrame(index=range(len(df_tr)),
                                   columns=["frame_prob_mean", "frame_prob_max", "frame_prob_std"],
                                   dtype=float)
            groups = df_tr[T.GROUP_COLUMN].values
            n_inner = min(5, len(np.unique(groups)))
            for in_tr, in_va in GroupKFold(n_splits=n_inner).split(df_tr, y_tr, groups):
                fm = fit_frame_model(df_tr.iloc[in_tr], np.asarray(y_tr)[in_tr], fe, fid,
                                     cfg["frame_model_C"])
                part = frame_meta_features(df_tr.iloc[in_va], fe, fid, fm)
                meta_tr.iloc[in_va] = part.values
        else:
            meta_tr = frame_meta_features(df_tr, fe, fid, fm_all)

        if feats.get("neigh"):
            n_tr, _ = T.build_neighborhood_features_from_probs(
                df_tr, meta_tr["frame_prob_mean"].values,
                past_sec=cfg["neigh_past_sec"], future_sec=cfg["neigh_future_sec"])
            n_te, _ = T.build_neighborhood_features_from_probs(
                df_te, meta_te["frame_prob_mean"].values,
                past_sec=cfg["neigh_past_sec"], future_sec=cfg["neigh_future_sec"])
            blocks_tr.append(n_tr.reset_index(drop=True))
            blocks_te.append(n_te.reset_index(drop=True))
        if feats.get("frame_meta"):
            blocks_tr.append(meta_tr.reset_index(drop=True).astype(float))
            blocks_te.append(meta_te.reset_index(drop=True).astype(float))

    if not blocks_tr:
        raise ValueError("Nenhum bloco de feature habilitado.")
    X_tr = pd.concat(blocks_tr, axis=1)
    X_te = pd.concat(blocks_te, axis=1)
    X_tr.columns = X_te.columns = [str(c) for c in X_tr.columns]
    return X_tr, X_te


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def binary_metrics(y_true, y_pred, prefix):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    esp = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * sens / (prec + sens) if (tp > 0) else 0.0
    return {f"{prefix}_sens": sens, f"{prefix}_esp": esp,
            f"{prefix}_bal_acc": np.nanmean([sens, esp]),
            f"{prefix}_f1": f1, f"{prefix}_prec": prec,
            f"{prefix}_n": tp + tn + fp + fn}


def epoch_pairs(df_te, y_pred, done, epoch_sec):
    """(gt, pred) por época de epoch_sec, só em arquivos 100% revisados."""
    n_ep = int(np.ceil(CLIP_DURATION_SEC / epoch_sec))
    df = df_te[[T.GROUP_COLUMN, "source_file", "global_start_sec", "global_end_sec", "target"]].copy()
    df["pred"] = np.asarray(y_pred).astype(int)
    gts, preds = [], []
    for pid in df[T.GROUP_COLUMN].unique():
        files = done.get(pid, set())
        ev_p = df[df[T.GROUP_COLUMN] == pid]
        for src in files:
            gt = np.zeros(n_ep, int)
            pr = np.zeros(n_ep, int)
            ev = ev_p[ev_p["source_file"] == src]
            for s, e, t, p in ev[["global_start_sec", "global_end_sec", "target", "pred"]].itertuples(index=False):
                a = max(0, int(np.floor(s / epoch_sec)))
                b = min(n_ep - 1, int(np.ceil(e / epoch_sec)) - 1)
                if b < a:
                    continue
                if t == 1:
                    gt[a:b + 1] = 1
                if p == 1:
                    pr[a:b + 1] = 1
            gts.append(gt)
            preds.append(pr)
    if not gts:
        return np.array([], int), np.array([], int)
    return np.concatenate(gts), np.concatenate(preds)


def split_metrics(df_te, y_te, proba, thr, done, epoch_sec):
    pred = (proba >= thr).astype(int)
    m = binary_metrics(y_te, pred, "event")
    g, p = epoch_pairs(df_te, pred, done, epoch_sec)
    m.update(binary_metrics(g, p, "epoch"))
    try:
        m["event_auc"] = roc_auc_score(y_te, proba)
    except ValueError:
        m["event_auc"] = np.nan
    return m


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

def make_splits(df, y, repeats):
    splits = []
    for r in range(repeats):
        gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=CV_SEED + r)
        splits.extend(gss.split(df, y, groups=df[T.GROUP_COLUMN]))
    return splits


def run_cv(cfg, d, df_dev, repeats):
    y = df_dev["target"].astype(int).reset_index(drop=True)
    df_dev = df_dev.reset_index(drop=True)
    splits = make_splits(df_dev, y, repeats)
    per_split, sweep = [], {t: [] for t in SWEEP_THRESHOLDS}

    for i, (tr, te) in enumerate(splits):
        t0 = time.time()
        df_tr, df_te = df_dev.iloc[tr].reset_index(drop=True), df_dev.iloc[te].reset_index(drop=True)
        y_tr, y_te = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        X_tr, X_te = build_features(df_tr, df_te, y_tr, d, cfg)
        model = make_model(cfg["model"], cfg.get("calibrate", False))
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        m = split_metrics(df_te, y_te, proba, cfg["threshold"], d["done"], cfg["epoch_sec"])
        per_split.append(m)
        for t in SWEEP_THRESHOLDS:
            sweep[t].append(split_metrics(df_te, y_te, proba, t, d["done"], cfg["epoch_sec"])[PRIMARY_METRIC])
        print(f"  split {i + 1}/{len(splits)}  {PRIMARY_METRIC}={m[PRIMARY_METRIC]:.4f}  "
              f"epoch sens={m['epoch_sens']:.3f} esp={m['epoch_esp']:.3f}  "
              f"event F1={m['event_f1']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return per_split, sweep, X_tr.shape[1]


def summarize(per_split):
    keys = per_split[0].keys()
    out = {}
    for k in keys:
        arr = np.array([m[k] for m in per_split], float)
        out[f"{k}_mean"] = float(np.nanmean(arr))
        out[f"{k}_std"] = float(np.nanstd(arr))
    return out


def append_log(row):
    df_row = pd.DataFrame([row])
    if os.path.exists(LOG_CSV):
        df_row.to_csv(LOG_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(LOG_CSV, index=False)


def cmd_run(cfg_path, repeats, data_root):
    with open(cfg_path, encoding="utf-8") as f:
        user_cfg = json.load(f)
    cfg = deep_merge(DEFAULT_CONFIG, user_cfg)
    if not cfg.get("exp_id"):
        cfg["exp_id"] = os.path.splitext(os.path.basename(cfg_path))[0]
    exp_id = cfg["exp_id"] + (f"_r{repeats}" if repeats > 1 else "")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(os.path.join(RESULTS_DIR, f"{exp_id}.json")):
        raise SystemExit(f"{exp_id} já existe em results/. Use outro exp_id (resultados não são sobrescritos).")

    reviewed_dir, frames_dir = resolve_dirs(data_root)
    d = load_all(reviewed_dir, frames_dir)
    holdout = load_holdout()
    df = d["df"]
    df_dev = df[~df[T.GROUP_COLUMN].isin(holdout)].copy()
    assert not set(df_dev[T.GROUP_COLUMN]) & set(holdout), "vazamento do holdout"
    print(f"[{exp_id}] dev: {df_dev[T.GROUP_COLUMN].nunique()} pacientes, {len(df_dev)} eventos "
          f"| holdout trancado: {len(holdout)} pacientes | dados={d['signature']}")

    t0 = time.time()
    per_split, sweep, n_feat = run_cv(cfg, d, df_dev, repeats)
    summ = summarize(per_split)
    sweep_mean = {str(t): float(np.nanmean(v)) for t, v in sweep.items()}
    best_thr = max(sweep_mean, key=sweep_mean.get)
    commit, dirty = git_info()

    result = {
        "exp_id": exp_id, "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "config": cfg, "repeats": repeats, "n_splits_total": len(per_split),
        "n_features": n_feat, "data_signature": d["signature"],
        "n_dev_patients": int(df_dev[T.GROUP_COLUMN].nunique()), "n_dev_events": int(len(df_dev)),
        "git_commit": commit, "git_dirty": dirty, "runtime_sec": round(time.time() - t0, 1),
        "summary": summ, "per_split": per_split,
        "threshold_sweep_primary": sweep_mean, "best_sweep_threshold": float(best_thr),
    }
    with open(os.path.join(RESULTS_DIR, f"{exp_id}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=float)

    append_log({
        "exp_id": exp_id, "timestamp": result["timestamp"], "description": cfg.get("description", ""),
        "primary": PRIMARY_METRIC,
        "primary_mean": round(summ[f"{PRIMARY_METRIC}_mean"], 4),
        "primary_std": round(summ[f"{PRIMARY_METRIC}_std"], 4),
        "epoch_sens": round(summ["epoch_sens_mean"], 4), "epoch_esp": round(summ["epoch_esp_mean"], 4),
        "event_f1": round(summ["event_f1_mean"], 4), "event_auc": round(summ["event_auc_mean"], 4),
        "threshold": cfg["threshold"], "best_sweep_thr": float(best_thr),
        "best_sweep_primary": round(sweep_mean[best_thr], 4),
        "repeats": repeats, "data_sig": d["signature"], "git": commit + ("*" if dirty else ""),
        "runtime_sec": result["runtime_sec"],
    })

    print(f"\n[{exp_id}] {PRIMARY_METRIC} = {summ[f'{PRIMARY_METRIC}_mean']:.4f} ± "
          f"{summ[f'{PRIMARY_METRIC}_std']:.4f} | epoch sens={summ['epoch_sens_mean']:.4f} "
          f"esp={summ['epoch_esp_mean']:.4f} | event F1={summ['event_f1_mean']:.4f} "
          f"AUC={summ['event_auc_mean']:.4f}")
    print(f"  melhor limiar na varredura: {best_thr} -> {sweep_mean[best_thr]:.4f}")


def load_result(exp_id):
    p = os.path.join(RESULTS_DIR, f"{exp_id}.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def cmd_compare(a, b):
    """Comparação PAREADA (mesmos splits): b - a, split a split."""
    ra, rb = load_result(a), load_result(b)
    if ra["data_signature"] != rb["data_signature"]:
        print("ATENÇÃO: dados diferentes entre os experimentos — comparação inválida. "
              "Rode o baseline de novo nos dados atuais.")
    if ra["n_splits_total"] != rb["n_splits_total"]:
        raise SystemExit("Número de splits diferente — rode ambos com o mesmo --repeats.")
    va = np.array([m[PRIMARY_METRIC] for m in ra["per_split"]])
    vb = np.array([m[PRIMARY_METRIC] for m in rb["per_split"]])
    diff = vb - va
    wins = int((diff > 0).sum())
    print(f"{b} vs {a} ({PRIMARY_METRIC}, pareado em {len(diff)} splits)")
    print(f"  média da diferença: {diff.mean():+.4f}  (dp {diff.std():.4f})")
    print(f"  splits em que {b} ganhou: {wins}/{len(diff)}")
    promoted = diff.mean() >= 0.005 and wins >= int(np.ceil(0.7 * len(diff)))
    print(f"  critério de promoção (Δ≥+0.005 e ≥70% dos splits): {'SIM' if promoted else 'NÃO'}")


def cmd_leaderboard(n=15):
    if not os.path.exists(LOG_CSV):
        raise SystemExit("Sem log ainda.")
    df = pd.read_csv(LOG_CSV)
    cols = ["exp_id", "primary_mean", "primary_std", "epoch_sens", "epoch_esp",
            "event_f1", "best_sweep_thr", "best_sweep_primary", "repeats", "data_sig", "description"]
    print(df.sort_values("primary_mean", ascending=False)[cols].head(n).to_string(index=False))


def cmd_final(cfg_path, data_root):
    """Treina em TODOS os pacientes de desenvolvimento e avalia UMA vez no holdout."""
    with open(cfg_path, encoding="utf-8") as f:
        cfg = deep_merge(DEFAULT_CONFIG, json.load(f))
    reviewed_dir, frames_dir = resolve_dirs(data_root)
    d = load_all(reviewed_dir, frames_dir)
    holdout = load_holdout()
    df = d["df"].reset_index(drop=True)
    y = df["target"].astype(int)
    te_mask = df[T.GROUP_COLUMN].isin(holdout).values
    df_tr, df_te = df[~te_mask].reset_index(drop=True), df[te_mask].reset_index(drop=True)
    y_tr, y_te = y[~te_mask].reset_index(drop=True), y[te_mask].reset_index(drop=True)
    X_tr, X_te = build_features(df_tr, df_te, y_tr, d, cfg)
    model = make_model(cfg["model"], cfg.get("calibrate", False))
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    m = split_metrics(df_te, y_te, proba, cfg["threshold"], d["done"], cfg["epoch_sec"])
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {"config": cfg, "holdout_patients": holdout, "metrics": m, "timestamp": stamp,
           "data_signature": d["signature"], "git": git_info()}
    path = os.path.join(EXP_DIR, f"FINAL_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    print(json.dumps(m, indent=2, default=float))
    print(f"\nSalvo em {path}")


def resolve_dirs(data_root):
    if data_root:
        return (os.path.join(data_root, "M4A_classification"),
                os.path.join(data_root, "M4A_classification_v5_frames"))
    return T.REVIEWED_CSV_FOLDER, T.FRAMES_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--repeats", type=int, default=1, help="repetições do CV com seeds diferentes")
    ap.add_argument("--data-root", default=None, help="sobrepõe a pasta snore2026 (testes)")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "NOVO"))
    ap.add_argument("--leaderboard", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--i-am-diego", action="store_true")
    args = ap.parse_args()

    if args.compare:
        cmd_compare(*args.compare)
    elif args.leaderboard:
        cmd_leaderboard()
    elif args.final:
        if not args.i_am_diego:
            raise SystemExit("--final avalia o holdout e só deve ser rodado pelo Diego "
                             "(adicione --i-am-diego). O loop autônomo NÃO roda isto.")
        cmd_final(args.config, args.data_root)
    elif args.config:
        cmd_run(args.config, args.repeats, args.data_root)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
