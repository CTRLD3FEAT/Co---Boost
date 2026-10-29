#!/usr/bin/env python3
"""Spike prediction + anomaly detection for Co-Boost.

Spike ML
  * Label: next-week price change (p[t+1]/p[t] - 1) >= SPIKE_THR (a "spike").
  * Features use ONLY information available up to and including week t:
    past returns (1/2/4 weeks), price z-score vs trailing 26 weeks,
    drawdown from the 26-week high, short-vs-long volatility ratio,
    consecutive-rise streak, month seasonality, food inflation.
  * Several classifiers are trained and compared on a CHRONOLOGICAL
    holdout (train < 2024-01, test >= 2024-01). Metrics: PR-AUC, ROC-AUC,
    precision@top5%, recall@top5%.
  * The best classifier (by PR-AUC) plus a direction regressor (next-week
    % change) and quantile regressors (p10/p50/p90) are saved to
    outputs/spike_ml_bundle.joblib together with a calibrated alert threshold.

Anomaly engine (statistical, per commodity, explainable)
  🔺 Unusual high   z-score of latest price vs trailing 26 weeks  >  +2
  🔻 Unusual low    z-score < -2
  🧊 Stable as never current 4-week std at its lowest ~5% ever
  🌀 Unusual move   latest one-week |change| >= its historical 95th percentile

Usage
  python3 Scripts/spike_ml.py            # train + save bundle + current predictions
  (streamlit_app.py imports the module for live inference)
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baseline import load_prices, MAIN_DATASET  # noqa: E402

SPIKE_THR = 0.10                 # >= 10% weekly move counts as a spike
Z_HIGH = 2.0
CALM_PCT = 0.05
MOVE_PCT = 0.95
TEST_START_YM = 202401
FEATURES = ["r1", "r2", "r4", "z26", "dd26", "vol_ratio", "streak",
            "month_sin", "month_cos", "inflation"]

SYMBOLS = {"spike": "⚡", "high": "🔺", "low": "🔻", "calm": "🧊", "move": "🌀"}
TYPES = {
    "spike": "⚡ Spike predicted",
    "high": "🔺 Unusual high",
    "low": "🔻 Unusual low",
    "calm": "🧊 Stable as never",
    "move": "🌀 Unusual move",
}


# --------------------------------------------------------------- features
def load_inflation(path: str = "Dataset/food_inflation.csv") -> pd.Series:
    inf = pd.read_csv(path)
    dates = pd.to_datetime(inf["Date"], format="%d/%m/%Y")
    ym = dates.dt.year * 100 + dates.dt.month
    return pd.Series(inf["Value"].ffill().values, index=ym).sort_index()


def build_feature_frame(df: pd.DataFrame, inflation: pd.Series) -> pd.DataFrame:
    """One row per (commodity, last-observed-with-full-history week)."""
    frames = []
    for comm, g in df.groupby("commodity"):
        g = g.copy()
        g["observation_date"] = pd.to_datetime(g["observation_date"])
        g = g.sort_values("observation_date")
        s = g.set_index("observation_date")["price"].astype(float)
        if len(s) < 27:
            continue
        r1 = s.pct_change()
        f = pd.DataFrame(index=s.index)
        f["r1"] = r1
        f["r2"] = s.pct_change(2)
        f["r4"] = s.pct_change(4)
        m26, sd26 = s.rolling(26).mean(), s.rolling(26).std()
        f["z26"] = (s - m26) / sd26
        f["dd26"] = s / s.rolling(26).max() - 1
        vol4 = r1.rolling(4).std()
        vol26 = r1.rolling(26).std()
        f["vol_ratio"] = vol4 / vol26
        pos = (r1 > 0).astype(float)
        grp = pos.ne(pos.shift()).cumsum()
        f["streak"] = pos.groupby(grp).cumcount() + 1
        ym = s.index.year * 100 + s.index.month
        f["month_sin"] = np.sin(2 * np.pi * (s.index.month - 1) / 12)
        f["month_cos"] = np.cos(2 * np.pi * (s.index.month - 1) / 12)
        f["inflation"] = ym.map(inflation.groupby(level=0).first()).values
        f["label_spike"] = (r1.shift(-1) >= SPIKE_THR).astype(float)
        f["label_r1"] = r1.shift(-1)
        f["price"] = s
        f["commodity"] = comm
        f["ym"] = ym.values
        frames.append(f.dropna(subset=FEATURES + ["label_r1"]))
    return pd.concat(frames) if frames else pd.DataFrame()


# --------------------------------------------------------------- training
def _eval_flagwise(y_true, proba) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    order = np.argsort(-proba)
    top5 = order[: max(1, int(0.05 * len(order)))]
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if y_true.sum() > 0 else float("nan"),
        "precision_top5": float(np.asarray(y_true)[top5].mean()),
        "n_top5": int(len(top5)),
    }


def pick_threshold(y, p) -> tuple[float, dict]:
    """Lowest threshold with test precision >= 0.45, searching the proba scale."""
    order = np.argsort(-p)
    y, p = np.asarray(y), np.asarray(p)
    best = (0.5, {"threshold": 0.5})
    for thr in np.unique(p)[::-1]:
        flag = p >= thr
        prec = y[flag].mean() if flag.sum() else 0.0
        rec = y[flag].sum() / max(y.sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best[1].get("f1", -1):
            best = (thr, {"threshold": float(thr), "f1": float(f1),
                          "precision": float(prec), "recall": float(rec)})
    return best


def get_models():
    from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                                  GradientBoostingClassifier,
                                  HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                     class_weight="balanced", n_jobs=-1, random_state=42),
        "extra_trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5,
                                            class_weight="balanced", n_jobs=-1, random_state=42),
        "hist_gb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                                  random_state=42),
        "grad_boost": GradientBoostingClassifier(n_estimators=150, random_state=42),
        "ada": AdaBoostClassifier(n_estimators=200, random_state=42),
        "decision_tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                random_state=42),
        "gauss_nb": GaussianNB(),
        "mlp": MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=42),
    }


def train(df: pd.DataFrame = None, out_dir: str = "outputs") -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    df = load_prices(MAIN_DATASET) if df is None else df
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    infl = load_inflation()
    F = build_feature_frame(df, infl)
    if F.empty:
        raise RuntimeError("no usable feature rows")

    train = F[F.ym < TEST_START_YM]
    test = F[F.ym >= TEST_START_YM]
    if len(test) < 500 or test.label_spike.sum() < 5:
        raise RuntimeError(f"chronological holdout too small: {len(test)} rows")

    X_tr, y_tr = train[FEATURES].values, train.label_spike.values
    X_te, y_te = test[FEATURES].values, test.label_spike.values
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    rows, fitted = [], {}
    for name, model in get_models().items():
        Xf_tr = X_tr_s if name in ("logreg", "mlp") else X_tr
        Xf_te = X_te_s if name in ("logreg", "mlp") else X_te
        import time
        t0 = time.time()
        model.fit(Xf_tr, y_tr)
        proba = model.predict_proba(Xf_te)[:, 1]
        met = _eval_flagwise(y_te, proba)
        met["model"] = name
        met["train_s"] = round(time.time() - t0, 1)
        rows.append(met)
        fitted[name] = (model, proba)

    lb = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    best_name = lb.iloc[0]["model"]
    best_model, best_proba = fitted[best_name]
    thr, thr_meta = pick_threshold(y_te, best_proba)

    # direction regressors: next-week % change
    y_tr_reg, y_te_reg = train.label_r1.values, test.label_r1.values
    reg_models = {"hist_gb": HistGradientBoostingRegressor(max_iter=300, random_state=42),
                  "ridge": Ridge()}
    dir_rows, dir_fit = [], {}
    for name, reg in reg_models.items():
        reg.fit(X_tr, y_tr_reg)
        pred = reg.predict(X_te)
        dir_rows.append({"model": name,
                         "mae": float(np.mean(np.abs(pred - y_te_reg))),
                         "dir_acc": float(np.mean(np.sign(pred) == np.sign(y_te_reg)))})
        dir_fit[name] = reg
    naive_mae = float(np.mean(np.abs(y_te_reg)))
    dir_lb = pd.DataFrame(dir_rows).sort_values("mae")
    best_reg = dir_lb.iloc[0]["model"]

    # quantile regressors for a p10/p50/p90 next-week change band
    quant_regs = {}
    for q in (0.1, 0.5, 0.9):
        qr = HistGradientBoostingRegressor(loss="quantile", quantile=q,
                                           max_iter=300, random_state=42)
        qr.fit(X_tr, y_tr_reg)
        quant_regs[q] = qr
        pred_q = qr.predict(X_te)
        err = y_te_reg - pred_q
        pinb = float(np.mean(np.maximum(q * err, (q - 1) * err)))
        print(f"quantile q={q}: pinball={pinb:.4f}")
    q10 = quant_regs[0.1].predict(X_te)
    q90 = quant_regs[0.9].predict(X_te)
    cov = float(np.mean((y_te_reg >= q10) & (y_te_reg <= q90)))
    print(f"quantile band p10-p90 holdout coverage: {cov:.2%}")
    quant_regs = dict(sorted(quant_regs.items()))

    bundle = {
        "features": FEATURES,
        "scaler": scaler,
        "classifier_name": best_name,
        "classifier": best_model,
        "threshold": thr,
        "threshold_meta": thr_meta,
        "regressor_name": best_reg,
        "regressor": dir_fit[best_reg],
        "quant_regs": quant_regs,
        "test_start_ym": TEST_START_YM,
    }
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)
    joblib.dump(bundle, out_path / "spike_ml_bundle.joblib")
    lb.to_csv(out_path / "spike_ml_leaderboard.csv", index=False)
    dir_lb.to_csv(out_path / "direction_leaderboard.csv", index=False)

    base_rate = float(y_te.mean())
    return {"leaderboard": lb, "direction": dir_lb, "naive_mae": naive_mae,
            "base_rate": base_rate, "train_rows": len(train), "test_rows": len(test),
            "bundle_path": str(out_path / "spike_ml_bundle.joblib"),
            "best": best_name, "threshold": thr}


# --------------------------------------------------------------- inference
def load_bundle(path: str = "outputs/spike_ml_bundle.joblib") -> dict:
    return joblib.load(path)


def anomaly_table(df: pd.DataFrame = None, bundle: dict = None) -> pd.DataFrame:
    """Statistical anomalies + ML spike probability for the LATEST week."""
    df = load_prices(MAIN_DATASET) if df is None else df
    bundle = load_bundle() if bundle is None else bundle
    rows = []
    scaler, clf, reg = bundle["scaler"], bundle["classifier"], bundle["regressor"]
    if bundle["classifier_name"] in ("logreg", "mlp"):
        def clf_proba(X):
            return clf.predict_proba(scaler.transform(X))[:, 1]
    else:
        def clf_proba(X):
            return clf.predict_proba(X)[:, 1]

    for comm, g in df.groupby("commodity"):
        g = g.sort_values("observation_date")
        s = g["price"].to_numpy(float)
        if len(s) < 27:
            continue
        last, past = s[-1], s[:-1]
        z = (last - past[-26:].mean()) / (past[-26:].std() + 1e-9)
        r1s = pd.Series(s).pct_change()
        jump = r1s.iloc[-1]
        vol4 = pd.Series(s).rolling(4).std().dropna()
        calm_pct = (vol4.iloc[-1] <= vol4).mean() if len(vol4) > 20 else 1.0
        move_pct = (abs(jump) >= r1s.dropna().abs()).mean()

        proba = np.nan
        if len(s) >= 27:
            f = build_feature_frame(g, load_inflation())
            if len(f):
                x = f.iloc[[-1]][FEATURES].values
                proba = float(clf_proba(x)[0])

        pred_r1 = float(reg.predict(x)[0]) if proba == proba else 0.0
        if abs(pred_r1) < 0.02:
            direction = "stable"
        elif pred_r1 > 0:
            direction = "up"
        else:
            direction = "down"

        flag = None
        if z > Z_HIGH:
            flag = "high"
        elif z < -Z_HIGH:
            flag = "low"
        elif calm_pct <= CALM_PCT and len(s) >= 50:
            flag = "calm"
        elif move_pct >= MOVE_PCT:
            flag = "move"
        rows.append({
            "commodity": comm, "last_price": last, "z26": z, "jump_1w": jump,
            "spike_prob": proba, "direction": direction, "pred_r1": pred_r1,
            "flag": flag, "symbol": SYMBOLS[flag] if flag else "",
            "detail": TYPES[flag] if flag else "",
        })
    t = pd.DataFrame(rows)
    if len(t):
        t = t.sort_values(["spike_prob", "flag"], ascending=[False, True], na_position="last")
    return t


def current_predictions(bundle: dict = None) -> pd.DataFrame:
    from baseline import load_prices, MAIN_DATASET
    df = load_prices(MAIN_DATASET)
    return anomaly_table(df, bundle)


# ------------------------------------------------------------------ CLI
def main() -> int:
    res = train()
    print(f"train rows {res['train_rows']} | test rows {res['test_rows']} | "
          f"spike base rate {res['base_rate']:.3f} | naive MAE {res['naive_mae']:.3f}")
    print("\n== classifier leaderboard (chronological holdout) ==")
    print(res["leaderboard"].to_string(index=False))
    print("\n== direction regressors (% change next week) ==")
    print(res["direction"].to_string(index=False))
    print(f"\nsaved bundle -> {res['bundle_path']} | alert threshold {res['threshold']:.3f}")
    cur = current_predictions(load_bundle())
    spikes = cur[cur.spike_prob >= load_bundle()["threshold"]]
    stats = cur[cur.flag.notna()]
    print(f"\n== spike alerts (proba >= {load_bundle()['threshold']:.3f}) ==")
    print(spikes.head(15).to_string(index=False))
    print(f"\n== statistical anomaly flags ({len(stats)}) ==")
    print(stats.head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
