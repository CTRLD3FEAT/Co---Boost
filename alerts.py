"""Alert logic - ML spike probabilities (Option 1).

Replaces fixed-threshold binary alerts with calibrated spike probabilities
from outputs/spike_ml_bundle.joblib, trained on 2019-2023 DCS data and
validated on the chronological 2024-2026 holdout (PR-AUC ~0.20 vs 0.05
random base rate). No price-level forecasting here - naive still wins that
at 2.7% median MAPE. This layer only answers: "which commodities are most
likely to spike next week?"
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

MIN_HISTORY = 26          # below this: "not enough data" monitoring badge
ML_HISTORY = 27           # model needs 26-week trailing stats + 1 lag
BUNDLE_PATH = Path("outputs/spike_ml_bundle.joblib")
FEATURES = ["r1", "r2", "r4", "z26", "dd26", "vol_ratio", "streak",
            "month_sin", "month_cos", "inflation"]
# Confidence tiers anchored to chronological-holdout precision
# (of weeks scored above X, % that really spiked; base rate ~5.4%):
#   >= 0.40 -> precision ~34%   High
#   0.20-0.40 -> precision ~22% Medium
HIGH_PROBA = 0.40
MED_PROBA = 0.20


def load_inflation(path: str = "Dataset/food_inflation.csv") -> pd.Series:
    inf = pd.read_csv(path)
    dates = pd.to_datetime(inf["Date"], format="%d/%m/%Y")
    ym = dates.dt.year * 100 + dates.dt.month
    return pd.Series(inf["Value"].ffill().values, index=ym).sort_index()


def load_bundle(path: str = BUNDLE_PATH) -> dict:
    return joblib.load(path)


def _features_last_row(s: pd.Series, infl_map: pd.Series) -> np.ndarray | None:
    """Feature vector for the most recent week of one commodity, using only
    information available up to and including that week."""
    if len(s) < ML_HISTORY:
        return None
    r1 = s.pct_change()
    m26, sd26 = s.rolling(26).mean(), s.rolling(26).std()
    vol4, vol26 = r1.rolling(4).std(), r1.rolling(26).std()
    pos = (r1 > 0).astype(float)
    grp = pos.ne(pos.shift()).cumsum()
    streak = pos.groupby(grp).cumcount() + 1
    last_ts = s.index[-1]
    ym = last_ts.year * 100 + last_ts.month
    infl_v = infl_map.get(ym, np.nan)
    if pd.isna(infl_v) and len(infl_map):
        prev = infl_map[infl_map.index <= ym]
        infl_v = prev.iloc[-1] if len(prev) else infl_map.iloc[0]
    return np.array([[
        float(r1.iloc[-1]),
        float(s.pct_change(2).iloc[-1]),
        float(s.pct_change(4).iloc[-1]),
        float((s.iloc[-1] - m26.iloc[-1]) / (sd26.iloc[-1] + 1e-9)),
        float(s.iloc[-1] / s.rolling(26).max().iloc[-1] - 1),
        float(vol4.iloc[-1] / (vol26.iloc[-1] + 1e-9)),
        float(streak.iloc[-1]),
        float(np.sin(2 * np.pi * (last_ts.month - 1) / 12)),
        float(np.cos(2 * np.pi * (last_ts.month - 1) / 12)),
        float(infl_v),
    ]])


def _confidence(proba: float) -> str:
    if proba >= HIGH_PROBA:
        return "High"
    if proba >= MED_PROBA:
        return "Medium"
    return "Low"


def quantile_pred(x: np.ndarray, bundle: dict) -> dict:
    """p10/p50/p90 next-week % change from bundle quantile regressors
    (trained on unscaled X, chronological holdout). Missing bundle key ->
    NaNs (defensive for older bundles)."""
    regs = bundle.get("quant_regs") or {}
    return {q: float(reg.predict(x)[0]) for q, reg in regs.items()} \
        if regs else {q: float("nan") for q in (0.1, 0.5, 0.9)}


def compute_spike_alerts(dy: pd.DataFrame, latest: pd.Timestamp):
    """ML spike alerts for the latest week.

    Returns (alerts_df, bundle). One row per commodity with >= ML_HISTORY
    weeks of history. alert = spike_prob >= bundle threshold (F1-optimal on
    the chronological holdout).
    """
    bundle = load_bundle()
    clf, scaler, reg = bundle["classifier"], bundle["scaler"], bundle["regressor"]
    infl_map = load_inflation().groupby(level=0).first()
    use_scaler = bundle["classifier_name"] in ("logreg", "mlp")
    thr = float(bundle["threshold"])

    rows = []
    for comm, g in dy.groupby("commodity"):
        g = g.sort_values("week")
        s = g.set_index("week")["price"].astype(float)
        weeks = int(len(s))
        if weeks < ML_HISTORY:
            continue
        x = _features_last_row(s, infl_map)
        X = scaler.transform(x) if use_scaler else x
        proba = float(clf.predict_proba(X)[0, 1])
        pred_r1 = float(reg.predict(x)[0])
        qpred = quantile_pred(x, bundle)
        sigma_pct = float(s.pct_change().rolling(26).std().iloc[-1])
        rows.append({
            "item": comm,
            "category": g["parent"].iat[0] if len(g) else "",
            "weeks": weeks,
            "last_price": float(s.iloc[-1]),
            "spike_prob": proba,
            "pred_r1": pred_r1,
            "pred_move_pct": pred_r1 * 100.0,
            "sigma_pct": sigma_pct,
            "alert": bool(proba >= thr),
            "score": proba,
            "threshold": thr,
            "confidence": _confidence(proba),
            "pred_q10": qpred[0.1],
            "pred_q50": qpred[0.5],
            "pred_q90": qpred[0.9],
        })
    t = pd.DataFrame(rows)
    if len(t):
        t = t.sort_values("score", ascending=False).reset_index(drop=True)
    return t, bundle


def monitoring_items(dy: pd.DataFrame, latest: pd.Timestamp) -> pd.DataFrame:
    """Commodities with < 26 weeks of history in total: visible, selectable,
    no predictive signals."""
    g = (dy.sort_values("week")
         .groupby("commodity")
         .agg(weeks=("price", "size"),
              last_price=("price", "last"),
              last_week=("week", "last"),
              category=("parent", lambda s: s.mode().iat[0])))
    g = g[g["weeks"] < MIN_HISTORY]
    return (g.reset_index()
             .rename(columns={"commodity": "item"})
             [["item", "category", "weeks", "last_price", "last_week"]])


def proba_label(p: float) -> str:
    # truncate so the shown % can never contradict its tier
    return f"{int(p * 100)}%"


def move_label(pct: float) -> str:
    return f"{pct:+.1f}%"
