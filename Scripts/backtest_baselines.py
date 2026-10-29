#!/usr/bin/env python3
"""Walk-forward (rolling-origin) backtest of baseline forecasters on
outputs/official_prices_wrp.csv - one step ahead (next observation, ~1 week).

Protocol
  * Series: price history per commodity, ordered by observation date
    (2019-12-21 -> 2026-10-07, ~4 weeks / month, occasional gaps).
  * For every observation t with at least MIN_HISTORY earlier observations,
    forecast price[t] using ONLY observations up to t-1.
  * No metric ever touches the future; the last real observation is never
    a training/fitting shortcut.

Models (match baseline.py definitions)
  naive_last        price[t] = last observed price
  residual_trend    linear trend over all history, extrapolated one week
                    (baseline.py residual_based forecast)
  median            mean-revert to the historical median
                    (baseline.py median_based forecast)
  seasonal_change   last price + mean of the last 3 week-on-week changes
                    (baseline.py seasonal_based forecast)

Metrics per commodity x model: MAE, RMSE, MAPE (n = points evaluated)

Outputs
  outputs/backtest_error_metrics.json   per-commodity metrics
  outputs/backtest_leaderboard.csv      aggregated leaderboard (+ win counts)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baseline import load_prices, MAIN_DATASET  # noqa: E402

MODELS = ["naive_last", "residual_trend", "median", "seasonal_change"]
MIN_HISTORY = 12          # at least 3 months of history before we score a forecast


def forecast(model: str, hist: np.ndarray) -> float:
    """One-step-ahead forecast from the history array (prices only)."""
    if model == "naive_last":
        return float(hist[-1])
    if model == "median":
        return float(np.median(hist))
    if model == "seasonal_change":
        if len(hist) < 4:
            return float(hist[-1])
        return float(hist[-1] + np.mean(np.diff(hist[-4:])))
    if model == "residual_trend":
        return float(np.polyval(np.polyfit(np.arange(len(hist)), hist, 1), len(hist)))
    raise ValueError(model)


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = actual - pred
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_err = np.abs(err / actual) * 100.0
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(pct_err)),
        "n": int(len(err)),
    }


def backtest_commodity(g: pd.DataFrame) -> dict[str, dict]:
    prices = g["price"].to_numpy(dtype=float)
    act_n, preds = {m: [] for m in MODELS}, {m: [] for m in MODELS}
    actual = []
    for i in range(MIN_HISTORY, len(prices)):
        hist = prices[:i]
        actual.append(prices[i])
        for m in MODELS:
            preds[m].append(forecast(m, hist))
    actual = np.asarray(actual)
    out = {}
    for m in MODELS:
        out[m] = metrics(actual, np.asarray(preds[m]))
    return out


def main() -> int:
    df = load_prices(MAIN_DATASET)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    results, leaderboard_rows = {}, []
    commodities = sorted(df["commodity"].dropna().unique())

    for comm in commodities:
        g = (df[df["commodity"] == comm]
             .dropna(subset=["price"])
             .sort_values("observation_date"))
        if len(g) <= MIN_HISTORY:
            continue
        scores = backtest_commodity(g)
        results[comm] = scores

    # aggregate
    win_counts = {m: 0 for m in MODELS}
    mape_all = {m: [] for m in MODELS}
    mae_all = {m: [] for m in MODELS}
    for comm, scores in results.items():
        best = min(scores, key=lambda m: scores[m]["mape"])
        win_counts[best] += 1
        for m in MODELS:
            mape_all[m].append(scores[m]["mape"])
            mae_all[m].append(scores[m]["mae"])

    agg = {}
    for m in MODELS:
        agg[m] = {
            "median_mape": float(np.median(mape_all[m])),
            "mean_mape": float(np.mean(mape_all[m])),
            "median_mae": float(np.median(mae_all[m])),
            "mean_mae": float(np.mean(mae_all[m])),
            "wins_by_mape": win_counts[m],
        }
        leaderboard_rows.append({
            "model": m,
            "best_per_commodity_wins": win_counts[m],
            "median_mape": agg[m]["median_mape"],
            "mean_mape": agg[m]["mean_mape"],
            "median_mae": agg[m]["median_mae"],
            "mean_mae": agg[m]["mean_mae"],
        })

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    payload = {
        "protocol": {
            "walk_forward": "one-step-ahead, expanding window",
            "min_history": MIN_HISTORY,
            "dataset": MAIN_DATASET,
            "commodities_scored": len(results),
        },
        "models": MODELS,
        "leaderboard": agg,
        "per_commodity": results,
    }
    (out_dir / "backtest_error_metrics.json").write_text(json.dumps(payload, indent=2))
    lb = pd.DataFrame(leaderboard_rows).sort_values("median_mape")
    lb.to_csv(out_dir / "backtest_leaderboard.csv", index=False)

    n_points = sum(s[m]["n"] for s in results.values() for m in [list(s)[0]])
    print(f"commodities scored: {len(results)} | forecast points per model: {n_points}")
    print(lb.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
