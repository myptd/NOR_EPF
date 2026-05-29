"""
Baseline models for day-ahead electricity price forecasting.

Three baselines of increasing sophistication:

1. Naive-24h       — predict hour t using price at hour t-24  (same hour yesterday)
2. Naive-168h      — predict hour t using price at hour t-168 (same hour last week)
3. Linear (ARX)    — OLS regression on lag features + calendar + external regressors
                     (load forecast, weather, commodities, reservoir)

Run:
    python models/baseline.py [--data data/cleaned/NO1_hourly.parquet]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.utils import (
    load_zone, add_lag_features, build_feature_matrix,
    train_test_split_temporal, train_val_test_split, evaluate, print_results,
    SEED, set_seed, save_results,
)

TARGET = "price_eur_mwh"


# ---------------------------------------------------------------------------
# Naive baselines
# ---------------------------------------------------------------------------

def naive_lag(df: pd.DataFrame, lag: int,
              val_start: str = "2024-01-01",
              test_start: str = "2025-01-01") -> tuple[dict, dict]:
    """Return (val_metrics, test_metrics) for a naïve lag-k predictor."""
    label  = f"Naive-{lag}h"
    series = df[TARGET].copy()
    v = pd.Timestamp(val_start,  tz="Europe/Oslo")
    t = pd.Timestamp(test_start, tz="Europe/Oslo")
    val  = series[(series.index >= v) & (series.index < t)]
    test = series[series.index >= t]
    pred_val  = series.shift(lag)[val.index]
    pred_test = series.shift(lag)[test.index]
    ok_val  = val.notna()  & pred_val.notna()
    ok_test = test.notna() & pred_test.notna()
    return (evaluate(val[ok_val],   pred_val[ok_val],   label),
            evaluate(test[ok_test], pred_test[ok_test], label))


# ---------------------------------------------------------------------------
# Linear ARX
# ---------------------------------------------------------------------------

def linear_arx(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_val:   pd.Series,
    y_test:  pd.Series,
) -> tuple[dict, dict]:
    """
    Ridge regression with standard scaling.
    Trains on train, selects alpha on val, reports val + test metrics.
    """
    from sklearn.model_selection import PredefinedSplit
    import numpy as np

    # Combine train+val for PredefinedSplit so Ridge sees the same scaler
    X_tv = pd.concat([X_train, X_val])
    y_tv = pd.concat([y_train, y_val])
    # -1 = train fold, 0 = validation fold
    split_idx = np.concatenate([
        np.full(len(X_train), -1),
        np.zeros(len(X_val), dtype=int),
    ])
    ps = PredefinedSplit(split_idx)

    from sklearn.model_selection import GridSearchCV
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge()),
    ])
    grid = GridSearchCV(pipe, {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]},
                        cv=ps, scoring="neg_mean_absolute_error", refit=True, n_jobs=1)
    grid.fit(X_tv, y_tv)

    val_pred  = grid.predict(X_val)
    test_pred = grid.predict(X_test)
    return (evaluate(y_val,  val_pred,  "Linear-ARX (Ridge)"),
            evaluate(y_test, test_pred, "Linear-ARX (Ridge)"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_path: str = "data/cleaned/NO1_hourly.parquet",
         val_start:  str = "2024-01-01",
         test_start: str = "2025-01-01") -> None:
    set_seed()
    zone = Path(data_path).stem.replace("_hourly", "")
    print(f"\nLoading {data_path} ...")
    df = load_zone(data_path)
    print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    print(f"  Date range: {df.index.min().date()} → {df.index.max().date()}")

    val_results:  list[dict] = []
    test_results: list[dict] = []

    # ----- Naive baselines --------
    print("\nFitting naive baselines ...")
    for lag in (24, 168):
        v, t = naive_lag(df, lag=lag, val_start=val_start, test_start=test_start)
        val_results.append(v); test_results.append(t)

    # ----- Linear ARX -------
    print("Building feature matrix for ARX ...")
    X, y = build_feature_matrix(df, target=TARGET)
    X_tr, X_val, X_te, y_tr, y_val, y_te = train_val_test_split(
        X, y, val_start=val_start, test_start=test_start)
    print(f"  Train: {len(y_tr):,}  Val: {len(y_val):,}  Test: {len(y_te):,}  Features: {X_tr.shape[1]}")

    print("Fitting Linear-ARX (Ridge, alpha tuned on val) ...")
    v, t = linear_arx(X_tr, X_val, X_te, y_tr, y_val, y_te)
    val_results.append(v); test_results.append(t)

    print("\n--- Validation set ---")
    print_results(val_results)
    print("--- Test set ---")
    print_results(test_results)
    save_results(test_results, f"{zone}_baseline_test")
    save_results(val_results, f"{zone}_baseline_val")
    return val_results, test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline electricity price models")
    parser.add_argument("--data", default="data/cleaned/NO1_hourly.parquet")
    parser.add_argument("--val-start",  default="2024-01-01")
    parser.add_argument("--test-start", default="2025-01-01")
    args = parser.parse_args()
    main(args.data, args.val_start, args.test_start)
