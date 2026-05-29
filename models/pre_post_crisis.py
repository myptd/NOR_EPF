"""
models/pre_post_crisis.py
=========================
Pre-crisis vs post-crisis training-window experiment for LightGBM.

Trains LightGBM under two additional windows while keeping val (2024)
and test (2025) fixed.  The full-window results (2019--2023) are NOT
re-run; they are read directly from the model_results/ CSVs produced
by step 1a.

Training windows
----------------
  pre_crisis  : 2019-01-01 to 2021-12-31
  post_crisis : 2022-01-01 to 2023-12-31
  full        : read from model_results/{ZONE}_test.csv  (step 1a output)

Val  (early stopping) : 2024  — fixed
Test (evaluation)     : 2025  — fixed

Outputs written to model_results/
  {ZONE}_test_pre_crisis.csv
  {ZONE}_test_post_crisis.csv

Run
---
  python models/pre_post_crisis.py [--zones NO1 NO2 NO3 NO4 NO5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.utils import (
    load_zone, build_feature_matrix,
    train_val_test_split, evaluate, save_results, set_seed,
)
from models.advanced import LGBM_PARAMS

TARGET     = "price_eur_mwh"
VAL_START  = "2024-01-01"
TEST_START = "2025-01-01"
TZ         = "Europe/Oslo"

WINDOWS = {
    "pre_crisis":  {"train_end":   "2022-01-01", "train_start": None},
    "post_crisis": {"train_start": "2022-01-01", "train_end":   None},
}


def fit_window(
    X: pd.DataFrame,
    y: pd.Series,
    train_start: str | None,
    train_end:   str | None,
    label:       str,
) -> dict:
    v = pd.Timestamp(VAL_START,  tz=TZ)
    t = pd.Timestamp(TEST_START, tz=TZ)

    X_val = X[(X.index >= v) & (X.index < t)]
    X_te  = X[X.index >= t]
    y_val = y[(y.index >= v) & (y.index < t)]
    y_te  = y[y.index >= t]

    # Build training slice
    mask = X.index < v           # everything before val by default
    if train_start is not None:
        mask = mask & (X.index >= pd.Timestamp(train_start, tz=TZ))
    if train_end is not None:
        mask = mask & (X.index <  pd.Timestamp(train_end,   tz=TZ))
    X_tr, y_tr = X[mask], y[mask]

    print(f"  [{label}] train={len(y_tr):,}  val={len(y_val):,}  test={len(y_te):,}")

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )
    print(f"  [{label}] best_iteration={model.best_iteration_}")

    metrics = evaluate(y_te.values, model.predict(X_te), label="LightGBM")
    metrics.pop("_errors", None)
    return metrics


def run_zone(data_path: str) -> None:
    zone = Path(data_path).stem.split("_")[0]
    print(f"\n{'=' * 60}\n  Zone: {zone}\n{'=' * 60}")

    df = load_zone(data_path)
    X, y = build_feature_matrix(df, target=TARGET)
    print(f"  Features: {X.shape[1]}")

    for window_name, bounds in WINDOWS.items():
        metrics = fit_window(X, y, bounds["train_start"], bounds["train_end"], window_name)
        metrics["zone"] = zone
        save_results([metrics], f"{zone}_test_{window_name}")
        print(f"  Saved → model_results/{zone}_test_{window_name}.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre/post-crisis LightGBM training-window experiment"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--data",  nargs="+", help="Explicit parquet file paths")
    group.add_argument(
        "--zones", nargs="+",
        default=["NO1", "NO2", "NO3", "NO4", "NO5"],
    )
    args = parser.parse_args()
    set_seed()

    paths = args.data if args.data else [
        f"data/cleaned/{z}_hourly.parquet" for z in args.zones
    ]
    for path in paths:
        if not Path(path).exists():
            print(f"WARNING: {path} not found — skipping.")
            continue
        run_zone(path)


if __name__ == "__main__":
    main()
