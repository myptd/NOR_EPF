"""
Run all models (baseline + advanced) on one or more zone files and print a
combined comparison table.

Usage:
    python models/run_all.py
    python models/run_all.py --data data/cleaned/NO1_hourly.parquet
    python models/run_all.py --zones NO1 NO2 NO3    # shorthand: looks in data/cleaned/
    python models/run_all.py --val-start 2024-01-01 --test-start 2025-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.utils import (
    load_zone, build_feature_matrix,
    train_val_test_split, evaluate, print_results, dm_table, save_results,
)
import joblib
from models.utils import set_seed, SEED
from models.baseline import naive_lag, linear_arx
from models.advanced import fit_lgbm, fit_xgb

TARGET = "price_eur_mwh"


def run_zone(data_path: str,
             val_start:  str = "2024-01-01",
             test_start: str = "2025-01-01") -> tuple[list[dict], list[dict]]:
    set_seed()
    zone = Path(data_path).stem.replace("_hourly", "")
    print(f"\n{'=' * 60}")
    print(f"  Zone file: {data_path}")
    print(f"{'=' * 60}")

    df = load_zone(data_path)
    print(f"  Rows: {len(df):,}  |  Cols: {df.shape[1]}"
          f"  |  {df.index.min().date()} → {df.index.max().date()}")

    val_results:  list[dict] = []
    test_results: list[dict] = []

    # ---- Naive baselines ---------------------------------------------------
    for lag in (24, 168):
        v, t = naive_lag(df, lag=lag, val_start=val_start, test_start=test_start)
        v["zone"] = zone; t["zone"] = zone
        val_results.append(v); test_results.append(t)

    # ---- Feature matrix for regression / tree models -----------------------
    X, y = build_feature_matrix(df, target=TARGET)
    X_tr, X_val, X_te, y_tr, y_val, y_te = train_val_test_split(
        X, y, val_start=val_start, test_start=test_start)
    print(f"  Train: {len(y_tr):,}  Val: {len(y_val):,}  Test: {len(y_te):,}  Features: {X_tr.shape[1]}")

    # Linear ARX
    v, t = linear_arx(X_tr, X_val, X_te, y_tr, y_val, y_te)
    v["zone"] = zone; t["zone"] = zone
    val_results.append(v); test_results.append(t)

    # LightGBM
    print("  Fitting LightGBM ...")
    lv, lt, lgbm_model = fit_lgbm(X_tr, X_val, X_te, y_tr, y_val, y_te)
    print(f"    Best iteration: {lgbm_model.best_iteration_}")
    lv["zone"] = zone; lt["zone"] = zone
    val_results.append(lv); test_results.append(lt)
    Path("model_weights").mkdir(parents=True, exist_ok=True)
    joblib.dump(lgbm_model, f"model_weights/lgbm_{zone}.pkl")
    print(f"    Saved → model_weights/lgbm_{zone}.pkl")

    # XGBoost
    print("  Fitting XGBoost ...")
    xv, xt, xgb_model = fit_xgb(X_tr, X_val, X_te, y_tr, y_val, y_te)
    print(f"    Best iteration: {xgb_model.best_iteration}")
    xv["zone"] = zone; xt["zone"] = zone
    val_results.append(xv); test_results.append(xt)
    xgb_model.save_model(f"model_weights/xgb_{zone}.json")
    print(f"    Saved → model_weights/xgb_{zone}.json")

    print("\n  -- Validation (2024) --")
    print_results(val_results, benchmark="Naive-24h")
    print("  -- Test (2025) --")
    print_results(test_results, benchmark="Naive-24h")
    save_results(val_results,  f"{zone}_val")
    save_results(test_results, f"{zone}_test")
    return val_results, test_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all electricity price models")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--data", nargs="+",
        default=["data/cleaned/NO1_hourly.parquet"],
        help="One or more parquet file paths",
    )
    group.add_argument(
        "--zones", nargs="+",
        metavar="NO1",
        help="Zone names (shorthand for data/cleaned/<ZONE>_hourly.parquet)",
    )
    parser.add_argument("--val-start",  default="2024-01-01")
    parser.add_argument("--test-start", default="2025-01-01")
    args = parser.parse_args()

    if args.zones:
        paths = [f"data/cleaned/{z}_hourly.parquet" for z in args.zones]
    else:
        paths = args.data

    all_val:  list[dict] = []
    all_test: list[dict] = []
    for path in paths:
        v, t = run_zone(path, val_start=args.val_start, test_start=args.test_start)
        all_val.extend(v); all_test.extend(t)

    if len(paths) > 1:
        for label, results in [("VALIDATION", all_val), ("TEST", all_test)]:
            print("\n" + "=" * 70)
            print(f"COMBINED {label} RESULTS — all zones")
            print("=" * 70)
            df = pd.DataFrame(results).set_index(["zone", "model"])
            print(df.to_string())
        print("=" * 70 + "\n")
        save_results(all_val,  "all_zones_val")
        save_results(all_test, "all_zones_test")


if __name__ == "__main__":
    main()
