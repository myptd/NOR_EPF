"""
Feature group ablation study for day-ahead electricity price forecasting.

Measures how much each feature group contributes to LightGBM performance
using two complementary approaches:

1. Leave-one-group-out (LOGO): train without each feature group in turn.
   Δ MAE = MAE_without_group − MAE_full  (positive = group is useful)

2. Group-only model: train using *only* that feature group + price lags
   (price lags always included as a baseline; removing them would make
   the task trivially hard for all groups).

Feature groups:
  lags        — autoregressive price lags + rolling stats
  calendar    — hour_sin/cos, dow_sin/cos, month_sin/cos
  weather     — wx_* columns
  reservoir   — res_* columns
  commodities — com_* columns
  load_wsf    — load_forecast_mw, wsf_* wind/solar forecasts

Run:
    python models/ablation.py [--data data/cleaned/NO1_hourly.parquet]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.utils import (
    load_zone, build_feature_matrix,
    train_val_test_split, evaluate,
)
from models.advanced import LGBM_PARAMS, fit_lgbm

WEIGHTS_DIR = Path("model_weights")

TARGET = "price_eur_mwh"

# ── Feature group definitions ────────────────────────────────────────────────

def classify_features(columns: pd.Index) -> dict[str, list[str]]:
    """Map each feature column to its group."""
    groups: dict[str, list[str]] = {
        "lags":        [],
        "calendar":    [],
        "weather":     [],
        "reservoir":   [],
        "commodities": [],
        "load_wsf":    [],
    }
    for c in columns:
        if f"{TARGET}_lag" in c or f"{TARGET}_roll" in c:
            groups["lags"].append(c)
        elif c.startswith("wx_"):
            groups["weather"].append(c)
        elif c.startswith("res_"):
            groups["reservoir"].append(c)
        elif c.startswith("com_"):
            groups["commodities"].append(c)
        elif c in ("load_forecast_mw",) or c.startswith("wsf_"):
            groups["load_wsf"].append(c)
        elif c.endswith("_sin") or c.endswith("_cos"):
            groups["calendar"].append(c)
    return groups


# ── LightGBM fit helper (no early stopping for ablation speed) ──────────────

def fit_lgbm_fixed(
    X_tr: pd.DataFrame, X_val: pd.DataFrame, X_te: pd.DataFrame,
    y_tr: pd.Series,   y_val: pd.Series,   y_te: pd.Series,
    n_estimators: int,
) -> tuple[dict, lgb.LGBMRegressor]:
    params = {**LGBM_PARAMS, "n_estimators": n_estimators, "verbose": -1}
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr)
    return evaluate(y_te.values, model.predict(X_te), "LightGBM"), model


# ── Main ─────────────────────────────────────────────────────────────────────

def main(
    data_path:  str = "data/cleaned/NO1_hourly.parquet",
    val_start:  str = "2024-01-01",
    test_start: str = "2025-01-01",
) -> pd.DataFrame:
    zone = Path(data_path).stem.split("_")[0]
    print(f"\n=== Ablation: {zone} ===")
    print(f"\nLoading {data_path} ...")
    df = load_zone(data_path)

    print("Building feature matrix ...")
    X, y = build_feature_matrix(df, target=TARGET)
    X_tr, X_val, X_te, y_tr, y_val, y_te = train_val_test_split(
        X, y, val_start=val_start, test_start=test_start)
    print(f"  Train: {len(y_tr):,}  Val: {len(y_val):,}  Test: {len(y_te):,}  Features: {X.shape[1]}")

    groups = classify_features(X.columns)
    print("\n  Feature group sizes:")
    for g, cols in groups.items():
        print(f"    {g:<14} {len(cols):>3} cols")

    # ── 1. Full model (baseline) ─────────────────────────────────────────────
    print("\nFitting full model (calibrate n_estimators) ...")
    _, full_test, lgbm_model = fit_lgbm(X_tr, X_val, X_te, y_tr, y_val, y_te)
    n_est = lgbm_model.best_iteration_
    print(f"  Full model → MAE={full_test['MAE']:.3f}  RMSE={full_test['RMSE']:.3f}  "
          f"R²={full_test['R2']:.4f}  (n_est={n_est})")

    records = [dict(experiment="full_model", groups_used="all",
                    MAE=full_test["MAE"], RMSE=full_test["RMSE"],
                    R2=full_test["R2"], delta_MAE=0.0)]

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 2. Leave-one-group-out ───────────────────────────────────────────────
    print("\nLeave-one-group-out ablation:")
    for drop_group, drop_cols in groups.items():
        keep = [c for c in X.columns if c not in drop_cols]
        if not keep:
            continue
        res, model = fit_lgbm_fixed(
            X_tr[keep], X_val[keep], X_te[keep],
            y_tr, y_val, y_te, n_est)
        delta = round(res["MAE"] - full_test["MAE"], 3)
        sign  = "+" if delta >= 0 else ""
        print(f"  drop {drop_group:<14} → MAE={res['MAE']:.3f}  ΔMAE={sign}{delta:.3f}"
              f"  R²={res['R2']:.4f}")
        records.append(dict(
            experiment=f"drop_{drop_group}",
            groups_used=f"all minus {drop_group}",
            MAE=res["MAE"], RMSE=res["RMSE"], R2=res["R2"],
            delta_MAE=delta,
        ))
        w_path = WEIGHTS_DIR / f"ablation_drop_{drop_group}_{zone}.pkl"
        joblib.dump({"model": model, "features": keep}, w_path)

    # ── 3. Group-only models (lags always included) ──────────────────────────
    print("\nGroup-only models (+ price lags always):")
    lag_cols = groups["lags"]
    for g, g_cols in groups.items():
        use = list(set(lag_cols) | set(g_cols))
        use = [c for c in X.columns if c in use]  # preserve column order
        res, model = fit_lgbm_fixed(
            X_tr[use], X_val[use], X_te[use],
            y_tr, y_val, y_te, n_est)
        print(f"  only {g:<14} → MAE={res['MAE']:.3f}  RMSE={res['RMSE']:.3f}  R²={res['R2']:.4f}")
        records.append(dict(
            experiment=f"only_{g}",
            groups_used=f"lags + {g}" if g != "lags" else "lags",
            MAE=res["MAE"], RMSE=res["RMSE"], R2=res["R2"],
            delta_MAE=round(res["MAE"] - full_test["MAE"], 3),
        ))
        w_path = WEIGHTS_DIR / f"ablation_only_{g}_{zone}.pkl"
        joblib.dump({"model": model, "features": use}, w_path)
        print(f"    Saved {w_path}")

    result_df = pd.DataFrame(records).set_index("experiment")

    print("\n" + "=" * 65)
    print("Ablation summary (test set 2025)")
    print("=" * 65)
    print(result_df.to_string())
    print("=" * 65 + "\n")

    out = Path(f"model_results/ablation_{zone}.csv")
    result_df.to_csv(out)
    print(f"  Results saved → {out}")
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default="data/cleaned/NO1_hourly.parquet")
    parser.add_argument("--zones",       nargs="+", default=None,
                        help="List of zones to run (overrides --data)")
    parser.add_argument("--val-start",   default="2024-01-01")
    parser.add_argument("--test-start",  default="2025-01-01")
    args = parser.parse_args()

    if args.zones:
        for z in args.zones:
            main(f"data/cleaned/{z}_hourly.parquet",
                 args.val_start, args.test_start)
    else:
        main(args.data, args.val_start, args.test_start)
