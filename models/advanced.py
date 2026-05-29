"""
Advanced models for day-ahead electricity price forecasting.

Two gradient-boosting models with the full feature set:

1. LightGBM  — fast GBDT, handles missing values natively
2. XGBoost   — GBDT with regularisation, complementary to LightGBM

Both use the same feature matrix built by models/utils.py:
  - Price lag / rolling features (1h … 168h)
  - Calendar cyclical encodings (hour_sin/cos, dow_sin/cos, month_sin/cos)
  - Weather (wx_*)
  - Reservoir (res_*)
  - Commodity prices & indicators (com_*)
  - Load forecast, wind/solar forecasts (wsf_*)

A simple feature-importance plot is saved to models/feature_importance.png.

Run:
    python models/advanced.py [--data data/cleaned/NO1_hourly.parquet]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.utils import (
    load_zone, build_feature_matrix,
    train_val_test_split, evaluate, print_results,
    SEED, set_seed, save_results,
)

TARGET = "price_eur_mwh"

# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

LGBM_PARAMS = dict(
    objective        = "regression_l1",   # MAE loss — robust to price spikes
    n_estimators     = 1000,
    learning_rate    = 0.05,
    num_leaves       = 63,
    min_child_samples= 20,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_lambda       = 1.0,
    random_state     = SEED,
    n_jobs           = -1,
    verbose          = -1,
)


def fit_lgbm(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_val:   pd.Series,
    y_test:  pd.Series,
) -> tuple[dict, dict, lgb.LGBMRegressor]:
    """Train on train, early-stop on val, evaluate on both val and test."""
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )
    val_pred  = model.predict(X_val)
    test_pred = model.predict(X_test)
    return (evaluate(y_val,  val_pred,  "LightGBM"),
            evaluate(y_test, test_pred, "LightGBM"),
            model)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

XGB_PARAMS = dict(
    objective        = "reg:absoluteerror",
    n_estimators     = 1000,
    learning_rate    = 0.05,
    max_depth        = 6,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_lambda       = 1.0,
    random_state     = SEED,
    n_jobs           = -1,
    tree_method      = "hist",
    early_stopping_rounds = 50,
)


def fit_xgb(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_val:   pd.Series,
    y_test:  pd.Series,
) -> tuple[dict, dict, xgb.XGBRegressor]:
    """Train on train, early-stop on val, evaluate on both val and test."""
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    val_pred  = model.predict(X_val)
    test_pred = model.predict(X_test)
    return (evaluate(y_val,  val_pred,  "XGBoost"),
            evaluate(y_test, test_pred, "XGBoost"),
            model)


# ---------------------------------------------------------------------------
# Feature importance plot
# ---------------------------------------------------------------------------

def save_importance(lgbm_model: lgb.LGBMRegressor, output_path: str = "models/feature_importance.png") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        feat_imp = pd.Series(
            lgbm_model.feature_importances_,
            index=lgbm_model.feature_name_,
        ).sort_values(ascending=False).head(30)

        fig, ax = plt.subplots(figsize=(8, 10))
        feat_imp.sort_values().plot.barh(ax=ax)
        ax.set_title("LightGBM — top 30 feature importances (gain)")
        ax.set_xlabel("Importance (split count)")
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        print(f"  Feature importance saved → {output_path}")
    except ImportError:
        print("  (matplotlib not installed — skipping importance plot)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_path: str = "data/cleaned/NO1_hourly.parquet",
         zone: str | None = None,
         val_start:  str = "2024-01-01",
         test_start: str = "2025-01-01") -> None:
    set_seed()
    if zone is None:
        zone = Path(data_path).stem.replace("_hourly", "")
    print(f"\nLoading {data_path} ...")
    df = load_zone(data_path)
    print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    print(f"  Date range: {df.index.min().date()} → {df.index.max().date()}")

    print("\nBuilding feature matrix ...")
    X, y = build_feature_matrix(df, target=TARGET)
    X_tr, X_val, X_te, y_tr, y_val, y_te = train_val_test_split(
        X, y, val_start=val_start, test_start=test_start)
    print(f"  Train: {len(y_tr):,}  Val: {len(y_val):,}  Test: {len(y_te):,}  Features: {X_tr.shape[1]}")

    val_results:  list[dict] = []
    test_results: list[dict] = []

    print("\nFitting LightGBM ...")
    lv, lt, lgbm_model = fit_lgbm(X_tr, X_val, X_te, y_tr, y_val, y_te)
    print(f"  Best iteration: {lgbm_model.best_iteration_}")
    val_results.append(lv); test_results.append(lt)

    print("\nFitting XGBoost ...")
    xv, xt, xgb_model = fit_xgb(X_tr, X_val, X_te, y_tr, y_val, y_te)
    print(f"  Best iteration: {xgb_model.best_iteration}")
    val_results.append(xv); test_results.append(xt)

    print("\n--- Validation set (2024) ---")
    print_results(val_results)
    print("--- Test set (2025) ---")
    print_results(test_results)

    save_importance(lgbm_model)

    # Save models
    Path("model_weights").mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(lgbm_model, f"model_weights/lgbm_{zone}.pkl")
    lgbm_model.booster_.save_model(f"model_weights/lgbm_{zone}.txt")
    print(f"  LightGBM saved → model_weights/lgbm_{zone}.pkl / .txt")
    xgb_model.save_model(f"model_weights/xgb_{zone}.json")
    print(f"  XGBoost saved → model_weights/xgb_{zone}.json")
    save_results(test_results, f"{zone}_test")
    save_results(val_results, f"{zone}_val")
    return val_results, test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced electricity price models")
    parser.add_argument("--data", default="data/cleaned/NO1_hourly.parquet")
    parser.add_argument("--val-start",  default="2024-01-01")
    parser.add_argument("--test-start", default="2025-01-01")
    args = parser.parse_args()
    main(args.data, args.val_start, args.test_start)
