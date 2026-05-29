"""
Walk-forward (rolling-origin) backtesting for day-ahead electricity price forecasting.

Instead of a single train/test split, the origin moves forward in weekly steps:
  - Initial training window: 2019-01-01 → 2023-12-31
  - Each step: extend training by 1 week, predict the next week
  - Collect per-week MAE/RMSE/sMAPE, then report mean ± std

This produces a *distribution* of errors rather than a single point estimate,
which is the standard evaluation protocol in the EPF literature:
  Lago et al. (2021), Marcjasz et al. (2020).

Models evaluated:
  - Naive-24h   (no fitting needed; always uses lag-24h)
  - LightGBM    (refit each step on expanded window)

LightGBM is refitted without early stopping (uses best_iteration from the
initial full fit as a fixed n_estimators) to keep runtime tractable.

Run:
    python models/walk_forward.py [--data data/cleaned/NO1_hourly.parquet]
    python models/walk_forward.py --step-weeks 4 --max-steps 26
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.utils import (
    load_zone, build_feature_matrix,
    evaluate, print_results, diebold_mariano,
)
from models.advanced import LGBM_PARAMS

TARGET = "price_eur_mwh"
LOCAL_TZ = "Europe/Oslo"


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    train_end:   str = "2023-12-31",
    test_start:  str = "2025-01-01",
    step_weeks:  int = 1,
    max_steps:   int | None = None,
    lgbm_n_estimators: int = 800,
) -> pd.DataFrame:
    """
    Rolling-origin backtest.

    For each step i:
      - train on all data up to (train_end + i * step_weeks weeks)
      - predict the following step_weeks * 168 hours
    Stops when the prediction window extends past test_start + all data,
    or after max_steps steps.

    Returns a DataFrame with one row per step: week_start, MAE, RMSE, sMAPE
    for both Naive-24h and LightGBM.
    """
    train_end_ts  = pd.Timestamp(train_end,  tz=LOCAL_TZ) + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    test_start_ts = pd.Timestamp(test_start, tz=LOCAL_TZ)

    # The first prediction window starts at test_start
    step_delta = pd.Timedelta(weeks=step_weeks)

    # Fixed LightGBM params (no early stopping — use fixed n_estimators)
    lgbm_params = {**LGBM_PARAMS}
    lgbm_params["n_estimators"] = lgbm_n_estimators
    lgbm_params.pop("verbose", None)

    records = []
    origin = test_start_ts
    step = 0

    while True:
        window_end = origin + step_delta - pd.Timedelta(hours=1)
        if window_end > X.index[-1]:
            break
        if max_steps is not None and step >= max_steps:
            break

        # Training mask: all data up to (and including) origin - 1h
        train_mask = X.index < origin
        test_mask  = (X.index >= origin) & (X.index <= window_end)

        if train_mask.sum() < 1000 or test_mask.sum() == 0:
            break

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask],  y[test_mask]

        # ---- Naive-24h ----
        # lag-24 of y: shift raw series by 24 rows within the test window
        lag24_col = f"{TARGET}_lag24h"
        if lag24_col in X_te.columns:
            naive_pred = X_te[lag24_col].values
        else:
            naive_pred = y_te.shift(24).fillna(method="bfill").values
        valid = ~np.isnan(naive_pred)
        naive_res = evaluate(y_te.values[valid], naive_pred[valid], "Naive-24h")

        # ---- LightGBM ----
        model = lgb.LGBMRegressor(**lgbm_params, verbose=-1)
        model.fit(X_tr, y_tr)
        lgbm_pred = model.predict(X_te)
        lgbm_res = evaluate(y_te.values, lgbm_pred, "LightGBM")

        records.append({
            "step":         step + 1,
            "window_start": origin.date().isoformat(),
            "window_end":   window_end.date().isoformat(),
            "n_hours":      test_mask.sum(),
            "naive_MAE":    naive_res["MAE"],
            "naive_RMSE":   naive_res["RMSE"],
            "naive_sMAPE":  naive_res["sMAPE"],
            "lgbm_MAE":     lgbm_res["MAE"],
            "lgbm_RMSE":    lgbm_res["RMSE"],
            "lgbm_sMAPE":   lgbm_res["sMAPE"],
            "_naive_err":   naive_res["_errors"][valid],
            "_lgbm_err":    lgbm_res["_errors"],
        })

        origin += step_delta
        step  += 1
        if step % 4 == 0:
            print(f"    Step {step:3d} | origin={records[-1]['window_start']} "
                  f"| LightGBM MAE={lgbm_res['MAE']:.2f}")

    return pd.DataFrame(records)


def summarise(wf: pd.DataFrame) -> None:
    metrics = ["MAE", "RMSE", "sMAPE"]
    print("\n" + "=" * 65)
    print(f"Walk-forward summary  ({len(wf)} steps of {wf['n_hours'].iloc[0]}h)")
    print("=" * 65)

    for model in ("naive", "lgbm"):
        label = "Naive-24h" if model == "naive" else "LightGBM"
        row   = {m: f"{wf[f'{model}_{m}'].mean():.3f} ± {wf[f'{model}_{m}'].std():.3f}"
                 for m in metrics}
        print(f"  {label:<20} " + "  ".join(f"{m}: {row[m]}" for m in metrics))

    # Pooled DM test across all steps (concatenate error arrays)
    all_naive = np.concatenate(wf["_naive_err"].tolist())
    all_lgbm  = np.concatenate(wf["_lgbm_err"].tolist())
    dm, p = diebold_mariano(all_lgbm, all_naive, h=24, power=2)
    sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "ns"))
    print(f"\n  DM test LightGBM vs Naive-24h (pooled): stat={dm:.3f}, p={p:.4f} {sig}")
    print("  (negative stat = LightGBM has smaller squared errors)")
    print("=" * 65 + "\n")


def main(
    data_path:  str  = "data/cleaned/NO1_hourly.parquet",
    train_end:  str  = "2023-12-31",
    test_start: str  = "2025-01-01",
    step_weeks: int  = 1,
    max_steps:  int  = None,
) -> pd.DataFrame:
    # extract zone from filename e.g. NO1_hourly.parquet -> NO1
    zone = Path(data_path).stem.split("_")[0]
    print(f"\n=== Walk-forward: {zone} ===")
    print(f"Loading {data_path} ...")
    df = load_zone(data_path)

    print("Building feature matrix ...")
    X, y = build_feature_matrix(df, target=TARGET)

    print("Calibrating LightGBM n_estimators on initial train/val split ...")
    tr_mask  = X.index <= pd.Timestamp(train_end,  tz=LOCAL_TZ)
    val_mask = (X.index > pd.Timestamp(train_end, tz=LOCAL_TZ)) & \
               (X.index < pd.Timestamp(test_start, tz=LOCAL_TZ))
    calib = lgb.LGBMRegressor(**{**LGBM_PARAMS, "verbose": -1})
    calib.fit(
        X[tr_mask], y[tr_mask],
        eval_set=[(X[val_mask], y[val_mask])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    n_est = calib.best_iteration_
    print(f"  Using n_estimators={n_est} for all walk-forward steps")

    print(f"\nRunning walk-forward backtest (step={step_weeks}w) ...")
    wf = walk_forward(X, y,
                      train_end=train_end,
                      test_start=test_start,
                      step_weeks=step_weeks,
                      max_steps=max_steps,
                      lgbm_n_estimators=n_est)

    summarise(wf)

    out = Path(f"model_results/walk_forward_{zone}.csv")
    save_cols = [c for c in wf.columns if not c.startswith("_")]
    wf[save_cols].to_csv(out, index=False)
    print(f"  Per-step results saved → {out}")
    return wf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default="data/cleaned/NO1_hourly.parquet")
    parser.add_argument("--zones",       nargs="+", default=None,
                        help="List of zones to run (overrides --data)")
    parser.add_argument("--train-end",   default="2023-12-31")
    parser.add_argument("--test-start",  default="2025-01-01")
    parser.add_argument("--step-weeks",  type=int, default=1)
    parser.add_argument("--max-steps",   type=int, default=None)
    args = parser.parse_args()

    if args.zones:
        for z in args.zones:
            main(f"data/cleaned/{z}_hourly.parquet",
                 args.train_end, args.test_start,
                 args.step_weeks, args.max_steps)
    else:
        main(args.data, args.train_end, args.test_start,
             args.step_weeks, args.max_steps)
