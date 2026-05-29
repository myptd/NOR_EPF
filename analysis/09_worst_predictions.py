"""
Worst prediction analysis: deep-dive into largest forecast errors.

Identifies the N worst predicted hours by LightGBM on NO1 test set (2025),
then explains each via the available regime features:
  - res_fill_anomaly_vs_median (reservoir scarcity signal)
  - com_gas_ttf_close          (gas price proxy)
  - load_forecast_mw           (demand signal)
  - wx_temperature_2m          (cold-spell indicator)
  - net_import_NO_1_mw         (congestion signal)
  - price_eur_mwh at t-24, t-168 (what the model "saw")

Generates:
  model_results/worst_predictions.csv
  paper_outputs/fig10.pdf / fig10.png

Run:
    python analysis/09_worst_predictions.py [--n 20]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.utils import (
    load_zone, build_feature_matrix, train_val_test_split, SEED, set_seed,
)

matplotlib.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'figure.dpi': 150,
    'font.family': 'sans-serif',
    'axes.spines.top': False, 'axes.spines.right': False,
})

PAPER_OUT = Path("paper_outputs")
MODEL_RES  = Path("model_results")
TARGET     = "price_eur_mwh"
EXPLAIN_COLS = [
    "res_fill_anomaly_vs_median",
    "com_gas_ttf_close",
    "load_forecast_mw",
    "wx_temperature_2m",
    "net_import_NO_1_mw",
]


def get_worst(zone: str = "NO1", n: int = 20) -> pd.DataFrame | None:
    set_seed()

    lgbm_path = Path(f"model_weights/lgbm_{zone}.pkl")
    if not lgbm_path.exists():
        print(f"  {lgbm_path} not found — run models/run_all.py first")
        return None

    lgbm_booster = joblib.load(lgbm_path)

    df_full = load_zone(f"data/cleaned/{zone}_hourly.parquet")
    X, y = build_feature_matrix(df_full)
    X_tr, X_val, X_te, y_tr, y_val, y_te = train_val_test_split(X, y)

    lgbm_pred    = lgbm_booster.predict(X_te)

    residuals = y_te.values - lgbm_pred
    abs_err   = np.abs(residuals)
    worst_idx = np.argsort(abs_err)[::-1][:n]

    rows = []
    for rank, i in enumerate(worst_idx, 1):
        ts = y_te.index[i]
        actual    = y_te.values[i]
        predicted = lgbm_pred[i]
        error     = residuals[i]

        row = {
            "rank":       rank,
            "datetime":   ts.isoformat(),
            "actual":     round(actual, 2),
            "predicted":  round(predicted, 2),
            "error":      round(error, 2),
            "abs_error":  round(abs(error), 2),
            "pct_error":  round(abs(error) / max(abs(actual), 1) * 100, 1),
        }

        # Look up explanatory features at that timestamp
        for col in EXPLAIN_COLS:
            if col in df_full.columns:
                val = df_full.loc[ts, col] if ts in df_full.index else np.nan
                row[col] = round(float(val), 3) if pd.notna(val) else np.nan

        # Add lag-24 and lag-168 price (what the model "saw")
        ts_lag24  = ts - pd.Timedelta(hours=24)
        ts_lag168 = ts - pd.Timedelta(hours=168)
        row["price_lag24h"]  = round(float(df_full.loc[ts_lag24,  TARGET]), 2) \
                               if ts_lag24  in df_full.index else np.nan
        row["price_lag168h"] = round(float(df_full.loc[ts_lag168, TARGET]), 2) \
                               if ts_lag168 in df_full.index else np.nan

        # Day/hour context
        row["day_of_week"] = ts.day_name()
        row["hour"]        = ts.hour

        rows.append(row)

    return pd.DataFrame(rows)


def make_fig10(worst_df: pd.DataFrame, zone: str = "NO1") -> None:
    """4-panel figure: time series context, scatter, error vs features."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axes_flat = axes.flatten()
    panel_labels = ["A", "B", "C", "D"]

    # Load full predictions for context
    try:
        from models.utils import build_feature_matrix, train_val_test_split, load_zone
        lgbm_booster = joblib.load(f"model_weights/lgbm_{zone}.pkl")
        df_full = load_zone(f"data/cleaned/{zone}_hourly.parquet")
        X, y = build_feature_matrix(df_full)
        _, _, X_te, _, _, y_te = train_val_test_split(X, y)
        lgbm_pred = lgbm_booster.predict(X_te)
        have_full = True
    except Exception:
        have_full = False

    # Panel A: First 60 days of test period with worst errors marked
    ax = axes_flat[0]
    if have_full:
        n_days = min(60 * 24, len(y_te))
        idx_slice = y_te.index[:n_days]
        ax.plot(idx_slice, y_te.values[:n_days], color="black",
                linewidth=0.7, label="Actual", zorder=2)
        ax.plot(idx_slice, lgbm_pred[:n_days], color="steelblue",
                linewidth=0.7, linestyle="--", label="LightGBM", zorder=2, alpha=0.8)

        # Mark worst predictions that fall in this window
        for _, row in worst_df.iterrows():
            ts = pd.Timestamp(row["datetime"])
            if ts in idx_slice:
                ax.scatter([ts], [row["actual"]], color="red", s=60, zorder=5,
                           marker="*", label="Worst" if _ == worst_df.index[0] else "")
                ax.annotate(f"#{row['rank']}", (ts, row["actual"]),
                            textcoords="offset points", xytext=(0, 6),
                            fontsize=6, color="red", ha="center")
        ax.set_xlabel("Date (first 60 days)")
        ax.set_ylabel("Price (EUR/MWh)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        # deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels)
                  if not (l in seen or seen.add(l))]
        ax.legend(*zip(*unique), loc="upper right")

    # Panel B: Actual vs error — scatter, highlight worst
    ax = axes_flat[1]
    if have_full:
        all_err = np.abs(y_te.values - lgbm_pred)
        ax.scatter(y_te.values, y_te.values - lgbm_pred,
                   s=1, alpha=0.2, color="steelblue", label="All test hours")
        # Highlight worst
        ax.scatter(worst_df["actual"], worst_df["error"],
                   s=40, color="red", zorder=5, label=f"Top {len(worst_df)} worst")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Actual Price (EUR/MWh)")
        ax.set_ylabel("Error (Actual − Predicted, EUR/MWh)")
        ax.legend()

    # Panel C: Error vs reservoir anomaly
    ax = axes_flat[2]
    col = "res_fill_anomaly_vs_median"
    if col in worst_df.columns and have_full:
        # Full test set (scatter)
        res_full = df_full[col].reindex(y_te.index).values
        err_full  = y_te.values - lgbm_pred
        valid = np.isfinite(res_full)
        ax.scatter(res_full[valid], err_full[valid], s=2, alpha=0.2,
                   color="steelblue", label="All hours")
        # Worst
        ax.scatter(worst_df[col], worst_df["error"], s=50, color="red",
                   zorder=5, label=f"Top {len(worst_df)} worst")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Reservoir Anomaly vs Median")
        ax.set_ylabel("Error (EUR/MWh)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Reservoir data unavailable",
                transform=ax.transAxes, ha="center", va="center")

    # Panel D: Error vs TTF gas price
    ax = axes_flat[3]
    col_ttf = "com_gas_ttf_close"
    if col_ttf in worst_df.columns and have_full:
        ttf_full = df_full[col_ttf].reindex(y_te.index).values
        valid = np.isfinite(ttf_full)
        ax.scatter(ttf_full[valid], err_full[valid], s=2, alpha=0.2,
                   color="steelblue", label="All hours")
        ax.scatter(worst_df[col_ttf], worst_df["error"], s=50, color="red",
                   zorder=5, label=f"Top {len(worst_df)} worst")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("TTF Gas Price (EUR/MWh)")
        ax.set_ylabel("Error (EUR/MWh)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "TTF data unavailable",
                transform=ax.transAxes, ha="center", va="center")

    for ax, lbl in zip(axes_flat, panel_labels):
        ax.text(-0.10, 1.02, lbl, transform=ax.transAxes,
                fontweight="bold", fontsize=12, va="bottom")

    fig.tight_layout()
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_OUT / "fig10.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(PAPER_OUT / "fig10.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  Saved fig10.pdf / fig10.png")


def main(n: int = 20) -> None:
    # Check if pre-computed per-zone CSVs exist
    all_worst = {}
    for z in ["NO1", "NO2", "NO3", "NO4", "NO5"]:
        csv_path = MODEL_RES / f"worst_predictions_{z}.csv"
        if csv_path.exists():
            all_worst[z] = pd.read_csv(csv_path)
            print(f"  Loaded {csv_path}")

    if all_worst:
        print(f"  Using pre-computed worst predictions from model_results/")
    else:
        # Fall back to model weight loading
        zones_to_run = ["NO1", "NO2", "NO3", "NO4", "NO5"]
        for z in zones_to_run:
            wd = get_worst(z, n=n)
            if wd is not None:
                all_worst[z] = wd
                MODEL_RES.mkdir(parents=True, exist_ok=True)
                wd.to_csv(MODEL_RES / f"worst_predictions_{z}.csv", index=False)
                print(f"  Saved model_results/worst_predictions_{z}.csv")

        if not all_worst:
            print("No model weights found. Run models/run_all.py first.")
            return

        # Keep legacy file for backward compatibility
        if "NO1" in all_worst:
            all_worst["NO1"].to_csv(MODEL_RES / "worst_predictions.csv", index=False)

    # Generate fig10 for NO1 (main)
    if "NO1" in all_worst:
        print("\nGenerating fig10 (NO1) ...")
        make_fig10(all_worst["NO1"], zone="NO1")

    # Save comprehensive text for all zones
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of worst predictions to analyse")
    args = parser.parse_args()
    main(args.n)
