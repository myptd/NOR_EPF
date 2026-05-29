"""
Conditional regime analysis: LightGBM error decomposition by market regime.

Regimes defined from features in the dataset (NO1 test set, 2025):
  (a) Reservoir anomaly: res_fill_anomaly_vs_median > 0 = HIGH (above seasonal normal)
                                                      < 0 = LOW  (below seasonal normal)
  (b) TTF gas price:     com_gas_ttf_close > 2025 median = HIGH gas price
                                            < 2025 median = LOW gas price

Reports:
  - LightGBM sMAPE/RMSE for each of the 4 regime cells (high/low res × high/low TTF)
  - Quartile breakdown of reservoir anomaly (Q1/Q4)
  - Hours in each cell and their price statistics

Also runs regime analysis for the lags+calendar ablation model across all zones (fig9).

Generates:
  model_results/regime_analysis_{zone}.csv
  model_results/regime_quartile_{zone}.csv
  model_results/regime_calendar_{zone}.csv
  paper_outputs/fig9.pdf / fig9.png
  paper_outputs/regime_cells.csv
  paper_outputs/regime_calendar.csv

Run:
    python analysis/08_regime_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))

matplotlib.rcParams.update({
    'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 150,
    'font.family': 'sans-serif',
    'axes.spines.top': False, 'axes.spines.right': False,
})

PAPER_OUT = Path("paper_outputs")
MODEL_RES  = Path("model_results")
TARGET = "price_eur_mwh"


def get_all_predictions(zone: str = "NO1") -> dict | None:
    """Return dict with actual, lgbm_pred + raw regime features for the test set."""
    import joblib
    from models.utils import load_zone, build_feature_matrix, train_val_test_split, set_seed
    set_seed()
    data_path = f"data/cleaned/{zone}_hourly.parquet"
    lgbm_path = Path(f"model_weights/lgbm_{zone}.pkl")

    if not lgbm_path.exists():
        print(f"  {lgbm_path} not found — run models/run_all.py first")
        return None

    lgbm_booster = joblib.load(lgbm_path)

    df_full = load_zone(data_path)
    X, y = build_feature_matrix(df_full)
    _, _, X_te, _, _, y_te = train_val_test_split(X, y)

    lgbm_pred = lgbm_booster.predict(X_te)

    # Pull raw regime features from the full df, aligned to test index
    test_idx  = X_te.index
    res_anom  = df_full["res_fill_anomaly_vs_median"].reindex(test_idx)
    ttf_price = df_full["com_gas_ttf_close"].reindex(test_idx)

    return {
        "idx":       test_idx,
        "y_true":    y_te.values,
        "lgbm_pred": lgbm_pred,
        "res_anom":  res_anom.values,
        "ttf_price": ttf_price.values,
        "zone":      zone,
    }


def cell_stats(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    err = y_true - y_pred
    # sMAPE: symmetric, bounded [0, 200%], robust to near-zero actual prices
    denom = (np.abs(y_true) + np.abs(y_pred))
    nonzero = denom > 0.1
    smape = np.mean(2 * np.abs(err[nonzero]) / denom[nonzero]) * 100 if nonzero.sum() > 0 else np.nan
    rmse = np.sqrt(np.mean(err**2))
    bias = np.mean(err)
    return {"label": label, "n": len(y_true),
            "price_mean": round(np.mean(y_true), 3),
            "price_std":  round(np.std(y_true),  3),
            "sMAPE": round(smape, 3),
            "RMSE": round(rmse, 3),
            "bias": round(bias, 3)}


def run_regime_analysis(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true    = data["y_true"]
    lgbm_pred = data["lgbm_pred"]
    res_anom  = data["res_anom"]
    ttf_price = data["ttf_price"]

    # --- Regime masks: median split per variable, per zone ---
    valid = np.isfinite(res_anom) & np.isfinite(ttf_price)
    res_med = np.nanmedian(res_anom[valid])
    ttf_med = np.nanmedian(ttf_price[valid])

    res_high = res_anom >= res_med
    res_low  = res_anom <  res_med
    ttf_high = ttf_price >= ttf_med
    ttf_low  = ttf_price <  ttf_med

    cells = {
        "High res / High TTF": res_high & ttf_high & valid,
        "High res / Low TTF":  res_high & ttf_low  & valid,
        "Low res  / High TTF": res_low  & ttf_high & valid,
        "Low res  / Low TTF":  res_low  & ttf_low  & valid,
    }

    records = []
    for cell_name, mask in cells.items():
        if mask.sum() < 5:
            continue
        s = cell_stats(y_true[mask], lgbm_pred[mask], label="LightGBM")
        s["cell"] = cell_name
        records.append(s)

    cell_df = pd.DataFrame(records)

    # --- Quartile breakdown of reservoir anomaly ---
    res_clean = res_anom[valid]
    q25 = np.nanpercentile(res_clean, 25)
    q75 = np.nanpercentile(res_clean, 75)

    quartile_masks = {
        f"Q1 res (≤{q25:.2f})": valid & (res_anom <= q25),
        f"Q2 res ({q25:.2f}–0)": valid & (res_anom > q25) & (res_anom < 0),
        f"Q3 res (0–{q75:.2f})": valid & (res_anom >= 0) & (res_anom < q75),
        f"Q4 res (≥{q75:.2f})": valid & (res_anom >= q75),
    }
    q_recs = []
    for qname, qmask in quartile_masks.items():
        if qmask.sum() < 5:
            continue
        yt = y_true[qmask]
        res_vals = res_anom[qmask]
        s = cell_stats(yt, lgbm_pred[qmask], label="LightGBM")
        s["quartile"] = qname
        s["res_anom_mean"] = round(np.nanmean(res_vals), 3)
        q_recs.append(s)

    quartile_df = pd.DataFrame(q_recs)
    return cell_df, quartile_df


ZONES = ["NO1", "NO2", "NO3", "NO4", "NO5"]
ZONE_COLORS = {"NO1": "#1f77b4", "NO2": "#ff7f0e", "NO3": "#2ca02c",
               "NO4": "#d62728", "NO5": "#9467bd"}


def get_calendar_model_predictions(zone: str) -> dict | None:
    """Load ablation_only_calendar_{zone}.pkl and return predictions + regime features."""
    import joblib
    from models.utils import load_zone, build_feature_matrix, train_val_test_split, set_seed
    set_seed()

    w_path = Path(f"model_weights/ablation_only_calendar_{zone}.pkl")
    if not w_path.exists():
        print(f"  {w_path} not found — run models/ablation.py first")
        return None

    payload  = joblib.load(w_path)
    model    = payload["model"]
    features = payload["features"]

    data_path = f"data/cleaned/{zone}_hourly.parquet"
    df_full = load_zone(data_path)
    X, y = build_feature_matrix(df_full)
    _, _, X_te, _, _, y_te = train_val_test_split(X, y)

    cal_pred  = model.predict(X_te[features])

    # Also load full LightGBM for comparison
    full_path = Path(f"model_weights/lgbm_{zone}.pkl")
    if full_path.exists():
        full_model = joblib.load(full_path)
        full_pred  = full_model.predict(X_te)
    else:
        full_pred = np.full(len(y_te), np.nan)

    test_idx  = X_te.index
    res_anom  = df_full["res_fill_anomaly_vs_median"].reindex(test_idx).values
    ttf_price = df_full["com_gas_ttf_close"].reindex(test_idx).values

    return {
        "zone":      zone,
        "idx":       test_idx,
        "y_true":    y_te.values,
        "cal_pred":  cal_pred,
        "full_pred": full_pred,
        "res_anom":  res_anom,
        "ttf_price": ttf_price,
    }


def run_regime_calendar(data: dict) -> pd.DataFrame:
    """Run 2×2 regime analysis comparing Lags+Calendar vs Full LightGBM."""
    y_true    = data["y_true"]
    cal_pred  = data["cal_pred"]
    full_pred = data["full_pred"]
    res_anom  = data["res_anom"]
    ttf_price = data["ttf_price"]

    # Regime masks: median split — same definition as run_regime_analysis.
    valid = np.isfinite(res_anom) & np.isfinite(ttf_price)
    res_med = np.nanmedian(res_anom[valid])
    ttf_med = np.nanmedian(ttf_price[valid])

    cells = {
        "High res / High TTF": (res_anom >= res_med) & (ttf_price >= ttf_med) & valid,
        "High res / Low TTF":  (res_anom >= res_med) & (ttf_price <  ttf_med) & valid,
        "Low res  / High TTF": (res_anom <  res_med) & (ttf_price >= ttf_med) & valid,
        "Low res  / Low TTF":  (res_anom <  res_med) & (ttf_price <  ttf_med) & valid,
    }

    records = []
    for cell_name, mask in cells.items():
        if mask.sum() < 5:
            continue
        yt = y_true[mask]
        for model_name, preds in [
            ("Lags+Calendar", cal_pred[mask]),
            ("Full LightGBM", full_pred[mask]),
        ]:
            s = cell_stats(yt, preds, label=model_name)
            s["cell"] = cell_name
            s["zone"] = data["zone"]
            records.append(s)

    return pd.DataFrame(records)


def make_fig9(all_regime_cal: dict[str, pd.DataFrame]) -> None:
    """All-zone regime heatmap for the lags+calendar model.

    Panel A: Marginal effect of reservoir — Lo-Res minus Hi-Res sMAPE per zone
    Panel B: Marginal effect of TTF gas   — Hi-TTF minus Lo-TTF sMAPE per zone
    Panel C: Lags+Calendar 4-cell × 5-zone heatmap (col-normalised)
    Panel D: Full LightGBM  4-cell × 5-zone heatmap (col-normalised)
    """
    zones      = list(all_regime_cal.keys())
    cell_order = ["High res / High TTF", "High res / Low TTF",
                  "Low res  / High TTF", "Low res  / Low TTF"]

    def _build_mat(model_label: str) -> np.ndarray:
        """Rows = cells (4), Cols = zones (5)."""
        mat = np.full((len(cell_order), len(zones)), np.nan)
        for j, z in enumerate(zones):
            df = all_regime_cal[z]
            sub = df[df["label"] == model_label].set_index("cell")
            for i, c in enumerate(cell_order):
                if c in sub.index:
                    mat[i, j] = sub.loc[c, "sMAPE"]
        return mat

    cal_mat  = _build_mat("Lags+Calendar")
    full_mat = _build_mat("Full LightGBM")

    # --- Marginal effects ---
    # Reservoir marginal: Lo-Res average − Hi-Res average (positive = Lo-Res harder)
    # Hi-Res rows: 0,1  |  Lo-Res rows: 2,3
    cal_res_marg  = np.nanmean(cal_mat[2:, :],  axis=0) - np.nanmean(cal_mat[:2, :],  axis=0)
    full_res_marg = np.nanmean(full_mat[2:, :], axis=0) - np.nanmean(full_mat[:2, :], axis=0)
    # TTF marginal: Hi-TTF average − Lo-TTF average (positive = Hi-TTF harder)
    # Hi-TTF rows: 0,2  |  Lo-TTF rows: 1,3
    cal_ttf_marg  = np.nanmean(cal_mat[[0,2], :],  axis=0) - np.nanmean(cal_mat[[1,3], :],  axis=0)
    full_ttf_marg = np.nanmean(full_mat[[0,2], :], axis=0) - np.nanmean(full_mat[[1,3], :], axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes_flat = axes.flatten()
    x = np.arange(len(zones))
    width = 0.35

    # --- Panel A: Reservoir marginal ---
    ax = axes_flat[0]
    bars_c = ax.bar(x - width/2, cal_res_marg,  width, label="Lags+Calendar",
                    color="#2ca02c", alpha=0.85)
    bars_f = ax.bar(x + width/2, full_res_marg, width, label="Full LightGBM",
                    color="#2166ac", alpha=0.85)
    for bars, vals in [(bars_c, cal_res_marg), (bars_f, full_res_marg)]:
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        v + (0.1 if v >= 0 else -0.4),
                        f"{v:+.2f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(zones)
    ax.set_ylabel("ΔsMAPE: Lo-Res − Hi-Res (%)")
    ax.set_title("Marginal effect of reservoir anomaly\n"
                 "(positive = low reservoir is harder to predict)")
    ax.legend(fontsize=8)
    ax.text(-0.12, 1.02, "A", transform=ax.transAxes,
            fontweight="bold", fontsize=12, va="bottom")

    # --- Panel B: TTF marginal ---
    ax = axes_flat[1]
    bars_c = ax.bar(x - width/2, cal_ttf_marg,  width, label="Lags+Calendar",
                    color="#2ca02c", alpha=0.85)
    bars_f = ax.bar(x + width/2, full_ttf_marg, width, label="Full LightGBM",
                    color="#2166ac", alpha=0.85)
    for bars, vals in [(bars_c, cal_ttf_marg), (bars_f, full_ttf_marg)]:
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        v + (0.1 if v >= 0 else -0.4),
                        f"{v:+.2f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(zones)
    ax.set_ylabel("ΔsMAPE: Hi-TTF − Lo-TTF (%)")
    ax.set_title("Marginal effect of TTF gas price\n"
                 "(positive = high gas price is harder to predict)")
    ax.legend(fontsize=8)
    ax.text(-0.12, 1.02, "B", transform=ax.transAxes,
            fontweight="bold", fontsize=12, va="bottom")

    # --- Shared heatmap helper ---
    def _heatmap_rc(ax, mat, title, panel_label, fmt="{:.2f}%"):
        """Column-normalised heatmap: colours show within-zone rank,
        annotations show actual sMAPE values."""
        disp = np.full_like(mat, np.nan)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            cmin, cmax = np.nanmin(col), np.nanmax(col)
            disp[:, j] = (col - cmin) / (cmax - cmin) if cmax > cmin else np.full(len(col), 0.5)
        im = ax.imshow(disp, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(zones))); ax.set_xticklabels(zones, fontsize=8)
        ax.set_yticks(range(len(cell_order)))
        ax.set_yticklabels(["Hi-Res/Hi-TTF", "Hi-Res/Lo-TTF",
                             "Lo-Res/Hi-TTF", "Lo-Res/Lo-TTF"], fontsize=8)
        for i in range(len(cell_order)):
            for j in range(len(zones)):
                v, dv = mat[i, j], disp[i, j]
                if not np.isnan(v):
                    tc = "white" if (not np.isnan(dv) and dv > 0.6) else "black"
                    ax.text(j, i, fmt.format(v), ha="center", va="center",
                            fontsize=7, color=tc)
        cbar = plt.colorbar(im, ax=ax, shrink=0.75)
        cbar.set_label("relative (col-normalised)", fontsize=7)
        cbar.set_ticks([])
        ax.set_title(title, fontsize=10)
        ax.text(-0.12, 1.02, panel_label, transform=ax.transAxes,
                fontweight="bold", fontsize=12, va="bottom")

    _heatmap_rc(axes_flat[2], cal_mat,
                "Lags+Calendar sMAPE by regime (%)\n(colour: within-zone normalised)", "C")
    _heatmap_rc(axes_flat[3], full_mat,
                "Full LightGBM sMAPE by regime (%)\n(colour: within-zone normalised)", "D")

    fig.tight_layout()
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_OUT / "fig9.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(PAPER_OUT / "fig9.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  Saved fig9.pdf / fig9.png")


def save_regime_csv(
    all_regime_cells: dict[str, pd.DataFrame],
    all_regime_cal: dict[str, pd.DataFrame],
) -> None:
    """Save two tidy CSV tables analogous to ablation_drop.csv / ablation_only.csv.

    Table 1 (regime_cells.csv):
        One row per (zone, cell) — LightGBM only.

    Table 2 (regime_calendar.csv):
        One row per (zone, model, cell) — Lags+Calendar and Full LightGBM only.
    """
    PAPER_OUT.mkdir(parents=True, exist_ok=True)

    # Table 1 — full model + lags+calendar cells (all zones)
    # Prefer all_regime_cal (has both "Lags+Calendar" and "Full LightGBM", consistent
    # with fig9 and regime_calendar.csv). Fall back to all_regime_cells (LightGBM only).
    source_cells = all_regime_cal if all_regime_cal else all_regime_cells
    if source_cells:
        rows = []
        for z, df in source_cells.items():
            for _, row in df.iterrows():
                rows.append({
                    "zone":       z,
                    "model":      row["label"],
                    "cell":       row["cell"],
                    "n":          int(row["n"]),
                    "price_mean": round(row["price_mean"], 3),
                    "price_std":  round(row["price_std"],  3),
                    "sMAPE":      round(row["sMAPE"],      3),
                    "RMSE":       round(row["RMSE"],       3),
                })
        pd.DataFrame(rows).to_csv(
            PAPER_OUT / "regime_cells.csv", index=False, float_format="%.3f"
        )
        print("  Saved regime_cells.csv")

    # Table 2 — calendar ablation cells (all zones)
    if all_regime_cal:
        rows = []
        for z, df in all_regime_cal.items():
            for _, row in df.iterrows():
                rows.append({
                    "zone":       z,
                    "model":      row["label"],
                    "cell":       row["cell"],
                    "n":          int(row["n"]),
                    "price_mean": round(row["price_mean"], 3),
                    "price_std":  round(row["price_std"],  3),
                    "sMAPE":      round(row["sMAPE"],      3),
                    "RMSE":       round(row["RMSE"],       3),
                })
        pd.DataFrame(rows).to_csv(
            PAPER_OUT / "regime_calendar.csv", index=False, float_format="%.3f"
        )
        print("  Saved regime_calendar.csv")


def main() -> None:
    print("\n=== Regime Analysis: All Zones ===\n")

    # ── Full LightGBM regime analysis ────────────────────────────────────────
    all_regime_cells: dict[str, pd.DataFrame] = {}
    MODEL_RES.mkdir(parents=True, exist_ok=True)
    for z in ZONES:
        d = get_all_predictions(z)
        if d is None:
            continue
        print(f"\nRunning regime analysis for {z} ...")
        cell_df, quartile_df = run_regime_analysis(d)

        lgbm_c = cell_df[cell_df["label"] == "LightGBM"]
        print(lgbm_c[["cell", "n", "price_mean", "sMAPE", "RMSE"]].to_string(index=False))

        cell_df.to_csv(MODEL_RES / f"regime_analysis_{z}.csv", index=False)
        print(f"  Saved model_results/regime_analysis_{z}.csv")
        quartile_df.to_csv(MODEL_RES / f"regime_quartile_{z}.csv", index=False)
        print(f"  Saved model_results/regime_quartile_{z}.csv")

        if z == "NO1":
            cell_df.to_csv(MODEL_RES / "regime_analysis.csv", index=False)

        all_regime_cells[z] = cell_df

    if not all_regime_cells:
        print("No full model weights found. Run models/run_all.py first.")

    # ── Lags+Calendar regime analysis across all zones ────────────────────────
    print("\n=== Regime Analysis: Lags+Calendar model (all zones) ===")
    all_regime_cal: dict[str, pd.DataFrame] = {}
    for z in ZONES:
        d = get_calendar_model_predictions(z)
        if d is not None:
            df_cal = run_regime_calendar(d)
            all_regime_cal[z] = df_cal
            df_cal.to_csv(MODEL_RES / f"regime_calendar_{z}.csv", index=False)
            print(f"  Saved model_results/regime_calendar_{z}.csv")

    if all_regime_cal:
        print("\nGenerating fig9 ...")
        make_fig9(all_regime_cal)

    save_regime_csv(all_regime_cells, all_regime_cal)
    print("\nDone.")


if __name__ == "__main__":
    main()
