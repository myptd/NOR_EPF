"""
Feature group ablation figures.
Loads model_results/ablation_{zone}.csv for all available zones.

Outputs:
  paper_outputs/fig6.pdf/png   -- All-zone ablation delta heatmaps (panels A-D)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

PAPER_OUT     = Path("paper_outputs")
MODEL_RESULTS = Path("model_results")
ZONES         = ["NO1", "NO2", "NO3", "NO4", "NO5"]
ZONE_COLORS   = {"NO1": "#1f77b4", "NO2": "#ff7f0e", "NO3": "#2ca02c",
                 "NO4": "#d62728", "NO5": "#9467bd"}
GROUP_COLORS  = {
    "lags":        "#d73027",
    "calendar":    "#fc8d59",
    "weather":     "#4575b4",
    "reservoir":   "#74add1",
    "commodities": "#f46d43",
    "load_wsf":    "#abd9e9",
}

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 150,
})


def load_ablation_data() -> dict[str, pd.DataFrame]:
    dfs = {}
    for z in ZONES:
        p = MODEL_RESULTS / f"ablation_{z}.csv"
        if p.exists():
            dfs[z] = pd.read_csv(p, index_col=0)
    return dfs




def make_fig6(all_abl: dict[str, pd.DataFrame]) -> None:
    """4-panel heatmap for all zones:
    A. LOGO ΔMAE vs full model (drop experiments)
    B. LOGO ΔR² vs full model (drop experiments)
    C. Lags+Feature ΔMAE vs full model (only experiments)
    D. Lags+Feature ΔR² vs full model (only experiments)
    Cell values shown to 3 decimal places.
    """
    zones = list(all_abl.keys())
    groups_order = ["lags", "calendar", "weather", "reservoir", "commodities", "load_wsf"]

    drop_delta_mae = pd.DataFrame(index=groups_order, columns=zones, dtype=float)
    drop_delta_r2  = pd.DataFrame(index=groups_order, columns=zones, dtype=float)
    only_delta_mae = pd.DataFrame(index=groups_order, columns=zones, dtype=float)
    only_delta_r2  = pd.DataFrame(index=groups_order, columns=zones, dtype=float)

    for z, df in all_abl.items():
        full_mae = df.loc["full_model", "MAE"] if "full_model" in df.index else np.nan
        full_r2  = df.loc["full_model", "R2"]  if "full_model" in df.index else np.nan
        for g in groups_order:
            drop_key = f"drop_{g}"
            if drop_key in df.index:
                drop_delta_mae.loc[g, z] = df.loc[drop_key, "delta_MAE"]
                drop_delta_r2.loc[g, z]  = full_r2 - df.loc[drop_key, "R2"]
            only_key = f"only_{g}"
            if only_key in df.index:
                only_delta_mae.loc[g, z] = df.loc[only_key, "MAE"] - full_mae
                only_delta_r2.loc[g, z]  = df.loc[only_key, "R2"] - full_r2

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    only_row_labels = ["lags (only)", "calendar", "weather", "reservoir", "commodities", "load_wsf"]

    def _heatmap(ax, mat_df, title, panel_label, cmap, row_labels=None):
        if row_labels is None:
            row_labels = groups_order
        mat = mat_df.values.astype(float)
        vabs = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 1.0
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vabs, vmax=vabs)
        ax.set_xticks(range(len(zones)))
        ax.set_xticklabels(zones)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        for i in range(len(groups_order)):
            for j in range(len(zones)):
                v = mat[i, j]
                if not np.isnan(v):
                    txt_color = "white" if abs(v) > 0.6 * vabs else "black"
                    ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                            fontsize=7, color=txt_color)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=10)
        ax.text(-0.10, 1.02, panel_label, transform=ax.transAxes,
                fontweight="bold", fontsize=12, va="bottom")

    # A: LOGO ΔMAE — positive = important group (removing hurts MAE); red = important
    _heatmap(axes[0], drop_delta_mae,
             "LOGO \u0394MAE vs Full model (EUR/MWh)", "A", "RdYlGn_r")
    # B: LOGO ΔR² = R²_full − R²_drop — positive = important group; red = important
    _heatmap(axes[1], drop_delta_r2,
             "LOGO \u0394R\u00b2 vs Full model", "B", "RdYlGn_r")
    # C: Only ΔMAE = MAE_only − MAE_full — positive = only model worse than full; red = large gap
    _heatmap(axes[2], only_delta_mae,
             "Lags+Feature \u0394MAE vs Full model (EUR/MWh)", "C", "RdYlGn_r",
             row_labels=only_row_labels)
    # D: Only ΔR² = R²_only − R²_full — negative = only model worse than full; red = large gap
    _heatmap(axes[3], only_delta_r2,
             "Lags+Feature \u0394R\u00b2 vs Full model", "D", "RdYlGn",
             row_labels=only_row_labels)

    fig.tight_layout()
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_OUT / "fig6.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(PAPER_OUT / "fig6.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  Saved fig6.pdf / fig6.png")


def save_ablation_csv(all_abl: dict[str, pd.DataFrame]) -> None:
    """Save ablation numbers as two tidy CSV tables (drop / only)."""
    groups_order = ["lags", "calendar", "weather", "reservoir", "commodities", "load_wsf"]
    drop_rows = []
    only_rows = []
    for z, df in all_abl.items():
        full_mae = df.loc["full_model", "MAE"] if "full_model" in df.index else float("nan")
        full_r2  = df.loc["full_model", "R2"]  if "full_model" in df.index else float("nan")
        for g in groups_order:
            dk = f"drop_{g}"
            ok = f"only_{g}"
            if dk in df.index:
                drop_rows.append({
                    "zone": z,
                    "feature_group": g,
                    "MAE": round(df.loc[dk, "MAE"], 3),
                    "R2":  round(df.loc[dk, "R2"],  3),
                    "delta_MAE": round(df.loc[dk, "delta_MAE"], 3),
                    "delta_R2":  round(full_r2 - df.loc[dk, "R2"], 3),
                })
            if ok in df.index:
                only_rows.append({
                    "zone": z,
                    "feature_group": "lags (only)" if g == "lags" else g,
                    "MAE": round(df.loc[ok, "MAE"], 3),
                    "R2":  round(df.loc[ok, "R2"],  3),
                    "delta_MAE_vs_full": round(df.loc[ok, "MAE"] - full_mae, 3),
                    "delta_R2_vs_full":  round(df.loc[ok, "R2"]  - full_r2,  3),
                })
    pd.DataFrame(drop_rows).to_csv(
        PAPER_OUT / "ablation_drop.csv", index=False, float_format="%.3f")
    pd.DataFrame(only_rows).to_csv(
        PAPER_OUT / "ablation_only.csv", index=False, float_format="%.3f")
    print("  Saved ablation_drop.csv / ablation_only.csv")


if __name__ == "__main__":
    print("Loading ablation data ...")
    all_abl = load_ablation_data()
    if not all_abl:
        print("No ablation data found. Run models/ablation.py first.")
    else:
        available = list(all_abl.keys())
        print(f"  Loaded zones: {available}")
        print("Generating fig6 (all zones) ...")
        make_fig6(all_abl)
        save_ablation_csv(all_abl)
    print("Done.")
