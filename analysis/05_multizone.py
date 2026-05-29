"""
Multi-zone model comparison figures.
All five zones (NO1-NO5).

Generates:
  paper_outputs/fig7.pdf/png  -- multi-zone comparison

Run:
    python analysis/05_multizone.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

matplotlib.rcParams.update({
    'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 150,
    'font.family': 'sans-serif',
    'axes.spines.top': False, 'axes.spines.right': False,
})

ZONES = ["NO1", "NO2", "NO3", "NO4", "NO5"]
ZONE_COLORS = {"NO1": "#1f77b4", "NO2": "#ff7f0e", "NO3": "#2ca02c",
               "NO4": "#d62728", "NO5": "#9467bd"}
MODEL_ORDER = ["Naive-24h", "Naive-168h", "Linear-ARX (Ridge)",
               "LightGBM", "XGBoost", "LSTM", "TCN", "Transformer"]
MODEL_COLORS = {"Naive-24h": "#aec7e8", "Naive-168h": "#c5b0d5",
                "Linear-ARX (Ridge)": "#ffbb78", "LightGBM": "#98df8a",
                "XGBoost": "#ff9896", "LSTM": "#c49c94",
                "TCN": "#17becf", "Transformer": "#bcbd22"}
PAPER_OUT = Path("paper_outputs")
MODEL_RESULTS = Path("model_results")


def load_results():
    test_p = MODEL_RESULTS / "all_zones_test.csv"
    val_p  = MODEL_RESULTS / "all_zones_val.csv"
    test_df = pd.read_csv(test_p) if test_p.exists() else None
    val_df  = pd.read_csv(val_p)  if val_p.exists() else None
    # Append deep learning model results per zone
    for split in ("test", "val"):
        rows = []
        for z in ZONES:
            for model in ("lstm", "tcn", "transformer"):
                lp = MODEL_RESULTS / f"{z}_{model}_{split}.csv"
                if lp.exists():
                    row = pd.read_csv(lp)
                    row["zone"] = z
                    rows.append(row)
        if rows:
            extra_df = pd.concat(rows, ignore_index=True)
            if split == "test" and test_df is not None:
                test_df = pd.concat([test_df, extra_df], ignore_index=True)
            elif split == "val" and val_df is not None:
                val_df = pd.concat([val_df, extra_df], ignore_index=True)
    if test_df is not None:
        test_df["model"] = test_df["model"].str.strip().replace(
            {"TRANSFORMER": "Transformer", "LSTM": "LSTM", "TCN": "TCN"})
    if val_df is not None:
        val_df["model"] = val_df["model"].str.strip().replace(
            {"TRANSFORMER": "Transformer", "LSTM": "LSTM", "TCN": "TCN"})
    return val_df, test_df


def get_ordered_models(df: pd.DataFrame) -> list:
    available = df["model"].unique().tolist()
    ordered = [m for m in MODEL_ORDER if m in available]
    return ordered + [m for m in available if m not in ordered]


# ---------------------------------------------------------------------------
# FIG 7
# ---------------------------------------------------------------------------

def make_fig7(val_df, test_df) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    axes = axes.flatten()

    ref_df = test_df if test_df is not None else val_df
    if ref_df is None:
        for ax, lbl in zip(axes, ["A","B","C","D"]):
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.text(-0.10, 1.02, lbl, transform=ax.transAxes,
                    fontweight='bold', fontsize=12, va='bottom')
        fig.tight_layout()
        PAPER_OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(PAPER_OUT / "fig7.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(PAPER_OUT / "fig7.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        return

    models = get_ordered_models(ref_df)
    n_models = len(models)

    # Panel A: MAE by zone -- grouped horizontal bars
    ax = axes[0]
    y = np.arange(len(ZONES))
    height = 0.8 / n_models
    for mi, m in enumerate(models):
        vals = []
        for z in ZONES:
            sub = ref_df[(ref_df["model"] == m) & (ref_df["zone"] == z)]
            vals.append(sub["MAE"].values[0] if len(sub) > 0 else np.nan)
        offset = (mi - n_models / 2 + 0.5) * height
        ax.barh(y + offset, vals, height, label=m,
                color=MODEL_COLORS.get(m, "#888"), alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(ZONES)
    ax.set_xlabel("MAE (EUR/MWh)")
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.10, 1.02, "A", transform=ax.transAxes,
            fontweight='bold', fontsize=12, va='bottom')

    # Panel B: RMSE heatmap -- model x zone
    ax = axes[1]
    rmse_mat = np.zeros((n_models, len(ZONES)))
    for mi, m in enumerate(models):
        for zi, z in enumerate(ZONES):
            sub = ref_df[(ref_df["model"] == m) & (ref_df["zone"] == z)]
            rmse_mat[mi, zi] = sub["RMSE"].values[0] if len(sub) > 0 else np.nan
    im = ax.imshow(rmse_mat, cmap="RdYlGn_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="RMSE (EUR/MWh)")
    ax.set_xticks(range(len(ZONES)))
    ax.set_xticklabels(ZONES)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(models, fontsize=8)
    for mi in range(n_models):
        for zi in range(len(ZONES)):
            val = rmse_mat[mi, zi]
            if not np.isnan(val):
                ax.text(zi, mi, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    ax.set_xlabel("Zone")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.text(-0.10, 1.02, "B", transform=ax.transAxes,
            fontweight='bold', fontsize=12, va='bottom')

    # Panel C: R2 heatmap -- model x zone
    ax = axes[2]
    r2_mat = np.zeros((n_models, len(ZONES)))
    for mi, m in enumerate(models):
        for zi, z in enumerate(ZONES):
            sub = ref_df[(ref_df["model"] == m) & (ref_df["zone"] == z)]
            r2_mat[mi, zi] = sub["R2"].values[0] if len(sub) > 0 else np.nan
    vmin = max(-0.5, float(np.nanmin(r2_mat)))
    im2 = ax.imshow(r2_mat, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=1.0)
    plt.colorbar(im2, ax=ax, label="R2")
    ax.set_xticks(range(len(ZONES)))
    ax.set_xticklabels(ZONES)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(models, fontsize=8)
    for mi in range(n_models):
        for zi in range(len(ZONES)):
            val = r2_mat[mi, zi]
            if not np.isnan(val):
                ax.text(zi, mi, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color="black")
    ax.set_xlabel("Zone")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.text(-0.10, 1.02, "C", transform=ax.transAxes,
            fontweight='bold', fontsize=12, va='bottom')

    # Panel D: MAE improvement over Naive-24h (%)
    ax = axes[3]
    non_naive = [m for m in models if m not in ("Naive-24h", "Naive-168h")]
    n_nn = len(non_naive)
    x = np.arange(len(ZONES))
    width = 0.8 / max(n_nn, 1)
    for mi, m in enumerate(non_naive):
        improvements = []
        for z in ZONES:
            nsub = ref_df[(ref_df["model"] == "Naive-24h") & (ref_df["zone"] == z)]
            msub = ref_df[(ref_df["model"] == m) & (ref_df["zone"] == z)]
            if len(nsub) > 0 and len(msub) > 0:
                pct = 100 * (nsub["MAE"].values[0] - msub["MAE"].values[0]) / nsub["MAE"].values[0]
                improvements.append(pct)
            else:
                improvements.append(np.nan)
        offset = (mi - n_nn / 2 + 0.5) * width
        ax.bar(x + offset, improvements, width, label=m,
               color=MODEL_COLORS.get(m, "#888"), alpha=0.85)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ZONES)
    ax.set_ylabel("MAE improvement over Naive-24h (%)")
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.10, 1.02, "D", transform=ax.transAxes,
            fontweight='bold', fontsize=12, va='bottom')

    fig.tight_layout()
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_OUT / "fig7.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(PAPER_OUT / "fig7.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  Saved fig7.pdf / fig7.png")


# ---------------------------------------------------------------------------
# TEXT OUTPUT
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading multi-zone results ...")
    val_df, test_df = load_results()

    print("\nGenerating fig7 ...")
    make_fig7(val_df, test_df)
    print("\nDone.")


if __name__ == "__main__":
    main()
