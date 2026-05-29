"""
Walk-forward rolling-origin backtest figures.
Loads model_results/walk_forward_{zone}.csv for all available zones.

Outputs:
  paper_outputs/fig5.pdf/png   -- NO1 detailed (panels A-D)
  paper_outputs/fig5b.pdf/png  -- All-zone summary (panels A-D)
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

PAPER_OUT   = Path("paper_outputs")
MODEL_RESULTS = Path("model_results")
ZONES       = ["NO1", "NO2", "NO3", "NO4", "NO5"]
ZONE_COLORS = {"NO1": "#1f77b4", "NO2": "#ff7f0e", "NO3": "#2ca02c",
               "NO4": "#d62728", "NO5": "#9467bd"}

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 150,
})


def load_wf_data() -> dict[str, pd.DataFrame]:
    """Load walk-forward CSVs for all available zones."""
    dfs = {}
    for z in ZONES:
        p = MODEL_RESULTS / f"walk_forward_{z}.csv"
        if p.exists():
            dfs[z] = pd.read_csv(p)
    return dfs


def make_fig5(wf: pd.DataFrame, zone: str = "NO1") -> None:
    """Detailed 4-panel walk-forward figure for one zone."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axes = axes.flatten()
    color = ZONE_COLORS.get(zone, "#1f77b4")

    steps = np.arange(1, len(wf) + 1)
    n_steps = len(wf)

    # Panel A: weekly MAE time series
    ax = axes[0]
    ax.plot(steps, wf["lgbm_MAE"], color=color, linewidth=1.2, label="LightGBM", marker="o", markersize=3)
    ax.plot(steps, wf["naive_MAE"], color="gray", linewidth=1.0, linestyle="--", label="Nave-24h", marker="s", markersize=3)
    ax.set_xlabel("Week step")
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title(f"Weekly MAE: {zone}")
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.10, 1.02, "A", transform=ax.transAxes, fontweight='bold', fontsize=12, va='bottom')

    # Panel B: LightGBM beats Naive (scatter, one dot per week)
    ax = axes[1]
    wins = (wf["lgbm_MAE"] < wf["naive_MAE"]).sum()
    win_rate = 100 * wins / n_steps
    ax.scatter(steps[wf["lgbm_MAE"] < wf["naive_MAE"]],
               wf["lgbm_MAE"][wf["lgbm_MAE"] < wf["naive_MAE"]],
               color=color, s=20, alpha=0.8, label=f"LightGBM wins ({wins}/{n_steps})")
    ax.scatter(steps[wf["lgbm_MAE"] >= wf["naive_MAE"]],
               wf["lgbm_MAE"][wf["lgbm_MAE"] >= wf["naive_MAE"]],
               color="red", s=30, marker="x", alpha=0.8, label="LightGBM loses")
    ax.set_xlabel("Week step")
    ax.set_ylabel("LightGBM MAE (EUR/MWh)")
    ax.set_title(f"Win rate {win_rate:.0f}%: {zone}")
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.10, 1.02, "B", transform=ax.transAxes, fontweight='bold', fontsize=12, va='bottom')

    # Panel C: Seasonal distribution (quarterly box plots of LightGBM MAE)
    ax = axes[2]
    wf2 = wf.copy()
    wf2["window_start_dt"] = pd.to_datetime(wf2["window_start"])
    wf2["quarter"] = wf2["window_start_dt"].dt.quarter.map({1: "Q1\n(Jan-Mar)", 2: "Q2\n(Apr-Jun)", 3: "Q3\n(Jul-Sep)", 4: "Q4\n(Oct-Dec)"})
    quarters = ["Q1\n(Jan-Mar)", "Q2\n(Apr-Jun)", "Q3\n(Jul-Sep)", "Q4\n(Oct-Dec)"]
    data_by_q = [wf2[wf2["quarter"] == q]["lgbm_MAE"].values for q in quarters]
    bp = ax.boxplot(data_by_q, tick_labels=quarters, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("LightGBM MAE (EUR/MWh)")
    ax.set_title(f"Seasonal distribution: {zone}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.10, 1.02, "C", transform=ax.transAxes, fontweight='bold', fontsize=12, va='bottom')

    # Panel D: Cumulative distribution of MAE reduction (%) over Naive
    ax = axes[3]
    reduction = 100 * (wf["naive_MAE"] - wf["lgbm_MAE"]) / wf["naive_MAE"]
    sorted_r = np.sort(reduction)
    cum_frac = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
    ax.plot(sorted_r, cum_frac, color=color, linewidth=1.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("MAE reduction over Nave-24h (%)")
    ax.set_ylabel("Cumulative fraction of weeks")
    ax.set_title(f"CDF of MAE reduction: {zone}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.10, 1.02, "D", transform=ax.transAxes, fontweight='bold', fontsize=12, va='bottom')

    fig.tight_layout()
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_OUT / "fig5.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(PAPER_OUT / "fig5.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  Saved fig5.pdf / fig5.png")


def make_fig5b(all_wf: dict[str, pd.DataFrame]) -> None:
    """Multi-zone summary 4-panel figure."""
    zones = list(all_wf.keys())
    if not zones:
        print("  No walk-forward data available for multi-zone fig5b")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axes = axes.flatten()
    colors = [ZONE_COLORS.get(z, "#888") for z in zones]

    # Panel A: Mean weekly LightGBM MAE per zone (bar chart)
    ax = axes[0]
    means = [all_wf[z]["lgbm_MAE"].mean() for z in zones]
    stds  = [all_wf[z]["lgbm_MAE"].std() for z in zones]
    bars = ax.bar(zones, means, color=colors, alpha=0.85, width=0.6)
    ax.errorbar(zones, means, yerr=stds, fmt='none', color='black', capsize=4, linewidth=1)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.1, f"{m:.2f}",
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel("Mean weekly MAE (EUR/MWh)")
    ax.set_title("LightGBM walk-forward MAE by zone")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.10, 1.02, "A", transform=ax.transAxes, fontweight='bold', fontsize=12, va='bottom')

    # Panel B: Win rate per zone
    ax = axes[1]
    win_rates = []
    for z in zones:
        wf = all_wf[z]
        wr = 100 * (wf["lgbm_MAE"] < wf["naive_MAE"]).sum() / len(wf)
        win_rates.append(wr)
    bars = ax.bar(zones, win_rates, color=colors, alpha=0.85, width=0.6)
    for bar, wr in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, wr + 0.5, f"{wr:.0f}%",
                ha='center', va='bottom', fontsize=8)
    ax.axhline(50, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylim(0, 110)
    ax.set_ylabel("% of weeks LightGBM < Nave-24h")
    ax.set_title("Win rate by zone")
    ax.text(-0.10, 1.02, "B", transform=ax.transAxes, fontweight='bold', fontsize=12, va='bottom')

    # Panel C: Box plots of weekly LightGBM MAE, one box per zone
    ax = axes[2]
    data_by_zone = [all_wf[z]["lgbm_MAE"].values for z in zones]
    bp = ax.boxplot(data_by_zone, tick_labels=zones, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Weekly LightGBM MAE (EUR/MWh)")
    ax.set_title("Distribution of weekly MAE by zone")
    ax.text(-0.10, 1.02, "C", transform=ax.transAxes, fontweight='bold', fontsize=12, va='bottom')

    # Panel D: LightGBM vs Naive mean MAE comparison
    ax = axes[3]
    x = np.arange(len(zones))
    w = 0.35
    lgbm_means  = [all_wf[z]["lgbm_MAE"].mean() for z in zones]
    naive_means = [all_wf[z]["naive_MAE"].mean() for z in zones]
    b1 = ax.bar(x - w/2, lgbm_means,  w, color=colors, alpha=0.85, label="LightGBM")
    b2 = ax.bar(x + w/2, naive_means, w, color='lightgray', alpha=0.85, label="Nave-24h")
    ax.set_xticks(x)
    ax.set_xticklabels(zones)
    ax.set_ylabel("Mean weekly MAE (EUR/MWh)")
    ax.set_title("LightGBM vs Nave-24h by zone")
    ax.legend()
    ax.text(-0.10, 1.02, "D", transform=ax.transAxes, fontweight='bold', fontsize=12, va='bottom')

    fig.tight_layout()
    fig.savefig(PAPER_OUT / "fig5b.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(PAPER_OUT / "fig5b.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  Saved fig5b.pdf / fig5b.png")


if __name__ == "__main__":
    print("Loading walk-forward data ...")
    all_wf = load_wf_data()
    if not all_wf:
        print("No walk-forward data found. Run models/walk_forward.py first.")
    else:
        available = list(all_wf.keys())
        print(f"  Loaded zones: {available}")
        main_zone = "NO1" if "NO1" in all_wf else available[0]
        print("Generating fig5 (detailed, NO1) ...")
        make_fig5(all_wf[main_zone], zone=main_zone)
        print("Generating fig5b (multi-zone summary) ...")
        make_fig5b(all_wf)
    print("Done.")
