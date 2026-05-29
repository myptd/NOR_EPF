"""
Data overview analysis for Norwegian electricity price forecasting.
All five zones (NO1-NO5).

Generates:
  paper_outputs/fig1.pdf/png  -- price time series and distribution
  paper_outputs/fig2.pdf/png  -- seasonality, ACF, met drivers

Run:
    python analysis/01_data_overview.py
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.utils import load_zone

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
TARGET = "price_eur_mwh"
PAPER_OUT = Path("paper_outputs")
DATA_DIR = Path("data/cleaned")

VAL_START  = pd.Timestamp("2024-01-01", tz="Europe/Oslo")
TEST_START = pd.Timestamp("2025-01-01", tz="Europe/Oslo")


def load_all_zones() -> dict[str, pd.DataFrame]:
    dfs = {}
    for z in ZONES:
        p = DATA_DIR / f"{z}_hourly.parquet"
        if p.exists():
            df = load_zone(str(p))
            dfs[z] = df
            print(f"  Loaded {z}: {len(df)} rows")
        else:
            print(f"  Warning: {p} not found")
    return dfs


def acf_manual(series: np.ndarray, max_lag: int = 200) -> np.ndarray:
    """Compute ACF without statsmodels."""
    s = series - series.mean()
    var = np.dot(s, s)
    acf_vals = np.zeros(max_lag + 1)
    acf_vals[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf_vals[lag] = np.dot(s[:-lag], s[lag:]) / var
    return acf_vals


def season_of(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    return "Autumn"


# ---------------------------------------------------------------------------
# FIG 1
# ---------------------------------------------------------------------------

def make_fig1(dfs: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axes = axes.flatten()
    zone_list = list(dfs.keys())

    # Panel A: daily mean price time series, all zones overlaid
    ax = axes[0]
    for z, df in dfs.items():
        daily = df[TARGET].resample("D").mean()
        ax.plot(daily.index, daily.values, color=ZONE_COLORS[z],
                linewidth=0.7, label=z, alpha=0.85)
    xmin = min(df.index.min() for df in dfs.values())
    xmax = max(df.index.max() for df in dfs.values())
    ax.axvspan(xmin, VAL_START,  alpha=0.05, color="blue")
    ax.axvspan(VAL_START,  TEST_START, alpha=0.10, color="orange")
    ax.axvspan(TEST_START, xmax,        alpha=0.10, color="red")
    ax.axvline(VAL_START,  color="grey", linestyle="--", linewidth=0.9)
    ax.axvline(TEST_START, color="grey", linestyle="--", linewidth=0.9)
    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=7, ncol=5)
    ax.text(-0.10, 1.02, "A", transform=ax.transAxes,
            fontweight='bold', fontsize=12, va='bottom')

    # Panel B: yearly grouped box plots (7 years x 5 zones = 35 boxes)
    ax = axes[1]
    years = sorted({y for df in dfs.values() for y in df.index.year.unique()})
    n_zones = len(zone_list)
    width = 0.8 / n_zones
    positions_all, data_all, colors_all = [], [], []
    for yi, year in enumerate(years):
        for zi, z in enumerate(zone_list):
            df = dfs[z]
            vals = df.loc[df.index.year == year, TARGET].dropna().values
            pos = yi * 1.1 + (zi - n_zones / 2 + 0.5) * width
            positions_all.append(pos)
            data_all.append(vals)
            colors_all.append(ZONE_COLORS[z])
    bp = ax.boxplot(data_all, positions=positions_all, widths=width * 0.85,
                    patch_artist=True, sym="", whis=[5, 95],
                    medianprops=dict(color="black", linewidth=1.2))
    for patch, color in zip(bp["boxes"], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks([yi * 1.1 for yi in range(len(years))])
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_xlabel("Year")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=ZONE_COLORS[z], label=z) for z in zone_list],
              fontsize=7, ncol=3)
    ax.text(-0.10, 1.02, "B", transform=ax.transAxes,
            fontweight='bold', fontsize=12, va='bottom')

    # Panel C: hourly price profile, mean +/- 1 sigma
    ax = axes[2]
    for z, df in dfs.items():
        hp = df.groupby(df.index.hour)[TARGET]
        hmean, hstd = hp.mean(), hp.std()
        ax.plot(hmean.index, hmean.values, color=ZONE_COLORS[z], linewidth=1.2, label=z)
        ax.fill_between(hmean.index, hmean - hstd, hmean + hstd,
                        color=ZONE_COLORS[z], alpha=0.12)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Price (EUR/MWh)")
    ax.set_xticks(range(0, 24, 4))
    ax.legend(fontsize=7)
    ax.text(-0.10, 1.02, "C", transform=ax.transAxes,
            fontweight='bold', fontsize=12, va='bottom')

    # Panel D: violin plots per zone
    ax = axes[3]
    vdata = [dfs[z][TARGET].dropna().values for z in zone_list]
    parts = ax.violinplot(vdata, positions=range(len(zone_list)), showmedians=True,
                          showextrema=False)
    for pc, z in zip(parts["bodies"], zone_list):
        pc.set_facecolor(ZONE_COLORS[z])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(zone_list)))
    ax.set_xticklabels(zone_list)
    ax.set_ylabel("Price (EUR/MWh)")
    ax.text(-0.10, 1.02, "D", transform=ax.transAxes,
            fontweight='bold', fontsize=12, va='bottom')

    fig.tight_layout()
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_OUT / "fig1.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(PAPER_OUT / "fig1.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  Saved fig1.pdf / fig1.png")


# ---------------------------------------------------------------------------
# FIG 2
# ---------------------------------------------------------------------------

def make_fig2(dfs: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axes = axes.flatten()
    zone_list = list(dfs.keys())

    # Panel A: seasonal price profiles by month
    ax = axes[0]
    months = list(range(1, 13))
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    for z, df in dfs.items():
        mp = df.groupby(df.index.month)[TARGET]
        mm  = mp.mean()
        mq1 = mp.quantile(0.25)
        mq3 = mp.quantile(0.75)
        ax.plot(months, [mm.get(m, np.nan) for m in months],
                color=ZONE_COLORS[z], linewidth=1.2, label=z)
        ax.fill_between(months,
                        [mq1.get(m, np.nan) for m in months],
                        [mq3.get(m, np.nan) for m in months],
                        color=ZONE_COLORS[z], alpha=0.12)
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels, rotation=30)
    ax.set_ylabel("Mean Price (EUR/MWh)")
    ax.set_xlabel("Month")
    ax.legend(fontsize=7)
    ax.text(-0.10, 1.02, "A", transform=ax.transAxes,
            fontweight='bold', fontsize=12, va='bottom')

    # Panel B: ACF for all zones up to lag 200
    ax = axes[1]
    for z, df in dfs.items():
        s = df[TARGET].dropna().values[:8760 * 3]
        acf_v = acf_manual(s, max_lag=200)
        ax.plot(range(201), acf_v, color=ZONE_COLORS[z], linewidth=0.9, label=z)
    ax.axvline(24,  color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(168, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(24,  0.05,  "24h",  fontsize=7, ha="center", color="grey", transform=ax.get_xaxis_transform())
    ax.text(168, 0.05, "168h", fontsize=7, ha="center", color="grey", transform=ax.get_xaxis_transform())
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("ACF")
    ax.set_xlim(0, 200)
    ax.legend(fontsize=7)
    ax.text(-0.10, 1.02, "B", transform=ax.transAxes,
            fontweight='bold', fontsize=12, va='bottom')

    # Panel C: mean price by day-of-week, grouped bars
    ax = axes[2]
    dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    n_zones = len(zone_list)
    width = 0.8 / n_zones
    x = np.arange(7)
    for zi, z in enumerate(zone_list):
        dp = dfs[z].groupby(dfs[z].index.dayofweek)[TARGET].mean()
        vals = [dp.get(d, np.nan) for d in range(7)]
        offset = (zi - n_zones / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=z, color=ZONE_COLORS[z], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(dow_labels)
    ax.set_ylabel("Mean Price (EUR/MWh)")
    ax.set_xlabel("Day of Week")
    ax.legend(fontsize=7)
    ax.text(-0.10, 1.02, "C", transform=ax.transAxes,
            fontweight='bold', fontsize=12, va='bottom')

    # Panel D: scatter price vs temperature, all zones
    ax = axes[3]
    rng = np.random.default_rng(42)
    for z, df in dfs.items():
        if "wx_temperature_2m" not in df.columns:
            continue
        sub = df[["wx_temperature_2m", TARGET]].dropna()
        if len(sub) > 2000:
            idx = rng.choice(len(sub), 2000, replace=False)
            sub = sub.iloc[idx]
        ax.scatter(sub["wx_temperature_2m"], sub[TARGET],
                   color=ZONE_COLORS[z], alpha=0.2, s=3, label=z)
        slope, intercept, *_ = stats.linregress(
            sub["wx_temperature_2m"].values, sub[TARGET].values)
        tx = np.array([sub["wx_temperature_2m"].min(), sub["wx_temperature_2m"].max()])
        ax.plot(tx, slope * tx + intercept, color=ZONE_COLORS[z], linewidth=1.5)
    ax.set_xlabel("Temperature (deg C)")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.legend(fontsize=7, markerscale=3)
    ax.text(-0.10, 1.02, "D", transform=ax.transAxes,
            fontweight='bold', fontsize=12, va='bottom')

    fig.tight_layout(pad=1.5, h_pad=2.5)
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_OUT / "fig2.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(PAPER_OUT / "fig2.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  Saved fig2.pdf / fig2.png")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading all zones ...")
    dfs = load_all_zones()
    if not dfs:
        print("ERROR: No data loaded. Check data/cleaned/*.parquet files.")
        return

    print("\nGenerating fig1 ...")
    make_fig1(dfs)

    print("\nGenerating fig2 ...")
    make_fig2(dfs)

    print("\nDone.")


if __name__ == "__main__":
    main()
