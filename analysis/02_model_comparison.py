"""
Model performance comparison figures for the paper.
All five zones (NO1-NO5), all models including LSTM.

Outputs (paper_outputs/):
  fig3.pdf/png  -- model metric comparison, all zones, all models
  fig4.pdf/png  -- LightGBM prediction quality, NO1
  table3.csv        -- per-zone results table (→ manuscript Table 3)
  table4.csv        -- cross-zone mean results table (→ manuscript Table 4)

All data read from model_results/*.csv only.

Run:
    python analysis/02_model_comparison.py
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

# Canonical model order (including all DL models)
MODEL_ORDER = ["Naive-24h", "Naive-168h", "Linear-ARX (Ridge)",
               "LightGBM", "XGBoost", "LSTM", "TCN", "Transformer"]
MODEL_COLORS = {
    "Naive-24h":          "#aec7e8",
    "Naive-168h":         "#c5b0d5",
    "Linear-ARX (Ridge)": "#ffbb78",
    "LightGBM":           "#98df8a",
    "XGBoost":            "#ff9896",
    "LSTM":               "#c49c94",
    "TCN":                "#17becf",
    "Transformer":        "#bcbd22",
}

PAPER_OUT    = Path("paper_outputs")
MODEL_RESULTS = Path("model_results")


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_all_zones_results() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load all_zones_{val,test}.csv and append LSTM rows from per-zone CSVs."""
    def _read(split: str) -> pd.DataFrame | None:
        p = MODEL_RESULTS / f"all_zones_{split}.csv"
        if not p.exists():
            print(f"  Warning: {p} not found.")
            return None
        df = pd.read_csv(p)
        # Append deep learning model results per zone
        dl_rows = []
        for z in ZONES:
            for model in ("lstm", "tcn", "transformer"):
                lp = MODEL_RESULTS / f"{z}_{model}_{split}.csv"
                if lp.exists():
                    row = pd.read_csv(lp)
                    row["zone"] = z
                    dl_rows.append(row)
        if dl_rows:
            dl_df = pd.concat(dl_rows, ignore_index=True)
            df = pd.concat([df, dl_df], ignore_index=True)
        _NAME_NORM = {"TRANSFORMER": "Transformer", "LSTM": "LSTM", "TCN": "TCN"}
        df["model"] = df["model"].str.strip().replace(_NAME_NORM)
        return df

    return _read("val"), _read("test")


def get_ordered_models(df: pd.DataFrame) -> list[str]:
    available = df["model"].unique().tolist()
    ordered   = [m for m in MODEL_ORDER if m in available]
    return ordered + [m for m in available if m not in ordered]


def season_of(month: int) -> str:
    if month in (12, 1, 2):  return "Winter"
    if month in (3, 4, 5):   return "Spring"
    if month in (6, 7, 8):   return "Summer"
    return "Autumn"


# ---------------------------------------------------------------------------
# TABLE CSV OUTPUTS
# ---------------------------------------------------------------------------

def write_table3_csv(test_df: pd.DataFrame) -> None:
    """Write per-zone per-model results → table3.csv (manuscript Table 3)."""
    models = get_ordered_models(test_df)
    rows = []
    for z in ZONES:
        for m in models:
            sub = test_df[(test_df["zone"] == z) & (test_df["model"] == m)]
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            rows.append({
                "zone": z, "model": m,
                "MAE":   round(r["MAE"],   3),
                "RMSE":  round(r["RMSE"],  3),
                "sMAPE": round(r["sMAPE"], 2),
                "R2":    round(r["R2"],    4),
            })
    pd.DataFrame(rows).to_csv(PAPER_OUT / "table3.csv", index=False)
    print("  Saved table3.csv")


def write_table4_csv(test_df: pd.DataFrame) -> None:
    """Write cross-zone mean results → table4.csv (manuscript Table 4)."""
    models = get_ordered_models(test_df)
    rows = []
    for m in models:
        sub = test_df[test_df["model"] == m]
        if len(sub) == 0:
            continue
        rows.append({
            "model":  m,
            "MAE":    round(sub["MAE"].mean(),   3),
            "RMSE":   round(sub["RMSE"].mean(),  3),
            "sMAPE":  round(sub["sMAPE"].mean(), 2),
            "R2":     round(sub["R2"].mean(),    4),
        })
    pd.DataFrame(rows).to_csv(PAPER_OUT / "table4.csv", index=False)
    print("  Saved table4.csv")


# ---------------------------------------------------------------------------
# FIG 3 — All-zone, all-model comparison (includes LSTM)
# ---------------------------------------------------------------------------

def make_fig3(val_df: pd.DataFrame | None, test_df: pd.DataFrame | None) -> None:
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    if test_df is None:
        print("  Skipping fig3: no test data")
        return

    models       = get_ordered_models(test_df)
    model_colors = [MODEL_COLORS.get(m, "#888888") for m in models]
    n_models     = len(models)
    n_zones      = len(ZONES)
    width        = 0.8 / n_models

    def grouped_bars(ax, df, metric):
        x = np.arange(n_zones)
        for mi, (m, mc) in enumerate(zip(models, model_colors)):
            vals = []
            for z in ZONES:
                sub = df[(df["model"] == m) & (df["zone"] == z)]
                vals.append(float(sub[metric].values[0]) if len(sub) > 0 else np.nan)
            offset = (mi - n_models / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=m, color=mc, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(ZONES)
        ax.set_ylabel(metric)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    axes = axes.flatten()
    panel_labels = ["A", "B", "C", "D", "E", "F"]

    # Panels A–D: MAE, RMSE, R2, sMAPE grouped bars (test set)
    for ax, metric, lbl in zip(axes[:4], ["MAE", "RMSE", "R2", "sMAPE"], panel_labels[:4]):
        grouped_bars(ax, test_df, metric)
        ax.set_title(f"{metric} by zone (test 2025)", fontsize=9)
        if lbl == "A":
            ax.legend(fontsize=6, ncol=2)
        ax.text(-0.10, 1.02, lbl, transform=ax.transAxes,
                fontweight='bold', fontsize=12, va='bottom')

    # Panel E: val vs test MAE for LightGBM per zone
    ax = axes[4]
    x = np.arange(n_zones)
    w = 0.35
    lgbm_val, lgbm_test = [], []
    for z in ZONES:
        sv = (val_df[(val_df["model"] == "LightGBM") & (val_df["zone"] == z)]
              if val_df is not None else pd.DataFrame())
        st = test_df[(test_df["model"] == "LightGBM") & (test_df["zone"] == z)]
        lgbm_val.append(float(sv["MAE"].values[0]) if len(sv) > 0 else np.nan)
        lgbm_test.append(float(st["MAE"].values[0]) if len(st) > 0 else np.nan)
    ax.bar(x - w/2, lgbm_val,  w, label="Val 2024",  color="#2166ac", alpha=0.85)
    ax.bar(x + w/2, lgbm_test, w, label="Test 2025", color="#d6604d", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(ZONES)
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("LightGBM: val vs test MAE", fontsize=9)
    ax.legend(fontsize=7)
    ax.text(-0.10, 1.02, "E", transform=ax.transAxes,
            fontweight='bold', fontsize=12, va='bottom')

    # Panel F: MAE % improvement over Naive-24h, lines per non-naïve model
    ax = axes[5]
    x = np.arange(n_zones)
    for m in models:
        if m in ("Naive-24h", "Naive-168h"):
            continue
        improvements = []
        for z in ZONES:
            nsub = test_df[(test_df["model"] == "Naive-24h") & (test_df["zone"] == z)]
            msub = test_df[(test_df["model"] == m) & (test_df["zone"] == z)]
            if len(nsub) > 0 and len(msub) > 0:
                improvements.append(100 * (float(nsub["MAE"].values[0]) -
                                           float(msub["MAE"].values[0])) /
                                    float(nsub["MAE"].values[0]))
            else:
                improvements.append(np.nan)
        ax.plot(x, improvements, marker='o', markersize=5,
                color=MODEL_COLORS.get(m, "#888"), linewidth=1.5, label=m)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(ZONES)
    ax.set_ylabel("MAE improvement over Naïve-24h (%)")
    ax.set_title("MAE reduction vs Naïve-24h", fontsize=9)
    ax.legend(fontsize=6)
    ax.text(-0.10, 1.02, "F", transform=ax.transAxes,
            fontweight='bold', fontsize=12, va='bottom')

    fig.suptitle("Model Performance Comparison – All Zones (Test 2025)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PAPER_OUT / "fig3.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(PAPER_OUT / "fig3.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  Saved fig3.pdf / fig3.png")


# ---------------------------------------------------------------------------
# FIG 4 — LightGBM quality, NO1 (data from model_results)
# ---------------------------------------------------------------------------

def make_fig4(val_df: pd.DataFrame | None, test_df: pd.DataFrame | None) -> None:
    """4-panel figure using only model_results data (no model weights required)."""
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    if test_df is None:
        print("  Skipping fig4: no test data"); return

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axes = axes.flatten()

    # Panel A: all models MAE/R2 for NO1 (horizontal grouped bar)
    ax = axes[0]
    no1 = test_df[test_df["zone"] == "NO1"]
    models_no1 = get_ordered_models(no1)
    y_pos = np.arange(len(models_no1))
    maes  = [float(no1[no1["model"] == m]["MAE"].values[0]) for m in models_no1]
    cols  = [MODEL_COLORS.get(m, "#888") for m in models_no1]
    bars  = ax.barh(y_pos, maes, color=cols, alpha=0.85)
    for bar, v in zip(bars, maes):
        ax.text(v + 0.1, bar.get_y() + bar.get_height()/2,
                f"{v:.2f}", va="center", ha="left", fontsize=8)
    ax.set_yticks(y_pos); ax.set_yticklabels(models_no1, fontsize=8)
    ax.set_xlabel("MAE (EUR/MWh)")
    ax.set_title("NO1 – all models (test 2025)", fontsize=9)
    ax.text(-0.14, 1.02, "A", transform=ax.transAxes,
            fontweight='bold', fontsize=12, va='bottom')

    # Panel B: scatter plot of MAE vs R2 for all zones + all models (bubble chart)
    ax = axes[1]
    for m in models_no1:
        maes_all = []
        r2s_all  = []
        for z in ZONES:
            sub = test_df[(test_df["zone"] == z) & (test_df["model"] == m)]
            if len(sub) > 0:
                maes_all.append(float(sub["MAE"].values[0]))
                r2s_all.append(float(sub["R2"].values[0]))
        if maes_all:
            ax.scatter(maes_all, r2s_all, color=MODEL_COLORS.get(m, "#888"),
                       s=60, alpha=0.8, label=m, zorder=3)
    ax.set_xlabel("MAE (EUR/MWh)")
    ax.set_ylabel("R²")
    ax.set_title("MAE vs R² (all zones × all models)", fontsize=9)
    ax.legend(fontsize=6, ncol=1)
    ax.text(-0.14, 1.02, "B", transform=ax.transAxes,
            fontweight='bold', fontsize=12, va='bottom')

    # Panel C: MAE improvement over Naive-24h per zone, all ML models (heatmap)
    ax = axes[2]
    ml_models = [m for m in get_ordered_models(test_df) if m not in ("Naive-24h", "Naive-168h")]
    imp_matrix = np.full((len(ml_models), len(ZONES)), np.nan)
    for mi, m in enumerate(ml_models):
        for zi, z in enumerate(ZONES):
            nsub = test_df[(test_df["model"] == "Naive-24h") & (test_df["zone"] == z)]
            msub = test_df[(test_df["model"] == m) & (test_df["zone"] == z)]
            if len(nsub) > 0 and len(msub) > 0:
                imp_matrix[mi, zi] = (100 * (float(nsub["MAE"].values[0]) -
                                              float(msub["MAE"].values[0])) /
                                      float(nsub["MAE"].values[0]))
    im = ax.imshow(imp_matrix, cmap="RdYlGn", vmin=-20, vmax=80, aspect="auto")
    ax.set_xticks(range(len(ZONES))); ax.set_xticklabels(ZONES, fontsize=8)
    ax.set_yticks(range(len(ml_models))); ax.set_yticklabels(ml_models, fontsize=8)
    for mi in range(len(ml_models)):
        for zi in range(len(ZONES)):
            v = imp_matrix[mi, zi]
            if not np.isnan(v):
                ax.text(zi, mi, f"{v:.1f}%", ha="center", va="center", fontsize=7,
                        color="black" if abs(v) < 60 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8, label="MAE reduction (%)")
    ax.set_title("MAE reduction vs Naïve-24h (%)", fontsize=9)
    ax.text(-0.14, 1.02, "C", transform=ax.transAxes,
            fontweight='bold', fontsize=12, va='bottom')

    # Panel D: val vs test MAE for all models (scatter, diagonal = equal)
    ax = axes[3]
    if val_df is not None:
        for m in get_ordered_models(test_df):
            val_maes, test_maes = [], []
            for z in ZONES:
                vsub = val_df[(val_df["model"] == m) & (val_df["zone"] == z)]
                tsub = test_df[(test_df["model"] == m) & (test_df["zone"] == z)]
                if len(vsub) > 0 and len(tsub) > 0:
                    val_maes.append(float(vsub["MAE"].values[0]))
                    test_maes.append(float(tsub["MAE"].values[0]))
            if val_maes:
                ax.scatter(val_maes, test_maes, color=MODEL_COLORS.get(m, "#888"),
                           s=45, alpha=0.85, label=m, zorder=3)
        lims_max = max(test_df["MAE"].max(), val_df["MAE"].max()) * 1.05
        ax.plot([0, lims_max], [0, lims_max], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Val MAE 2024 (EUR/MWh)")
        ax.set_ylabel("Test MAE 2025 (EUR/MWh)")
        ax.legend(fontsize=6)
    else:
        ax.text(0.5, 0.5, "No validation data", transform=ax.transAxes, ha="center")
    ax.set_title("Validation vs test MAE (all zones)", fontsize=9)
    ax.text(-0.14, 1.02, "D", transform=ax.transAxes,
            fontweight='bold', fontsize=12, va='bottom')

    fig.tight_layout()
    fig.savefig(PAPER_OUT / "fig4.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(PAPER_OUT / "fig4.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  Saved fig4.pdf / fig4.png")


# ---------------------------------------------------------------------------
# TEXT OUTPUTS
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading results (model_results/*.csv) ...")
    val_df, test_df = load_all_zones_results()
    if test_df is not None:
        print(f"  Test models: {sorted(test_df['model'].unique().tolist())}")
        print(f"  Zones: {sorted(test_df['zone'].unique().tolist())}")

    print("\nWriting Table 3 + Table 4 CSVs ...")
    if test_df is not None:
        write_table3_csv(test_df)
        write_table4_csv(test_df)

    print("\nGenerating fig3 ...")
    make_fig3(val_df, test_df)

    print("\nGenerating fig4 ...")
    make_fig4(val_df, test_df)

    print("\nDone.")


if __name__ == "__main__":
    main()

