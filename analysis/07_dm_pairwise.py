"""
Pairwise Diebold-Mariano tests: LightGBM vs Linear-ARX (Ridge), all zones.

Uses one-sided DM test (H1: LightGBM better = negative DM stat) with
Harvey-Leybourne-Newbold (1997) small-sample correction and Newey-West HAC
variance estimator for h=24 day-ahead horizon.

Generates:
  model_results/dm_pairwise.csv
  paper_outputs/fig8.pdf / fig8.png

Run:
    python analysis/07_dm_pairwise.py
"""
from __future__ import annotations

import sys
import faulthandler; faulthandler.enable()
from pathlib import Path

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
ZONE_COLORS = {
    "NO1": "#1f77b4", "NO2": "#ff7f0e", "NO3": "#2ca02c",
    "NO4": "#d62728", "NO5": "#9467bd"
}
PAPER_OUT = Path("paper_outputs")
MODEL_RES  = Path("model_results")
H = 24  # day-ahead horizon correction


def get_errors(zone: str) -> dict[str, np.ndarray] | None:
    """Fit Ridge and load LightGBM to extract test-set error arrays."""
    import joblib
    from scipy import stats
    from models.utils import (
        load_zone, build_feature_matrix, train_val_test_split,
        diebold_mariano, SEED, set_seed,
    )
    from models.baseline import linear_arx
    set_seed()

    data_path = f"data/cleaned/{zone}_hourly.parquet"
    if not Path(data_path).exists():
        print(f"  Skipping {zone}: {data_path} not found")
        return None

    lgbm_path = Path(f"model_weights/lgbm_{zone}.pkl")
    if not lgbm_path.exists():
        print(f"  Skipping {zone}: {lgbm_path} not found (run models/run_all.py first)")
        return None

    lgbm_booster = joblib.load(lgbm_path)

    df = load_zone(data_path)
    X, y = build_feature_matrix(df)
    X_tr, X_val, X_te, y_tr, y_val, y_te = train_val_test_split(X, y)

    # LightGBM errors
    lgbm_pred    = lgbm_booster.predict(X_te)
    lgbm_err   = y_te.values - lgbm_pred

    # Ridge ARX errors (refit — same pipeline as baseline.py)
    _, ridge_res = linear_arx(X_tr, X_val, X_te, y_tr, y_val, y_te)
    # ridge_res is a metrics dict — we need raw predictions; refit to get them
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV, PredefinedSplit

    X_tv = pd.concat([X_tr, X_val])
    y_tv = pd.concat([y_tr, y_val])
    split_idx = np.concatenate([
        np.full(len(X_tr), -1), np.zeros(len(X_val), dtype=int)
    ])
    ps = PredefinedSplit(split_idx)
    pipe = Pipeline([("sc", StandardScaler()), ("ridge", Ridge())])
    grid = GridSearchCV(pipe, {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]},
                        cv=ps, scoring="neg_mean_absolute_error", refit=True, n_jobs=1)
    grid.fit(X_tv, y_tv)
    ridge_pred = grid.predict(X_te)
    ridge_err  = y_te.values - ridge_pred

    # Also Naive-24h errors for reference
    naive_pred = y_te.values.copy()
    # shift by 24: use values 24 steps before test start
    # grab series and shift
    full_price = df["price_eur_mwh"]
    test_start = X_te.index[0]
    naive_series = full_price.shift(24)[X_te.index]
    ok = naive_series.notna()
    naive_err = (y_te[ok].values - naive_series[ok].values)

    n_min = min(len(lgbm_err), len(ridge_err))
    return {
        "lgbm":  lgbm_err[-n_min:],
        "ridge": ridge_err[-n_min:],
        "naive": naive_err[-(n_min):] if len(naive_err) >= n_min else naive_err,
        "zone":  zone,
        "n":     n_min,
    }


def run_dm_pairwise(errors_a: np.ndarray, errors_b: np.ndarray,
                    h: int = H, one_sided: bool = True) -> tuple[float, float]:
    """
    Diebold-Mariano test with HLN correction.
    one_sided=True tests H1: model A has lower MSE (negative DM = A is better).
    Returns (dm_stat, p_value).
    """
    from scipy import stats
    n_min = min(len(errors_a), len(errors_b))
    ea = np.asarray(errors_a[-n_min:], dtype=float)
    eb = np.asarray(errors_b[-n_min:], dtype=float)

    d = ea ** 2 - eb ** 2   # positive = A is worse
    n = len(d)
    d_bar = d.mean()

    # Newey-West HAC variance
    gamma0 = np.mean((d - d_bar) ** 2)
    nw_var = gamma0
    for lag in range(1, h):
        gk = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        nw_var += 2 * (1 - lag / h) * gk
    nw_var = max(nw_var, 1e-14)

    # HLN correction factor
    hlp = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_raw = d_bar / np.sqrt(nw_var / n)
    dm_stat = float(dm_raw * hlp)

    if one_sided:
        # H1: A better → dm_stat should be negative
        p_value = float(stats.t.cdf(dm_stat, df=n - 1))
    else:
        p_value = float(2 * stats.t.sf(abs(dm_stat), df=n - 1))

    return round(dm_stat, 4), round(p_value, 5)


def make_fig8(records: list[dict]) -> None:
    """Heatmap + bar of DM test results across zones."""
    df = pd.DataFrame(records)

    pairs = [
        ("lgbm_vs_ridge", "LightGBM vs Ridge"),
        ("lgbm_vs_naive", "LightGBM vs Naive-24h"),
        ("ridge_vs_naive", "Ridge vs Naive-24h"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
    panel_labels = ["A", "B", "C"]

    for ax, (col_dm, label), plbl in zip(axes, pairs, panel_labels):
        col_p = col_dm.replace("_vs_", "_p_").replace("lgbm", "lgbm").replace("ridge", "ridge").replace("naive","naive")
        # map pair names to correct col prefixes
        dm_col = col_dm + "_dm"
        p_col  = col_dm + "_p"

        if dm_col not in df.columns:
            ax.text(0.5, 0.5, f"No data for\n{label}", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        zones = df["zone"].tolist()
        dm_vals = df[dm_col].values
        p_vals  = df[p_col].values

        colors = []
        for dm, p in zip(dm_vals, p_vals):
            if p < 0.01:
                colors.append("#2166ac" if dm < 0 else "#d73027")  # significant
            elif p < 0.05:
                colors.append("#74add1" if dm < 0 else "#f46d43")
            else:
                colors.append("#aaaaaa")

        bars = ax.barh(zones, dm_vals, color=colors, alpha=0.9, height=0.6)
        ax.axvline(0, color="black", linewidth=0.8)

        # Annotate with p-value stars — place text inside the bar
        for i, (dm, p) in enumerate(zip(dm_vals, p_vals)):
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            # Place label just inside the bar end
            if abs(dm) > 0.5:
                xpos = dm * 0.85
                ha = "center"
                col = "white"
            else:
                xpos = dm + (0.05 if dm >= 0 else -0.05)
                ha = "left" if dm >= 0 else "right"
                col = "black"
            ax.text(xpos, i, sig, va="center", ha=ha, fontsize=7, color=col, fontweight="bold")

        # Compute symmetric xlim with 10% padding
        abs_max = max(abs(dm_vals.min()), abs(dm_vals.max()), 0.5)
        pad = abs_max * 0.15
        ax.set_xlim(-abs_max - pad, abs_max + pad)

        ax.set_title(label, fontsize=9)
        ax.set_xlabel("DM statistic  (negative = left model better)")
        ax.set_ylabel("Zone")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Legend
        from matplotlib.patches import Patch
        legend_els = [
            Patch(facecolor="#2166ac", label="p<0.01 (better)"),
            Patch(facecolor="#d73027", label="p<0.01 (worse)"),
            Patch(facecolor="#aaaaaa", label="ns"),
        ]
        ax.legend(handles=legend_els, loc="lower right", fontsize=7)
        ax.text(-0.10, 1.02, plbl, transform=ax.transAxes,
                fontweight="bold", fontsize=12, va="bottom")

    fig.tight_layout()
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_OUT / "fig8.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(PAPER_OUT / "fig8.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  Saved fig8.pdf / fig8.png")


def main() -> None:
    print("\n=== Pairwise DM Tests: LightGBM vs Ridge ARX, all zones ===\n")

    # Prefer pre-computed results from model_results/dm_pairwise.csv
    csv_path = MODEL_RES / "dm_pairwise.csv"
    if csv_path.exists():
        print(f"  Loading pre-computed results from {csv_path}")
        result_df = pd.read_csv(csv_path)
        records = result_df.to_dict("records")
    else:
        # Fall back to computing from model weights
        records = []
        for zone in ZONES:
            print(f"  Processing {zone} ...")
            errs = get_errors(zone)
            if errs is None:
                continue

            lgbm  = errs["lgbm"]
            ridge = errs["ridge"]
            naive = errs["naive"]

            n_min3 = min(len(lgbm), len(ridge), len(naive))
            lgbm  = lgbm[-n_min3:]
            ridge = ridge[-n_min3:]
            naive = naive[-n_min3:]

            dm_lr, p_lr = run_dm_pairwise(lgbm, ridge, one_sided=True)
            dm_ln, p_ln = run_dm_pairwise(lgbm, naive, one_sided=True)
            dm_rn, p_rn = run_dm_pairwise(ridge, naive, one_sided=True)

            sig_lr = "***" if p_lr < 0.001 else ("**" if p_lr < 0.01 else ("*" if p_lr < 0.05 else "ns"))
            print(f"    LGBM vs Ridge:  DM={dm_lr:+.4f}  p={p_lr:.5f}  {sig_lr}")

            records.append({
                "zone": zone, "n": n_min3,
                "lgbm_mae":   round(np.mean(np.abs(lgbm)), 4),
                "ridge_mae":  round(np.mean(np.abs(ridge)), 4),
                "naive_mae":  round(np.mean(np.abs(naive)), 4),
                "lgbm_rmse":  round(np.sqrt(np.mean(lgbm**2)), 4),
                "ridge_rmse": round(np.sqrt(np.mean(ridge**2)), 4),
                "lgbm_vs_ridge_dm": dm_lr, "lgbm_vs_ridge_p": p_lr,
                "lgbm_vs_naive_dm": dm_ln, "lgbm_vs_naive_p": p_ln,
                "ridge_vs_naive_dm": dm_rn, "ridge_vs_naive_p": p_rn,
            })

        if not records:
            print("No results — run models first or ensure dm_pairwise.csv exists.")
            return

        result_df = pd.DataFrame(records)
        MODEL_RES.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(csv_path, index=False)
        print(f"\n  Saved {csv_path}")

    print("\nGenerating fig8 ...")
    make_fig8(records)
    print("\nDone.")


if __name__ == "__main__":
    main()
