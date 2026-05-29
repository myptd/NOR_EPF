"""
analysis/10_pre_post_crisis_table.py
=====================================
Collect LightGBM test-set metrics for three training windows and write
a comparison table matching the manuscript format:

    Zone | Train 2019-2021 (MAE RMSE sMAPE R2)
         | Train 2022-2023 (MAE RMSE sMAPE R2)
         | Train 2019-2023 (MAE RMSE sMAPE R2)

Source CSVs (model_results/):
  {ZONE}_test_pre_crisis.csv   — produced by models/pre_post_crisis.py
  {ZONE}_test_post_crisis.csv  — produced by models/pre_post_crisis.py
  {ZONE}_test.csv              — produced by step 1a (full window, not re-run)

Outputs:
  model_results/pre_post_crisis.csv     — raw long-format rows
  paper_outputs/tab_pre_post_crisis.csv — wide table (zone rows, window columns),
                                          best metric per row marked with *
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ZONES   = ["NO1", "NO2", "NO3", "NO4", "NO5"]
METRICS = ["MAE", "RMSE", "sMAPE", "R2"]
MODEL   = "LightGBM"

# (display label, CSV suffix)
WINDOWS = [
    ("Train 2019-2021", "_pre_crisis"),
    ("Train 2022-2023", "_post_crisis"),
    ("Train 2019-2023", ""),            # full window — from step 1a
]

IN_DIR  = Path("model_results")
OUT_DIR = Path("paper_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_lgbm_row(zone: str, suffix: str) -> dict | None:
    path = IN_DIR / f"{zone}_test{suffix}.csv"
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping.")
        return None
    df = pd.read_csv(path)
    row = df[df["model"] == MODEL]
    if row.empty:
        print(f"  WARNING: no '{MODEL}' row in {path}.")
        return None
    return row.iloc[0].to_dict()


def main() -> None:
    long_rows: list[dict] = []

    for zone in ZONES:
        for win_label, suffix in WINDOWS:
            rec = load_lgbm_row(zone, suffix)
            if rec is None:
                continue
            long_rows.append({
                "zone":   zone,
                "window": win_label,
                **{m: rec[m] for m in METRICS if m in rec},
            })

    if not long_rows:
        print("No data found.  Run step 1a and models/pre_post_crisis.py first.")
        return

    df_long = pd.DataFrame(long_rows)
    long_path = IN_DIR / "pre_post_crisis.csv"
    df_long.to_csv(long_path, index=False)
    print(f"Saved → {long_path}")

    # ── Wide table: zone as rows, (window, metric) as columns ─────────────────
    df_wide = df_long.pivot_table(
        index="zone", columns="window", values=METRICS, aggfunc="first"
    ).round(3)

    # Reorder columns to manuscript layout: pre | post | full
    win_labels = [w for w, _ in WINDOWS]
    ordered = [(m, w) for w in win_labels for m in METRICS]
    df_wide = df_wide[[c for c in ordered if c in df_wide.columns]]

    # Flatten multi-level column index
    label_to_key = {
        "Train 2019-2021": "pre",
        "Train 2022-2023": "post",
        "Train 2019-2023": "full",
    }
    df_wide.columns = [
        f"{m}_{label_to_key[w]}" for m, w in df_wide.columns
    ]

    # Mark best value per metric per zone with *
    # For MAE/RMSE/sMAPE lower is better; for R2 higher is better
    df_marked = df_wide.copy().astype(object)
    for metric in METRICS:
        cols = [c for c in df_marked.columns if c.startswith(metric + "_")]
        if not cols:
            continue
        higher_is_better = (metric == "R2")
        for zone in df_marked.index:
            vals = df_wide.loc[zone, cols]
            best = vals.max() if higher_is_better else vals.min()
            for c in cols:
                v = df_wide.loc[zone, c]
                df_marked.loc[zone, c] = f"{v}*" if v == best else str(v)

    wide_path = OUT_DIR / "tab_pre_post_crisis.csv"
    df_marked.to_csv(wide_path)
    print(f"Saved → {wide_path}")

    mae_cols = [c for c in df_wide.columns if c.startswith("MAE_")]
    print("\n=== LightGBM MAE — pre / post / full (test 2025) ===")
    print(df_wide[mae_cols].to_string())


if __name__ == "__main__":
    main()
