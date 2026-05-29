"""
Shared utilities: data loading, feature engineering, walk-forward split, metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_zone(path: str) -> pd.DataFrame:
    """Load a cleaned hourly parquet file and return it with sorted index."""
    df = pd.read_parquet(path)
    df = df.sort_index()
    return df


# ---------------------------------------------------------------------------
# Feature engineering — lag features
# ---------------------------------------------------------------------------

LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 168]   # 168 h = 1 week
ROLL_WINDOWS = [24, 168]                       # 24-h and 1-week rolling stats

def add_lag_features(df: pd.DataFrame, target: str = "price_eur_mwh") -> pd.DataFrame:
    """Add autoregressive lag and rolling features for *target* column."""
    df = df.copy()
    for lag in LAG_HOURS:
        df[f"{target}_lag{lag}h"] = df[target].shift(lag)
    for w in ROLL_WINDOWS:
        df[f"{target}_roll{w}h_mean"] = df[target].shift(1).rolling(w).mean()
        df[f"{target}_roll{w}h_std"]  = df[target].shift(1).rolling(w).std()
    return df


def build_feature_matrix(df: pd.DataFrame, target: str = "price_eur_mwh") -> tuple[pd.DataFrame, pd.Series]:
    """
    Return (X, y) ready for modelling.

    Feature groups included:
    - Lag / rolling features on the target price
    - Calendar cyclical encodings (already in the data)
    - Weather (wx_*), reservoir (res_*), commodity (com_*)
    - Load forecast (available day-ahead)
    - Wind / solar forecast (wsf_*)

    Columns with >50 % NaN are dropped. Remaining NaNs are forward-filled then
    back-filled (edge periods). Rows where target is NaN are removed.
    """
    df = add_lag_features(df, target)

    # Drop columns that are mostly missing (e.g. gen_other, gen_biomass)
    thresh = 0.50
    keep = df.columns[df.isnull().mean() < thresh]
    df = df[keep]

    # Drop columns not available at prediction time (post-delivery actuals):
    #   gen_*        — actual generation by type (only known after delivery hour)
    #   flow_*       — actual cross-border physical flows (post-delivery)
    #   net_import_* — actual net import (post-delivery)
    #   load_mw      — actual realised load (post-delivery; load_forecast_mw is kept)
    # Only day-ahead published forecasts (load_forecast_mw, wsf_*) are retained.
    drop_prefixes = ("gen_", "flow_", "net_import_")
    drop_exact    = {"zone", "hour", "day_of_week", "month", "week_of_year",
                     "is_weekend", "load_mw"}

    feature_cols = [
        c for c in df.columns
        if c != target
        and c not in drop_exact
        and not any(c.startswith(p) for p in drop_prefixes)
    ]

    y = df[target].dropna()
    X = df.loc[y.index, feature_cols]

    # Fill remaining NaN (commodity / weather edge period)
    X = X.ffill().bfill()

    # Align after ffill — drop any rows still containing NaN (lag warm-up period)
    valid = X.notna().all(axis=1)
    X = X[valid]
    y = y[valid]

    return X, y


# ---------------------------------------------------------------------------
# Temporal splits
# ---------------------------------------------------------------------------

def train_test_split_temporal(
    X: pd.DataFrame,
    y: pd.Series,
    test_start: str = "2025-01-01",
    tz: str = "Europe/Oslo",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Binary train / test split (kept for backward compat)."""
    split = pd.Timestamp(test_start, tz=tz)
    mask_train = X.index < split
    mask_test  = X.index >= split
    return X[mask_train], X[mask_test], y[mask_train], y[mask_test]


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_start:  str = "2024-01-01",
    test_start: str = "2025-01-01",
    tz: str = "Europe/Oslo",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series,  pd.Series,  pd.Series]:
    """
    Three-way temporal split: train | validation | test.

    Default boundaries (matching literature practice for Norwegian EPF):
      train : 2019-01-01 → 2023-12-31  (5 years, ~43,819 hourly rows)
      val   : 2024-01-01 → 2024-12-31  (1 year,  ~8,783 rows)
      test  : 2025-01-01 → end         (1 year, ~8,736 rows)

    The validation set is used for early stopping / hyperparameter tuning.
    The test set is the final held-out evaluation set — never touched during training.
    """
    v = pd.Timestamp(val_start,  tz=tz)
    t = pd.Timestamp(test_start, tz=tz)
    X_tr  = X[X.index <  v]
    X_val = X[(X.index >= v) & (X.index < t)]
    X_te  = X[X.index >= t]
    y_tr  = y[y.index <  v]
    y_val = y[(y.index >= v) & (y.index < t)]
    y_te  = y[y.index >= t]
    return X_tr, X_val, X_te, y_tr, y_val, y_te




# ---------------------------------------------------------------------------
# Evaluation metrics (extended)
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray | pd.Series, y_pred: np.ndarray, label: str = "") -> dict:
    """
    Compute MAE, RMSE, sMAPE and R².

    sMAPE replaces MAPE: it is bounded [0, 200%] and symmetric around zero,
    so it handles near-zero and negative prices without exploding.
      sMAPE = mean(2|y-ŷ| / (|y|+|ŷ|+ε)) × 100
    Reference: Makridakis (1993); Hyndman & Koehler (2006).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err   = y_true - y_pred
    mae   = float(np.mean(np.abs(err)))
    rmse  = float(np.sqrt(np.mean(err ** 2)))
    # sMAPE — epsilon avoids 0/0 when both y and ŷ are zero
    eps   = 1e-8
    smape = float(np.mean(2 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100)
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2    = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return dict(model=label,
                MAE=round(mae, 3),
                RMSE=round(rmse, 3),
                sMAPE=round(smape, 2),
                R2=round(r2, 4),
                # keep raw arrays for DM test
                _errors=err)


# ---------------------------------------------------------------------------
# Diebold-Mariano (DM) test
# ---------------------------------------------------------------------------

def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
    power: int = 2,
) -> tuple[float, float]:
    """
    Two-sided Diebold-Mariano (1995) test for equal predictive accuracy.

    H0: E[d_t] = 0  where d_t = |e_a,t|^power − |e_b,t|^power

    Uses Harvey, Leybourne & Newbold (1997) small-sample correction and
    a Newey-West HAC variance estimator for horizon h > 1.

    Returns (dm_stat, p_value).  Reject H0 (unequal accuracy) if p < 0.05.
    Negative DM stat → model A is better; positive → model B is better.
    """
    from scipy import stats

    # Align arrays to the same length (edge effects from lag warm-up can cause ±1)
    n_min = min(len(errors_a), len(errors_b))
    errors_a = np.asarray(errors_a[-n_min:], dtype=float)
    errors_b = np.asarray(errors_b[-n_min:], dtype=float)

    d = np.abs(errors_a) ** power - np.abs(errors_b) ** power
    n = len(d)
    d_bar = np.mean(d)

    # Newey-West HAC variance (bandwidth = h-1)
    gamma0 = np.mean((d - d_bar) ** 2)
    nw_var = gamma0
    for lag in range(1, h):
        gamma_k = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        nw_var += 2 * (1 - lag / h) * gamma_k
    nw_var = max(nw_var, 1e-12)  # numerical floor

    # Harvey-Leybourne-Newbold small-sample correction
    hlp_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat = float(d_bar / (np.sqrt(nw_var / n)))
    dm_stat_corr = dm_stat * hlp_factor

    p_value = float(2 * stats.t.sf(np.abs(dm_stat_corr), df=n - 1))
    return round(dm_stat_corr, 3), round(p_value, 4)


def dm_table(results: list[dict], benchmark: str, h: int = 24) -> pd.DataFrame:
    """
    Print a DM-test table comparing all models against *benchmark*.
    Uses squared errors (power=2, tests RMSE-equivalent forecast accuracy).

    h=24: forecast horizon correction for day-ahead (24-step-ahead) predictions.
    """
    bench = next((r for r in results if r["model"] == benchmark), None)
    if bench is None:
        raise ValueError(f"Benchmark model '{benchmark}' not found in results.")

    rows = []
    for r in results:
        if r["model"] == benchmark:
            rows.append(dict(model=r["model"], DM_stat="—", p_value="—", sig="(baseline)"))
            continue
        dm, p = diebold_mariano(r["_errors"], bench["_errors"], h=h, power=2)
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "ns"))
        rows.append(dict(model=r["model"], DM_stat=dm, p_value=p, sig=sig))
    return pd.DataFrame(rows).set_index("model")


def print_results(results: list[dict], benchmark: str | None = None, h: int = 24) -> None:
    # Strip internal _errors before printing metrics
    clean = [{k: v for k, v in r.items() if k != "_errors"} for r in results]
    df = pd.DataFrame(clean).set_index("model")
    print("\n" + "=" * 65)
    print("Model comparison")
    print("=" * 65)
    print(df.to_string())
    if benchmark:
        print("\nDiebold-Mariano test vs. " + benchmark + f"  (h={h}, power=2)")
        print(dm_table(results, benchmark=benchmark, h=h).to_string())
    print("=" * 65 + "\n")


def save_results(results: list[dict], name: str, out_dir: str = "model_results") -> "Path":
    """Save a list of metric dicts to model_results/<name>.csv"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    clean = [{k: v for k, v in r.items() if k != "_errors"} for r in results]
    df = pd.DataFrame(clean)
    out = Path(out_dir) / f"{name}.csv"
    df.to_csv(out, index=False)
    return out
