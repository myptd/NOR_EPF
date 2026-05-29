#!/usr/bin/env python
"""
bin/data_preprocess.py
======================
Norwegian Electricity Price Forecasting — Data Preprocessing & Merging Pipeline

Merges four multi-frequency data sources into clean, analysis-ready panel datasets:

  Source              | Frequency | Scope
  --------------------|-----------|--------------------------------------------
  ENTSO-E             | Hourly    | Prices, load, generation, forecasts, flows
  Open-Meteo weather  | Hourly    | Temperature, wind, solar, precipitation
  NVE reservoir       | Weekly    | Hydro storage filling levels
  Yahoo Finance       | Daily     | Gas, carbon, oil, FX prices

Merging strategy
----------------
  1. Master index   : Hourly timestamps from day-ahead prices per zone,
                      normalised to Europe/Oslo (CET/CEST) timezone.
  2. Hourly sources : Joined directly on timestamp after tz-normalisation.
  3. Daily sources  : Commodity prices reindexed to daily range, ffilled to hourly.
  4. Weekly sources : Reservoir reindexed to daily, ffilled within each week,
                      then broadcast to hourly.
  5. Temporal feats : hour, day-of-week, month, ISO-week; sine/cosine encodings.

Outputs (in data/cleaned/)
--------------------------
  panel_hourly.parquet    — full multi-zone hourly panel (zone column included)
  {ZONE}_hourly.parquet   — per-zone hourly dataset (NO1 … NO5)
  panel_daily_avg.parquet — daily aggregation of the hourly panel
  CSV variants of all the above (written by default)

Usage
-----
  conda activate elec
  python bin/data_preprocess.py [options]

Options
-------
  --zones      Zones to process  [default: NO_1 NO_2 NO_3 NO_4 NO_5]
  --start      Start date YYYY-MM-DD  [default: first available]
  --end        End date YYYY-MM-DD    [default: last available]
  --no-parquet Skip parquet output
  --no-csv     Skip CSV output (strongly recommended for large runs)
  --com-all-cols  Include ALL commodity columns (not just key ones)
"""

import argparse
import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT  = DATA / "cleaned"
OUT.mkdir(parents=True, exist_ok=True)

# ─── Zone configuration ───────────────────────────────────────────────────────
ZONES = ["NO_1", "NO_2", "NO_3", "NO_4", "NO_5"]

# Maps ENTSOE zone key (NO_1) → weather file key (NO1)
ZONE_SHORT = {z: z.replace("_", "") for z in ZONES}

# Maps NVE omr_nr (1-5) → ENTSOE zone key
RESERVOIR_NR_TO_ZONE = {1: "NO_1", 2: "NO_2", 3: "NO_3", 4: "NO_4", 5: "NO_5"}

LOCAL_TZ = "Europe/Oslo"

# ─── Cross-border borders relevant per zone ───────────────────────────────────
# Each entry is a CSV stem in data/entsoe/
ZONE_BORDERS: dict[str, list[str]] = {
    "NO_1": [
        "crossborder_NO_1_to_NO_2",
        "crossborder_NO_2_to_NO_1",
        "crossborder_NO_1_to_NO_3",
        "crossborder_NO_3_to_NO_1",
        "crossborder_NO_1_to_NO_5",
        "crossborder_NO_5_to_NO_1",
    ],
    "NO_2": [
        "crossborder_NO_2_to_NO_1",
        "crossborder_NO_1_to_NO_2",
        "crossborder_NO_2_to_DE_LU",
        "crossborder_DE_LU_to_NO_2",
        "crossborder_NO_2_to_NL",
        "crossborder_NL_to_NO_2",
        "crossborder_NO_2_to_GB",
        "crossborder_GB_to_NO_2",
        "crossborder_NO_2_to_NO_5",
        "crossborder_NO_5_to_NO_2",
    ],
    "NO_3": [
        "crossborder_NO_3_to_NO_1",
        "crossborder_NO_1_to_NO_3",
        "crossborder_NO_3_to_NO_4",
        "crossborder_NO_4_to_NO_3",
    ],
    "NO_4": [
        "crossborder_NO_4_to_NO_3",
        "crossborder_NO_3_to_NO_4",
        "crossborder_NO_4_to_FI",
        "crossborder_FI_to_NO_4",
    ],
    "NO_5": [
        "crossborder_NO_5_to_NO_1",
        "crossborder_NO_1_to_NO_5",
        "crossborder_NO_5_to_NO_2",
        "crossborder_NO_2_to_NO_5",
    ],
}

# ─── Commodity columns to keep (curated subset) ───────────────────────────────
# Includes close prices and key engineered indicators.
# Use --com-all-cols to keep everything.
COMMODITY_COLS_KEEP = re.compile(
    r"(carbon_eua_lt_close|carbon_eua_lt_return|carbon_eua_lt_ma5|"
    r"carbon_eua_lt_ma20|carbon_eua_lt_vol20|carbon_eua_lt_log|"
    r"carbon_eua_close|carbon_eua_return|carbon_eua_ma5|carbon_eua_ma20|"
    r"carbon_eua_vol20|carbon_eua_log|"
    r"gas_ttf_close|gas_ttf_return|gas_ttf_ma5|gas_ttf_ma20|gas_ttf_vol20|gas_ttf_log|"
    r"oil_brent_close|oil_brent_return|oil_brent_ma5|oil_brent_ma20|"
    r"oil_brent_vol20|oil_brent_log|"
    r"fx_eurnok_close|fx_eurnok_return|fx_eurnok_ma5|fx_eurnok_ma20|"
    r"fx_eurnok_vol20|fx_eurnok_log|"
    r"gas_henryhub_close|ratio_gas_carbon|ratio_gas_oil|"
    r"carbon_momentum|gas_momentum)"
)

# ─── Reservoir columns to keep ────────────────────────────────────────────────
RESERVOIR_COLS_KEEP = [
    "fill_pct", "capacity_twh", "fill_twh", "fill_pct_100",
    "fill_pct_yoy", "fill_pct_4w_avg", "fill_trend_4w",
    "empty_twh", "fill_change_pct", "fill_acceleration",
    "fill_anomaly_vs_median", "fill_normalized_pos",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _slugify(text: str) -> str:
    """Convert arbitrary column text to a safe snake_case slug."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _parse_ts_index(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    """
    Parse timezone-aware datetime column, convert to Europe/Oslo, set as index.
    Handles both fixed-offset strings (+01:00 / +02:00) and UTC properly.
    """
    ts = pd.to_datetime(df[col], utc=True).dt.tz_convert(LOCAL_TZ)
    df = df.drop(columns=[col])
    df.index = ts
    df.index.name = "datetime"
    return df


def _date_key_from_tz_index(idx: pd.DatetimeIndex) -> pd.Index:
    """
    Return a string date index (YYYY-MM-DD) from a tz-aware DatetimeIndex
    (using the Oslo local date, not UTC date).
    """
    return idx.tz_convert(LOCAL_TZ).normalize().strftime("%Y-%m-%d")


def _broadcast_daily_to_hourly(
    master_idx: pd.DatetimeIndex,
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Given a daily DataFrame (DatetimeIndex, tz-naive, freq=D or irregular),
    broadcast its values to every hour in master_idx via forward-fill.

    Steps:
      1. Build a string-keyed mapping date→row from daily_df.
      2. Create date keys for each row in master_idx.
      3. Reindex and ffill.
    """
    # Ensure daily_df index is a DatetimeIndex without tz
    if not isinstance(daily_df.index, pd.DatetimeIndex):
        raise ValueError("daily_df must have a DatetimeIndex")

    # Full daily range covering master
    daily_range = pd.date_range(
        master_idx.min().normalize().tz_localize(None),
        master_idx.max().normalize().tz_localize(None),
        freq="D",
    )
    # Reindex to daily range and forward-fill (fills weekends/holidays)
    daily_reindexed = daily_df.reindex(daily_range).ffill()
    # Build {date_str: row} dict for fast lookup
    daily_reindexed.index = daily_reindexed.index.strftime("%Y-%m-%d")

    # Map via master date keys
    master_date_keys = _date_key_from_tz_index(master_idx)
    result = daily_reindexed.reindex(master_date_keys)
    result.index = master_idx
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Data loaders — ENTSO-E
# ═══════════════════════════════════════════════════════════════════════════════

def load_prices(zone: str) -> pd.DataFrame | None:
    path = DATA / "entsoe" / f"day_ahead_prices_{zone}.csv"
    if not path.exists():
        log.warning(f"Missing prices: {path}")
        return None
    df = pd.read_csv(path)
    return _parse_ts_index(df)


def load_actual_load(zone: str) -> pd.DataFrame | None:
    path = DATA / "entsoe" / f"actual_load_{zone}.csv"
    if not path.exists():
        return None
    return _parse_ts_index(pd.read_csv(path))


def load_load_forecast(zone: str) -> pd.DataFrame | None:
    path = DATA / "entsoe" / f"load_forecast_{zone}.csv"
    if not path.exists():
        return None
    return _parse_ts_index(pd.read_csv(path))


def load_generation(zone: str) -> pd.DataFrame | None:
    path = DATA / "entsoe" / f"generation_by_type_{zone}.csv"
    if not path.exists():
        return None
    df = _parse_ts_index(pd.read_csv(path))
    # Prefix: "Hydro Water Reservoir" → "gen_hydro_water_reservoir"
    df.columns = ["gen_" + _slugify(c) for c in df.columns]
    return df


def load_wind_solar_forecast(zone: str) -> pd.DataFrame | None:
    path = DATA / "entsoe" / f"wind_solar_forecast_{zone}.csv"
    if not path.exists():
        return None
    df = _parse_ts_index(pd.read_csv(path))
    df.columns = ["wsf_" + _slugify(c) for c in df.columns]
    return df


def load_net_import(zone: str) -> pd.DataFrame | None:
    path = DATA / "entsoe" / f"net_import_{zone}.csv"
    if not path.exists():
        return None
    return _parse_ts_index(pd.read_csv(path))


def load_crossborder_flows(zone: str) -> pd.DataFrame | None:
    """
    Load pairwise cross-border flow CSVs for the given zone.
    Each file has columns [datetime, flow_mw]; renamed to the border direction.
    Returns a joined DataFrame or None if no files found.
    """
    dfs = []
    for stem in ZONE_BORDERS.get(zone, []):
        path = DATA / "entsoe" / f"{stem}.csv"
        if not path.exists():
            continue
        df = _parse_ts_index(pd.read_csv(path))
        col = "flow_" + stem.replace("crossborder_", "")
        df = df.rename(columns={"flow_mw": col})
        dfs.append(df)

    if not dfs:
        return None
    result = dfs[0]
    for d in dfs[1:]:
        result = result.join(d, how="outer")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Data loaders — Weather
# ═══════════════════════════════════════════════════════════════════════════════

def load_weather(zone: str) -> pd.DataFrame | None:
    """
    Load per-zone weather CSV. Timestamps are UTC in the raw file; convert to
    Europe/Oslo to align with ENTSO-E. Prefix columns with 'wx_'.
    """
    short = ZONE_SHORT[zone]  # NO_1 → NO1
    candidates = sorted((DATA / "weather").glob(f"weather_{short}_*.csv"))
    if not candidates:
        log.warning(f"No weather file for zone {zone}")
        return None
    df = _parse_ts_index(pd.read_csv(candidates[0]))
    # Drop zone metadata (not features)
    meta_cols = {"bidding_zone", "zone_name", "latitude", "longitude"}
    df = df.drop(columns=[c for c in df.columns if c in meta_cols])
    df.columns = [f"wx_{c}" for c in df.columns]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Data loaders — Reservoir (shared, loaded once)
# ═══════════════════════════════════════════════════════════════════════════════

def load_reservoir_all() -> pd.DataFrame | None:
    """
    Load NVE elspot reservoir data for NO1–NO5.
    Returns a DataFrame pivot-ready with zone column and DatetimeIndex on 'date'.
    """
    path = DATA / "reservoir" / "reservoir_elspot_zones.csv"
    if not path.exists():
        log.warning(f"Missing reservoir: {path}")
        return None

    df = pd.read_csv(path, parse_dates=["date"])
    # Keep only Elspot bidding zones
    df = df[df["omr_type"] == "EL"].copy()
    df["zone"] = df["omr_nr"].map(RESERVOIR_NR_TO_ZONE)
    df = df.dropna(subset=["zone"])

    # Select features
    keep = ["zone", "date"] + [c for c in RESERVOIR_COLS_KEEP if c in df.columns]
    df = df[keep]

    # Prefix reservoir feature columns
    res_feat_cols = [c for c in df.columns if c not in ("zone", "date")]
    df = df.rename(columns={c: f"res_{c}" for c in res_feat_cols})
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Data loaders — Commodities (shared, loaded once)
# ═══════════════════════════════════════════════════════════════════════════════

def load_commodities(all_cols: bool = False) -> pd.DataFrame | None:
    """
    Load the forward-filled commodity dataset.
    Returns a daily DatetimeIndex DataFrame (tz-naive) with 'com_' prefix.
    """
    path = DATA / "commodities" / "commodities_combined_ffill.csv"
    if not path.exists():
        log.warning(f"Missing commodities: {path}")
        return None

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    df.index.name = "date"

    if not all_cols:
        selected = [c for c in df.columns if COMMODITY_COLS_KEEP.fullmatch(c)]
        df = df[selected]

    df.columns = [f"com_{c}" for c in df.columns]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal feature engineering
# ═══════════════════════════════════════════════════════════════════════════════

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar / cyclical temporal features to a tz-aware hourly DataFrame.
    Cyclical encoding (sin/cos) prevents the "midnight/year-end discontinuity"
    artefact that plain integer hour/month features introduce.
    """
    idx = df.index
    df = df.copy()
    df["hour"]         = idx.hour
    df["day_of_week"]  = idx.dayofweek          # 0 = Monday
    df["month"]        = idx.month
    df["week_of_year"] = idx.isocalendar().week.astype(int)
    df["is_weekend"]   = (idx.dayofweek >= 5).astype(int)
    # Cyclical encodings
    df["hour_sin"]     = np.sin(2 * np.pi * idx.hour        / 24)
    df["hour_cos"]     = np.cos(2 * np.pi * idx.hour        / 24)
    df["dow_sin"]      = np.sin(2 * np.pi * idx.dayofweek   /  7)
    df["dow_cos"]      = np.cos(2 * np.pi * idx.dayofweek   /  7)
    df["month_sin"]    = np.sin(2 * np.pi * idx.month       / 12)
    df["month_cos"]    = np.cos(2 * np.pi * idx.month       / 12)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Per-zone assembly
# ═══════════════════════════════════════════════════════════════════════════════

def build_zone_dataset(
    zone: str,
    reservoir_all: pd.DataFrame | None,
    commodities: pd.DataFrame | None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame | None:
    """
    Build the complete hourly feature matrix for a single bidding zone.

    Column groups in output:
      price_eur_mwh          — target variable
      load_mw                — actual load (ENTSO-E)
      load_forecast_mw       — day-ahead load forecast
      gen_*                  — generation by technology (MW)
      wsf_*                  — wind/solar forecast (MW)
      net_import_*_mw        — net physical import (MW)
      flow_*                 — pairwise cross-border flows (MW)
      wx_*                   — weather variables (temperature, wind, etc.)
      res_*                  — reservoir filling levels / hydro indicators
      com_*                  — commodity prices and indicators
      hour / dow / month ...  — temporal features + cyclical encodings
    """
    log.info(f"─── Zone {zone} ───")

    # ── 1. Master index from day-ahead prices ──────────────────────────────────
    prices = load_prices(zone)
    if prices is None:
        log.error(f"  Skipping {zone}: no price data.")
        return None

    if start_date:
        prices = prices[prices.index >= pd.Timestamp(start_date, tz=LOCAL_TZ)]
    if end_date:
        prices = prices[prices.index <= pd.Timestamp(end_date,   tz=LOCAL_TZ)]

    if prices.empty:
        log.error(f"  Skipping {zone}: no rows in requested date range.")
        return None

    master = prices.copy()
    log.info(f"  Master index: {master.index[0]} → {master.index[-1]}, {len(master):,} rows")

    # ── Enforce strict hourly resolution ──────────────────────────────────────
    # ENTSO-E occasionally publishes sub-hourly (e.g. 15-min) data for recent
    # periods. Resample to 1h (mean for prices, handles DST 23/25-h days) so
    # all downstream joins operate on a consistent hourly grid.
    raw_deltas = master.index.to_series().diff().dropna().unique()
    if any(d < pd.Timedelta("1h") for d in raw_deltas):
        n_before = len(master)
        master = master.resample("1h").mean()
        log.warning(
            f"  Sub-hourly timestamps detected — resampled to 1h "
            f"({n_before:,} → {len(master):,} rows)."
        )
    # ── 2. ENTSO-E hourly sources ──────────────────────────────────────────────
    entsoe_sources = [
        (load_actual_load,           "actual_load"),
        (load_load_forecast,         "load_forecast"),
        (load_generation,            "generation_by_type"),
        (load_wind_solar_forecast,   "wind_solar_forecast"),
        (load_net_import,            "net_import"),
    ]
    for loader, label in entsoe_sources:
        df = loader(zone)
        if df is not None:
            master = master.join(df, how="left")
            log.info(f"  [{label}] +{df.shape[1]} cols → {master.shape}")
        else:
            log.warning(f"  [{label}] not available for {zone}")

    # ── 3. Cross-border flows ──────────────────────────────────────────────────
    flows = load_crossborder_flows(zone)
    if flows is not None:
        master = master.join(flows, how="left")
        log.info(f"  [crossborder_flows] +{flows.shape[1]} cols → {master.shape}")
    else:
        log.warning(f"  [crossborder_flows] no data for {zone}")

    # ── 4. Weather (hourly, UTC → Oslo already in _parse_ts_index) ────────────
    weather = load_weather(zone)
    if weather is not None:
        # Weather starts from CET midnight (2018-12-31 23:00 UTC = 2019-01-01 00:00 Oslo)
        # so the join aligns naturally once both are in Europe/Oslo.
        master = master.join(weather, how="left")
        log.info(f"  [weather] +{weather.shape[1]} cols → {master.shape}")
    else:
        log.warning(f"  [weather] not available for {zone}")

    # ── 5. Reservoir (weekly → daily ffill → hourly broadcast) ────────────────
    if reservoir_all is not None:
        zone_res = reservoir_all[reservoir_all["zone"] == zone].copy()
        zone_res = zone_res.drop(columns=["zone"])
        zone_res["date"] = pd.to_datetime(zone_res["date"])
        zone_res = zone_res.set_index("date").sort_index()

        # Broadcast to master hourly index
        res_hourly = _broadcast_daily_to_hourly(master.index, zone_res)
        master = master.join(res_hourly, how="left")
        log.info(f"  [reservoir] +{zone_res.shape[1]} cols → {master.shape}")
    else:
        log.warning(f"  [reservoir] no data loaded")

    # ── 6. Commodities (daily → hourly broadcast) ─────────────────────────────
    if commodities is not None:
        com_hourly = _broadcast_daily_to_hourly(master.index, commodities)
        master = master.join(com_hourly, how="left")
        log.info(f"  [commodities] +{commodities.shape[1]} cols → {master.shape}")
    else:
        log.warning(f"  [commodities] no data loaded")

    # ── 7. Temporal features ───────────────────────────────────────────────────
    master = add_temporal_features(master)
    log.info(f"  [temporal_feats] added → {master.shape}")

    # ── 8. Zone identifier (insert at position 0) ─────────────────────────────
    master.insert(0, "zone", zone)

    return master


# ═══════════════════════════════════════════════════════════════════════════════
# Daily aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def _build_agg_rules(df: pd.DataFrame) -> dict:
    """
    Build column-level aggregation rules for hourly→daily reduction.

    Logic:
      gen_*, flow_*         → sum   (energy volumes)
      res_*, com_*          → last  (already constant within day/week)
      hour, hour_sin/cos    → skip  (sub-daily; meaningless at daily level)
      everything else       → mean  (prices, load, weather)
    """
    skip = {"zone", "hour", "hour_sin", "hour_cos",
            "day_of_week", "week_of_year", "is_weekend",
            "dow_sin", "dow_cos", "month_sin", "month_cos", "month"}
    rules = {}
    for col in df.columns:
        if col in skip:
            continue
        if col.startswith(("gen_", "flow_")):
            rules[col] = "sum"
        elif col.startswith(("res_", "com_")):
            rules[col] = "last"
        else:
            rules[col] = "mean"
    return rules


def make_daily_panel(zone_datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate all per-zone hourly DataFrames to daily resolution.
    """
    parts = []
    for zone, df in zone_datasets.items():
        hourly = df.drop(columns=["zone"], errors="ignore")
        agg_rules = _build_agg_rules(hourly)
        # Group by Oslo local date
        daily_key = hourly.index.tz_convert(LOCAL_TZ).normalize()
        daily = hourly.groupby(daily_key).agg(agg_rules)
        daily.index.name = "date"
        # Re-add daily-level temporal features
        dti = pd.DatetimeIndex(daily.index)
        daily["day_of_week"]  = dti.dayofweek
        daily["month"]        = dti.month
        daily["week_of_year"] = dti.isocalendar().week.astype(int)
        daily["is_weekend"]   = (dti.dayofweek >= 5).astype(int)
        daily["dow_sin"]      = np.sin(2 * np.pi * dti.dayofweek / 7)
        daily["dow_cos"]      = np.cos(2 * np.pi * dti.dayofweek / 7)
        daily["month_sin"]    = np.sin(2 * np.pi * dti.month     / 12)
        daily["month_cos"]    = np.cos(2 * np.pi * dti.month     / 12)
        daily.insert(0, "zone", zone)
        parts.append(daily)

    return pd.concat(parts, axis=0).sort_index()


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def report_missing(df: pd.DataFrame, zone: str, threshold: float = 5.0) -> None:
    """Log columns with more than `threshold`% missing values."""
    pct = df.isnull().mean() * 100
    high = pct[pct > threshold].sort_values(ascending=False)
    if high.empty:
        log.info(f"  [{zone}] All columns ≤ {threshold}% missing. ✓")
    else:
        log.warning(f"  [{zone}] {len(high)} columns above {threshold}% missing:")
        for col, val in high.items():
            log.warning(f"    {col:55s} {val:6.1f}%")


def save(df: pd.DataFrame, stem: str, no_parquet: bool, no_csv: bool) -> None:
    """Save DataFrame to parquet and/or CSV."""
    if not no_parquet:
        p = OUT / f"{stem}.parquet"
        df.to_parquet(p)
        log.info(f"  Saved {p.name}  ({len(df):,} rows × {df.shape[1]} cols)")
    if not no_csv:
        p = OUT / f"{stem}.csv"
        df.to_csv(p)
        log.info(f"  Saved {p.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Norwegian EPF — data preprocessing & merging pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--zones",       nargs="+", default=ZONES,
                        help="Bidding zones to process (default: all five)")
    parser.add_argument("--start",       default=None,
                        help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end",         default=None,
                        help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--no-parquet",  action="store_true",
                        help="Skip parquet output")
    parser.add_argument("--no-csv",      action="store_true",
                        help="Skip CSV output (recommended for large datasets)")
    parser.add_argument("--com-all-cols", action="store_true",
                        help="Keep all commodity columns (default: curated subset)")
    parser.add_argument("--missing-threshold", type=float, default=5.0,
                        help="% missing-data warning threshold (default: 5.0)")
    args = parser.parse_args()

    log.info("══════════════════════════════════════════════════")
    log.info("  Norwegian EPF — Data Preprocessing Pipeline")
    log.info("══════════════════════════════════════════════════")
    log.info(f"  Zones  : {args.zones}")
    log.info(f"  Range  : {args.start or 'all'} → {args.end or 'all'}")
    log.info(f"  Output : {OUT}")

    # ── Load shared (zone-agnostic) sources once ───────────────────────────────
    log.info("Loading shared data sources …")
    reservoir_all = load_reservoir_all()
    commodities   = load_commodities(all_cols=args.com_all_cols)
    log.info(f"  Reservoir : {len(reservoir_all):,} rows" if reservoir_all is not None else "  Reservoir : NOT FOUND")
    log.info(f"  Commodities: {len(commodities):,} rows × {commodities.shape[1]} cols"
             if commodities is not None else "  Commodities: NOT FOUND")

    # ── Build per-zone datasets ────────────────────────────────────────────────
    zone_datasets: dict[str, pd.DataFrame] = {}
    for zone in args.zones:
        df = build_zone_dataset(
            zone, reservoir_all, commodities,
            start_date=args.start,
            end_date=args.end,
        )
        if df is None:
            continue
        report_missing(df, zone, threshold=args.missing_threshold)
        zone_datasets[zone] = df

    if not zone_datasets:
        log.error("No zone datasets produced. Exiting.")
        return

    # ── Save per-zone files ────────────────────────────────────────────────────
    log.info("Saving per-zone files …")
    for zone, df in zone_datasets.items():
        short = ZONE_SHORT[zone]  # NO1 … NO5
        save(df, f"{short}_hourly", args.no_parquet, args.no_csv)

    # ── Save full hourly panel ─────────────────────────────────────────────────
    log.info("Building and saving full hourly panel …")
    panel_hourly = pd.concat(zone_datasets.values(), axis=0).sort_index()
    save(panel_hourly, "panel_hourly", args.no_parquet, args.no_csv)

    # ── Save daily aggregated panel ────────────────────────────────────────────
    log.info("Building and saving daily aggregated panel …")
    panel_daily = make_daily_panel(zone_datasets)
    save(panel_daily, "panel_daily_avg", args.no_parquet, args.no_csv)

    log.info("══════════════════════════════════════════════════")
    log.info("  Preprocessing complete.")
    log.info(f"  Hourly panel : {panel_hourly.shape[0]:,} rows × {panel_hourly.shape[1]} cols")
    log.info(f"  Daily  panel : {panel_daily.shape[0]:,} rows × {panel_daily.shape[1]} cols")
    log.info(f"  Output dir   : {OUT}")
    log.info("══════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
