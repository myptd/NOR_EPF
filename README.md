# Electricity price forecasting across Norway's five bidding zones in the post-crisis era

Code repository for the paper:

> My Thi Diem Phan, Trung Tuyen Truong, Hoai Phuong Ha, Dat Thanh Nguyen.
> **Electricity price forecasting across Norway's five bidding zones in the post-crisis era.**
> arXiv:2604.26634 [cs.LG], 2026.
> https://arxiv.org/abs/2604.26634

## Overview

This work presents a comprehensive benchmark for **short-horizon electricity price forecasting** across all five Norwegian Nord Pool bidding zones (NO1–NO5).
We construct a multimodal hourly dataset (2019–2025) and evaluate eight model families — including LightGBM, Ridge ARX, LSTM, TCN, and Transformer — using a strictly causal test set for short-term forecasting (1 step ahead prediction).
Experiments include rolling-origin walk-forward backtesting, leave-one-group-out feature ablation, and conditional regime analysis.

Key findings:
- LightGBM achieves the best performance in every zone (MAE 1.64–5.74 EUR/MWh on the 2025 test set).
- Ridge ARX remains a competitive linear baseline in northern zones.
- Lagged prices and calendar features alone yield surprisingly high accuracy; external features (reservoir levels, gas prices) remain critical for stratifying errors under stressed market regimes.

## Repository structure

```
bin/
    data_preprocess.py          # Data preprocessing and feature engineering
data/
    cleaned/                    # Preprocessed hourly parquet files (NO1–NO5)
    commodities/                # Raw commodity price data
    entsoe/                     # Raw ENTSO-E data
    reservoir/                  # Raw NVE reservoir data
    weather/                    # Raw open-meteo weather data
models/                         # Model implementations (LightGBM, Ridge ARX, LSTM, TCN, Transformer)
model_results/                  # Pre-computed result CSVs
model_weights/                  # Saved model weights (LightGBM, ablation variants)
analysis/
    01_data_overview.py         # Fig 1–2: price distributions and seasonality
    02_model_comparison.py      # Fig 3–4, Table 3–4: all-model comparison
    03_walk_forward.py          # Fig 5–5b: rolling-origin backtest
    04_feature_ablation.py      # Fig 6: feature group ablation heatmaps
    05_multizone.py             # Fig 7: multi-zone summary
    07_dm_pairwise.py           # Fig 8: Diebold-Mariano pairwise tests
    08_regime_analysis.py       # Fig 9: conditional regime analysis
    09_worst_predictions.py     # Fig 10: worst-prediction deep-dive
    10_pre_post_crisis_table.py # Table: pre/post-crisis training window
paper_outputs/                  # Generated figures and tables
```

## Environment setup

Requires Python 3.13 and Conda.

```bash
# Create and activate environment
conda env create -f env.yml
conda activate elec

# (Optional) install additional packages via pip
pip install -r requirements.txt
```

## Running the pipeline

Individual steps can be run separately:

```bash
bash 1a_train_models.sh        # Train all models (all zones)
bash 1b_walk_forward.sh        # Walk-forward backtest
bash 1c_ablation.sh            # Feature ablation experiments
bash 1d_deep_learning.sh       # LSTM, TCN, Transformer
bash 1e_pre_post_crisis.sh     # Pre/post-crisis training window
bash 1f_analysis.sh            # Generate all paper figures
```

Or run the complete pipeline at once:

```bash
bash 1_run_all.sh
```

## Data preprocessing

Raw data should be placed in `data/{entsoe,reservoir,weather,commodities}/`.
Run preprocessing to produce cleaned hourly parquet files:

```bash
python bin/data_preprocess.py
```

## Citation

```bibtex
@article{phan2026electricity,
  title   = {Electricity price forecasting across Norway's five bidding zones in the post-crisis era},
  author  = {Phan, My Thi Diem and Truong, Trung Tuyen and Ha, Hoai Phuong and Nguyen, Dat Thanh},
  journal = {arXiv preprint arXiv:2604.26634},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.26634}
}
```

