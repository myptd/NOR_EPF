#!/usr/bin/env bash
# 1d_lstm.sh  — Deep learning models (LSTM, TCN, Transformer) for all 5 zones
set -euo pipefail
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate elec

mkdir -p model_results model_weights

echo "=== Step 1d-a: LSTM training (NO1-NO5, 150 epochs) ==="
python models/deep_learning.py \
  --model lstm \
  --zones NO1 NO2 NO3 NO4 NO5 \
  --val-start 2024-01-01 --test-start 2025-01-01 \
  --epochs 150

echo "=== Step 1d-b: TCN training (NO1-NO5, 150 epochs) ==="
python models/deep_learning.py \
  --model tcn \
  --zones NO1 NO2 NO3 NO4 NO5 \
  --val-start 2024-01-01 --test-start 2025-01-01 \
  --epochs 150

echo "=== Step 1d-c: Transformer training (NO1-NO5, 150 epochs) ==="
python models/deep_learning.py \
  --model transformer \
  --zones NO1 NO2 NO3 NO4 NO5 \
  --val-start 2024-01-01 --test-start 2025-01-01 \
  --epochs 150

echo "=== Done: 1d_lstm ==="
