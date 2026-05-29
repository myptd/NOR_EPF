#!/usr/bin/env bash
# 1a_train_models. Train baseline + advanced models for all 5 zonessh 
set -euo pipefail
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate elec

mkdir -p model_results model_weights

echo "=== Step 1a: Train all models (NO1-NO5) ==="
python models/run_all.py \
  --zones NO1 NO2 NO3 NO4 NO5 \
  --val-start 2024-01-01 --test-start 2025-01-01
echo "=== Done: 1a_train_models ==="
