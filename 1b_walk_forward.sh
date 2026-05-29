#!/usr/bin/env bash
# 1b_walk_forward. Walk-forward backtest for all 5 zonessh 
set -euo pipefail
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate elec

mkdir -p model_results

echo "=== Step 1b: Walk-forward backtest (NO1-NO5) ==="
pids=()
for zone in NO1 NO2 NO3 NO4 NO5; do
  echo "  Starting zone $zone..."
  python models/walk_forward.py \
    --zones "$zone" \
    --train-end 2023-12-31 --test-start 2025-01-01 --step-weeks 1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
echo "=== Done: 1b_walk_forward ==="
