#!/usr/bin/env bash
# 1c_ablation. Feature ablation study for all 5 zonessh 
set -euo pipefail
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate elec

mkdir -p model_results

echo "=== Step 1c: Feature ablation (NO1-NO5) ==="
pids=()
for zone in NO1 NO2 NO3 NO4 NO5; do
  echo "  Starting zone $zone..."
  python models/ablation.py \
    --zones "$zone" \
    --val-start 2024-01-01 --test-start 2025-01-01 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
echo "=== Done: 1c_ablation ==="
