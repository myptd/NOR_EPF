#!/usr/bin/env bash
# 1e_pre_post_crisis.sh
# Train LightGBM under pre-crisis (2019-2021) and post-crisis (2022-2023)
# windows only.  The full-window (2019-2023) results already exist from
# step 1a and are NOT re-run.
# Finally, aggregate all three windows into the comparison table.
set -euo pipefail
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate elec

mkdir -p model_results paper_outputs

echo "=== Step 1e: Pre-crisis vs post-crisis training-window experiment ==="

echo "  Running pre-crisis (2019-2021) and post-crisis (2022-2023) ..."
python models/pre_post_crisis.py --zones NO1 NO2 NO3 NO4 NO5

echo "  Building comparison table ..."
python analysis/10_pre_post_crisis_table.py

echo "=== Done: 1e_pre_post_crisis ==="

