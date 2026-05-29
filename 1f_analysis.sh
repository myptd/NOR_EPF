#!/usr/bin/env bash
# 1f_analysis. Generate all paper figures (parallel + sequential)sh 
set -euo pipefail
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate elec

mkdir -p paper_outputs

echo "=== Step 1f: Paper figure analysis ==="

echo "  Batch 1 (independent, parallel) ..."
python analysis/01_data_overview.py     &
python analysis/05_multizone.py         &
python analysis/03_walk_forward.py      &
python analysis/04_feature_ablation.py  &
wait
echo "  Batch 1 done."

echo "  Batch 2 (model_results CSVs) ..."
python analysis/02_model_comparison.py
python analysis/07_dm_pairwise.py
python analysis/08_regime_analysis.py
python analysis/09_worst_predictions.py
python analysis/10_pre_post_crisis_table.py

echo "=== Done: 1f_analysis ==="
