#!/usr/bin/env bash
# =============================================================================
# 1_run_all. Run the complete Norwegian EPF pipelinesh 
#
# Run from project root:  bash 1_run_all.sh
#
# Steps:
#   1a: Train baseline + advanced models (all zones)
#   1b: Walk-forward backtest (all zones)
#   1c: Feature ablation (all zones)
#   1d: Deep learning models: LSTM, TCN, Transformer (all zones)
#   1e: Pre-crisis vs post-crisis training-window experiment
#   1f: Analysis + paper figures
#
# Individual steps:
#   bash 1a_train_models.sh
#   bash 1b_walk_forward.sh
#   bash 1c_ablation.sh
#   bash 1d_deep_learning.sh
#   bash 1e_pre_post_crisis.sh
#   bash 1f_analysis.sh
# =============================================================================
set -euo pipefail

echo "============================================================"
echo "  Norwegian EPF  Full Run"Pipeline 
echo "  $(date)"
echo "============================================================"

bash 1a_train_models.sh
bash 1b_walk_forward.sh
bash 1c_ablation.sh
bash 1d_deep_learning.sh
bash 1e_pre_post_crisis.sh
bash 1f_analysis.sh

echo ""
echo "============================================================"
echo "  All steps complete!"
echo "  $(date)"
echo "============================================================"
