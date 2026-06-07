#!/usr/bin/env bash
set -euo pipefail

: "${MGHDA_ROOT:=/xjtu-mlp-vepfs/wangruiyang/MGHDA}"
: "${MGHDA_DATA_ROOT:=/xjtu-mlp-vepfs/wangruiyang/MGHDA-data}"
: "${RUN_NAME:=medgemma_smoke}"

cd "$MGHDA_ROOT"
export PYTHONPATH="$MGHDA_ROOT/src"

python -m ghm.evaluation.visualize_results \
  --inputs "$MGHDA_DATA_ROOT/outputs/audits/${RUN_NAME}_g1_g2_h1_h2_score_summary.json" \
  --run-names "$RUN_NAME" \
  --output-dir "$MGHDA_DATA_ROOT/outputs/reports/${RUN_NAME}"
