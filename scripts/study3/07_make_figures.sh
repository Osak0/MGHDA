#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../lib/study3_env.sh"
require_env MGHDA_DATA_ROOT
require_study3_identity
: "${RUN_NAME:=model_study3_multiselect_v2}"
: "${BOOTSTRAP_SAMPLES:=10000}"

cd "$MGHDA_ROOT"
mkdir -p "$MGHDA_DATA_ROOT/outputs/study3/v2/figures"
python -m ghm.study3.reporting \
  --inputs \
    "$MGHDA_DATA_ROOT/outputs/study3/v2/scored/${RUN_NAME}_g1_scored.jsonl" \
    "$MGHDA_DATA_ROOT/outputs/study3/v2/scored/${RUN_NAME}_g2_scored.jsonl" \
  --summary "$MGHDA_DATA_ROOT/outputs/study3/v2/audits/${RUN_NAME}_g1_g2_score_summary.json" \
  --figures-dir "$MGHDA_DATA_ROOT/outputs/study3/v2/figures/$RUN_NAME" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --bootstrap-seed "${STUDY3_SAMPLE_SEED:-42}"
