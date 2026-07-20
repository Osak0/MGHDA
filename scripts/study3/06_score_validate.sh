#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../lib/study3_env.sh"
require_env MGHDA_DATA_ROOT
require_study3_identity
: "${RUN_NAME:=medgemma_study3_multiselect_v1}"
: "${BOOTSTRAP_SAMPLES:=10000}"

cd "$MGHDA_ROOT"
mkdir -p "$MGHDA_DATA_ROOT/outputs/study3/scored"
mkdir -p "$MGHDA_DATA_ROOT/outputs/study3/audits"
for granularity in g1 g2; do
  python -m ghm.study3.scoring \
    --raw "$MGHDA_DATA_ROOT/outputs/study3/raw_responses/${RUN_NAME}_${granularity}_raw.jsonl" \
    --eval-metadata "$MGHDA_DATA_ROOT/processed/study3/prompts/study3_${granularity}_multiselect_eval_metadata.jsonl" \
    --output "$MGHDA_DATA_ROOT/outputs/study3/scored/${RUN_NAME}_${granularity}_scored.jsonl" \
    --summary "$MGHDA_DATA_ROOT/outputs/study3/audits/${RUN_NAME}_${granularity}_score_summary.json" \
    --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
    --bootstrap-seed "${STUDY3_SAMPLE_SEED:-42}"
  python -m ghm.study3.validation \
    --model-inputs "$MGHDA_DATA_ROOT/processed/study3/prompts/study3_${granularity}_multiselect_model_inputs.jsonl" \
    --eval-metadata "$MGHDA_DATA_ROOT/processed/study3/prompts/study3_${granularity}_multiselect_eval_metadata.jsonl" \
    --raw "$MGHDA_DATA_ROOT/outputs/study3/raw_responses/${RUN_NAME}_${granularity}_raw.jsonl" \
    --scored "$MGHDA_DATA_ROOT/outputs/study3/scored/${RUN_NAME}_${granularity}_scored.jsonl" \
    --data-root "$MGHDA_DATA_ROOT" \
    --output "$MGHDA_DATA_ROOT/outputs/study3/audits/${RUN_NAME}_${granularity}_validation.json"
done
