#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../lib/study3_env.sh"
require_env MGHDA_DATA_ROOT
require_study3_identity
: "${RUN_NAME:=model_study3_multiselect_v2}"
: "${BOOTSTRAP_SAMPLES:=10000}"

cd "$MGHDA_ROOT"
mkdir -p "$MGHDA_DATA_ROOT/outputs/study3/v2/scored"
mkdir -p "$MGHDA_DATA_ROOT/outputs/study3/v2/audits"
if [[ "${STUDY3_MODE:-full}" == "smoke" ]]; then
  splits=(smoke)
else
  splits=(g1 g2)
fi
for granularity in "${splits[@]}"; do
  if [[ "$granularity" == "smoke" ]]; then
    input_path="$MGHDA_DATA_ROOT/processed/study3/v2/smoke/model_inputs.jsonl"
    metadata_path="$MGHDA_DATA_ROOT/processed/study3/v2/smoke/eval_metadata.jsonl"
  else
    input_path="$MGHDA_DATA_ROOT/processed/study3/v2/prompts/study3_${granularity}_multiselect_model_inputs.jsonl"
    metadata_path="$MGHDA_DATA_ROOT/processed/study3/v2/prompts/study3_${granularity}_multiselect_eval_metadata.jsonl"
  fi
  python -m ghm.study3.scoring \
    --raw "$MGHDA_DATA_ROOT/outputs/study3/v2/raw_responses/${RUN_NAME}_${granularity}_raw.jsonl" \
    --eval-metadata "$metadata_path" \
    --output "$MGHDA_DATA_ROOT/outputs/study3/v2/scored/${RUN_NAME}_${granularity}_scored.jsonl" \
    --summary "$MGHDA_DATA_ROOT/outputs/study3/v2/audits/${RUN_NAME}_${granularity}_score_summary.json" \
    --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
    --bootstrap-seed "${STUDY3_SAMPLE_SEED:-42}"
  python -m ghm.study3.validation \
    --model-inputs "$input_path" \
    --eval-metadata "$metadata_path" \
    --raw "$MGHDA_DATA_ROOT/outputs/study3/v2/raw_responses/${RUN_NAME}_${granularity}_raw.jsonl" \
    --scored "$MGHDA_DATA_ROOT/outputs/study3/v2/scored/${RUN_NAME}_${granularity}_scored.jsonl" \
    --data-root "$MGHDA_DATA_ROOT" \
    --output "$MGHDA_DATA_ROOT/outputs/study3/v2/audits/${RUN_NAME}_${granularity}_validation.json"
done
