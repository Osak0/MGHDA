#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT
require_env RUN_NAME
: "${STUDY2_SET:=full_v2}"
: "${RUN_PHASE:=full}"
version="${STUDY2_SET#full_}"
phase_root="$MGHDA_DATA_ROOT/outputs/study2/$STUDY2_SET/$RUN_NAME/$RUN_PHASE"
for split in g1 g2; do
  if [[ "$RUN_PHASE" == "ablation" ]]; then
    input_path="$MGHDA_DATA_ROOT/processed/study2/ablation/$version/${split}_model_inputs.jsonl"
    metadata_path="$MGHDA_DATA_ROOT/processed/study2/ablation/$version/${split}_eval_metadata.jsonl"
  else
    input_path="$MGHDA_DATA_ROOT/processed/study2/$version/study2_${split}_model_inputs.jsonl"
    metadata_path="$MGHDA_DATA_ROOT/processed/study2/$version/study2_${split}_eval_metadata.jsonl"
  fi
  python -m ghm.evaluation.validate_run \
    --model-inputs "$input_path" --eval-metadata "$metadata_path" \
    --raw "$phase_root/raw/${split}.jsonl" \
    --parsed "$phase_root/parsed/${split}.jsonl" \
    --scored "$phase_root/scored/${split}.jsonl" \
    --output "$phase_root/audits/${split}_validation.json"
done
