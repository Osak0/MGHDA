#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../lib/study2_env.sh"
require_env MGHDA_DATA_ROOT
MODEL_PATH="${MODEL_PATH:-${MEDGEMMA_MODEL_PATH:-}}"
require_env MODEL_PATH
require_env MODEL_NAME
require_env RUN_NAME
: "${STUDY2_SET:=full_v2}"
: "${RUN_PHASE:=full}"
: "${RUN_MODE:=run}"
: "${SEED:=42}"

if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: RUN_NAME contains unsupported characters" >&2
  exit 2
fi
case "$STUDY2_SET" in
  full_v1) version=v1 ;;
  full_v2) version=v2 ;;
  *) echo "error: STUDY2_SET must be full_v1 or full_v2" >&2; exit 2 ;;
esac
if [[ "$RUN_PHASE" != "ablation" && "$RUN_PHASE" != "full" ]]; then
  echo "error: RUN_PHASE must be ablation or full" >&2
  exit 2
fi
if [[ "$RUN_MODE" != "run" && "$RUN_MODE" != "dry-run" ]]; then
  echo "error: RUN_MODE must be run or dry-run" >&2
  exit 2
fi

run_root="$MGHDA_DATA_ROOT/outputs/study2/$STUDY2_SET/$RUN_NAME"
phase_root="$run_root/$RUN_PHASE"
mkdir -p "$phase_root/raw" "$phase_root/parsed" "$phase_root/scored" \
  "$phase_root/audits" "$run_root/checkpoints"

scored_paths=()
for split in g1 g2; do
  if [[ "$RUN_PHASE" == "ablation" ]]; then
    input_path="$MGHDA_DATA_ROOT/processed/study2/ablation/$version/${split}_model_inputs.jsonl"
    metadata_path="$MGHDA_DATA_ROOT/processed/study2/ablation/$version/${split}_eval_metadata.jsonl"
  else
    input_path="$MGHDA_DATA_ROOT/processed/study2/$version/study2_${split}_model_inputs.jsonl"
    metadata_path="$MGHDA_DATA_ROOT/processed/study2/$version/study2_${split}_eval_metadata.jsonl"
  fi

  if [[ "$RUN_MODE" == "dry-run" ]]; then
    python -m ghm.inference.medgemma_runner \
      --input "$input_path" --eval-metadata "$metadata_path" \
      --output "$phase_root/audits/${split}_dry_run.jsonl" \
      --model-path "$MODEL_PATH" --model-name "$MODEL_NAME" \
      --data-root "$MGHDA_DATA_ROOT" --dry-run
    continue
  fi

  python -m ghm.inference.medgemma_runner \
    --input "$input_path" --eval-metadata "$metadata_path" \
    --output "$phase_root/raw/${split}.jsonl" \
    --model-path "$MODEL_PATH" --model-name "$MODEL_NAME" \
    --data-root "$MGHDA_DATA_ROOT" \
    --batch-size "${BATCH_SIZE:-1}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-16}" \
    --temperature "${TEMPERATURE:-0.0}" \
    --dtype "${DTYPE:-bfloat16}" --device "${DEVICE:-cuda}" \
    --seed "$SEED" \
    --checkpoint "$run_root/checkpoints/${split}.jsonl" --resume
  python -m ghm.evaluation.parse_answer \
    --input "$phase_root/raw/${split}.jsonl" \
    --output "$phase_root/parsed/${split}.jsonl"
  python -m ghm.evaluation.score_closed_qa \
    --input "$phase_root/parsed/${split}.jsonl" \
    --output "$phase_root/scored/${split}.jsonl" \
    --summary "$phase_root/audits/${split}_score_summary.json"
  python -m ghm.evaluation.validate_run \
    --model-inputs "$input_path" --eval-metadata "$metadata_path" \
    --raw "$phase_root/raw/${split}.jsonl" \
    --parsed "$phase_root/parsed/${split}.jsonl" \
    --scored "$phase_root/scored/${split}.jsonl" \
    --output "$phase_root/audits/${split}_validation.json"
  scored_paths+=("$phase_root/scored/${split}.jsonl")
done

if [[ "$RUN_MODE" == "run" ]]; then
  python -m ghm.evaluation.metrics \
    --inputs "${scored_paths[@]}" \
    --output "$phase_root/audits/combined_score_summary.json"
  python -m ghm.evaluation.visualize_results \
    --inputs "$phase_root/audits/combined_score_summary.json" \
    --run-names "$RUN_NAME-$RUN_PHASE" \
    --output-dir "$phase_root/reports"
fi
