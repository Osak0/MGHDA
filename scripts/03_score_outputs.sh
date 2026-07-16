#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT
: "${RUN_NAME:=medgemma_study2}"

cd "$MGHDA_ROOT"
mkdir -p "$MGHDA_DATA_ROOT/outputs/raw_responses"
mkdir -p "$MGHDA_DATA_ROOT/outputs/scored"
mkdir -p "$MGHDA_DATA_ROOT/outputs/audits"

for split in study2_g1 study2_g2; do
  python -m ghm.evaluation.parse_answer \
    --input "$MGHDA_DATA_ROOT/outputs/raw_responses/${RUN_NAME}_${split}.jsonl" \
    --output "$MGHDA_DATA_ROOT/outputs/raw_responses/${RUN_NAME}_${split}_parsed.jsonl"

  python -m ghm.evaluation.score_closed_qa \
    --input "$MGHDA_DATA_ROOT/outputs/raw_responses/${RUN_NAME}_${split}_parsed.jsonl" \
    --output "$MGHDA_DATA_ROOT/outputs/scored/${RUN_NAME}_${split}_scored.jsonl" \
    --summary "$MGHDA_DATA_ROOT/outputs/audits/${RUN_NAME}_${split}_score_summary.json"
done

python -m ghm.evaluation.metrics \
  --inputs \
    "$MGHDA_DATA_ROOT/outputs/scored/${RUN_NAME}_study2_g1_scored.jsonl" \
    "$MGHDA_DATA_ROOT/outputs/scored/${RUN_NAME}_study2_g2_scored.jsonl" \
  --output "$MGHDA_DATA_ROOT/outputs/audits/${RUN_NAME}_study2_g1_g2_score_summary.json"
