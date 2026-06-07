#!/usr/bin/env bash
set -euo pipefail

: "${MGHDA_ROOT:=/xjtu-mlp-vepfs/wangruiyang/MGHDA}"
: "${MGHDA_DATA_ROOT:=/xjtu-mlp-vepfs/wangruiyang/MGHDA-data}"
: "${RUN_NAME:=medgemma_smoke}"

cd "$MGHDA_ROOT"
export PYTHONPATH="$MGHDA_ROOT/src"

mkdir -p "$MGHDA_DATA_ROOT/outputs/raw_responses"
mkdir -p "$MGHDA_DATA_ROOT/outputs/scored"
mkdir -p "$MGHDA_DATA_ROOT/outputs/audits"

for split in g1_h1 g1_h2 g2_h1 g2_h2; do
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
    "$MGHDA_DATA_ROOT/outputs/scored/${RUN_NAME}_g1_h1_scored.jsonl" \
    "$MGHDA_DATA_ROOT/outputs/scored/${RUN_NAME}_g1_h2_scored.jsonl" \
    "$MGHDA_DATA_ROOT/outputs/scored/${RUN_NAME}_g2_h1_scored.jsonl" \
    "$MGHDA_DATA_ROOT/outputs/scored/${RUN_NAME}_g2_h2_scored.jsonl" \
  --output "$MGHDA_DATA_ROOT/outputs/audits/${RUN_NAME}_g1_g2_h1_h2_score_summary.json"
