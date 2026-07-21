#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT
require_env RUN_NAME
: "${STUDY2_SET:=full_v2}"
: "${RUN_PHASE:=full}"
case "$STUDY2_SET" in full_v1|full_v2) ;; *) echo "error: invalid STUDY2_SET" >&2; exit 2 ;; esac
phase_root="$MGHDA_DATA_ROOT/outputs/study2/$STUDY2_SET/$RUN_NAME/$RUN_PHASE"
mkdir -p "$phase_root/parsed" "$phase_root/scored" "$phase_root/audits"
scored_paths=()
for split in g1 g2; do
  python -m ghm.evaluation.parse_answer \
    --input "$phase_root/raw/${split}.jsonl" \
    --output "$phase_root/parsed/${split}.jsonl"
  python -m ghm.evaluation.score_closed_qa \
    --input "$phase_root/parsed/${split}.jsonl" \
    --output "$phase_root/scored/${split}.jsonl" \
    --summary "$phase_root/audits/${split}_score_summary.json"
  scored_paths+=("$phase_root/scored/${split}.jsonl")
done
python -m ghm.evaluation.metrics \
  --inputs "${scored_paths[@]}" \
  --output "$phase_root/audits/combined_score_summary.json"
