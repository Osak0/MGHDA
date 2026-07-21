#!/usr/bin/env bash
set -euo pipefail

: "${RUN_NAME:=medgemma_study2_v2_full}"
: "${STUDY2_SET:=full_v2}"
: "${RUN_PHASE:=ablation}"
export RUN_NAME STUDY2_SET RUN_PHASE
exec bash "$(dirname "$0")/study2/run.sh"
