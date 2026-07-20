#!/usr/bin/env bash

SCRIPT_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MGHDA_ROOT:=$(cd "$SCRIPT_LIB_DIR/../.." && pwd)}"
: "${MGHDA_CONFIG:=$MGHDA_ROOT/configs/study3_medgemma.env}"

if [[ -f "$MGHDA_CONFIG" ]]; then
  # The config is private and owned by the local operator.
  # shellcheck disable=SC1090
  source "$MGHDA_CONFIG"
fi

export MGHDA_ROOT
export PYTHONPATH="$MGHDA_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "error: $name is required; copy configs/study3_medgemma.env.example to configs/study3_medgemma.env" >&2
    return 1
  fi
}

require_study3_identity() {
  : "${STUDY3_EXPERIMENT_ID:=study3_multiselect_v1}"
  if [[ "$STUDY3_EXPERIMENT_ID" != "study3_multiselect_v1" ]]; then
    echo "error: STUDY3_EXPERIMENT_ID must be study3_multiselect_v1" >&2
    return 1
  fi
}
