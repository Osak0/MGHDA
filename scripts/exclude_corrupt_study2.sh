#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: bash scripts/exclude_corrupt_study2.sh /private/bad_paths.txt [--apply]" >&2
  exit 2
fi
if [[ $# -eq 2 && "$2" != "--apply" ]]; then
  echo "error: the optional second argument must be --apply" >&2
  exit 2
fi

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT

args=(
  --data-root "$MGHDA_DATA_ROOT"
  --repair-list "$1"
)
if [[ $# -eq 2 ]]; then
  args+=(--apply)
fi

cd "$MGHDA_ROOT"
python -m ghm.data.exclude_corrupt_study2 "${args[@]}"
