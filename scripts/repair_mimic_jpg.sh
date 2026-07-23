#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash scripts/repair_mimic_jpg.sh /private/bad_image_paths.txt" >&2
  exit 2
fi

source "$(dirname "$0")/lib/study2_env.sh"
require_env MGHDA_DATA_ROOT

cd "$MGHDA_ROOT"
python -m ghm.data.repair_mimic_jpg \
  --data-root "$MGHDA_DATA_ROOT" \
  --repair-list "$1"
