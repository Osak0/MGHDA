#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: bash scripts/models/download_modelscope.sh <model-id> <local-dir>" >&2
  exit 2
fi
model_id="$1"
local_dir="$2"
case "$model_id" in
  google/medgemma-4b-it|google/medgemma-1.5-4b-it|Qwen/Qwen3-VL-8B-Instruct) ;;
  *)
    echo "error: model-id is not in the approved three-model registry" >&2
    exit 2
    ;;
esac
if [[ -e "$local_dir" && ! -d "$local_dir" ]]; then
  echo "error: local destination exists and is not a directory" >&2
  exit 2
fi
python - "$model_id" "$local_dir" <<'PY'
import sys
from modelscope import snapshot_download

snapshot_download(
    sys.argv[1],
    local_dir=sys.argv[2],
)
PY
