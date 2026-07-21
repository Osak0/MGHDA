#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash scripts/run_three_models_phase.sh <study2-ablation|study2-full|study3-smoke|study3-full>" >&2
  exit 2
fi
phase="$1"
source "$(dirname "$0")/lib/study2_env.sh"
: "${MODELS_CONFIG:=$MGHDA_ROOT/configs/models.env}"
if [[ ! -f "$MODELS_CONFIG" ]]; then
  echo "error: copy configs/models.env.example to configs/models.env" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$MODELS_CONFIG"
require_env MEDGEMMA4_MODEL_PATH
require_env MEDGEMMA15_MODEL_PATH
require_env QWEN3_VL_MODEL_PATH

model_names=(
  "google/medgemma-4b-it"
  "google/medgemma-1.5-4b-it"
  "Qwen/Qwen3-VL-8B-Instruct"
)
model_paths=(
  "$MEDGEMMA4_MODEL_PATH"
  "$MEDGEMMA15_MODEL_PATH"
  "$QWEN3_VL_MODEL_PATH"
)
model_slugs=(medgemma4 medgemma15 qwen3vl8b)

for index in "${!model_names[@]}"; do
  model_name="${model_names[$index]}"
  model_path="${model_paths[$index]}"
  slug="${model_slugs[$index]}"
  case "$phase" in
    study2-ablation)
      MODEL_NAME="$model_name" MODEL_PATH="$model_path" \
      STUDY2_SET=full_v1 RUN_PHASE=ablation \
      RUN_NAME="${slug}_study2_v1_full" \
        bash "$MGHDA_ROOT/scripts/study2/run.sh"
      MODEL_NAME="$model_name" MODEL_PATH="$model_path" \
      STUDY2_SET=full_v2 RUN_PHASE=ablation \
      RUN_NAME="${slug}_study2_v2_full" \
        bash "$MGHDA_ROOT/scripts/study2/run.sh"
      V1_RUN_NAME="${slug}_study2_v1_full" \
      V2_RUN_NAME="${slug}_study2_v2_full" \
      ABLATION_COMPARISON_NAME="${slug}_study2_v1_vs_v2" \
        bash "$MGHDA_ROOT/scripts/study2/compare_ablation.sh"
      ;;
    study2-full)
      MODEL_NAME="$model_name" MODEL_PATH="$model_path" \
      STUDY2_SET=full_v1 RUN_PHASE=full \
      RUN_NAME="${slug}_study2_v1_full" \
        bash "$MGHDA_ROOT/scripts/study2/run.sh"
      MODEL_NAME="$model_name" MODEL_PATH="$model_path" \
      STUDY2_SET=full_v2 RUN_PHASE=full \
      RUN_NAME="${slug}_study2_v2_full" \
        bash "$MGHDA_ROOT/scripts/study2/run.sh"
      ;;
    study3-smoke)
      MODEL_NAME="$model_name" MODEL_PATH="$model_path" \
      RUN_NAME="${slug}_study3_v2_smoke" \
        bash "$MGHDA_ROOT/scripts/study3/run_smoke.sh"
      ;;
    study3-full)
      MODEL_NAME="$model_name" MODEL_PATH="$model_path" \
      RUN_NAME="${slug}_study3_v2_full" \
        bash "$MGHDA_ROOT/scripts/study3/run_full.sh"
      ;;
    *)
      echo "error: unsupported phase: $phase" >&2
      exit 2
      ;;
  esac
done
