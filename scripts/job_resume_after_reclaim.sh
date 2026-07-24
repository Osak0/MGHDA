#!/usr/bin/env bash
set -euo pipefail

: "${WORKSPACE:=/opt/data/private/mghda}"
: "${RESUME_PHASE:=all}"

repo="$WORKSPACE/code/MGHDA"
venv="$WORKSPACE/envs/mghda"
patch_dir="$WORKSPACE/incoming/corrupt-exclusion"
patch_name="study2_study3_corrupt_exclusion_private.tar.gz"
patch="$patch_dir/$patch_name"
patch_sidecar="$patch.sha256"
marker="$WORKSPACE/outputs/audits/corrupt_exclusion_patch_applied.txt"
log_dir="$WORKSPACE/outputs/logs"
log_path="$log_dir/reclaimed_job_${RESUME_PHASE}.log"

mkdir -p "$log_dir" "$WORKSPACE/outputs/audits"
exec > >(tee -a "$log_path") 2>&1

on_exit() {
  status=$?
  echo "job exit code: $status"
  echo "job finished at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap on_exit EXIT

case "$RESUME_PHASE" in
  all|study2-full|study3-full) ;;
  *)
    echo "error: RESUME_PHASE must be all, study2-full, or study3-full" >&2
    exit 2
    ;;
esac

for path in "$repo" "$venv/bin/activate" "$patch" "$patch_sidecar"; do
  if [[ ! -e "$path" ]]; then
    echo "error: required persistent path is missing" >&2
    exit 2
  fi
done

source "$venv/bin/activate"
cd "$repo"

echo "job started at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "workspace mounted: OK"
echo "resume phase: $RESUME_PHASE"
nvidia-smi

python - <<'PY'
import torch
import transformers

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda_available:", torch.cuda.is_available())
print(
    "gpu:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else None,
)
print(
    "bf16:",
    torch.cuda.is_bf16_supported()
    if torch.cuda.is_available()
    else False,
)
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
assert tuple(
    int(value)
    for value in torch.__version__.split("+")[0].split(".")[:2]
) >= (2, 6)
PY

if [[ ! -f "$marker" ]]; then
  (
    cd "$patch_dir"
    sha256sum -c "$(basename "$patch_sidecar")"
  )
  python - "$patch" <<'PY'
import sys
import tarfile
from pathlib import PurePosixPath

archive = sys.argv[1]
allowed = (
    "processed/study2/v1/",
    "processed/study2/v2/",
    "processed/study3/v2/prompts/",
    "processed/study3/v2/items/",
    "outputs/audits/",
)
count = 0
with tarfile.open(archive, "r:gz") as tar:
    for member in tar.getmembers():
        name = member.name.removeprefix("./")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError("unsafe patch archive member")
        if member.isfile():
            if not name.startswith(allowed):
                raise RuntimeError("unexpected patch archive member")
            count += 1
if count == 0:
    raise RuntimeError("patch archive contains no files")
print("patch archive safety: OK")
print("patch files:", count)
PY

  backup="$WORKSPACE/outputs/audits/pre_corrupt_exclusion_remote"
  if [[ ! -d "$backup" ]]; then
    mkdir -p "$backup/study2" "$backup/study3"
    cp -a "$WORKSPACE/processed/study2/v1" "$backup/study2/v1"
    cp -a "$WORKSPACE/processed/study2/v2" "$backup/study2/v2"
    cp -a "$WORKSPACE/processed/study3/v2/prompts" "$backup/study3/prompts"
    cp -a "$WORKSPACE/processed/study3/v2/items" "$backup/study3/items"
  fi
  tar -xzf "$patch" -C "$WORKSPACE"
  {
    echo "patch=$patch_name"
    echo "applied_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$marker"
  echo "corrupt-image exclusion patch applied: OK"
else
  echo "corrupt-image exclusion patch already applied: OK"
fi

python - "$WORKSPACE" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
bad_list = root / "outputs/audits/study2_g1_truncated_private_paths.txt"
bad_paths = {
    line.strip()
    for line in bad_list.read_text(encoding="utf-8").splitlines()
    if line.strip()
}

def load(path):
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]

def ids(rows):
    values = [str(row["item_id"]) for row in rows]
    if len(values) != len(set(values)):
        raise RuntimeError("duplicate item IDs")
    return set(values)

for version in ("v1", "v2"):
    for split in ("g1", "g2"):
        directory = root / f"processed/study2/{version}"
        inputs = load(directory / f"study2_{split}_model_inputs.jsonl")
        metadata = load(directory / f"study2_{split}_eval_metadata.jsonl")
        if len(inputs) != 478 or ids(inputs) != ids(metadata):
            raise RuntimeError("unexpected reduced Study 2 layers")
        if any(str(row.get("image_path")) in bad_paths for row in inputs):
            raise RuntimeError("corrupt image remains in Study 2")
        print(f"study2_{version}_{split}: records=478")

for split, expected in (("g1", 1800), ("g2", 1780)):
    base = root / "processed/study3/v2"
    inputs = load(
        base / "prompts" / f"study3_{split}_multiselect_model_inputs.jsonl"
    )
    metadata = load(
        base / "prompts" / f"study3_{split}_multiselect_eval_metadata.jsonl"
    )
    items = load(base / "items" / f"study3_{split}_multiselect_items.jsonl")
    if (
        len(inputs) != expected
        or ids(inputs) != ids(metadata)
        or ids(inputs) != ids(items)
    ):
        raise RuntimeError("unexpected reduced Study 3 layers")
    if any(str(row.get("image_path")) in bad_paths for row in inputs):
        raise RuntimeError("corrupt image remains in Study 3")
    print(f"study3_{split}: records={expected}")

smoke_inputs = load(root / "processed/study3/v2/smoke/model_inputs.jsonl")
smoke_metadata = load(root / "processed/study3/v2/smoke/eval_metadata.jsonl")
if len(smoke_inputs) != 40 or ids(smoke_inputs) != ids(smoke_metadata):
    raise RuntimeError("unexpected Study 3 smoke layers")
print("study3_smoke: records=40")
print("reduced input validation: OK")
PY

pytest -q
bash scripts/dry_run_all_inputs.sh

if [[ "$RESUME_PHASE" == "all" || "$RESUME_PHASE" == "study2-full" ]]; then
  echo "starting Study 2 full checkpoint resume"
  bash scripts/run_three_models_phase.sh study2-full
fi

if [[ "$RESUME_PHASE" == "all" || "$RESUME_PHASE" == "study3-full" ]]; then
  echo "starting Study 3 full checkpoint resume"
  bash scripts/run_three_models_phase.sh study3-full
fi

echo "requested experiment phases completed successfully"
