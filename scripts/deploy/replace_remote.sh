#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: bash scripts/deploy/replace_remote.sh /persistent/workspace <commit>" >&2
  exit 2
fi
workspace="$(realpath -m "$1")"
commit="$2"
if [[ "$workspace" == "/" || -z "$workspace" || ! "$commit" =~ ^[0-9a-fA-F]+$ ]]; then
  echo "error: unsafe workspace or commit" >&2
  exit 2
fi
incoming="$workspace/incoming/$commit"
code_archive="$incoming/MGHDA-study2v1v2-study3v2-$commit.tar.gz"
private_archive="$incoming/MGHDA-private-study2v1v2-study3v2-$commit.tar.gz"
for path in \
  "$code_archive" "$code_archive.sha256" \
  "$private_archive" "$private_archive.sha256"; do
  if [[ ! -f "$path" ]]; then
    echo "error: missing upload: $path" >&2
    exit 2
  fi
done

(
  cd "$incoming"
  sha256sum -c "$(basename "$code_archive.sha256")"
  sha256sum -c "$(basename "$private_archive.sha256")"
)

python - "$code_archive" "$private_archive" <<'PY'
import sys
import tarfile
from pathlib import PurePosixPath

for archive_index, archive_path in enumerate(sys.argv[1:]):
    seen = set()
    allowed_roots = {"MGHDA"} if archive_index == 0 else {"files", "processed", "outputs"}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or member.issym()
                or member.islnk()
                or member.isdev()
                or member.name in seen
                or path.parts[0] not in allowed_roots
            ):
                raise SystemExit(f"unsafe archive member in {archive_path}")
            seen.add(member.name)
PY

targets=(
  "$workspace/code/MGHDA"
  "$workspace/files"
  "$workspace/processed"
  "$workspace/outputs"
  "$workspace/interim"
  "$workspace/raw"
)
for target in "${targets[@]}"; do
  resolved="$(realpath -m "$target")"
  case "$resolved" in
    "$workspace/code/MGHDA"|"$workspace/files"|"$workspace/processed"|\
    "$workspace/outputs"|"$workspace/interim"|"$workspace/raw") ;;
    *)
      echo "error: deletion target escaped allowlist: $resolved" >&2
      exit 2
      ;;
  esac
done

for target in "${targets[@]}"; do
  rm -rf -- "$target"
done
mkdir -p "$workspace/code"
tar --extract --gzip --file "$code_archive" --directory "$workspace/code" --no-same-owner
tar --extract --gzip --file "$private_archive" --directory "$workspace" --no-same-owner

cd "$workspace/code/MGHDA"
if [[ "$(tr -d '\r\n' < RELEASE_COMMIT)" != "$commit" ]]; then
  echo "error: code archive commit identity mismatch" >&2
  exit 2
fi
export PYTHONPATH="$workspace/code/MGHDA/src${PYTHONPATH:+:$PYTHONPATH}"
python -m ghm.migration.archive verify-internal --data-root "$workspace"
python -m ghm.migration.archive verify-commit \
  --data-root "$workspace" \
  --expected "$commit"
echo "Clean replacement completed. models/, envs/, and incoming/ were preserved."
