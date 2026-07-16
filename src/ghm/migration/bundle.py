"""Create and verify a minimum private Study 2 inference bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from ghm.granularity.common import read_jsonl


MANIFEST_NAME = "bundle_manifest.json"
CHECKSUM_NAME = "manifest.sha256"
PROMPT_DIR = Path("processed/prompts")
TRANSFER_CHECKSUM_NAME = "study2_files.sha256"
TRANSFER_FILE_LIST_NAME = "study2_files_from.txt"
TRANSFER_SUMMARY_NAME = "study2_transfer_summary.json"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_bundle(
    *,
    data_root: Path,
    output_dir: Path,
    model_input_paths: list[Path],
    eval_metadata_paths: list[Path],
    git_commit: str | None,
) -> dict[str, Any]:
    """Materialize prompts, metadata, and deduplicated referenced images."""

    if len(model_input_paths) != len(eval_metadata_paths):
        raise ValueError("model-input and eval-metadata file counts must match")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be empty to prevent stale private files")

    data_root = data_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_output_dir = output_dir / PROMPT_DIR
    prompt_output_dir.mkdir(parents=True, exist_ok=True)

    all_input_ids: set[str] = set()
    all_metadata_ids: set[str] = set()
    image_relpaths: set[Path] = set()
    prompt_template_ids: set[str] = set()
    copied_files: set[Path] = set()

    for model_path, metadata_path in zip(model_input_paths, eval_metadata_paths, strict=True):
        model_rows = read_jsonl(model_path)
        metadata_rows = read_jsonl(metadata_path)
        model_ids = _unique_item_ids(model_rows, label="model inputs")
        metadata_ids = _unique_item_ids(metadata_rows, label="eval metadata")
        if model_ids != metadata_ids:
            raise ValueError("model inputs and eval metadata must have identical item_id sets")
        overlap = all_input_ids.intersection(model_ids)
        if overlap:
            raise ValueError("item_id values must be unique across all input files")
        all_input_ids.update(model_ids)
        all_metadata_ids.update(metadata_ids)

        for row in model_rows:
            if "answer_label" in row:
                raise ValueError("model inputs must not contain answer_label")
            relpath = _portable_image_path(row.get("image_path"))
            image_relpaths.add(relpath)
            template_id = row.get("prompt_template_id")
            if template_id:
                prompt_template_ids.add(str(template_id))

        for source in (model_path, metadata_path):
            destination = prompt_output_dir / source.name
            shutil.copy2(source, destination)
            copied_files.add(destination)

    for relpath in sorted(image_relpaths, key=lambda path: path.as_posix()):
        source = (data_root / relpath).resolve()
        if not source.is_relative_to(data_root):
            raise ValueError("image path escapes data root")
        if not source.is_file():
            raise FileNotFoundError("one or more referenced images are missing")
        destination = output_dir / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_files.add(destination)

    total_bytes = sum(path.stat().st_size for path in copied_files)
    manifest = {
        "bundle_schema_version": 1,
        "experiment": "study2_g1_g2_claim_verification",
        "git_commit": git_commit,
        "model_input_records": len(all_input_ids),
        "eval_metadata_records": len(all_metadata_ids),
        "unique_images": len(image_relpaths),
        "copied_files": len(copied_files),
        "copied_bytes": total_bytes,
        "prompt_template_ids": sorted(prompt_template_ids),
    }
    manifest_path = output_dir / MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    copied_files.add(manifest_path)
    _write_checksum_manifest(output_dir, copied_files)
    return manifest


def verify_bundle(bundle_root: Path) -> dict[str, int]:
    """Verify every file named in the private SHA256 manifest."""

    bundle_root = bundle_root.resolve()
    checksum_path = bundle_root / CHECKSUM_NAME
    if not checksum_path.is_file():
        raise FileNotFoundError(f"missing {CHECKSUM_NAME}")
    checked = 0
    failed = 0
    with checksum_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            expected, relative_text = line.rstrip("\n").split("  ", 1)
            relative = _safe_manifest_path(relative_text)
            target = (bundle_root / relative).resolve()
            if not target.is_relative_to(bundle_root) or not target.is_file():
                failed += 1
            elif sha256_file(target) != expected:
                failed += 1
            checked += 1
    return {"checked_files": checked, "failed_files": failed}


def create_transfer_manifest(
    *,
    data_root: Path,
    output_dir: Path,
    model_input_paths: list[Path],
    eval_metadata_paths: list[Path],
    git_commit: str | None,
) -> dict[str, Any]:
    """Hash existing payload files in place without copying or moving data."""

    if len(model_input_paths) != len(eval_metadata_paths):
        raise ValueError("model-input and eval-metadata file counts must match")
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(data_root):
        raise ValueError("transfer manifest output must remain inside data root")
    output_dir.mkdir(parents=True, exist_ok=True)

    payload_paths: set[Path] = set()
    image_relpaths: set[Path] = set()
    all_ids: set[str] = set()
    prompt_template_ids: set[str] = set()
    for model_path, metadata_path in zip(model_input_paths, eval_metadata_paths, strict=True):
        model_path = model_path.resolve()
        metadata_path = metadata_path.resolve()
        for source in (model_path, metadata_path):
            if not source.is_relative_to(data_root) or not source.is_file():
                raise ValueError("prompt and metadata files must exist inside data root")
            payload_paths.add(source)
        model_rows = read_jsonl(model_path)
        metadata_rows = read_jsonl(metadata_path)
        model_ids = _unique_item_ids(model_rows, label="model inputs")
        metadata_ids = _unique_item_ids(metadata_rows, label="eval metadata")
        if model_ids != metadata_ids:
            raise ValueError("model inputs and eval metadata must have identical item_id sets")
        if all_ids.intersection(model_ids):
            raise ValueError("item_id values must be unique across all input files")
        all_ids.update(model_ids)
        for row in model_rows:
            if "answer_label" in row:
                raise ValueError("model inputs must not contain answer_label")
            relpath = _portable_image_path(row.get("image_path"))
            image_relpaths.add(relpath)
            template_id = row.get("prompt_template_id")
            if template_id:
                prompt_template_ids.add(str(template_id))

    for relpath in image_relpaths:
        image_path = (data_root / relpath).resolve()
        if not image_path.is_relative_to(data_root) or not image_path.is_file():
            raise FileNotFoundError("one or more referenced images are missing")
        payload_paths.add(image_path)

    checksum_path = output_dir / TRANSFER_CHECKSUM_NAME
    file_list_path = output_dir / TRANSFER_FILE_LIST_NAME
    summary_path = output_dir / TRANSFER_SUMMARY_NAME
    checksum_rows = sorted(
        (path.relative_to(data_root).as_posix(), sha256_file(path)) for path in payload_paths
    )
    with checksum_path.open("w", encoding="utf-8", newline="\n") as file:
        for relative, digest in checksum_rows:
            file.write(f"{digest}  {relative}\n")

    metadata_paths = {checksum_path, file_list_path, summary_path}
    with file_list_path.open("w", encoding="utf-8", newline="\n") as file:
        for path in sorted(payload_paths | metadata_paths):
            file.write(path.relative_to(data_root).as_posix())
            file.write("\n")

    summary = {
        "transfer_schema_version": 1,
        "experiment": "study2_g1_g2_claim_verification",
        "git_commit": git_commit,
        "model_input_records": len(all_ids),
        "eval_metadata_records": len(all_ids),
        "unique_images": len(image_relpaths),
        "payload_files": len(payload_paths),
        "payload_bytes": sum(path.stat().st_size for path in payload_paths),
        "prompt_template_ids": sorted(prompt_template_ids),
        "copies_created": 0,
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    return summary


def verify_checksum_file(*, data_root: Path, checksum_path: Path) -> dict[str, int]:
    """Verify an in-place transfer checksum file relative to the data root."""

    data_root = data_root.resolve()
    checksum_path = checksum_path.resolve()
    if not checksum_path.is_file():
        raise FileNotFoundError("transfer checksum file is missing")
    checked = 0
    failed = 0
    with checksum_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            expected, relative_text = line.rstrip("\n").split("  ", 1)
            relative = _safe_manifest_path(relative_text)
            target = (data_root / relative).resolve()
            if not target.is_relative_to(data_root) or not target.is_file():
                failed += 1
            elif sha256_file(target) != expected:
                failed += 1
            checked += 1
    return {"checked_files": checked, "failed_files": failed}


def _unique_item_ids(rows: list[dict[str, Any]], *, label: str) -> set[str]:
    ids: list[str] = []
    for row in rows:
        if row.get("item_id") is None:
            raise ValueError(f"{label} contains a row without item_id")
        ids.append(str(row["item_id"]))
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains duplicate item_id values")
    return set(ids)


def _portable_image_path(value: Any) -> Path:
    if not value:
        raise ValueError("model input contains an empty image_path")
    text = str(value).replace("\\", "/")
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("image_path must be portable and relative")
    path = Path(*pure.parts)
    if path.parts[:1] != ("files",):
        raise ValueError("image_path must be relative to data root under files/")
    return path


def _safe_manifest_path(text: str) -> Path:
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("checksum manifest contains an unsafe path")
    return Path(*pure.parts)


def _write_checksum_manifest(bundle_root: Path, paths: set[Path]) -> None:
    rows: list[tuple[str, str]] = []
    for path in paths:
        relative = path.relative_to(bundle_root).as_posix()
        rows.append((relative, sha256_file(path)))
    with (bundle_root / CHECKSUM_NAME).open("w", encoding="utf-8", newline="\n") as file:
        for relative, digest in sorted(rows):
            file.write(f"{digest}  {relative}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--data-root", type=Path, required=True)
    create.add_argument("--output-dir", type=Path, required=True)
    create.add_argument("--model-inputs", type=Path, nargs="+", required=True)
    create.add_argument("--eval-metadata", type=Path, nargs="+", required=True)
    create.add_argument("--git-commit", default=None)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--data-root", type=Path, required=True)
    manifest.add_argument("--output-dir", type=Path, required=True)
    manifest.add_argument("--model-inputs", type=Path, nargs="+", required=True)
    manifest.add_argument("--eval-metadata", type=Path, nargs="+", required=True)
    manifest.add_argument("--git-commit", default=None)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--bundle-root", type=Path, required=True)
    verify_transfer = subparsers.add_parser("verify-transfer")
    verify_transfer.add_argument("--data-root", type=Path, required=True)
    verify_transfer.add_argument("--checksum", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "create":
        summary = create_bundle(
            data_root=args.data_root,
            output_dir=args.output_dir,
            model_input_paths=args.model_inputs,
            eval_metadata_paths=args.eval_metadata,
            git_commit=args.git_commit,
        )
        print(
            "Created private inference bundle: "
            f"records={summary['model_input_records']}, "
            f"unique_images={summary['unique_images']}, files={summary['copied_files']}"
        )
        return 0

    if args.command == "manifest":
        summary = create_transfer_manifest(
            data_root=args.data_root,
            output_dir=args.output_dir,
            model_input_paths=args.model_inputs,
            eval_metadata_paths=args.eval_metadata,
            git_commit=args.git_commit,
        )
        print(
            "Created in-place transfer manifest: "
            f"records={summary['model_input_records']}, "
            f"unique_images={summary['unique_images']}, files={summary['payload_files']}, "
            "copies_created=0"
        )
        return 0

    if args.command == "verify-transfer":
        result = verify_checksum_file(data_root=args.data_root, checksum_path=args.checksum)
    else:
        result = verify_bundle(args.bundle_root)
    print(
        "Verified private inference bundle: "
        f"checked={result['checked_files']}, failed={result['failed_files']}"
    )
    return 1 if result["failed_files"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
