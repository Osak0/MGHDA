"""Create and verify an isolated Study 3 direct-transfer manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from ghm.granularity.common import read_jsonl
from ghm.study3.constants import (
    EXPERIMENT_ID,
    require_study3_experiment,
    require_study3_item_id,
    require_study3_output_path,
)


CHECKSUM_NAME = "study3_files.sha256"
FILE_LIST_NAME = "study3_files_from.txt"
SUMMARY_NAME = "study3_transfer_summary.json"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_transfer_manifest(
    *,
    data_root: Path,
    output_dir: Path,
    model_input_paths: list[Path],
    eval_metadata_paths: list[Path],
    git_commit: str | None,
) -> dict[str, Any]:
    """Hash Study 3 prompts, metadata, and referenced existing images in place."""

    if len(model_input_paths) != len(eval_metadata_paths):
        raise ValueError("model input and eval metadata file counts must match")
    if not model_input_paths:
        raise ValueError("at least one Study 3 prompt pair is required")
    require_study3_output_path(output_dir)
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(data_root):
        raise ValueError("Study 3 transfer directory must be inside data root")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Study 3 transfer directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    payload_paths: set[Path] = set()
    image_paths: set[Path] = set()
    all_ids: set[str] = set()
    template_ids: set[str] = set()
    counts: Counter[str] = Counter()
    for model_path, metadata_path in zip(
        model_input_paths,
        eval_metadata_paths,
        strict=True,
    ):
        model_path = model_path.resolve()
        metadata_path = metadata_path.resolve()
        for path in (model_path, metadata_path):
            if not path.is_relative_to(data_root) or not path.is_file():
                raise ValueError("Study 3 prompt files must exist inside data root")
            require_study3_output_path(path)
            payload_paths.add(path)
        model_rows = read_jsonl(model_path)
        metadata_rows = read_jsonl(metadata_path)
        model_ids = _validated_ids(model_rows, is_model_input=True)
        metadata_ids = _validated_ids(metadata_rows, is_model_input=False)
        if model_ids != metadata_ids:
            raise ValueError("Study 3 model input and metadata IDs differ")
        if all_ids & model_ids:
            raise ValueError("Study 3 item IDs overlap across input files")
        all_ids.update(model_ids)

        for row in model_rows:
            relpath = _portable_path(row.get("image_path"))
            image = (data_root / relpath).resolve()
            if not image.is_relative_to(data_root) or not image.is_file():
                raise FileNotFoundError("one or more Study 3 images are missing")
            image_paths.add(image)
            template_ids.add(str(row["prompt_template_id"]))
        for row in metadata_rows:
            counts[
                "|".join(
                    [
                        str(row.get("granularity")),
                        str(row.get("variant")),
                        str(row.get("query_relation")),
                    ]
                )
            ] += 1

    payload_paths.update(image_paths)
    checksum_path = output_dir / CHECKSUM_NAME
    file_list_path = output_dir / FILE_LIST_NAME
    summary_path = output_dir / SUMMARY_NAME
    summary = {
        "transfer_schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "git_commit": git_commit,
        "model_input_records": len(all_ids),
        "eval_metadata_records": len(all_ids),
        "unique_images": len(image_paths),
        "payload_files": len(payload_paths),
        "payload_bytes": sum(path.stat().st_size for path in payload_paths),
        "prompt_template_ids": sorted(template_ids),
        "counts_by_granularity_variant_relation": dict(sorted(counts.items())),
        "copies_created": 0,
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")

    checksum_sources = payload_paths | {summary_path}
    checksum_rows = sorted(
        (
            path.relative_to(data_root).as_posix(),
            sha256_file(path),
        )
        for path in checksum_sources
    )
    with checksum_path.open("w", encoding="utf-8", newline="\n") as file:
        for relative, digest in checksum_rows:
            file.write(f"{digest}  {relative}\n")

    transfer_metadata = {checksum_path, file_list_path, summary_path}
    with file_list_path.open("w", encoding="utf-8", newline="\n") as file:
        for path in sorted(
            payload_paths | transfer_metadata,
            key=lambda value: value.as_posix(),
        ):
            file.write(path.relative_to(data_root).as_posix())
            file.write("\n")
    return summary


def verify_transfer(
    *,
    data_root: Path,
    checksum_path: Path,
    summary_path: Path,
    expected_git_commit: str | None = None,
) -> dict[str, Any]:
    """Verify Study 3 checksums and experiment/commit identity."""

    data_root = data_root.resolve()
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    require_study3_experiment(summary.get("experiment_id"))
    errors: list[str] = []
    if (
        expected_git_commit
        and expected_git_commit != summary.get("git_commit")
    ):
        errors.append("current Git commit does not match Study 3 transfer summary")
    checked = 0
    with checksum_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            expected, relative_text = line.rstrip("\n").split("  ", 1)
            relative = _portable_path(relative_text)
            target = (data_root / relative).resolve()
            if (
                not target.is_relative_to(data_root)
                or not target.is_file()
                or sha256_file(target) != expected
            ):
                errors.append(f"checksum failed: {relative.as_posix()}")
            checked += 1
    try:
        layer_summary = _validate_transferred_layers(
            data_root,
            summary_path.parent / FILE_LIST_NAME,
        )
        if layer_summary["records"] != summary.get("model_input_records"):
            errors.append("transferred record count does not match transfer summary")
        if (
            layer_summary["counts_by_granularity_variant_relation"]
            != summary.get("counts_by_granularity_variant_relation")
        ):
            errors.append("transferred strata counts do not match transfer summary")
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"transferred prompt validation failed: {exc}")
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if not errors else "fail",
        "checked_files": checked,
        "errors": errors,
        "git_commit": summary.get("git_commit"),
    }


def _validate_transferred_layers(
    data_root: Path,
    file_list_path: Path,
) -> dict[str, Any]:
    """Re-read transferred prompt layers and recompute their aggregate counts."""

    if not file_list_path.is_file():
        raise FileNotFoundError(FILE_LIST_NAME)
    relative_paths: list[Path] = []
    with file_list_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                relative = _portable_path(line.strip())
                if any(part.lower().startswith("study2") for part in relative.parts):
                    raise ValueError("Study 2 path detected in Study 3 transfer list")
                relative_paths.append(relative)
    model_paths = {
        path.name.removesuffix("_model_inputs.jsonl"): data_root / path
        for path in relative_paths
        if path.name.endswith("_model_inputs.jsonl")
    }
    metadata_paths = {
        path.name.removesuffix("_eval_metadata.jsonl"): data_root / path
        for path in relative_paths
        if path.name.endswith("_eval_metadata.jsonl")
    }
    if not model_paths or set(model_paths) != set(metadata_paths):
        raise ValueError("transferred Study 3 prompt pairs are missing or mismatched")

    all_ids: set[str] = set()
    counts: Counter[str] = Counter()
    for prefix in sorted(model_paths):
        model_rows = read_jsonl(model_paths[prefix])
        metadata_rows = read_jsonl(metadata_paths[prefix])
        model_ids = _validated_ids(model_rows, is_model_input=True)
        metadata_ids = _validated_ids(metadata_rows, is_model_input=False)
        if model_ids != metadata_ids or all_ids & model_ids:
            raise ValueError("transferred Study 3 item IDs are mismatched or duplicated")
        all_ids.update(model_ids)
        for row in metadata_rows:
            counts[
                "|".join(
                    [
                        str(row.get("granularity")),
                        str(row.get("variant")),
                        str(row.get("query_relation")),
                    ]
                )
            ] += 1
    return {
        "records": len(all_ids),
        "counts_by_granularity_variant_relation": dict(sorted(counts.items())),
    }


def _validated_ids(
    rows: list[dict[str, Any]],
    *,
    is_model_input: bool,
) -> set[str]:
    ids: set[str] = set()
    for row in rows:
        require_study3_experiment(row.get("experiment_id"))
        require_study3_item_id(row.get("item_id"))
        item_id = str(row["item_id"])
        if item_id in ids:
            raise ValueError("duplicate Study 3 item_id in transfer inputs")
        ids.add(item_id)
        if is_model_input and any(
            key in row for key in ("gold_selected_options", "answer_label", "polarity")
        ):
            raise ValueError("Study 3 model inputs contain answer leakage")
    return ids


def _portable_path(value: object) -> Path:
    if value is None:
        raise ValueError("portable path is missing")
    pure = PurePosixPath(str(value).replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"invalid portable Study 3 path: {value!r}")
    return Path(*pure.parts)


def main(argv: list[str] | None = None) -> int:
    """Create or verify Study 3 transfer metadata."""

    parser = argparse.ArgumentParser(description="Manage Study 3 transfer manifests.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--data-root", type=Path, required=True)
    manifest.add_argument("--output-dir", type=Path, required=True)
    manifest.add_argument("--model-inputs", type=Path, nargs="+", required=True)
    manifest.add_argument("--eval-metadata", type=Path, nargs="+", required=True)
    manifest.add_argument("--git-commit", default=None)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--data-root", type=Path, required=True)
    verify.add_argument("--checksum", type=Path, required=True)
    verify.add_argument("--summary", type=Path, required=True)
    verify.add_argument("--expected-git-commit", default=None)
    args = parser.parse_args(argv)

    if args.command == "manifest":
        summary = create_transfer_manifest(
            data_root=args.data_root,
            output_dir=args.output_dir,
            model_input_paths=args.model_inputs,
            eval_metadata_paths=args.eval_metadata,
            git_commit=args.git_commit,
        )
        print(
            "Prepared Study 3 transfer: "
            f"records={summary['model_input_records']}, "
            f"images={summary['unique_images']}"
        )
        return 0
    result = verify_transfer(
        data_root=args.data_root,
        checksum_path=args.checksum,
        summary_path=args.summary,
        expected_git_commit=args.expected_git_commit,
    )
    print(
        f"Study 3 transfer verification: status={result['status']}, "
        f"checked={result['checked_files']}"
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
