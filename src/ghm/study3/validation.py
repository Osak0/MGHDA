"""Structural validation for Study 3 prompt, inference, and score layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ghm.granularity.common import read_jsonl
from ghm.study3.constants import (
    EXPERIMENT_ID,
    require_study3_experiment,
    require_study3_item_id,
    require_study3_output_path,
)


def validate_study3_run(
    model_inputs: list[dict[str, Any]],
    eval_metadata: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]] | None = None,
    scored_rows: list[dict[str, Any]] | None = None,
    *,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Validate Study 3 isolation, ID alignment, and image availability."""

    errors: list[str] = []
    input_ids = _validated_ids(model_inputs, "model_inputs", errors)
    metadata_ids = _validated_ids(eval_metadata, "eval_metadata", errors)
    raw_ids = (
        _validated_ids(raw_rows, "raw", errors) if raw_rows is not None else None
    )
    scored_ids = (
        _validated_ids(scored_rows, "scored", errors)
        if scored_rows is not None
        else None
    )
    if input_ids != metadata_ids:
        errors.append("model_inputs and eval_metadata item_id sets differ")
    if raw_ids is not None and raw_ids != input_ids:
        errors.append("raw and model_inputs item_id sets differ")
    if scored_ids is not None and scored_ids != input_ids:
        errors.append("scored and model_inputs item_id sets differ")

    leaked_model_rows = 0
    missing_images = 0
    for row in model_inputs:
        if any(
            key in row
            for key in (
                "gold_selected_options",
                "answer_label",
                "source_assertions",
                "polarity",
            )
        ):
            leaked_model_rows += 1
        image_path = row.get("image_path")
        if data_root is not None:
            resolved = data_root / str(image_path) if image_path else None
            if resolved is None or not resolved.is_file():
                missing_images += 1
    if leaked_model_rows:
        errors.append(f"{leaked_model_rows} model input rows contain answer metadata")
    if missing_images:
        errors.append(f"{missing_images} referenced images are missing")

    prompt_template_ids = sorted(
        {
            str(row.get("prompt_template_id"))
            for row in model_inputs
            if row.get("prompt_template_id")
        }
    )
    if any(not value.startswith("study3_") for value in prompt_template_ids):
        errors.append("non-Study-3 prompt template detected")
    model_ids = (
        {
            str(row.get("model_id") or row.get("model_name"))
            for row in raw_rows
            if row.get("model_id") or row.get("model_name")
        }
        if raw_rows is not None
        else set()
    )
    if raw_rows is not None and len(model_ids) != 1:
        errors.append("raw rows must contain exactly one model identity")

    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "model_inputs": len(model_inputs),
        "eval_metadata": len(eval_metadata),
        "raw_rows": len(raw_rows) if raw_rows is not None else None,
        "scored_rows": len(scored_rows) if scored_rows is not None else None,
        "leaked_model_rows": leaked_model_rows,
        "missing_images": missing_images,
        "prompt_template_ids": prompt_template_ids,
        "model_identity_variants": len(model_ids),
    }


def _validated_ids(
    rows: list[dict[str, Any]] | None,
    label: str,
    errors: list[str],
) -> set[str]:
    if rows is None:
        return set()
    ids: list[str] = []
    for row in rows:
        try:
            if "experiment_id" in row:
                require_study3_experiment(row.get("experiment_id"))
            require_study3_item_id(row.get("item_id"))
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            continue
        ids.append(str(row["item_id"]))
    if len(ids) != len(set(ids)):
        errors.append(f"{label} contains duplicate item_id values")
    return set(ids)


def main(argv: list[str] | None = None) -> int:
    """Validate one Study 3 run and write an aggregate report."""

    parser = argparse.ArgumentParser(description="Validate Study 3 run layers.")
    parser.add_argument("--model-inputs", type=Path, required=True)
    parser.add_argument("--eval-metadata", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--scored", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_study3_output_path(args.output)

    result = validate_study3_run(
        read_jsonl(args.model_inputs),
        read_jsonl(args.eval_metadata),
        read_jsonl(args.raw) if args.raw else None,
        read_jsonl(args.scored) if args.scored else None,
        data_root=args.data_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
        file.write("\n")
    print(
        f"Study 3 validation: status={result['status']}, "
        f"inputs={result['model_inputs']}, errors={len(result['errors'])}"
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
