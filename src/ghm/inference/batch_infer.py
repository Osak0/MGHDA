"""Batch inference utilities for prompt JSONL records.

The first implementation provides a mock runner only. It does not read images
or call a real model.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ghm.granularity.common import read_jsonl, write_jsonl


ORACLE_LABELS = {"Yes", "No", "Uncertain", "Supported", "Unsupported"}


def build_mock_outputs(
    prompt_records: list[dict[str, Any]],
    eval_metadata_rows: list[dict[str, Any]] | None = None,
    *,
    model_name: str = "mock",
    mode: str = "oracle",
) -> list[dict[str, Any]]:
    """Build schema-compatible raw responses without calling a model."""

    metadata_by_item_id = {
        str(row.get("item_id")): row for row in (eval_metadata_rows or []) if row.get("item_id")
    }
    if mode == "oracle" and not metadata_by_item_id:
        raise ValueError("oracle mock mode requires eval metadata with answer_label")

    outputs: list[dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for index, record in enumerate(prompt_records):
        metadata = metadata_by_item_id.get(str(record.get("item_id")), {})
        answer_label = metadata.get("answer_label")
        raw_response = _mock_response(
            answer_label=answer_label,
            mode=mode,
            index=index,
            hallucination_probe=metadata.get("hallucination_probe"),
        )
        outputs.append(
            {
                "item_id": record.get("item_id"),
                "model_name": model_name,
                "model_version": mode,
                "image_path": record.get("image_path"),
                "prompt_template_id": record.get("prompt_template_id"),
                "prompt": record.get("prompt"),
                "raw_response": raw_response,
                "parsed_answer": None,
                "parse_status": "unparsed",
                "generation_config": {"mode": mode},
                "runtime": {
                    "timestamp": timestamp,
                    "device": "mock",
                },
                "answer_label": answer_label,
                "granularity": metadata.get("granularity"),
                "question_type": metadata.get("question_type"),
                "hallucination_probe": metadata.get("hallucination_probe"),
            }
        )
    return outputs


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Run mock batch inference.")
    parser.add_argument("--input", type=Path, required=True, help="Input model-input JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Output response JSONL.")
    parser.add_argument(
        "--eval-metadata",
        type=Path,
        default=None,
        help="Evaluation metadata JSONL. Required for --mode oracle.",
    )
    parser.add_argument("--model-name", default="mock", help="Model name to record.")
    parser.add_argument(
        "--mode",
        choices=[
            "oracle",
            "always_yes",
            "always_no",
            "always_supported",
            "always_unsupported",
            "alternating",
            "invalid",
        ],
        default="oracle",
        help="Mock response mode.",
    )
    args = parser.parse_args(argv)

    prompt_records = read_jsonl(args.input)
    eval_metadata_rows = read_jsonl(args.eval_metadata) if args.eval_metadata else None
    try:
        outputs = build_mock_outputs(
            prompt_records,
            eval_metadata_rows,
            model_name=args.model_name,
            mode=args.mode,
        )
    except ValueError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")
    write_jsonl(outputs, args.output)
    print(
        "Built mock raw responses: "
        f"model_inputs={len(prompt_records)}, outputs={len(outputs)}, mode={args.mode}"
    )
    return 0


def _mock_response(
    *,
    answer_label: Any,
    mode: str,
    index: int,
    hallucination_probe: Any,
) -> str:
    if mode == "oracle":
        return str(answer_label) if answer_label in ORACLE_LABELS else "Uncertain"
    if mode == "always_yes":
        return "Yes"
    if mode == "always_no":
        return "No"
    if mode == "always_supported":
        return "Supported"
    if mode == "always_unsupported":
        return "Unsupported"
    if mode == "alternating":
        if hallucination_probe == "H2":
            return "Supported" if index % 2 == 0 else "Unsupported"
        return "Yes" if index % 2 == 0 else "No"
    return "This response intentionally does not follow the requested format."


if __name__ == "__main__":
    raise SystemExit(main())
