"""Render linked items into separated model-input and evaluation metadata JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ghm.granularity.common import read_jsonl, write_jsonl
from ghm.prompts.templates import (
    SUPPORTED_UNSUPPORTED_TEMPLATE_ID,
    YES_NO_UNCERTAIN_TEMPLATE_ID,
    render_supported_unsupported_prompt,
    render_yes_no_uncertain_prompt,
)


MODEL_INPUT_FIELDS = ["item_id", "image_path", "prompt_template_id", "prompt"]
EVAL_METADATA_FIELDS = [
    "item_id",
    "answer_label",
    "granularity",
    "question_type",
    "hallucination_probe",
    "target_finding",
    "target_anatomy",
    "evidence_sources",
]


def build_prompt_layers(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Build model inputs and evaluation metadata from linked model-ready items."""

    model_inputs: list[dict[str, Any]] = []
    eval_metadata: list[dict[str, Any]] = []
    skipped_missing_image_path = 0
    skipped_missing_question = 0

    for item in items:
        image_path = item.get("image_path")
        question = item.get("question")
        if not image_path:
            skipped_missing_image_path += 1
            continue
        if not question:
            skipped_missing_question += 1
            continue

        prompt_template_id, prompt = render_prompt_for_item(item, str(question))
        item_id = item.get("item_id")
        model_inputs.append(
            {
                "item_id": item_id,
                "image_path": image_path,
                "prompt_template_id": prompt_template_id,
                "prompt": prompt,
            }
        )
        eval_metadata.append(
            {
                "item_id": item_id,
                "answer_label": item.get("answer_label"),
                "granularity": item.get("granularity"),
                "question_type": item.get("question_type"),
                "hallucination_probe": item.get("hallucination_probe"),
                "target_finding": item.get("target_finding"),
                "target_anatomy": item.get("target_anatomy"),
                "evidence_sources": item.get("evidence_sources"),
            }
        )

    summary = {
        "input_items": len(items),
        "model_input_records": len(model_inputs),
        "eval_metadata_records": len(eval_metadata),
        "skipped_missing_image_path": skipped_missing_image_path,
        "skipped_missing_question": skipped_missing_question,
    }
    return model_inputs, eval_metadata, summary


def build_prompt_records(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Backward-compatible helper returning only model input records."""

    model_inputs, _, summary = build_prompt_layers(items)
    legacy_summary = {
        "input_items": summary["input_items"],
        "prompt_records": summary["model_input_records"],
        "skipped_missing_image_path": summary["skipped_missing_image_path"],
        "skipped_missing_question": summary["skipped_missing_question"],
    }
    return model_inputs, legacy_summary


def render_prompt_for_item(item: dict[str, Any], question: str) -> tuple[str, str]:
    """Render the prompt matching a probe's answer space."""

    if item.get("hallucination_probe") == "H2":
        return SUPPORTED_UNSUPPORTED_TEMPLATE_ID, render_supported_unsupported_prompt(question)
    return YES_NO_UNCERTAIN_TEMPLATE_ID, render_yes_no_uncertain_prompt(question)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="Build separated model-input and eval-metadata prompt JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input linked item JSONL.",
    )
    parser.add_argument(
        "--model-inputs-output",
        type=Path,
        required=True,
        help="Output model-input JSONL. Does not include answer_label.",
    )
    parser.add_argument(
        "--eval-metadata-output",
        type=Path,
        required=True,
        help="Output evaluation metadata JSONL keyed by item_id.",
    )
    args = parser.parse_args(argv)

    items = read_jsonl(args.input)
    model_inputs, eval_metadata, summary = build_prompt_layers(items)
    write_jsonl(model_inputs, args.model_inputs_output)
    write_jsonl(eval_metadata, args.eval_metadata_output)
    print(
        "Built prompt layers: "
        f"input_items={summary['input_items']}, "
        f"model_inputs={summary['model_input_records']}, "
        f"eval_metadata={summary['eval_metadata_records']}, "
        f"skipped_missing_image_path={summary['skipped_missing_image_path']}, "
        f"skipped_missing_question={summary['skipped_missing_question']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
