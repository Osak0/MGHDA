"""Build separated Study 3 model-input and evaluation-metadata layers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ghm.granularity.common import read_jsonl, write_jsonl
from ghm.study3.constants import (
    ABSENT,
    EXPERIMENT_ID,
    PRESENT,
    require_study3_experiment,
    require_study3_item_id,
    require_study3_output_path,
)


MODEL_INPUT_FIELDS = (
    "item_id",
    "experiment_id",
    "image_path",
    "prompt_template_id",
    "prompt",
)

EVAL_METADATA_FIELDS = (
    "item_id",
    "experiment_id",
    "granularity",
    "question_type",
    "query_relation",
    "variant",
    "controlled_k",
    "anchor_id",
    "option_set_id",
    "option_count",
    "natural_option_count",
    "answer_composition",
    "options",
    "gold_selected_options",
    "target_anatomy",
    "bbox",
    "source_assertions",
    "evidence_sources",
    "source_quality",
)


def build_prompt_layers(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Separate prompts from private gold metadata without answer leakage."""

    model_inputs: list[dict[str, Any]] = []
    eval_metadata: list[dict[str, Any]] = []
    for item in items:
        validate_item(item)
        prompt_template_id, prompt = render_prompt(item)
        model_row = {
            "item_id": item["item_id"],
            "experiment_id": EXPERIMENT_ID,
            "image_path": item.get("image_path"),
            "prompt_template_id": prompt_template_id,
            "prompt": prompt,
        }
        if any(
            key in model_row
            for key in ("gold_selected_options", "answer_label", "polarity")
        ):
            raise ValueError("Study 3 model input contains answer leakage")
        metadata_row = {field: item.get(field) for field in EVAL_METADATA_FIELDS}
        model_inputs.append(model_row)
        eval_metadata.append(metadata_row)

    input_ids = {str(row["item_id"]) for row in model_inputs}
    metadata_ids = {str(row["item_id"]) for row in eval_metadata}
    if len(input_ids) != len(model_inputs) or input_ids != metadata_ids:
        raise ValueError("Study 3 prompt layers must have identical unique item IDs")
    return model_inputs, eval_metadata, {
        "model_inputs": len(model_inputs),
        "eval_metadata": len(eval_metadata),
        "present_prompts": sum(
            row["query_relation"] == PRESENT for row in eval_metadata
        ),
        "absent_prompts": sum(
            row["query_relation"] == ABSENT for row in eval_metadata
        ),
    }


def render_prompt(item: dict[str, Any]) -> tuple[str, str]:
    """Render one strict multiple-select prompt."""

    relation = item["query_relation"]
    relation_word = "present" if relation == PRESENT else "absent"
    template_id = f"study3_multiselect_{relation}_v1"
    if item["granularity"] == "G1_finding_existence":
        scope = "in this chest X-ray"
    else:
        anatomy = str(item.get("target_anatomy") or "").strip()
        if not anatomy:
            raise ValueError("G2 Study 3 item requires target_anatomy")
        scope = f"in the {anatomy}"
    option_lines = "\n".join(
        f"{option['option_id']}. {option['label_name']}"
        for option in item["options"]
    )
    prompt = (
        "This is a multiple-select question.\n"
        "Considering only the listed findings, select all findings that are "
        f"{relation_word} {scope}.\n\n"
        f"{option_lines}\n\n"
        "Reply only with comma-separated option letters, or NONE if no option "
        "applies."
    )
    return template_id, prompt


def validate_item(item: dict[str, Any]) -> None:
    """Validate a private item before prompt separation."""

    require_study3_experiment(item.get("experiment_id"))
    require_study3_item_id(item.get("item_id"))
    if item.get("query_relation") not in {PRESENT, ABSENT}:
        raise ValueError("Study 3 query_relation must be present or absent")
    options = item.get("options")
    if not isinstance(options, list) or len(options) < 2:
        raise ValueError("Study 3 item requires at least two options")
    option_ids = [row.get("option_id") for row in options]
    if len(set(option_ids)) != len(option_ids) or any(not value for value in option_ids):
        raise ValueError("Study 3 option IDs must be non-empty and unique")
    gold = item.get("gold_selected_options")
    if not isinstance(gold, list) or not set(gold).issubset(set(option_ids)):
        raise ValueError("Study 3 gold options must be a subset of option IDs")
    if item.get("option_count") != len(options):
        raise ValueError("Study 3 option_count does not match options")
    if not item.get("image_path"):
        raise ValueError("Study 3 item must be linked to an existing image")


def main(argv: list[str] | None = None) -> int:
    """Build prompt and evaluation layers for one Study 3 granularity."""

    parser = argparse.ArgumentParser(description="Build Study 3 prompt layers.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-inputs-output", type=Path, required=True)
    parser.add_argument("--eval-metadata-output", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in (args.model_inputs_output, args.eval_metadata_output):
        require_study3_output_path(path)

    items = read_jsonl(args.input)
    model_inputs, eval_metadata, summary = build_prompt_layers(items)
    write_jsonl(model_inputs, args.model_inputs_output)
    write_jsonl(eval_metadata, args.eval_metadata_output)
    print(
        "Built Study 3 prompt layers: "
        f"inputs={summary['model_inputs']}, "
        f"present={summary['present_prompts']}, "
        f"absent={summary['absent_prompts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
