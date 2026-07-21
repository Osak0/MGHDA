"""Validate structural integrity of a completed Study 2 model run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ghm.granularity.common import read_jsonl


def validate_run_layers(
    *,
    model_inputs: list[dict[str, Any]],
    eval_metadata: list[dict[str, Any]],
    raw: list[dict[str, Any]],
    parsed: list[dict[str, Any]],
    scored: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check one-to-one item coverage while treating model invalids as metrics."""

    layers = {
        "model_inputs": model_inputs,
        "eval_metadata": eval_metadata,
        "raw": raw,
        "parsed": parsed,
        "scored": scored,
    }
    ids: dict[str, set[str]] = {}
    duplicate_counts: dict[str, int] = {}
    missing_id_counts: dict[str, int] = {}
    failures: list[str] = []
    for name, rows in layers.items():
        values = [str(row["item_id"]) for row in rows if row.get("item_id") is not None]
        ids[name] = set(values)
        duplicate_counts[name] = len(values) - len(ids[name])
        missing_id_counts[name] = len(rows) - len(values)
        if duplicate_counts[name]:
            failures.append(f"duplicate_item_id:{name}")
        if missing_id_counts[name]:
            failures.append(f"missing_item_id:{name}")

    expected = ids["model_inputs"]
    for name in ("eval_metadata", "raw", "parsed", "scored"):
        if ids[name] != expected:
            failures.append(f"item_id_set_mismatch:{name}")

    answer_leaks = sum(1 for row in model_inputs if "answer_label" in row)
    if answer_leaks:
        failures.append("answer_label_in_model_inputs")
    runtime_failures = sum(
        1 for row in raw if row.get("runtime", {}).get("status") != "success"
    )
    if runtime_failures:
        failures.append("runtime_failures")

    template_ids = sorted(
        {str(row.get("prompt_template_id")) for row in model_inputs if row.get("prompt_template_id")}
    )
    generation_configs = {
        json.dumps(row.get("generation_config"), sort_keys=True)
        for row in raw
        if row.get("generation_config") is not None
    }
    if len(generation_configs) > 1:
        failures.append("generation_config_mismatch")
    model_ids = {
        str(row.get("model_id") or row.get("model_name"))
        for row in raw
        if row.get("model_id") or row.get("model_name")
    }
    if len(model_ids) != 1:
        failures.append("model_identity_mismatch")

    parse_status_counts = _count_values(parsed, "parse_status")
    invalid_responses = len(parsed) - parse_status_counts.get("success", 0)
    return {
        "validation_schema_version": 1,
        "valid": not failures,
        "failures": sorted(failures),
        "record_counts": {name: len(rows) for name, rows in layers.items()},
        "duplicate_item_id_counts": duplicate_counts,
        "missing_item_id_counts": missing_id_counts,
        "answer_label_leaks": answer_leaks,
        "runtime_failures": runtime_failures,
        "invalid_model_responses": invalid_responses,
        "parse_status_counts": parse_status_counts,
        "prompt_template_ids": template_ids,
        "generation_config_variants": len(generation_configs),
        "model_identity_variants": len(model_ids),
    }


def _count_values(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field))
        counts[value] = counts.get(value, 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-inputs", type=Path, required=True)
    parser.add_argument("--eval-metadata", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--parsed", type=Path, required=True)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    result = validate_run_layers(
        model_inputs=read_jsonl(args.model_inputs),
        eval_metadata=read_jsonl(args.eval_metadata),
        raw=read_jsonl(args.raw),
        parsed=read_jsonl(args.parsed),
        scored=read_jsonl(args.scored),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
        file.write("\n")
    print(
        "Validated Study 2 run: "
        f"valid={result['valid']}, records={result['record_counts']['model_inputs']}, "
        f"runtime_failures={result['runtime_failures']}, "
        f"invalid_model_responses={result['invalid_model_responses']}"
    )
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
