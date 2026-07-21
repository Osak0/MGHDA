"""Build the fixed 40-record Study 3 smoke subset."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from ghm.granularity.common import read_jsonl, write_jsonl


def select_smoke_rows(
    model_inputs: list[dict[str, Any]],
    eval_metadata: list[dict[str, Any]],
    *,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Select one item per granularity/framing/relation/variant-K cell."""

    inputs = _index(model_inputs, "model inputs")
    metadata = _index(eval_metadata, "eval metadata")
    if set(inputs) != set(metadata):
        raise ValueError("smoke model-input and metadata IDs differ")
    cells: dict[tuple[str, str, str, str], list[str]] = {}
    for item_id, row in metadata.items():
        variant_key = (
            "natural"
            if row.get("variant") == "natural"
            else f"controlled_k{row.get('controlled_k')}"
        )
        key = (
            str(row.get("granularity")),
            str(row.get("prompt_framing")),
            str(row.get("query_relation")),
            variant_key,
        )
        cells.setdefault(key, []).append(item_id)
    expected = {
        (granularity, framing, relation, variant)
        for granularity in ("G1_finding_existence", "G2_anatomical_localization")
        for framing in ("state", "evidence")
        for relation in ("present", "absent")
        for variant in (
            "natural",
            "controlled_k2",
            "controlled_k3",
            "controlled_k4",
            "controlled_k5",
        )
    }
    missing = expected - set(cells)
    if missing:
        raise ValueError(f"Study 3 smoke coverage has missing cells: count={len(missing)}")
    selected_ids = [
        min(
            cells[key],
            key=lambda item_id: hashlib.sha256(
                f"{seed}:{item_id}".encode("utf-8")
            ).hexdigest(),
        )
        for key in sorted(expected)
    ]
    return (
        [inputs[item_id] for item_id in selected_ids],
        [metadata[item_id] for item_id in selected_ids],
        {"records": len(selected_ids), "coverage_cells": len(expected), "seed": seed},
    )


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = row.get("item_id")
        if item_id is None or str(item_id) in result:
            raise ValueError(f"{label} contain missing or duplicate item_id")
        result[str(item_id)] = row
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-metadata", type=Path, nargs="+", required=True)
    parser.add_argument("--model-inputs-output", type=Path, required=True)
    parser.add_argument("--eval-metadata-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    selected_inputs, selected_metadata, summary = select_smoke_rows(
        [row for path in args.model_inputs for row in read_jsonl(path)],
        [row for path in args.eval_metadata for row in read_jsonl(path)],
        seed=args.seed,
    )
    write_jsonl(selected_inputs, args.model_inputs_output)
    write_jsonl(selected_metadata, args.eval_metadata_output)
    print(
        "Built Study 3 fixed smoke subset: "
        f"records={summary['records']}, coverage_cells={summary['coverage_cells']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
