"""CLI for deterministic Study 3 G1/G2 item construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ghm.granularity.common import read_parquet_rows, write_jsonl
from ghm.study3.candidates import prepare_linked_anchors
from ghm.study3.constants import (
    DEFAULT_CONTROLLED_ANCHORS,
    DEFAULT_CONTROLLED_K,
    DEFAULT_NATURAL_ANCHORS,
    DEFAULT_SEED,
    EXPERIMENT_ID,
    require_study3_output_path,
)
from ghm.study3.construction import (
    ATTRIBUTE_COLUMNS,
    OBJECT_COLUMNS,
    build_multiselect_items,
    sample_study3_anchors,
)
from ghm.study3.linking import DEFAULT_BASE_URL, load_mimic_metadata


def build_for_granularity(
    anchors: list[dict[str, Any]],
    *,
    natural_count: int,
    controlled_count: int,
    controlled_k: tuple[int, ...],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample anchors and construct all paired Study 3 items."""

    selected, controlled_ids, sample_summary = sample_study3_anchors(
        anchors,
        natural_count=natural_count,
        controlled_count=controlled_count,
        seed=seed,
    )
    items, item_summary = build_multiselect_items(
        selected,
        controlled_ids,
        controlled_k=controlled_k,
        seed=seed,
    )
    return items, {
        "sampling": sample_summary,
        "items": item_summary,
    }


def parse_k_values(text: str) -> tuple[int, ...]:
    """Parse a comma-separated, sorted sequence such as ``2,3,4,5``."""

    try:
        values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("controlled K values must be integers") from exc
    if not values or min(values) < 2 or tuple(sorted(set(values))) != values:
        raise argparse.ArgumentTypeError(
            "controlled K values must be unique, sorted, and at least 2"
        )
    return values


def main(argv: list[str] | None = None) -> int:
    """Build linked and sampled Study 3 items."""

    parser = argparse.ArgumentParser(description="Build Study 3 multi-select items.")
    parser.add_argument(
        "--attributes",
        type=Path,
        default=Path("interim/ci_attribute_assertions.parquet"),
    )
    parser.add_argument(
        "--objects",
        type=Path,
        default=Path("interim/ci_objects.parquet"),
    )
    parser.add_argument("--mimic-metadata", type=Path, required=True)
    parser.add_argument("--mimic-split", type=Path, required=True)
    parser.add_argument("--files-root", type=Path, required=True)
    parser.add_argument("--image-path-root", type=Path, default=Path("files"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--g1-output",
        type=Path,
        default=Path("processed/study3/v2/items/study3_g1_multiselect_items.jsonl"),
    )
    parser.add_argument(
        "--g2-output",
        type=Path,
        default=Path("processed/study3/v2/items/study3_g2_multiselect_items.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/study3/v2/audits/study3_item_summary.json"),
    )
    parser.add_argument(
        "--natural-anchors",
        type=int,
        default=DEFAULT_NATURAL_ANCHORS,
    )
    parser.add_argument(
        "--controlled-anchors",
        type=int,
        default=DEFAULT_CONTROLLED_ANCHORS,
    )
    parser.add_argument(
        "--controlled-k",
        type=parse_k_values,
        default=DEFAULT_CONTROLLED_K,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    for path in (args.g1_output, args.g2_output, args.summary):
        require_study3_output_path(path)
    if args.natural_anchors < 1:
        parser.error("--natural-anchors must be positive")
    if args.controlled_anchors < 0:
        parser.error("--controlled-anchors must be non-negative")

    try:
        assertion_rows = read_parquet_rows(args.attributes, columns=ATTRIBUTE_COLUMNS)
        object_rows = read_parquet_rows(args.objects, columns=OBJECT_COLUMNS)
    except RuntimeError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")
    metadata = load_mimic_metadata(args.mimic_metadata, args.mimic_split)
    g1_anchors, g2_anchors, candidate_summary = prepare_linked_anchors(
        assertion_rows,
        object_rows,
        metadata,
        files_root=args.files_root,
        image_path_root=args.image_path_root,
        base_url=args.base_url,
    )
    g1_items, g1_summary = build_for_granularity(
        g1_anchors,
        natural_count=args.natural_anchors,
        controlled_count=args.controlled_anchors,
        controlled_k=args.controlled_k,
        seed=args.seed,
    )
    g2_items, g2_summary = build_for_granularity(
        g2_anchors,
        natural_count=args.natural_anchors,
        controlled_count=args.controlled_anchors,
        controlled_k=args.controlled_k,
        seed=args.seed,
    )
    write_jsonl(g1_items, args.g1_output)
    write_jsonl(g2_items, args.g2_output)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "candidate_audit": candidate_summary,
        "g1": g1_summary,
        "g2": g2_summary,
        "total_items": len(g1_items) + len(g2_items),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    print(
        "Built Study 3 items: "
        f"g1={len(g1_items)}, g2={len(g2_items)}, "
        f"total={summary['total_items']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
