"""Load, link, and audit Study 3 candidate anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ghm.study3.linking import (
    DEFAULT_BASE_URL,
    build_link_rows,
    collect_needed_images,
    load_mimic_metadata,
)
from ghm.granularity.common import read_parquet_rows
from ghm.study3.constants import EXPERIMENT_ID, require_study3_output_path
from ghm.study3.construction import (
    ATTRIBUTE_COLUMNS,
    OBJECT_COLUMNS,
    build_g1_anchors,
    build_g2_anchors,
    link_existing_anchors,
    summarize_anchors,
)


def prepare_linked_anchors(
    assertion_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    mimic_metadata: dict[tuple[str, str, str], dict[str, str | None]],
    *,
    files_root: Path,
    image_path_root: Path = Path("files"),
    base_url: str = DEFAULT_BASE_URL,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build G1/G2 candidates and retain only locally available images."""

    g1_anchors, g1_candidate_summary = build_g1_anchors(assertion_rows)
    g2_anchors, g2_candidate_summary = build_g2_anchors(
        assertion_rows,
        object_rows,
    )
    needed = collect_needed_images([*g1_anchors, *g2_anchors])
    link_rows, link_summary = build_link_rows(
        needed,
        mimic_metadata,
        files_root=files_root,
        base_url=base_url,
        image_path_root=image_path_root,
    )
    linked_g1, g1_link_summary = link_existing_anchors(g1_anchors, link_rows)
    linked_g2, g2_link_summary = link_existing_anchors(g2_anchors, link_rows)
    return linked_g1, linked_g2, {
        "experiment_id": EXPERIMENT_ID,
        "g1_candidates": g1_candidate_summary,
        "g2_candidates": g2_candidate_summary,
        "image_linking": link_summary,
        "g1_local": {
            **g1_link_summary,
            **summarize_anchors(linked_g1),
        },
        "g2_local": {
            **g2_link_summary,
            **summarize_anchors(linked_g2),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Write an aggregate-only Study 3 candidate audit."""

    parser = argparse.ArgumentParser(
        description="Audit locally available Study 3 multi-select candidates."
    )
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
        "--output",
        type=Path,
        default=Path("outputs/study3/audits/study3_candidate_audit.json"),
    )
    args = parser.parse_args(argv)
    require_study3_output_path(args.output)

    try:
        assertion_rows = read_parquet_rows(args.attributes, columns=ATTRIBUTE_COLUMNS)
        object_rows = read_parquet_rows(args.objects, columns=OBJECT_COLUMNS)
    except RuntimeError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")
    metadata = load_mimic_metadata(args.mimic_metadata, args.mimic_split)
    _, _, summary = prepare_linked_anchors(
        assertion_rows,
        object_rows,
        metadata,
        files_root=args.files_root,
        image_path_root=args.image_path_root,
        base_url=args.base_url,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    print(
        "Audited Study 3 candidates: "
        f"g1_local={summary['g1_local']['anchor_count']}, "
        f"g2_local={summary['g2_local']['anchor_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
