"""Aggregate Chest ImaGenome bbox-finding coverage for G2 construction."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ghm.granularity.common import read_parquet_rows


OBJECT_COLUMNS = [
    "image_id",
    "bbox_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "original_x1",
    "original_y1",
    "original_x2",
    "original_y2",
]

ASSERTION_COLUMNS = [
    "image_id",
    "bbox_name",
    "anatomy_bound",
    "category",
    "polarity",
    "label_name",
]

BBOX_FINDING_SUMMARY_COLUMNS = [
    "bbox_name",
    "label_name",
    "n_assertions",
    "n_images",
    "n_yes",
    "n_no",
    "n_conflict_image_bbox_finding",
    "bbox_object_rows",
    "bbox_images",
    "original_bbox_complete_rows",
    "original_bbox_missing_rows",
    "original_bbox_complete_rate",
    "resized_bbox_complete_rows",
    "resized_bbox_missing_rows",
    "resized_bbox_complete_rate",
    "g2_candidate_status",
]

BBOX_QUALITY_SUMMARY_COLUMNS = [
    "bbox_name",
    "bbox_images",
    "bbox_object_rows",
    "unique_findings",
    "n_anatomicalfinding_assertions",
    "n_yes",
    "n_no",
    "n_conflict_image_bbox_finding",
    "original_bbox_complete_rate",
    "resized_bbox_complete_rate",
    "coordinate_pattern_count",
    "recommended_g2_use",
]

ORIGINAL_COORD_COLUMNS = ["original_x1", "original_y1", "original_x2", "original_y2"]
RESIZED_COORD_COLUMNS = ["x1", "y1", "x2", "y2"]
COORD_COLUMNS = ORIGINAL_COORD_COLUMNS + RESIZED_COORD_COLUMNS

USE_FOR_G2 = "use_for_g2"
USE_EXPLICIT_ONLY = "use_explicit_only"
REVIEW = "review"
DROP_FOR_G2 = "drop_for_g2"

USABLE = "usable"
COORDINATE_INCOMPLETE = "coordinate_incomplete"
CONFLICT_OR_SPARSE = "conflict_or_sparse"


def build_bbox_finding_reference_tables(
    object_rows: list[dict[str, Any]],
    assertion_rows: list[dict[str, Any]],
    *,
    sparse_min_images: int = 2,
    coordinate_drop_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build aggregate bbox-finding and bbox-quality summaries.

    Outputs intentionally exclude patient, study, image, report, and path fields.
    """

    bbox_stats = _bbox_coordinate_stats(object_rows)
    assertion_groups = _group_anatomy_bound_findings(assertion_rows)
    conflict_keys = _conflict_keys(assertion_groups)
    finding_rows = _bbox_finding_rows(
        assertion_groups=assertion_groups,
        conflict_keys=conflict_keys,
        bbox_stats=bbox_stats,
        sparse_min_images=sparse_min_images,
    )
    quality_rows = _bbox_quality_rows(
        assertion_groups=assertion_groups,
        conflict_keys=conflict_keys,
        bbox_stats=bbox_stats,
        coordinate_drop_threshold=coordinate_drop_threshold,
    )
    summary = {
        "bbox_name_rows": len(quality_rows),
        "bbox_finding_rows": len(finding_rows),
        "bbox_names_with_objects": len(bbox_stats),
        "bbox_names_with_anatomicalfinding_evidence": len(
            {key[1] for key in assertion_groups}
        ),
        "total_bbox_object_rows": sum(row["bbox_object_rows"] for row in bbox_stats.values()),
        "total_anatomicalfinding_assertions": sum(
            len(rows) for rows in assertion_groups.values()
        ),
        "conflict_image_bbox_finding_groups": len(conflict_keys),
        "recommended_g2_use_counts": dict(
            Counter(row["recommended_g2_use"] for row in quality_rows)
        ),
        "g2_candidate_status_counts": dict(
            Counter(row["g2_candidate_status"] for row in finding_rows)
        ),
        "sparse_min_images": sparse_min_images,
        "coordinate_drop_threshold": coordinate_drop_threshold,
    }
    return finding_rows, quality_rows, summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for aggregate G2 bbox-finding audits."""

    parser = argparse.ArgumentParser(
        description="Build aggregate bbox-finding reference tables for G2."
    )
    parser.add_argument(
        "--objects",
        type=Path,
        default=Path("interim/ci_objects.parquet"),
        help="Chest ImaGenome object table parquet.",
    )
    parser.add_argument(
        "--assertions",
        type=Path,
        default=Path("interim/ci_attribute_assertions.parquet"),
        help="Chest ImaGenome attribute assertion table parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audits"),
        help="Directory for aggregate CSV/JSON audit outputs.",
    )
    parser.add_argument("--sparse-min-images", type=int, default=2)
    parser.add_argument("--coordinate-drop-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    if args.sparse_min_images < 1:
        parser.error("--sparse-min-images must be at least 1.")
    if not 0 <= args.coordinate_drop_threshold <= 1:
        parser.error("--coordinate-drop-threshold must be between 0 and 1.")

    try:
        object_rows = read_parquet_rows(args.objects, columns=OBJECT_COLUMNS)
        assertion_rows = read_parquet_rows(args.assertions, columns=ASSERTION_COLUMNS)
    except RuntimeError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    finding_rows, quality_rows, summary = build_bbox_finding_reference_tables(
        object_rows,
        assertion_rows,
        sparse_min_images=args.sparse_min_images,
        coordinate_drop_threshold=args.coordinate_drop_threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        finding_rows,
        args.output_dir / "bbox_finding_vocab_summary.csv",
        BBOX_FINDING_SUMMARY_COLUMNS,
    )
    _write_csv(
        quality_rows,
        args.output_dir / "bbox_name_quality_summary.csv",
        BBOX_QUALITY_SUMMARY_COLUMNS,
    )
    summary_path = args.output_dir / "bbox_finding_reference_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")

    print(
        "Built G2 bbox-finding reference tables: "
        f"bbox_names={summary['bbox_name_rows']}, "
        f"bbox_findings={summary['bbox_finding_rows']}, "
        f"conflicts={summary['conflict_image_bbox_finding_groups']}"
    )
    return 0


def _bbox_coordinate_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bbox_name = row.get("bbox_name")
        if not _present(bbox_name):
            continue
        grouped[str(bbox_name)].append(row)

    stats: dict[str, dict[str, Any]] = {}
    for bbox_name, bbox_rows in grouped.items():
        original_complete = sum(_coords_complete(row, ORIGINAL_COORD_COLUMNS) for row in bbox_rows)
        resized_complete = sum(_coords_complete(row, RESIZED_COORD_COLUMNS) for row in bbox_rows)
        object_count = len(bbox_rows)
        stats[bbox_name] = {
            "bbox_name": bbox_name,
            "bbox_object_rows": object_count,
            "bbox_images": _unique_count(row.get("image_id") for row in bbox_rows),
            "original_bbox_complete_rows": original_complete,
            "original_bbox_missing_rows": object_count - original_complete,
            "original_bbox_complete_rate": _rate(original_complete, object_count),
            "resized_bbox_complete_rows": resized_complete,
            "resized_bbox_missing_rows": object_count - resized_complete,
            "resized_bbox_complete_rate": _rate(resized_complete, object_count),
            "coordinate_pattern_count": len(
                {_coordinate_presence_pattern(row) for row in bbox_rows}
            ),
        }
    return stats


def _group_anatomy_bound_findings(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("category") != "anatomicalfinding":
            continue
        if row.get("polarity") not in {"yes", "no"}:
            continue
        if row.get("anatomy_bound") is not True:
            continue
        if not _present(row.get("image_id")):
            continue
        if not _present(row.get("bbox_name")) or not _present(row.get("label_name")):
            continue
        key = (str(row["image_id"]), str(row["bbox_name"]), str(row["label_name"]))
        groups[key].append(row)
    return groups


def _conflict_keys(
    groups: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> set[tuple[str, str, str]]:
    return {
        key
        for key, rows in groups.items()
        if {"yes", "no"}.issubset({row.get("polarity") for row in rows})
    }


def _bbox_finding_rows(
    *,
    assertion_groups: dict[tuple[str, str, str], list[dict[str, Any]]],
    conflict_keys: set[tuple[str, str, str]],
    bbox_stats: dict[str, dict[str, Any]],
    sparse_min_images: int,
) -> list[dict[str, Any]]:
    by_bbox_finding: dict[tuple[str, str], list[tuple[tuple[str, str, str], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for key, rows in assertion_groups.items():
        _, bbox_name, label_name = key
        for row in rows:
            by_bbox_finding[(bbox_name, label_name)].append((key, row))

    output: list[dict[str, Any]] = []
    for (bbox_name, label_name), keyed_rows in sorted(by_bbox_finding.items()):
        keys = {key for key, _ in keyed_rows}
        rows = [row for _, row in keyed_rows]
        stat = _stats_or_empty(bbox_name, bbox_stats)
        n_conflicts = sum(1 for key in keys if key in conflict_keys)
        row = {
            "bbox_name": bbox_name,
            "label_name": label_name,
            "n_assertions": len(rows),
            "n_images": _unique_count(row.get("image_id") for row in rows),
            "n_yes": sum(1 for row in rows if row.get("polarity") == "yes"),
            "n_no": sum(1 for row in rows if row.get("polarity") == "no"),
            "n_conflict_image_bbox_finding": n_conflicts,
            "bbox_object_rows": stat["bbox_object_rows"],
            "bbox_images": stat["bbox_images"],
            "original_bbox_complete_rows": stat["original_bbox_complete_rows"],
            "original_bbox_missing_rows": stat["original_bbox_missing_rows"],
            "original_bbox_complete_rate": stat["original_bbox_complete_rate"],
            "resized_bbox_complete_rows": stat["resized_bbox_complete_rows"],
            "resized_bbox_missing_rows": stat["resized_bbox_missing_rows"],
            "resized_bbox_complete_rate": stat["resized_bbox_complete_rate"],
        }
        row["g2_candidate_status"] = _candidate_status(
            row,
            sparse_min_images=sparse_min_images,
        )
        output.append(row)
    return output


def _bbox_quality_rows(
    *,
    assertion_groups: dict[tuple[str, str, str], list[dict[str, Any]]],
    conflict_keys: set[tuple[str, str, str]],
    bbox_stats: dict[str, dict[str, Any]],
    coordinate_drop_threshold: float,
) -> list[dict[str, Any]]:
    assertion_rows_by_bbox: dict[str, list[dict[str, Any]]] = defaultdict(list)
    findings_by_bbox: dict[str, set[str]] = defaultdict(set)
    conflicts_by_bbox: Counter[str] = Counter()

    for key, rows in assertion_groups.items():
        _, bbox_name, label_name = key
        assertion_rows_by_bbox[bbox_name].extend(rows)
        findings_by_bbox[bbox_name].add(label_name)
        if key in conflict_keys:
            conflicts_by_bbox[bbox_name] += 1

    bbox_names = sorted(set(bbox_stats) | set(assertion_rows_by_bbox))
    output: list[dict[str, Any]] = []
    for bbox_name in bbox_names:
        rows = assertion_rows_by_bbox.get(bbox_name, [])
        stat = _stats_or_empty(bbox_name, bbox_stats)
        row = {
            "bbox_name": bbox_name,
            "bbox_images": stat["bbox_images"],
            "bbox_object_rows": stat["bbox_object_rows"],
            "unique_findings": len(findings_by_bbox.get(bbox_name, set())),
            "n_anatomicalfinding_assertions": len(rows),
            "n_yes": sum(1 for row in rows if row.get("polarity") == "yes"),
            "n_no": sum(1 for row in rows if row.get("polarity") == "no"),
            "n_conflict_image_bbox_finding": conflicts_by_bbox[bbox_name],
            "original_bbox_complete_rate": stat["original_bbox_complete_rate"],
            "resized_bbox_complete_rate": stat["resized_bbox_complete_rate"],
            "coordinate_pattern_count": stat["coordinate_pattern_count"],
        }
        row["recommended_g2_use"] = _recommended_g2_use(
            row,
            coordinate_drop_threshold=coordinate_drop_threshold,
        )
        output.append(row)
    return output


def _candidate_status(row: dict[str, Any], *, sparse_min_images: int) -> str:
    if row["n_conflict_image_bbox_finding"] > 0 or row["n_images"] < sparse_min_images:
        return CONFLICT_OR_SPARSE
    if row["original_bbox_complete_rate"] < 1.0:
        return COORDINATE_INCOMPLETE
    return USABLE


def _recommended_g2_use(
    row: dict[str, Any], *, coordinate_drop_threshold: float
) -> str:
    if (
        row["bbox_object_rows"] == 0
        or row["original_bbox_complete_rate"] < coordinate_drop_threshold
    ):
        return DROP_FOR_G2
    if row["original_bbox_complete_rate"] < 1.0 or row["n_conflict_image_bbox_finding"] > 0:
        return REVIEW
    if row["unique_findings"] < 2:
        return USE_EXPLICIT_ONLY
    return USE_FOR_G2


def _stats_or_empty(bbox_name: str, stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return stats.get(
        bbox_name,
        {
            "bbox_object_rows": 0,
            "bbox_images": 0,
            "original_bbox_complete_rows": 0,
            "original_bbox_missing_rows": 0,
            "original_bbox_complete_rate": 0.0,
            "resized_bbox_complete_rows": 0,
            "resized_bbox_missing_rows": 0,
            "resized_bbox_complete_rate": 0.0,
            "coordinate_pattern_count": 0,
        },
    )


def _write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _coords_complete(row: dict[str, Any], columns: list[str]) -> bool:
    return all(row.get(column) is not None for column in columns)


def _coordinate_presence_pattern(row: dict[str, Any]) -> tuple[bool, ...]:
    return tuple(row.get(column) is not None for column in COORD_COLUMNS)


def _present(value: Any) -> bool:
    return value is not None and str(value) != ""


def _unique_count(values: Any) -> int:
    return len({value for value in values if _present(value)})


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


if __name__ == "__main__":
    raise SystemExit(main())
