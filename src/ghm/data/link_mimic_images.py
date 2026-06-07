"""Build an image_index table for future MIMIC-CXR-JPG linking.

This first-stage linker does not read image files or download data. It only
preserves identifiers already present in Chest ImaGenome intermediate tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ghm.granularity.common import read_parquet_rows, update_summary, write_parquet_rows


IMAGE_INDEX_COLUMNS = [
    "patient_id",
    "study_id",
    "image_id",
    "dicom_id",
    "image_path",
    "view_position",
    "has_chest_imagenome",
    "has_radgraph",
]


def build_image_index_rows(
    object_rows: list[dict[str, Any]],
    assertion_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create one image_index row per unique patient/study/image triple."""

    records: dict[tuple[str | None, str | None, str | None], dict[str, Any]] = {}
    for row in [*object_rows, *assertion_rows]:
        patient_id = _string_or_none(row.get("patient_id"))
        study_id = _string_or_none(row.get("study_id"))
        image_id = _string_or_none(row.get("image_id"))
        if image_id is None:
            continue
        key = (patient_id, study_id, image_id)
        records[key] = {
            "patient_id": patient_id,
            "study_id": study_id,
            "image_id": image_id,
            "dicom_id": image_id,
            "image_path": None,
            "view_position": None,
            "has_chest_imagenome": True,
            "has_radgraph": False,
        }
    return [records[key] for key in sorted(records, key=_sort_key)]


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a de-identified aggregate summary for image_index."""

    return {
        "image_index_rows": len(rows),
        "non_null_image_path_rows": sum(1 for row in rows if row.get("image_path")),
        "has_chest_imagenome_rows": sum(
            1 for row in rows if row.get("has_chest_imagenome")
        ),
        "has_radgraph_rows": sum(1 for row in rows if row.get("has_radgraph")),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="Build image_index.parquet from Chest ImaGenome intermediate tables."
    )
    parser.add_argument(
        "--objects",
        type=Path,
        default=Path("data/interim/ci_objects.parquet"),
        help="Path to ci_objects.parquet.",
    )
    parser.add_argument(
        "--attributes",
        type=Path,
        default=Path("data/interim/ci_attribute_assertions.parquet"),
        help="Path to ci_attribute_assertions.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/image_index.parquet"),
        help="Output image_index parquet path.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/audits/g1_g2_candidate_summary.json"),
        help="Aggregate summary JSON to update.",
    )
    args = parser.parse_args(argv)

    try:
        object_rows = read_parquet_rows(
            args.objects, columns=["patient_id", "study_id", "image_id"]
        )
        assertion_rows = read_parquet_rows(
            args.attributes, columns=["patient_id", "study_id", "image_id"]
        )
        image_index_rows = build_image_index_rows(object_rows, assertion_rows)
        write_parquet_rows(image_index_rows, args.output, IMAGE_INDEX_COLUMNS)
    except RuntimeError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    summary = build_summary(image_index_rows)
    update_summary(args.summary, "image_index", summary)
    print(
        "Built image_index: "
        f"rows={summary['image_index_rows']}, "
        f"non_null_image_path_rows={summary['non_null_image_path_rows']}"
    )
    return 0


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _sort_key(key: tuple[str | None, str | None, str | None]) -> tuple[str, str, str]:
    return tuple("" if value is None else value for value in key)


if __name__ == "__main__":
    raise SystemExit(main())
