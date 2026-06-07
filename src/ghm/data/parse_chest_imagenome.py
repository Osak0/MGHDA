"""Parse Chest ImaGenome scene graphs into normalized evidence tables.

The parser keeps this step limited to source normalization. It does not infer
Unknown/Conflict labels or build model-ready G1/G2/G3 items.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


OBJECT_COLUMNS = [
    "patient_id",
    "study_id",
    "image_id",
    "object_id",
    "bbox_name",
    "name",
    "synsets",
    "x1",
    "y1",
    "x2",
    "y2",
    "original_x1",
    "original_y1",
    "original_x2",
    "original_y2",
    "source_quality",
]

ASSERTION_COLUMNS = [
    "assertion_id",
    "patient_id",
    "study_id",
    "image_id",
    "object_id",
    "bbox_name",
    "anatomy_name",
    "anatomy_bound",
    "raw_label",
    "category",
    "polarity",
    "label_name",
    "attributes_id",
    "phrase_index",
    "phrase_id",
    "section",
    "phrase",
    "severity_cues",
    "temporal_cues",
    "texture_cues",
    "comparison_cues",
    "source_quality",
]

REQUIRED_TOP_LEVEL_KEYS = ("image_id", "objects", "attributes")
PHRASE_ALIGNED_KEYS = (
    "attributes",
    "attributes_ids",
    "phrases",
    "phrase_IDs",
    "sections",
    "comparison_cues",
    "temporal_cues",
    "severity_cues",
    "texture_cues",
)


@dataclass
class ParseResult:
    """Parsed rows and aggregate warnings for one parser run."""

    object_rows: list[dict[str, Any]] = field(default_factory=list)
    assertion_rows: list[dict[str, Any]] = field(default_factory=list)
    warnings: Counter[str] = field(default_factory=Counter)
    files_seen: int = 0


def discover_scene_graphs(input_path: Path) -> list[Path]:
    """Return scene graph JSON files from a file or directory path."""

    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return sorted(input_path.rglob("*_SceneGraph.json"))


def parse_label(raw_label: Any) -> tuple[str | None, str | None, str | None]:
    """Parse a Chest ImaGenome label of the form category|polarity|name."""

    if not isinstance(raw_label, str):
        return None, None, None
    parts = raw_label.split("|", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        return None, None, None
    return parts[0], parts[1], parts[2]


def parse_scene_graph_file(path: Path) -> ParseResult:
    """Parse one Chest ImaGenome scene graph JSON file."""

    with path.open("r", encoding="utf-8") as file:
        graph = json.load(file)

    result = ParseResult(files_seen=1)
    missing_keys = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in graph]
    for key in missing_keys:
        result.warnings[f"missing_top_level_key:{key}"] += 1
    if missing_keys:
        return result

    image_id = _string_or_none(graph.get("image_id"))
    patient_id = _string_or_none(graph.get("patient_id"))
    study_id = _string_or_none(graph.get("study_id"))

    for obj in _list_or_empty(graph.get("objects")):
        if not isinstance(obj, dict):
            result.warnings["invalid_object_entry"] += 1
            continue
        result.object_rows.append(
            {
                "patient_id": patient_id,
                "study_id": study_id,
                "image_id": image_id,
                "object_id": _string_or_none(obj.get("object_id")),
                "bbox_name": _string_or_none(obj.get("bbox_name")),
                "name": _string_or_none(obj.get("name")),
                "synsets": _list_or_empty(obj.get("synsets")),
                "x1": _number_or_none(obj.get("x1")),
                "y1": _number_or_none(obj.get("y1")),
                "x2": _number_or_none(obj.get("x2")),
                "y2": _number_or_none(obj.get("y2")),
                "original_x1": _number_or_none(obj.get("original_x1")),
                "original_y1": _number_or_none(obj.get("original_y1")),
                "original_x2": _number_or_none(obj.get("original_x2")),
                "original_y2": _number_or_none(obj.get("original_y2")),
                "source_quality": "silver",
            }
        )

    for attribute_index, attr_dict in enumerate(_list_or_empty(graph.get("attributes"))):
        if not isinstance(attr_dict, dict):
            result.warnings["invalid_attribute_entry"] += 1
            continue
        _extend_attribute_assertions(
            result=result,
            attr_dict=attr_dict,
            attribute_index=attribute_index,
            patient_id=patient_id,
            study_id=study_id,
            image_id=image_id,
        )

    return result


def parse_scene_graphs(paths: list[Path]) -> ParseResult:
    """Parse many Chest ImaGenome scene graph files."""

    combined = ParseResult()
    for path in paths:
        parsed = parse_scene_graph_file(path)
        combined.object_rows.extend(parsed.object_rows)
        combined.assertion_rows.extend(parsed.assertion_rows)
        combined.warnings.update(parsed.warnings)
        combined.files_seen += parsed.files_seen
    return combined


def write_parquet_outputs(result: ParseResult, output_dir: Path) -> None:
    """Write normalized tables as Parquet files."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Parquet output requires pyarrow. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    object_rows = _ensure_columns(result.object_rows, OBJECT_COLUMNS)
    assertion_rows = _ensure_columns(result.assertion_rows, ASSERTION_COLUMNS)

    pq.write_table(pa.Table.from_pylist(object_rows), output_dir / "ci_objects.parquet")
    pq.write_table(
        pa.Table.from_pylist(assertion_rows),
        output_dir / "ci_attribute_assertions.parquet",
    )


def write_summary(result: ParseResult, output_dir: Path) -> None:
    """Write a de-identified aggregate parser summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "files_seen": result.files_seen,
        "ci_objects_rows": len(result.object_rows),
        "ci_attribute_assertions_rows": len(result.assertion_rows),
        "warnings": dict(sorted(result.warnings.items())),
        "output_tables": [
            "ci_objects.parquet",
            "ci_attribute_assertions.parquet",
        ],
    }
    with (output_dir / "parse_chest_imagenome_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="Parse Chest ImaGenome scene graphs into Parquet tables."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/fixtures"),
        help="Scene graph JSON file or directory containing *_SceneGraph.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim"),
        help="Directory for Parquet tables and aggregate parser summary.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of sorted scene graph files to parse.",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer when provided.")

    paths = discover_scene_graphs(args.input)
    if args.limit is not None:
        paths = paths[: args.limit]
    result = parse_scene_graphs(paths)
    try:
        write_parquet_outputs(result, args.output)
    except RuntimeError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")
    write_summary(result, args.output)

    print(
        "Parsed Chest ImaGenome scene graphs: "
        f"files={result.files_seen}, "
        f"ci_objects={len(result.object_rows)}, "
        f"ci_attribute_assertions={len(result.assertion_rows)}, "
        f"warning_types={len(result.warnings)}"
    )
    return 0


def _extend_attribute_assertions(
    *,
    result: ParseResult,
    attr_dict: dict[str, Any],
    attribute_index: int,
    patient_id: str | None,
    study_id: str | None,
    image_id: str | None,
) -> None:
    bbox_name = _valid_bbox_name(attr_dict.get("bbox_name"))
    anatomy_name = _string_or_none(attr_dict.get("name"))
    object_id = _string_or_none(attr_dict.get("object_id"))
    anatomy_bound = bool(bbox_name and attr_dict.get(bbox_name) is True)

    phrase_count = max(
        [len(_list_or_empty(attr_dict.get(key))) for key in PHRASE_ALIGNED_KEYS]
        or [0]
    )
    for key in PHRASE_ALIGNED_KEYS:
        values = _list_or_empty(attr_dict.get(key))
        if values and len(values) != phrase_count:
            result.warnings[f"misaligned_phrase_key:{key}"] += 1

    for phrase_index in range(phrase_count):
        labels = _list_or_empty(_list_get(attr_dict.get("attributes"), phrase_index))
        label_ids = _list_or_empty(_list_get(attr_dict.get("attributes_ids"), phrase_index))
        for label_index, raw_label in enumerate(labels):
            category, polarity, label_name = parse_label(raw_label)
            if category is None:
                result.warnings["malformed_label"] += 1
            result.assertion_rows.append(
                {
                    "assertion_id": _assertion_id(
                        image_id=image_id,
                        attribute_index=attribute_index,
                        phrase_index=phrase_index,
                        label_index=label_index,
                    ),
                    "patient_id": patient_id,
                    "study_id": study_id,
                    "image_id": image_id,
                    "object_id": object_id,
                    "bbox_name": bbox_name if anatomy_bound else None,
                    "anatomy_name": anatomy_name if anatomy_bound else None,
                    "anatomy_bound": anatomy_bound,
                    "raw_label": raw_label if isinstance(raw_label, str) else None,
                    "category": category,
                    "polarity": polarity,
                    "label_name": label_name,
                    "attributes_id": _string_or_none(_list_get(label_ids, label_index)),
                    "phrase_index": phrase_index,
                    "phrase_id": _string_or_none(
                        _list_get(attr_dict.get("phrase_IDs"), phrase_index)
                    ),
                    "section": _string_or_none(
                        _list_get(attr_dict.get("sections"), phrase_index)
                    ),
                    "phrase": _string_or_none(
                        _list_get(attr_dict.get("phrases"), phrase_index)
                    ),
                    "severity_cues": _list_or_empty(
                        _list_get(attr_dict.get("severity_cues"), phrase_index)
                    ),
                    "temporal_cues": _list_or_empty(
                        _list_get(attr_dict.get("temporal_cues"), phrase_index)
                    ),
                    "texture_cues": _list_or_empty(
                        _list_get(attr_dict.get("texture_cues"), phrase_index)
                    ),
                    "comparison_cues": _list_or_empty(
                        _list_get(attr_dict.get("comparison_cues"), phrase_index)
                    ),
                    "source_quality": "silver",
                }
            )


def _assertion_id(
    *,
    image_id: str | None,
    attribute_index: int,
    phrase_index: int,
    label_index: int,
) -> str:
    safe_image_id = image_id or "unknown_image"
    return f"ci_attr::{safe_image_id}::{attribute_index:04d}::{phrase_index:03d}::{label_index:03d}"


def _ensure_columns(
    rows: list[dict[str, Any]], columns: list[str]
) -> list[dict[str, Any]]:
    return [{column: row.get(column) for column in columns} for row in rows]


def _list_get(value: Any, index: int) -> Any:
    if isinstance(value, list) and 0 <= index < len(value):
        return value[index]
    return None


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    return str(value)


def _valid_bbox_name(value: Any) -> str | None:
    if value is False or value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
