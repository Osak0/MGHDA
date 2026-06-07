"""Shared helpers for model-ready item construction."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


SENSITIVE_SOURCE_FIELDS = {"phrase"}


def require_pyarrow() -> tuple[Any, Any]:
    """Import pyarrow lazily so pure construction logic stays testable."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Parquet IO requires pyarrow. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return pa, pq


def read_parquet_rows(path: Path, columns: list[str] | None = None) -> list[dict[str, Any]]:
    """Read Parquet rows as dictionaries."""

    _, pq = require_pyarrow()
    table = pq.read_table(path, columns=columns)
    return table.to_pylist()


def write_parquet_rows(
    rows: list[dict[str, Any]], path: Path, columns: list[str]
) -> None:
    """Write rows to Parquet with a stable column order."""

    pa, pq = require_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [{column: row.get(column) for column in columns} for row in rows]
    pq.write_table(pa.Table.from_pylist(normalized), path)


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write model-ready items as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows for tests and local inspection."""

    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def update_summary(path: Path, section: str, payload: dict[str, Any]) -> None:
    """Update one section in an aggregate summary JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
    summary[section] = payload
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")


def stable_item_id(prefix: str, components: dict[str, Any]) -> str:
    """Create a deterministic item ID without using report text."""

    payload = json.dumps(components, ensure_ascii=True, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def clean_source_assertion(row: dict[str, Any]) -> dict[str, Any]:
    """Keep structured traceability fields while excluding raw report text."""

    keys = [
        "assertion_id",
        "raw_label",
        "category",
        "polarity",
        "label_name",
        "bbox_name",
        "phrase_id",
        "phrase_index",
        "source_quality",
    ]
    return {
        key: row.get(key)
        for key in keys
        if key not in SENSITIVE_SOURCE_FIELDS and key in row
    }


def image_index_by_key(
    rows: list[dict[str, Any]],
) -> dict[tuple[str | None, str | None, str | None], dict[str, Any]]:
    """Index image records by patient, study, and image identifiers."""

    return {
        (row.get("patient_id"), row.get("study_id"), row.get("image_id")): row
        for row in rows
    }


def get_image_record(
    index: dict[tuple[str | None, str | None, str | None], dict[str, Any]],
    *,
    patient_id: str | None,
    study_id: str | None,
    image_id: str | None,
) -> dict[str, Any]:
    """Return an image_index row or a null-path fallback with identifiers."""

    key = (patient_id, study_id, image_id)
    return index.get(
        key,
        {
            "patient_id": patient_id,
            "study_id": study_id,
            "image_id": image_id,
            "dicom_id": image_id,
            "image_path": None,
        },
    )


def first_non_null(*values: Any) -> Any:
    """Return the first value that is not None."""

    for value in values:
        if value is not None:
            return value
    return None


def safe_tuple_sort_key(key: tuple[Any, ...]) -> tuple[str, ...]:
    """Sort tuple keys deterministically even when some values are None."""

    return tuple("" if value is None else str(value) for value in key)


def balance_binary_label_items(
    items: list[dict[str, Any]],
    *,
    positive_label: str,
    negative_label: str,
    negative_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically downsample two-label items to an approximate target fraction."""

    if not 0 < negative_fraction < 1:
        raise ValueError("negative_fraction must be between 0 and 1")

    positive_items = [item for item in items if item.get("answer_label") == positive_label]
    negative_items = [item for item in items if item.get("answer_label") == negative_label]
    passthrough_items = [
        item
        for item in items
        if item.get("answer_label") not in {positive_label, negative_label}
    ]

    positive_before = len(positive_items)
    negative_before = len(negative_items)
    if positive_before == 0 or negative_before == 0:
        balanced = _stable_sample_items(
            positive_items, positive_before, seed=seed, label=positive_label
        ) + _stable_sample_items(
            negative_items, negative_before, seed=seed, label=negative_label
        )
        balanced.extend(passthrough_items)
        return _restore_original_order(items, balanced), {
            "balance_applied": False,
            "balance_reason": "single_label_or_empty",
            "positive_label": positive_label,
            "negative_label": negative_label,
            "target_negative_fraction": negative_fraction,
            "positive_before": positive_before,
            "negative_before": negative_before,
            "positive_after": positive_before,
            "negative_after": negative_before,
            "excluded_by_ratio": 0,
            "sampling_seed": seed,
        }

    target_fraction = Fraction(str(negative_fraction)).limit_denominator(1000)
    target_negative_per_positive = target_fraction / (1 - target_fraction)
    keep_positive = min(
        positive_before,
        (negative_before * target_negative_per_positive.denominator)
        // target_negative_per_positive.numerator,
    )
    keep_negative = min(
        negative_before,
        (positive_before * target_negative_per_positive.numerator)
        // target_negative_per_positive.denominator,
    )
    keep_positive = max(1, keep_positive)
    keep_negative = max(1, keep_negative)

    kept_positive = _stable_sample_items(
        positive_items, keep_positive, seed=seed, label=positive_label
    )
    kept_negative = _stable_sample_items(
        negative_items, keep_negative, seed=seed, label=negative_label
    )
    balanced = _restore_original_order(items, kept_positive + kept_negative + passthrough_items)
    positive_after = len(kept_positive)
    negative_after = len(kept_negative)
    return balanced, {
        "balance_applied": True,
        "balance_reason": "target_fraction_downsample",
        "positive_label": positive_label,
        "negative_label": negative_label,
        "target_negative_fraction": negative_fraction,
        "positive_before": positive_before,
        "negative_before": negative_before,
        "positive_after": positive_after,
        "negative_after": negative_after,
        "excluded_by_ratio": positive_before
        + negative_before
        - positive_after
        - negative_after,
        "sampling_seed": seed,
    }


def _stable_sample_items(
    items: list[dict[str, Any]], count: int, *, seed: int, label: str
) -> list[dict[str, Any]]:
    if count >= len(items):
        return list(items)
    ranked = sorted(
        items,
        key=lambda item: stable_item_id(
            "sample",
            {
                "seed": seed,
                "label": label,
                "item_id": item.get("item_id"),
            },
        ),
    )
    return ranked[:count]


def _restore_original_order(
    original_items: list[dict[str, Any]], kept_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    kept_ids = {item.get("item_id") for item in kept_items}
    return [item for item in original_items if item.get("item_id") in kept_ids]
