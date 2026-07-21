"""Construct Study 3 G1/G2 multi-select anchors and items.

All functions in this module operate on in-memory dictionaries so they can be
tested with synthetic fixtures. Real restricted tables are read only by the CLI
entrypoints on the user's trusted data machine.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from ghm.granularity.common import clean_source_assertion, stable_item_id
from ghm.study3.constants import (
    ABSENT,
    CONTROLLED,
    DEFAULT_CONTROLLED_K,
    EVIDENCE,
    EXPERIMENT_ID,
    NATURAL,
    PRESENT,
    PROMPT_FRAMINGS,
    QUESTION_TYPE,
    STATE,
)
from ghm.study3.linking import normalize_id


ATTRIBUTE_COLUMNS = [
    "assertion_id",
    "patient_id",
    "study_id",
    "image_id",
    "bbox_name",
    "anatomy_bound",
    "raw_label",
    "category",
    "polarity",
    "label_name",
    "phrase_id",
    "phrase_index",
    "source_quality",
]

OBJECT_COLUMNS = [
    "patient_id",
    "study_id",
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
    "source_quality",
]


def build_g1_anchors(
    assertion_rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate explicit image-level findings into eligible G1 anchors."""

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    ignored_rows = 0
    for row in assertion_rows:
        if not _eligible_assertion(row):
            ignored_rows += 1
            continue
        key = (
            row.get("patient_id"),
            row.get("study_id"),
            row.get("image_id"),
            str(row["label_name"]),
        )
        groups[key].append(row)

    options_by_anchor: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    conflicts = 0
    for key in sorted(groups, key=_tuple_sort_key):
        patient_id, study_id, image_id, label_name = key
        polarities = {str(row["polarity"]) for row in groups[key]}
        if polarities == {"yes", "no"}:
            conflicts += 1
            continue
        polarity = next(iter(polarities))
        anchor_key = (patient_id, study_id, image_id)
        options_by_anchor[anchor_key].append(
            _option_candidate(label_name, polarity, groups[key])
        )

    anchors: list[dict[str, Any]] = []
    excluded_too_few = 0
    for key in sorted(options_by_anchor, key=_tuple_sort_key):
        options = sorted(options_by_anchor[key], key=lambda row: row["label_name"])
        if len(options) < 2:
            excluded_too_few += 1
            continue
        patient_id, study_id, image_id = key
        anchors.append(
            _anchor(
                granularity="G1_finding_existence",
                prefix="study3_g1_anchor",
                patient_id=patient_id,
                study_id=study_id,
                image_id=image_id,
                target_anatomy=None,
                bbox=None,
                options=options,
            )
        )

    return anchors, {
        "granularity": "G1_finding_existence",
        "eligible_anchors": len(anchors),
        "ignored_assertion_rows": ignored_rows,
        "excluded_conflict_findings": conflicts,
        "excluded_anchors_fewer_than_two_options": excluded_too_few,
        **summarize_anchors(anchors),
    }


def build_g2_anchors(
    assertion_rows: Iterable[dict[str, Any]],
    object_rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate explicit anatomy-bound findings into eligible G2 anchors."""

    object_index = _valid_object_index(object_rows)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    ignored_rows = 0
    for row in assertion_rows:
        if not _eligible_assertion(row):
            ignored_rows += 1
            continue
        if row.get("anatomy_bound") is not True or not _text(row.get("bbox_name")):
            ignored_rows += 1
            continue
        key = (
            row.get("patient_id"),
            row.get("study_id"),
            row.get("image_id"),
            str(row["bbox_name"]),
            str(row["label_name"]),
        )
        groups[key].append(row)

    options_by_anchor: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    conflicts = 0
    missing_bbox_findings = 0
    for key in sorted(groups, key=_tuple_sort_key):
        patient_id, study_id, image_id, bbox_name, label_name = key
        anchor_key = (patient_id, study_id, image_id, bbox_name)
        if anchor_key not in object_index:
            missing_bbox_findings += 1
            continue
        polarities = {str(row["polarity"]) for row in groups[key]}
        if polarities == {"yes", "no"}:
            conflicts += 1
            continue
        polarity = next(iter(polarities))
        options_by_anchor[anchor_key].append(
            _option_candidate(label_name, polarity, groups[key])
        )

    anchors: list[dict[str, Any]] = []
    excluded_too_few = 0
    for key in sorted(options_by_anchor, key=_tuple_sort_key):
        options = sorted(options_by_anchor[key], key=lambda row: row["label_name"])
        if len(options) < 2:
            excluded_too_few += 1
            continue
        patient_id, study_id, image_id, bbox_name = key
        object_row = object_index[key]
        anchors.append(
            _anchor(
                granularity="G2_anatomical_localization",
                prefix="study3_g2_anchor",
                patient_id=patient_id,
                study_id=study_id,
                image_id=image_id,
                target_anatomy=bbox_name,
                bbox=_bbox_from_object(object_row),
                options=options,
            )
        )

    return anchors, {
        "granularity": "G2_anatomical_localization",
        "eligible_anchors": len(anchors),
        "ignored_assertion_rows": ignored_rows,
        "excluded_conflict_findings": conflicts,
        "excluded_findings_without_valid_bbox": missing_bbox_findings,
        "excluded_anchors_fewer_than_two_options": excluded_too_few,
        **summarize_anchors(anchors),
    }


def link_existing_anchors(
    anchors: Iterable[dict[str, Any]],
    link_rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Attach portable image paths and keep only images that exist locally."""

    anchor_rows = list(anchors)
    link_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for link in link_rows:
        patient_id = normalize_id(link.get("patient_id"))
        study_id = normalize_id(link.get("study_id"))
        dicom_id = normalize_id(link.get("dicom_id"))
        if patient_id and study_id and dicom_id:
            link_index[(patient_id, study_id, dicom_id)] = link

    linked: list[dict[str, Any]] = []
    excluded_missing_link = 0
    excluded_missing_file = 0
    for anchor in anchor_rows:
        patient_id = normalize_id(anchor.get("patient_id"))
        study_id = normalize_id(anchor.get("study_id"))
        dicom_id = normalize_id(anchor.get("dicom_id") or anchor.get("image_id"))
        link = (
            link_index.get((patient_id, study_id, dicom_id))
            if patient_id and study_id and dicom_id
            else None
        )
        if not link or link.get("link_status") != "matched" or not link.get("image_path"):
            excluded_missing_link += 1
            continue
        local_path = link.get("_local_image_path")
        if not local_path or not Path(str(local_path)).is_file():
            excluded_missing_file += 1
            continue
        updated = dict(anchor)
        updated["patient_id"] = patient_id
        updated["study_id"] = study_id
        updated["image_id"] = link.get("image_id") or dicom_id
        updated["dicom_id"] = dicom_id
        updated["image_path"] = str(link["image_path"]).replace("\\", "/")
        linked.append(updated)

    return linked, {
        "input_anchors": len(anchor_rows),
        "linked_existing_anchors": len(linked),
        "excluded_missing_link": excluded_missing_link,
        "excluded_missing_image_file": excluded_missing_file,
    }


def sample_study3_anchors(
    anchors: list[dict[str, Any]],
    *,
    natural_count: int,
    controlled_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    """Select natural anchors across K bins and a nested controlled subset."""

    if natural_count < 1:
        raise ValueError("natural_count must be positive")
    if controlled_count < 0:
        raise ValueError("controlled_count must be non-negative")

    bins = ("2", "3", "4", "5plus")
    by_bin: dict[str, list[dict[str, Any]]] = {name: [] for name in bins}
    for anchor in anchors:
        by_bin[_option_count_bin(int(anchor["natural_option_count"]))].append(anchor)

    base, remainder = divmod(natural_count, len(bins))
    quotas = {
        name: base + (1 if index < remainder else 0)
        for index, name in enumerate(bins)
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    realized_by_bin: Counter[str] = Counter()
    for name in bins:
        bucket = _composition_round_robin(by_bin[name], seed=seed, label=f"natural:{name}")
        for anchor in bucket[: quotas[name]]:
            selected.append(anchor)
            selected_ids.add(str(anchor["anchor_id"]))
            realized_by_bin[name] += 1

    if len(selected) < natural_count:
        remaining = [row for row in anchors if str(row["anchor_id"]) not in selected_ids]
        remaining = _stable_rank(remaining, seed=seed, label="natural:refill")
        for anchor in remaining[: natural_count - len(selected)]:
            selected.append(anchor)
            selected_ids.add(str(anchor["anchor_id"]))
            realized_by_bin[_option_count_bin(int(anchor["natural_option_count"]))] += 1

    controlled_candidates = [
        row for row in selected if int(row["natural_option_count"]) >= 5
    ]
    controlled_ranked = _composition_round_robin(
        controlled_candidates,
        seed=seed,
        label="controlled",
    )
    controlled_ids = {
        str(row["anchor_id"]) for row in controlled_ranked[:controlled_count]
    }
    selected = sorted(selected, key=lambda row: str(row["anchor_id"]))
    return selected, controlled_ids, {
        "requested_natural_anchors": natural_count,
        "selected_natural_anchors": len(selected),
        "requested_controlled_anchors": controlled_count,
        "selected_controlled_anchors": len(controlled_ids),
        "controlled_candidate_anchors": len(controlled_candidates),
        "natural_bin_quotas": quotas,
        "natural_bin_realized": dict(sorted(realized_by_bin.items())),
        "sampling_seed": seed,
        "natural_shortage": max(0, natural_count - len(selected)),
        "controlled_shortage": max(0, controlled_count - len(controlled_ids)),
    }


def build_multiselect_items(
    selected_anchors: Iterable[dict[str, Any]],
    controlled_anchor_ids: set[str],
    *,
    controlled_k: tuple[int, ...] = DEFAULT_CONTROLLED_K,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create four framing/relation records per natural or controlled option set."""

    if not controlled_k or min(controlled_k) < 2:
        raise ValueError("controlled_k must contain integers of at least 2")
    if tuple(sorted(set(controlled_k))) != controlled_k:
        raise ValueError("controlled_k must be unique and sorted")

    items: list[dict[str, Any]] = []
    anchor_count = 0
    controlled_anchor_count = 0
    for anchor in selected_anchors:
        anchor_count += 1
        natural_options = _ordered_options(
            anchor["options"],
            seed=seed,
            label=f"natural:{anchor['anchor_id']}",
        )
        items.extend(_paired_items(anchor, natural_options, variant=NATURAL, k=None))

        if str(anchor["anchor_id"]) not in controlled_anchor_ids:
            continue
        controlled_anchor_count += 1
        controlled_options = _ordered_options(
            anchor["options"],
            seed=seed + 1,
            label=f"controlled:{anchor['anchor_id']}",
        )
        for k in controlled_k:
            if len(controlled_options) < k:
                continue
            items.extend(
                _paired_items(
                    anchor,
                    controlled_options[:k],
                    variant=CONTROLLED,
                    k=k,
                )
            )

    item_ids = [str(item["item_id"]) for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Study 3 construction produced duplicate item_id values")
    return items, {
        "selected_natural_anchors": anchor_count,
        "selected_controlled_anchors": controlled_anchor_count,
        "items_written": len(items),
        "natural_items": sum(item["variant"] == NATURAL for item in items),
        "controlled_items": sum(item["variant"] == CONTROLLED for item in items),
        "present_items": sum(item["query_relation"] == PRESENT for item in items),
        "absent_items": sum(item["query_relation"] == ABSENT for item in items),
        "state_items": sum(item["prompt_framing"] == STATE for item in items),
        "evidence_items": sum(item["prompt_framing"] == EVIDENCE for item in items),
        "controlled_k": list(controlled_k),
        "sampling_seed": seed,
    }


def summarize_anchors(anchors: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return aggregate-only anchor distributions safe for audit logs."""

    count_bins: Counter[str] = Counter()
    compositions: Counter[str] = Counter()
    total = 0
    controlled_eligible = 0
    for anchor in anchors:
        total += 1
        count = int(anchor["natural_option_count"])
        count_bins[_option_count_bin(count)] += 1
        compositions[str(anchor["answer_composition"])] += 1
        if count >= 5:
            controlled_eligible += 1
    return {
        "anchor_count": total,
        "option_count_bins": dict(sorted(count_bins.items())),
        "answer_compositions": dict(sorted(compositions.items())),
        "controlled_k5_eligible_anchors": controlled_eligible,
    }


def option_id(index: int) -> str:
    """Return Excel-style option IDs: A..Z, AA..AZ, and so on."""

    if index < 0:
        raise ValueError("option index must be non-negative")
    value = index + 1
    result = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _eligible_assertion(row: dict[str, Any]) -> bool:
    return (
        row.get("category") == "anatomicalfinding"
        and row.get("polarity") in {"yes", "no"}
        and bool(_text(row.get("image_id")))
        and bool(_text(row.get("label_name")))
    )


def _option_candidate(
    label_name: str,
    polarity: str,
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "label_name": label_name,
        "polarity": polarity,
        "source_assertions": [clean_source_assertion(row) for row in rows],
    }


def _anchor(
    *,
    granularity: str,
    prefix: str,
    patient_id: Any,
    study_id: Any,
    image_id: Any,
    target_anatomy: str | None,
    bbox: dict[str, Any] | None,
    options: list[dict[str, Any]],
) -> dict[str, Any]:
    anchor_id = stable_item_id(
        prefix,
        {
            "patient_id": patient_id,
            "study_id": study_id,
            "image_id": image_id,
            "target_anatomy": target_anatomy,
        },
    )
    return {
        "anchor_id": anchor_id,
        "experiment_id": EXPERIMENT_ID,
        "source_dataset": "ChestImaGenome",
        "patient_id": patient_id,
        "study_id": study_id,
        "image_id": image_id,
        "dicom_id": image_id,
        "image_path": None,
        "granularity": granularity,
        "target_anatomy": target_anatomy,
        "bbox": bbox,
        "options": options,
        "natural_option_count": len(options),
        "answer_composition": _composition(options),
        "source_quality": "silver",
    }


def _paired_items(
    anchor: dict[str, Any],
    ordered_candidates: list[dict[str, Any]],
    *,
    variant: str,
    k: int | None,
) -> list[dict[str, Any]]:
    public_options = [
        {"option_id": option_id(index), "label_name": row["label_name"]}
        for index, row in enumerate(ordered_candidates)
    ]
    option_evidence = [
        {
            "option_id": option_id(index),
            "label_name": row["label_name"],
            "polarity": row["polarity"],
            "source_assertions": row["source_assertions"],
        }
        for index, row in enumerate(ordered_candidates)
    ]
    option_set_id = stable_item_id(
        "study3_option_set",
        {
            "anchor_id": anchor["anchor_id"],
            "variant": variant,
            "k": k,
            "labels": [row["label_name"] for row in ordered_candidates],
        },
    )
    results: list[dict[str, Any]] = []
    for prompt_framing in PROMPT_FRAMINGS:
        for relation in (PRESENT, ABSENT):
            selected_polarity = "yes" if relation == PRESENT else "no"
            gold = [
                option_id(index)
                for index, row in enumerate(ordered_candidates)
                if row["polarity"] == selected_polarity
            ]
            granularity_prefix = (
                "study3_g1_ms"
                if anchor["granularity"] == "G1_finding_existence"
                else "study3_g2_ms"
            )
            item_id = stable_item_id(
                granularity_prefix,
                {
                    "option_set_id": option_set_id,
                    "prompt_framing": prompt_framing,
                    "query_relation": relation,
                },
            )
            results.append(
                {
                "item_id": item_id,
                "experiment_id": EXPERIMENT_ID,
                "source_dataset": anchor["source_dataset"],
                "patient_id": anchor.get("patient_id"),
                "study_id": anchor.get("study_id"),
                "image_id": anchor.get("image_id"),
                "dicom_id": anchor.get("dicom_id"),
                "image_path": anchor.get("image_path"),
                "granularity": anchor["granularity"],
                "question_type": QUESTION_TYPE,
                "prompt_framing": prompt_framing,
                "query_relation": relation,
                "variant": variant,
                "controlled_k": k,
                "anchor_id": anchor["anchor_id"],
                "option_set_id": option_set_id,
                "option_count": len(public_options),
                "natural_option_count": anchor["natural_option_count"],
                "answer_composition": _composition(ordered_candidates),
                "options": public_options,
                "gold_selected_options": gold,
                "target_anatomy": anchor.get("target_anatomy"),
                "bbox": anchor.get("bbox"),
                "source_assertions": option_evidence,
                "evidence_sources": (
                    ["E2_structured_label"]
                    if anchor["granularity"] == "G1_finding_existence"
                    else ["E1_bbox", "E2_anatomy_finding_pair"]
                ),
                "source_quality": anchor["source_quality"],
                "valid_for_training": False,
                }
            )
    return results


def _valid_object_index(
    object_rows: Iterable[dict[str, Any]],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in object_rows:
        bbox_name = _text(row.get("bbox_name"))
        image_id = _text(row.get("image_id"))
        if not bbox_name or not image_id or not _has_complete_bbox(row):
            continue
        key = (
            row.get("patient_id"),
            row.get("study_id"),
            row.get("image_id"),
            bbox_name,
        )
        index.setdefault(key, row)
    return index


def _has_complete_bbox(row: dict[str, Any]) -> bool:
    normalized = [row.get(name) for name in ("x1", "y1", "x2", "y2")]
    original = [
        row.get(name)
        for name in ("original_x1", "original_y1", "original_x2", "original_y2")
    ]
    return all(value is not None for value in normalized) or all(
        value is not None for value in original
    )


def _bbox_from_object(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "bbox_name": row.get("bbox_name"),
        "x1": row.get("x1"),
        "y1": row.get("y1"),
        "x2": row.get("x2"),
        "y2": row.get("y2"),
        "original_x1": row.get("original_x1"),
        "original_y1": row.get("original_y1"),
        "original_x2": row.get("original_x2"),
        "original_y2": row.get("original_y2"),
        "coordinate_space": "original_image",
    }


def _ordered_options(
    options: Iterable[dict[str, Any]],
    *,
    seed: int,
    label: str,
) -> list[dict[str, Any]]:
    return sorted(
        options,
        key=lambda row: stable_item_id(
            "study3_option_rank",
            {
                "seed": seed,
                "label": label,
                "finding": row["label_name"],
            },
        ),
    )


def _stable_rank(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    label: str,
) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: stable_item_id(
            "study3_anchor_rank",
            {
                "seed": seed,
                "label": label,
                "anchor_id": row["anchor_id"],
            },
        ),
    )


def _composition_round_robin(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    label: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {
        "all_yes": [],
        "all_no": [],
        "mixed": [],
    }
    for row in rows:
        groups[str(row["answer_composition"])].append(row)
    for name in groups:
        groups[name] = _stable_rank(groups[name], seed=seed, label=f"{label}:{name}")

    ordered: list[dict[str, Any]] = []
    while any(groups.values()):
        for name in ("all_yes", "all_no", "mixed"):
            if groups[name]:
                ordered.append(groups[name].pop(0))
    return ordered


def _composition(options: Iterable[dict[str, Any]]) -> str:
    polarities = {str(row["polarity"]) for row in options}
    if polarities == {"yes"}:
        return "all_yes"
    if polarities == {"no"}:
        return "all_no"
    return "mixed"


def _option_count_bin(count: int) -> str:
    if count <= 2:
        return "2"
    if count == 3:
        return "3"
    if count == 4:
        return "4"
    return "5plus"


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _tuple_sort_key(key: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple("" if value is None else str(value) for value in key)
