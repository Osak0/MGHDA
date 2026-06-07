"""Build G1 H1/H2 model-ready items from Chest ImaGenome assertions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from ghm.granularity.common import (
    balance_binary_label_items,
    clean_source_assertion,
    get_image_record,
    image_index_by_key,
    read_parquet_rows,
    safe_tuple_sort_key,
    stable_item_id,
    update_summary,
    write_jsonl,
)


ATTRIBUTE_COLUMNS = [
    "assertion_id",
    "patient_id",
    "study_id",
    "image_id",
    "bbox_name",
    "raw_label",
    "category",
    "polarity",
    "label_name",
    "phrase_id",
    "phrase_index",
    "source_quality",
]

IMAGE_INDEX_COLUMNS = [
    "patient_id",
    "study_id",
    "image_id",
    "dicom_id",
    "image_path",
]


def build_g1_h1_items(
    assertion_rows: list[dict[str, Any]],
    image_index_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build G1 H1 yes/no finding-existence items."""

    groups = _group_anatomicalfindings(assertion_rows, include_bbox=False)
    image_index = image_index_by_key(image_index_rows or [])
    items: list[dict[str, Any]] = []
    conflict_groups = 0
    yes_items = 0
    no_items = 0

    for key in sorted(groups, key=safe_tuple_sort_key):
        patient_id, study_id, image_id, label_name = key
        rows = groups[key]
        polarities = {row.get("polarity") for row in rows}
        if "yes" in polarities and "no" in polarities:
            conflict_groups += 1
            continue
        answer_label = "Yes" if "yes" in polarities else "No"
        if answer_label == "Yes":
            yes_items += 1
        else:
            no_items += 1
        image_record = get_image_record(
            image_index,
            patient_id=patient_id,
            study_id=study_id,
            image_id=image_id,
        )
        items.append(
            _base_item(
                prefix="ci_g1_h1",
                image_record=image_record,
                target_finding=label_name,
                target_anatomy=None,
                question=f"Is there evidence of {label_name} in this chest X-ray?",
                answer_label=answer_label,
                hallucination_probe="H1",
                question_type="h1_yes_no_qa",
                source_assertions=[clean_source_assertion(row) for row in rows],
                evidence_sources=["E2_structured_label"],
                extra_id_components={"target_finding": label_name},
            )
        )

    return items, {
        "candidate_groups": len(groups),
        "items_written": len(items),
        "yes_items": yes_items,
        "no_items": no_items,
        "excluded_conflict_groups": conflict_groups,
    }


def build_g1_h2_items(
    assertion_rows: list[dict[str, Any]],
    image_index_rows: list[dict[str, Any]] | None = None,
    *,
    unsupported_per_image: int = 1,
    balance_h2: bool = True,
    h2_unsupported_fraction: float = 0.8,
    h2_sampling_seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build G1 H2 claim-support items."""

    groups = _group_anatomicalfindings(assertion_rows, include_bbox=False)
    image_index = image_index_by_key(image_index_rows or [])
    finding_vocab = sorted(
        {
            row.get("label_name")
            for row in assertion_rows
            if row.get("category") == "anatomicalfinding"
            and row.get("polarity") in {"yes", "no"}
            and row.get("label_name")
        }
    )
    mentioned_by_image: dict[tuple[str | None, str | None, str | None], set[str]]
    mentioned_by_image = defaultdict(set)
    items: list[dict[str, Any]] = []
    supported_items = 0
    unsupported_items = 0
    conflict_groups = 0

    for key in sorted(groups, key=safe_tuple_sort_key):
        patient_id, study_id, image_id, label_name = key
        rows = groups[key]
        mentioned_by_image[(patient_id, study_id, image_id)].add(label_name)
        polarities = {row.get("polarity") for row in rows}
        if "yes" in polarities and "no" in polarities:
            conflict_groups += 1
            continue
        polarity = "yes" if "yes" in polarities else "no"
        claim = _claim(label_name, polarity=polarity, anatomy=None)
        image_record = get_image_record(
            image_index,
            patient_id=patient_id,
            study_id=study_id,
            image_id=image_id,
        )
        items.append(
            _base_item(
                prefix="ci_g1_h2",
                image_record=image_record,
                target_finding=label_name,
                target_anatomy=None,
                question=_support_question(claim),
                answer_label="Supported",
                hallucination_probe="H2",
                question_type="h2_claim_support",
                source_assertions=[clean_source_assertion(row) for row in rows],
                evidence_sources=["E2_structured_label"],
                extra_id_components={"target_finding": label_name, "claim": claim},
            )
        )
        supported_items += 1

    for image_key in sorted(mentioned_by_image, key=safe_tuple_sort_key):
        patient_id, study_id, image_id = image_key
        missing_findings = [
            finding
            for finding in _stable_missing_findings(finding_vocab, mentioned_by_image[image_key], image_id)
            if finding not in mentioned_by_image[image_key]
        ][:unsupported_per_image]
        for finding in missing_findings:
            claim = _claim(finding, polarity="yes", anatomy=None)
            image_record = get_image_record(
                image_index,
                patient_id=patient_id,
                study_id=study_id,
                image_id=image_id,
            )
            items.append(
                _base_item(
                    prefix="ci_g1_h2",
                    image_record=image_record,
                    target_finding=finding,
                    target_anatomy=None,
                    question=_support_question(claim),
                    answer_label="Unsupported",
                    hallucination_probe="H2",
                    question_type="h2_claim_support",
                    source_assertions=[
                        {
                            "source": "constructed_missing_evidence_probe",
                            "label_name": finding,
                            "scope": "image",
                        }
                    ],
                    evidence_sources=["E5_task_context"],
                    extra_id_components={"target_finding": finding, "claim": claim},
                )
            )
            unsupported_items += 1

    balance_summary: dict[str, Any] = {
        "balance_applied": False,
        "balance_reason": "disabled",
        "positive_label": "Supported",
        "negative_label": "Unsupported",
        "target_negative_fraction": h2_unsupported_fraction,
        "positive_before": supported_items,
        "negative_before": unsupported_items,
        "positive_after": supported_items,
        "negative_after": unsupported_items,
        "excluded_by_ratio": 0,
        "sampling_seed": h2_sampling_seed,
    }
    if balance_h2:
        items, balance_summary = balance_binary_label_items(
            items,
            positive_label="Supported",
            negative_label="Unsupported",
            negative_fraction=h2_unsupported_fraction,
            seed=h2_sampling_seed,
        )

    return items, {
        "finding_vocab_size": len(finding_vocab),
        "candidate_supported_groups": len(groups),
        "items_written": len(items),
        "supported_items": balance_summary["positive_after"],
        "unsupported_items": balance_summary["negative_after"],
        "supported_items_before_balance": supported_items,
        "unsupported_items_before_balance": unsupported_items,
        "excluded_conflict_groups": conflict_groups,
        "unsupported_per_image": unsupported_per_image,
        "h2_balance": balance_summary,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Build G1 H1/H2 items.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/interim/ci_attribute_assertions.parquet"),
    )
    parser.add_argument(
        "--image-index",
        type=Path,
        default=Path("data/interim/image_index.parquet"),
    )
    parser.add_argument(
        "--h1-output",
        type=Path,
        default=Path("data/processed/items/g1_h1_items.jsonl"),
    )
    parser.add_argument(
        "--h2-output",
        type=Path,
        default=Path("data/processed/items/g1_h2_items.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/audits/g1_g2_candidate_summary.json"),
    )
    parser.add_argument("--unsupported-per-image", type=int, default=1)
    parser.add_argument(
        "--h2-unsupported-fraction",
        type=float,
        default=0.8,
        help="Pilot target fraction of Unsupported labels in G1 H2 items.",
    )
    parser.add_argument("--h2-sampling-seed", type=int, default=42)
    parser.add_argument("--no-balance-h2", action="store_true")
    args = parser.parse_args(argv)
    if args.unsupported_per_image < 1:
        parser.error("--unsupported-per-image must be positive.")
    if not 0 < args.h2_unsupported_fraction < 1:
        parser.error("--h2-unsupported-fraction must be between 0 and 1.")

    try:
        assertion_rows = read_parquet_rows(args.input, columns=ATTRIBUTE_COLUMNS)
        image_index_rows = read_parquet_rows(args.image_index, columns=IMAGE_INDEX_COLUMNS)
    except RuntimeError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    h1_items, h1_summary = build_g1_h1_items(assertion_rows, image_index_rows)
    h2_items, h2_summary = build_g1_h2_items(
        assertion_rows,
        image_index_rows,
        unsupported_per_image=args.unsupported_per_image,
        balance_h2=not args.no_balance_h2,
        h2_unsupported_fraction=args.h2_unsupported_fraction,
        h2_sampling_seed=args.h2_sampling_seed,
    )
    write_jsonl(h1_items, args.h1_output)
    write_jsonl(h2_items, args.h2_output)
    update_summary(args.summary, "g1_h1", h1_summary)
    update_summary(args.summary, "g1_h2", h2_summary)
    print(
        "Built G1 items: "
        f"h1={h1_summary['items_written']}, "
        f"h2={h2_summary['items_written']}, "
        f"h1_conflicts={h1_summary['excluded_conflict_groups']}, "
        f"h2_conflicts={h2_summary['excluded_conflict_groups']}"
    )
    return 0


def _group_anatomicalfindings(
    rows: list[dict[str, Any]],
    *,
    include_bbox: bool,
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("category") != "anatomicalfinding":
            continue
        if row.get("polarity") not in {"yes", "no"}:
            continue
        if not row.get("image_id") or not row.get("label_name"):
            continue
        if include_bbox:
            key = (
                row.get("patient_id"),
                row.get("study_id"),
                row.get("image_id"),
                row.get("bbox_name"),
                row.get("label_name"),
            )
        else:
            key = (
                row.get("patient_id"),
                row.get("study_id"),
                row.get("image_id"),
                row.get("label_name"),
            )
        groups[key].append(row)
    return groups


def _base_item(
    *,
    prefix: str,
    image_record: dict[str, Any],
    target_finding: str,
    target_anatomy: str | None,
    question: str,
    answer_label: str,
    hallucination_probe: str,
    question_type: str,
    source_assertions: list[dict[str, Any]],
    evidence_sources: list[str],
    extra_id_components: dict[str, Any],
) -> dict[str, Any]:
    item_id = stable_item_id(
        prefix,
        {
            "patient_id": image_record.get("patient_id"),
            "study_id": image_record.get("study_id"),
            "image_id": image_record.get("image_id"),
            "target_anatomy": target_anatomy,
            **extra_id_components,
        },
    )
    return {
        "item_id": item_id,
        "source_dataset": "ChestImaGenome",
        "patient_id": image_record.get("patient_id"),
        "study_id": image_record.get("study_id"),
        "image_id": image_record.get("image_id"),
        "dicom_id": image_record.get("dicom_id"),
        "image_path": image_record.get("image_path"),
        "granularity": "G1_finding_existence",
        "question_type": question_type,
        "hallucination_probe": hallucination_probe,
        "question": question,
        "answer_label": answer_label,
        "target_finding": target_finding,
        "target_anatomy": target_anatomy,
        "bbox": None,
        "source_assertions": source_assertions,
        "evidence_sources": evidence_sources,
        "source_quality": "silver",
        "valid_for_clean_qa": hallucination_probe == "H1",
        "valid_for_false_premise": False,
        "valid_for_roi_mask": False,
        "valid_for_roi_only": False,
        "valid_for_training": False,
        "exclusion_flag": None,
    }


def _claim(finding: str, *, polarity: str, anatomy: str | None) -> str:
    if anatomy is None:
        if polarity == "yes":
            return f"There is evidence of {finding}."
        return f"There isn't evidence of {finding}."
    if polarity == "yes":
        return f"There is evidence of {finding} in the {anatomy}."
    return f"There isn't evidence of {finding} in the {anatomy}."


def _support_question(claim: str) -> str:
    return (
        "Is the following claim supported by the chest X-ray? "
        f"Claim: {claim}"
    )


def _stable_missing_findings(
    finding_vocab: list[str],
    mentioned: set[str],
    image_id: Any,
) -> list[str]:
    missing = [finding for finding in finding_vocab if finding not in mentioned]
    return sorted(missing, key=lambda finding: stable_item_id("rank", {"image": image_id, "finding": finding}))


if __name__ == "__main__":
    raise SystemExit(main())
