"""Build Study 2 G1 claim-verification items from Chest ImaGenome assertions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from ghm.granularity.common import (
    clean_source_assertion,
    get_image_record,
    image_index_by_key,
    read_parquet_rows,
    safe_tuple_sort_key,
    stable_item_id,
    update_summary,
    write_jsonl,
)
from ghm.granularity.study2 import (
    CLAIM_NEGATIVE,
    CLAIM_POSITIVE,
    EVIDENCE_AFFIRMED,
    EVIDENCE_NEGATED,
    EVIDENCE_NOT_ENOUGH,
    MISSING_FINDING_SAMPLE_SIZE,
    QUESTION_TYPE,
    answer_for_claim,
    claim_text,
    question_for_claim,
    stable_missing_findings,
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


def build_g1_claim_verification_items(
    assertion_rows: list[dict[str, Any]],
    image_index_rows: list[dict[str, Any]] | None = None,
    *,
    missing_finding_sample_size: int = MISSING_FINDING_SAMPLE_SIZE,
    missing_finding_seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build G1 image-level Study 2 claim-verification items."""

    groups = _group_anatomicalfindings(assertion_rows)
    image_index = image_index_by_key(image_index_rows or [])
    mentioned_by_image: dict[tuple[str | None, str | None, str | None], set[str]]
    mentioned_by_image = defaultdict(set)
    rows_by_image: dict[tuple[str | None, str | None, str | None], list[dict[str, Any]]]
    rows_by_image = defaultdict(list)
    items: list[dict[str, Any]] = []
    conflict_groups = 0
    affirmed_groups = 0
    negated_groups = 0
    missing_findings_sampled = 0

    for key in sorted(groups, key=safe_tuple_sort_key):
        patient_id, study_id, image_id, label_name = key
        rows = groups[key]
        image_key = (patient_id, study_id, image_id)
        mentioned_by_image[image_key].add(label_name)
        rows_by_image[image_key].extend(rows)
        polarities = {row.get("polarity") for row in rows}
        if "yes" in polarities and "no" in polarities:
            conflict_groups += 1
            continue
        evidence_state = EVIDENCE_AFFIRMED if "yes" in polarities else EVIDENCE_NEGATED
        if evidence_state == EVIDENCE_AFFIRMED:
            affirmed_groups += 1
        else:
            negated_groups += 1
        image_record = get_image_record(
            image_index,
            patient_id=patient_id,
            study_id=study_id,
            image_id=image_id,
        )
        items.extend(
            _claim_items(
                prefix="study2_g1_claim",
                image_record=image_record,
                target_finding=label_name,
                evidence_state=evidence_state,
                source_assertions=[clean_source_assertion(row) for row in rows],
                evidence_sources=["E2_structured_label"],
                missing_finding_sample_size=missing_finding_sample_size,
            )
        )

    for image_key in sorted(mentioned_by_image, key=safe_tuple_sort_key):
        patient_id, study_id, image_id = image_key
        sampled_findings = stable_missing_findings(
            mentioned=mentioned_by_image[image_key],
            sample_size=missing_finding_sample_size,
            seed=missing_finding_seed,
            scope_components={
                "granularity": "G1",
                "patient_id": patient_id,
                "study_id": study_id,
                "image_id": image_id,
            },
        )
        image_record = get_image_record(
            image_index,
            patient_id=patient_id,
            study_id=study_id,
            image_id=image_id,
        )
        for finding in sampled_findings:
            items.extend(
                _claim_items(
                    prefix="study2_g1_claim",
                    image_record=image_record,
                    target_finding=finding,
                    evidence_state=EVIDENCE_NOT_ENOUGH,
                    source_assertions=[
                        {
                            "source": "constructed_missing_evidence_probe",
                            "label_name": finding,
                            "scope": "image",
                        }
                    ],
                    evidence_sources=["E5_task_context"],
                    missing_finding_sample_size=missing_finding_sample_size,
                )
            )
            missing_findings_sampled += 1

    return items, {
        "candidate_groups": len(groups),
        "items_written": len(items),
        "affirmed_groups": affirmed_groups,
        "negated_groups": negated_groups,
        "explicit_claim_items": 2 * (affirmed_groups + negated_groups),
        "missing_findings_sampled": missing_findings_sampled,
        "missing_claim_items": 2 * missing_findings_sampled,
        "missing_finding_sample_size": missing_finding_sample_size,
        "missing_finding_seed": missing_finding_seed,
        "excluded_conflict_groups": conflict_groups,
    }


def build_g1_h1_items(
    assertion_rows: list[dict[str, Any]],
    image_index_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Backward-compatible alias for the Study 2 G1 builder."""

    return build_g1_claim_verification_items(assertion_rows, image_index_rows)


def build_g1_h2_items(
    assertion_rows: list[dict[str, Any]],
    image_index_rows: list[dict[str, Any]] | None = None,
    **_: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Backward-compatible alias for the Study 2 G1 builder."""

    return build_g1_claim_verification_items(assertion_rows, image_index_rows)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Build Study 2 G1 claim items.")
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
        "--output",
        type=Path,
        default=Path("data/processed/items/study2_g1_claim_verification_items.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/audits/study2_candidate_summary.json"),
    )
    parser.add_argument("--missing-finding-sample-size", type=int, default=2)
    parser.add_argument("--missing-finding-seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.missing_finding_sample_size < 0:
        parser.error("--missing-finding-sample-size must be non-negative.")

    try:
        assertion_rows = read_parquet_rows(args.input, columns=ATTRIBUTE_COLUMNS)
        image_index_rows = read_parquet_rows(args.image_index, columns=IMAGE_INDEX_COLUMNS)
    except RuntimeError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    items, summary = build_g1_claim_verification_items(
        assertion_rows,
        image_index_rows,
        missing_finding_sample_size=args.missing_finding_sample_size,
        missing_finding_seed=args.missing_finding_seed,
    )
    write_jsonl(items, args.output)
    update_summary(args.summary, "study2_g1_claim_verification", summary)
    print(
        "Built Study 2 G1 claim items: "
        f"items={summary['items_written']}, "
        f"affirmed={summary['affirmed_groups']}, "
        f"negated={summary['negated_groups']}, "
        f"missing_findings={summary['missing_findings_sampled']}, "
        f"conflicts={summary['excluded_conflict_groups']}"
    )
    return 0


def _group_anatomicalfindings(
    rows: list[dict[str, Any]],
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("category") != "anatomicalfinding":
            continue
        if row.get("polarity") not in {"yes", "no"}:
            continue
        if not row.get("image_id") or not row.get("label_name"):
            continue
        key = (
            row.get("patient_id"),
            row.get("study_id"),
            row.get("image_id"),
            row.get("label_name"),
        )
        groups[key].append(row)
    return groups


def _claim_items(
    *,
    prefix: str,
    image_record: dict[str, Any],
    target_finding: str,
    evidence_state: str,
    source_assertions: list[dict[str, Any]],
    evidence_sources: list[str],
    missing_finding_sample_size: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for claim_polarity in [CLAIM_POSITIVE, CLAIM_NEGATIVE]:
        claim = claim_text(target_finding, claim_polarity=claim_polarity, anatomy=None)
        answer_label = answer_for_claim(
            evidence_state=evidence_state,
            claim_polarity=claim_polarity,
        )
        item_id = stable_item_id(
            prefix,
            {
                "patient_id": image_record.get("patient_id"),
                "study_id": image_record.get("study_id"),
                "image_id": image_record.get("image_id"),
                "target_finding": target_finding,
                "claim_polarity": claim_polarity,
                "claim": claim,
            },
        )
        items.append(
            {
                "item_id": item_id,
                "source_dataset": "ChestImaGenome",
                "patient_id": image_record.get("patient_id"),
                "study_id": image_record.get("study_id"),
                "image_id": image_record.get("image_id"),
                "dicom_id": image_record.get("dicom_id"),
                "image_path": image_record.get("image_path"),
                "granularity": "G1_finding_existence",
                "question_type": QUESTION_TYPE,
                "hallucination_probe": None,
                "question": question_for_claim(claim),
                "claim": claim,
                "claim_polarity": claim_polarity,
                "evidence_state": evidence_state,
                "answer_label": answer_label,
                "target_finding": target_finding,
                "target_anatomy": None,
                "bbox": None,
                "source_assertions": source_assertions,
                "evidence_sources": evidence_sources,
                "source_quality": "silver",
                "valid_for_clean_qa": False,
                "valid_for_false_premise": True,
                "valid_for_roi_mask": False,
                "valid_for_roi_only": False,
                "valid_for_training": False,
                "missing_finding_sample_size": missing_finding_sample_size,
                "exclusion_flag": None,
            }
        )
    return items


if __name__ == "__main__":
    raise SystemExit(main())
