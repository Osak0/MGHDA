"""Score Study 2 closed claim-verification outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ghm.granularity.common import read_jsonl, write_jsonl
from ghm.granularity.study2 import (
    ANSWER_CONTRADICTED,
    ANSWER_NOT_ENOUGH,
    ANSWER_SUPPORTED,
)


STUDY2_LABELS = {ANSWER_SUPPORTED, ANSWER_CONTRADICTED, ANSWER_NOT_ENOUGH}


def score_closed_qa_row(row: dict[str, Any]) -> dict[str, Any]:
    """Score one parsed Study 2 claim-verification response."""

    answer_label = row.get("answer_label")
    parsed_answer = row.get("parsed_answer")
    parse_status = row.get("parse_status")

    score = "requires_manual_review"
    hallucination_type = None
    evidence_relation = "U_uncertain"
    error_category = None
    h1_error_direction = None
    is_correct: bool | None = None
    requires_manual_review = False

    if parse_status != "success" or parsed_answer is None:
        score = "invalid_response"
        is_correct = False
    elif answer_label == parsed_answer:
        score = "correct"
        evidence_relation = _evidence_relation_for_label(answer_label)
        is_correct = True
    elif answer_label == ANSWER_CONTRADICTED and parsed_answer == ANSWER_SUPPORTED:
        score = "H1_evidence_contradicted"
        hallucination_type = "H1"
        evidence_relation = "S-1_contradicted"
        error_category = "contradicted_claim_supported"
        is_correct = False
    elif answer_label == ANSWER_NOT_ENOUGH and parsed_answer == ANSWER_SUPPORTED:
        score = "H2_evidence_unsupported"
        hallucination_type = "H2"
        evidence_relation = "S0_unsupported"
        error_category = "unsupported_claim_supported"
        is_correct = False
    elif answer_label in STUDY2_LABELS and parsed_answer in STUDY2_LABELS:
        score = "incorrect_non_hallucination"
        evidence_relation = _evidence_relation_for_label(answer_label)
        error_category = "wrong_claim_verification_label"
        is_correct = False
    else:
        requires_manual_review = True
        is_correct = None

    return {
        "item_id": row.get("item_id"),
        "model_name": row.get("model_name"),
        "granularity": row.get("granularity"),
        "question_type": row.get("question_type"),
        "hallucination_probe": row.get("hallucination_probe"),
        "claim_polarity": row.get("claim_polarity"),
        "evidence_state": row.get("evidence_state"),
        "answer_label": answer_label,
        "parsed_answer": parsed_answer,
        "score": score,
        "is_correct": is_correct,
        "hallucination_type": hallucination_type,
        "h1_error_direction": h1_error_direction,
        "evidence_relation": evidence_relation,
        "error_category": error_category,
        "requires_manual_review": requires_manual_review,
        "notes": None,
    }


def score_closed_qa_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score parsed response rows."""

    return [score_closed_qa_row(row) for row in rows]


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize scores overall and by Study 2 metadata."""

    return {
        "overall": _summarize_group(rows),
        "by_granularity": _summarize_by(rows, "granularity"),
        "by_claim_polarity": _summarize_by(rows, "claim_polarity"),
        "by_evidence_state": _summarize_by(rows, "evidence_state"),
        "by_answer_label": _summarize_by(rows, "answer_label"),
        "by_parsed_answer": _summarize_by(rows, "parsed_answer"),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Score parsed closed-QA responses.")
    parser.add_argument("--input", type=Path, required=True, help="Parsed response JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Scored output JSONL.")
    parser.add_argument("--summary", type=Path, required=True, help="Aggregate summary JSON.")
    args = parser.parse_args(argv)

    parsed_rows = read_jsonl(args.input)
    scored_rows = score_closed_qa_rows(parsed_rows)
    write_jsonl(scored_rows, args.output)
    summary = summarize_scores(scored_rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    overall = summary["overall"]
    print(
        "Scored Study 2 claim verification: "
        f"items={overall['items']}, "
        f"accuracy={overall['accuracy']}, "
        f"contradicted_supported_count={overall['h1_count']}, "
        f"not_enough_supported_count={overall['h2_count']}, "
        f"invalid_count={overall['invalid_count']}, "
        f"manual_review_count={overall['manual_review_count']}"
    )
    return 0


def _evidence_relation_for_label(label: Any) -> str:
    if label == ANSWER_SUPPORTED:
        return "S2_directly_supported"
    if label == ANSWER_CONTRADICTED:
        return "S-1_contradicted"
    if label == ANSWER_NOT_ENOUGH:
        return "S0_unsupported"
    return "U_uncertain"


def _summarize_by(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(field))
        groups.setdefault(key, []).append(row)
    return {key: _summarize_group(groups[key]) for key in sorted(groups)}


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    items = len(rows)
    correct = sum(1 for row in rows if row.get("is_correct") is True)
    scored = sum(1 for row in rows if row.get("is_correct") is not None)
    h1_count = sum(1 for row in rows if row.get("score") == "H1_evidence_contradicted")
    h2_count = sum(1 for row in rows if row.get("score") == "H2_evidence_unsupported")
    return {
        "items": items,
        "scored_items": scored,
        "correct": correct,
        "accuracy": round(correct / scored, 6) if scored else None,
        "h1_count": h1_count,
        "h1_false_positive_count": h1_count,
        "h1_false_negative_count": 0,
        "h2_count": h2_count,
        "invalid_count": sum(1 for row in rows if row.get("score") == "invalid_response"),
        "manual_review_count": sum(
            1 for row in rows if row.get("requires_manual_review") is True
        ),
        "uncertain_or_abstention_count": 0,
        "incorrect_non_hallucination_count": sum(
            1 for row in rows if row.get("score") == "incorrect_non_hallucination"
        ),
        "answer_distribution": _count_values(rows, "parsed_answer"),
    }


def _count_values(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
