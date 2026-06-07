"""Score closed QA outputs against constructed answer labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ghm.granularity.common import read_jsonl, write_jsonl


def score_closed_qa_row(row: dict[str, Any]) -> dict[str, Any]:
    """Score one parsed closed-QA response."""

    answer_label = row.get("answer_label")
    parsed_answer = row.get("parsed_answer")
    parse_status = row.get("parse_status")
    hallucination_probe = row.get("hallucination_probe")

    score = "requires_manual_review"
    hallucination_type = None
    evidence_relation = "U_uncertain"
    error_category = None
    h1_error_direction = None
    is_correct: bool | None = None
    requires_manual_review = False

    if parse_status != "success" or parsed_answer is None:
        score = "invalid_response"
        evidence_relation = "U_uncertain"
        requires_manual_review = False
        is_correct = False
    elif answer_label == parsed_answer:
        score = "correct"
        evidence_relation = (
            "S0_unsupported" if answer_label == "Unsupported" else "S2_directly_supported"
        )
        is_correct = True
    elif answer_label == "No" and parsed_answer == "Yes":
        score = "H1_evidence_contradicted"
        hallucination_type = "H1"
        h1_error_direction = "false_positive"
        evidence_relation = "S-1_contradicted"
        is_correct = False
    elif answer_label == "Yes" and parsed_answer == "No":
        score = "H1_evidence_contradicted"
        hallucination_type = "H1"
        h1_error_direction = "false_negative"
        evidence_relation = "S-1_contradicted"
        is_correct = False
    elif answer_label == "Unsupported" and parsed_answer == "Supported":
        score = "H2_evidence_unsupported"
        hallucination_type = "H2"
        evidence_relation = "S0_unsupported"
        is_correct = False
    elif answer_label == "Supported" and parsed_answer == "Unsupported":
        score = "incorrect_non_hallucination"
        evidence_relation = "S0_unsupported"
        error_category = "unsupported_rejection"
        is_correct = False
    elif parsed_answer == "Uncertain" and answer_label in {
        "Yes",
        "No",
        "Supported",
        "Unsupported",
    }:
        score = "uncertain_or_abstention"
        evidence_relation = "U_uncertain"
        is_correct = False
    elif answer_label == "Uncertain" and parsed_answer in {
        "Yes",
        "No",
        "Supported",
        "Unsupported",
    }:
        score = "requires_manual_review"
        evidence_relation = "U_uncertain"
        error_category = "uncertain_case"
        is_correct = None
        requires_manual_review = True
    else:
        requires_manual_review = True

    return {
        "item_id": row.get("item_id"),
        "model_name": row.get("model_name"),
        "granularity": row.get("granularity"),
        "question_type": row.get("question_type"),
        "hallucination_probe": hallucination_probe,
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
    """Summarize scores overall and by core item metadata."""

    return {
        "overall": _summarize_group(rows),
        "by_granularity": _summarize_by(rows, "granularity"),
        "by_hallucination_probe": _summarize_by(rows, "hallucination_probe"),
        "by_answer_label": _summarize_by(rows, "answer_label"),
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
        "Scored closed QA: "
        f"items={overall['items']}, "
        f"accuracy={overall['accuracy']}, "
        f"h1_count={overall['h1_count']}, "
        f"h1_false_positive_count={overall['h1_false_positive_count']}, "
        f"h1_false_negative_count={overall['h1_false_negative_count']}, "
        f"h2_count={overall['h2_count']}, "
        f"invalid_count={overall['invalid_count']}, "
        f"manual_review_count={overall['manual_review_count']}"
    )
    return 0


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
    h1_false_positive = sum(
        1 for row in rows if row.get("h1_error_direction") == "false_positive"
    )
    h1_false_negative = sum(
        1 for row in rows if row.get("h1_error_direction") == "false_negative"
    )
    return {
        "items": items,
        "scored_items": scored,
        "correct": correct,
        "accuracy": round(correct / scored, 6) if scored else None,
        "h1_count": h1_false_positive + h1_false_negative,
        "h1_false_positive_count": h1_false_positive,
        "h1_false_negative_count": h1_false_negative,
        "h2_count": sum(1 for row in rows if row.get("score") == "H2_evidence_unsupported"),
        "invalid_count": sum(1 for row in rows if row.get("score") == "invalid_response"),
        "manual_review_count": sum(
            1 for row in rows if row.get("requires_manual_review") is True
        ),
        "uncertain_or_abstention_count": sum(
            1 for row in rows if row.get("score") == "uncertain_or_abstention"
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
