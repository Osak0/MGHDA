"""Compute aggregate metrics for scored model outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ghm.evaluation.score_closed_qa import STUDY2_LABELS, summarize_scores
from ghm.granularity.common import read_jsonl


def summarize_scored_files(paths: list[Path]) -> dict[str, Any]:
    """Summarize one or more scored JSONL files."""

    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    summary = summarize_scores(rows)
    summary["data_quality"] = _summarize_data_quality(rows, input_files=len(paths))
    return summary


def _summarize_data_quality(
    rows: list[dict[str, Any]],
    *,
    input_files: int,
) -> dict[str, Any]:
    """Report aggregate integrity checks without exposing identifiers."""

    item_ids = [str(row["item_id"]) for row in rows if row.get("item_id") is not None]
    required_fields = (
        "item_id",
        "granularity",
        "claim_polarity",
        "evidence_state",
        "answer_label",
        "parsed_answer",
        "score",
        "is_correct",
    )
    return {
        "input_files": input_files,
        "records": len(rows),
        "unique_item_ids": len(set(item_ids)),
        "missing_item_id_count": len(rows) - len(item_ids),
        "duplicate_item_id_count": len(item_ids) - len(set(item_ids)),
        "null_field_counts": {
            field: sum(1 for row in rows if row.get(field) is None)
            for field in required_fields
        },
        "unexpected_answer_label_count": sum(
            1 for row in rows if row.get("answer_label") not in STUDY2_LABELS
        ),
        "unexpected_parsed_answer_count": sum(
            1
            for row in rows
            if row.get("parsed_answer") is not None
            and row.get("parsed_answer") not in STUDY2_LABELS
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Summarize scored JSONL files.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    summary = summarize_scored_files(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    overall = summary["overall"]
    print(
        "Summarized scored outputs: "
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


if __name__ == "__main__":
    raise SystemExit(main())
