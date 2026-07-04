"""Parse raw model responses into normalized closed-form answers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from ghm.granularity.common import read_jsonl, write_jsonl
from ghm.granularity.study2 import (
    ANSWER_CONTRADICTED,
    ANSWER_NOT_ENOUGH,
    ANSWER_SUPPORTED,
)


LABEL_PATTERN = re.compile(
    r"\b(not enough evidence|contradicted|unsupported|supported|uncertain|yes|no)\b",
    re.IGNORECASE,
)
LETTER_PATTERN = re.compile(r"(?:^|[\s:])([ABC])(?:[\s\).:,-]|$)")
CANONICAL = {
    "a": ANSWER_SUPPORTED,
    "b": ANSWER_CONTRADICTED,
    "c": ANSWER_NOT_ENOUGH,
    "supported": ANSWER_SUPPORTED,
    "contradicted": ANSWER_CONTRADICTED,
    "not enough evidence": ANSWER_NOT_ENOUGH,
    # Legacy labels are retained so older outputs fail gracefully instead of
    # becoming unparsable during transition runs.
    "yes": "Yes",
    "no": "No",
    "uncertain": "Uncertain",
    "unsupported": "Unsupported",
}


def parse_closed_answer(raw_response: Any) -> tuple[str | None, str]:
    """Parse a closed-form model answer."""

    if raw_response is None:
        return None, "empty_response"
    text = str(raw_response).strip()
    if not text:
        return None, "empty_response"

    matches: list[str] = []
    if text.lower() in {"a", "b", "c"}:
        matches.append(CANONICAL[text.lower()])
    matches.extend(CANONICAL[match.group(1).lower()] for match in LETTER_PATTERN.finditer(text))
    matches.extend(CANONICAL[match.group(1).lower()] for match in LABEL_PATTERN.finditer(text))
    unique = list(dict.fromkeys(matches))
    if not unique:
        return None, "invalid_format"
    if len(unique) > 1:
        return None, "multiple_answers"
    return unique[0], "success"


def parse_response_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add parsed_answer and parse_status to raw response rows."""

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        parsed_answer, parse_status = parse_closed_answer(row.get("raw_response"))
        updated = dict(row)
        updated["parsed_answer"] = parsed_answer
        updated["parse_status"] = parse_status
        parsed_rows.append(updated)
    return parsed_rows


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Parse closed-form model answers.")
    parser.add_argument("--input", type=Path, required=True, help="Raw response JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Parsed response JSONL.")
    args = parser.parse_args(argv)

    rows = read_jsonl(args.input)
    parsed_rows = parse_response_rows(rows)
    write_jsonl(parsed_rows, args.output)
    status_counts: dict[str, int] = {}
    for row in parsed_rows:
        status = str(row.get("parse_status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    print(f"Parsed responses: rows={len(parsed_rows)}, status_counts={status_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
