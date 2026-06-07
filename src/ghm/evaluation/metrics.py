"""Compute aggregate metrics for scored model outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ghm.evaluation.score_closed_qa import summarize_scores
from ghm.granularity.common import read_jsonl


def summarize_scored_files(paths: list[Path]) -> dict[str, Any]:
    """Summarize one or more scored JSONL files."""

    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return summarize_scores(rows)


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
