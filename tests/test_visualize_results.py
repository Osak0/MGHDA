import json

from ghm.evaluation.visualize_results import flatten_summaries, write_tables


def test_flatten_summaries_and_write_tables(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "overall": {
                    "items": 10,
                    "scored_items": 10,
                    "correct": 8,
                    "accuracy": 0.8,
                    "h1_count": 1,
                    "h1_false_positive_count": 1,
                    "h1_false_negative_count": 0,
                    "h2_count": 2,
                    "invalid_count": 0,
                    "manual_review_count": 0,
                    "uncertain_or_abstention_count": 1,
                },
                "by_hallucination_probe": {
                    "H1": {
                        "items": 5,
                        "scored_items": 5,
                        "correct": 4,
                        "accuracy": 0.8,
                        "h1_count": 1,
                        "h1_false_positive_count": 1,
                        "h1_false_negative_count": 0,
                        "h2_count": 0,
                        "invalid_count": 0,
                        "manual_review_count": 0,
                        "uncertain_or_abstention_count": 0,
                    },
                    "H2": {
                        "items": 5,
                        "scored_items": 5,
                        "correct": 4,
                        "accuracy": 0.8,
                        "h1_count": 0,
                        "h1_false_positive_count": 0,
                        "h1_false_negative_count": 0,
                        "h2_count": 2,
                        "invalid_count": 0,
                        "manual_review_count": 0,
                        "uncertain_or_abstention_count": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    rows = flatten_summaries([summary_path], ["smoke"])
    overall_path, group_path = write_tables(rows, tmp_path / "report")

    overall_rows = overall_path.read_text(encoding="utf-8").splitlines()
    group_rows = group_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3
    assert "h1_rate" in overall_rows[0]
    assert ",0.1,0.2,0.0,0.1" in overall_rows[1]
    assert len(group_rows) == 3
