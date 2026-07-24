from __future__ import annotations

import csv
import json
from pathlib import Path

from ghm.evaluation.experiment_preferences import analyze_result_directories


def test_aggregate_preference_analysis_is_complete_and_safe(tmp_path: Path) -> None:
    study2 = tmp_path / "study2"
    study3 = tmp_path / "study3"
    output = tmp_path / "aggregate"
    study2.mkdir()
    study3.mkdir()

    six_cells = {}
    for evidence in ("affirmed", "negated", "not_enough_evidence"):
        six_cells[evidence] = {}
        for polarity in ("positive", "negative"):
            six_cells[evidence][polarity] = {
                "items": 4,
                "accuracy": 0.5,
                "answer_distribution": {
                    "Supported": 2,
                    "Contradicted": 1,
                    "Not enough evidence": 1,
                },
            }
    (study2 / "toy_combined_score_summary.json").write_text(
        json.dumps({"by_evidence_state_and_claim_polarity": six_cells}),
        encoding="utf-8",
    )
    (study2 / "empty_v1_vs_v2.json").write_text("", encoding="utf-8")

    rows = [
        _scored_row("x-present", "set-x", "present", ["A"], ["A"], "valid_options"),
        _scored_row("x-absent", "set-x", "absent", ["B"], ["B"], "valid_options"),
        _scored_row("y-present", "set-y", "present", [], [], "valid_none"),
        _scored_row(
            "y-absent",
            "set-y",
            "absent",
            ["A", "B"],
            ["A", "B"],
            "valid_options",
        ),
        _scored_row(
            "bad",
            "set-bad",
            "present",
            ["A"],
            None,
            "invalid_format",
            raw_response="Yes, it is present.",
        ),
    ]
    _write_jsonl(study3 / "toy_g1_scored.jsonl", rows)
    _write_jsonl(
        study3 / "toy_g1_raw.jsonl",
        [
            {
                "item_id": "private-id",
                "raw_response": "A",
                "runtime": {"status": "success", "latency_seconds": 2.0},
                "generation_config": {"max_new_tokens": 8, "do_sample": False},
            }
        ],
    )
    (study3 / "toy_g1_g2_score_summary.json").write_text(
        json.dumps(
            {
                "overall": {
                    "items": 5,
                    "valid_items": 4,
                    "invalid_items": 1,
                    "invalid_rate": 0.2,
                    "exact_set_accuracy": 0.5,
                    "mean_hamming_accuracy": 0.75,
                    "micro_hamming_accuracy": 0.75,
                    "micro_precision": 0.8,
                    "micro_recall": 0.8,
                    "micro_f1": 0.8,
                    "none_accuracy": 1.0,
                    "true_positive": 4,
                    "false_positive": 1,
                    "false_negative": 1,
                    "true_negative": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    (study3 / "missing_g2_scored.jsonl").write_text("", encoding="utf-8")

    report = analyze_result_directories(
        study2_dir=study2,
        study3_dir=study3,
        output_dir=output,
    )

    assert report["source_status"]["empty_files"] == 2
    study2_rows = _read_csv(output / "study2_answer_distribution.csv")
    assert len(study2_rows) == 6
    assert study2_rows[0]["supported_count"] == "2"
    summary_rows = _read_csv(output / "study3_summary_metrics.csv")
    assert summary_rows[0]["end_to_end_exact_set_accuracy"] == "0.4"
    assert summary_rows[0]["selection_gap"] == "0.0"

    metrics = _read_csv(output / "study3_group_metrics.csv")
    data_rows = [row for row in metrics if row["granularity"] != "__data_quality__"]
    assert sum(int(row["invalid_items"]) for row in data_rows) == 1

    invalid = _read_csv(output / "study3_invalid_shape_distribution.csv")
    assert invalid[0]["response_shape"] == "prose_or_other"
    pairs = _read_csv(output / "study3_pair_consistency.csv")
    assert pairs[0]["exact_pair_consistency"] == "1.0"

    for path in output.iterdir():
        text = path.read_text(encoding="utf-8-sig")
        assert "private-id" not in text
        assert "Yes, it is present." not in text
        assert "raw_response" not in text
        assert "item_id" not in text


def _scored_row(
    item_id: str,
    option_set_id: str,
    relation: str,
    gold: list[str],
    predicted: list[str] | None,
    parse_status: str,
    *,
    raw_response: str = "",
) -> dict:
    option_ids = ["A", "B"]
    valid = predicted is not None
    predicted_set = set(predicted or [])
    gold_set = set(gold)
    tp = len(gold_set & predicted_set) if valid else 0
    fp = len(predicted_set - gold_set) if valid else 0
    fn = len(gold_set - predicted_set) if valid else 0
    tn = len(set(option_ids) - gold_set - predicted_set) if valid else 0
    return {
        "item_id": item_id,
        "option_set_id": option_set_id,
        "granularity": "G1_finding_existence",
        "prompt_framing": "state",
        "query_relation": relation,
        "variant": "natural",
        "option_count": 2,
        "options": [{"option_id": value, "text": "private"} for value in option_ids],
        "gold_selected_options": gold,
        "parsed_selected_options": predicted,
        "parse_status": parse_status,
        "raw_response": raw_response,
        "valid_response": valid,
        "exact_set_correct": valid and gold_set == predicted_set,
        "hamming_accuracy": (tp + tn) / 2 if valid else None,
        "jaccard": (
            len(gold_set & predicted_set) / len(gold_set | predicted_set)
            if valid and gold_set | predicted_set
            else (1.0 if valid else None)
        ),
        "set_f1": (
            2 * tp / (2 * tp + fp + fn)
            if valid and 2 * tp + fp + fn
            else (1.0 if valid else None)
        ),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "gold_is_none": not gold,
        "predicted_is_none": not predicted if valid else None,
        "none_correct": not predicted if valid and not gold else None,
        "option_results": [
            {
                "option_id": option,
                "position": index + 1,
                "gold_selected": option in gold_set,
                "predicted_selected": option in predicted_set if valid else None,
                "correct": (
                    (option in gold_set) == (option in predicted_set)
                    if valid
                    else None
                ),
            }
            for index, option in enumerate(option_ids)
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))
