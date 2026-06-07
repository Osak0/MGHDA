from pathlib import Path

import pytest

from ghm.evaluation.metrics import summarize_scored_files
from ghm.evaluation.parse_answer import parse_closed_answer, parse_response_rows
from ghm.evaluation.score_closed_qa import score_closed_qa_rows, summarize_scores
from ghm.granularity.common import read_jsonl, write_jsonl
from ghm.inference.batch_infer import build_mock_outputs
from ghm.prompts.build_prompts import build_prompt_layers
from ghm.prompts.templates import (
    SUPPORTED_UNSUPPORTED_TEMPLATE_ID,
    YES_NO_UNCERTAIN_TEMPLATE_ID,
)


def test_prompt_builder_splits_model_inputs_and_eval_metadata():
    items = [
        _linked_item("a", answer_label="Yes", hallucination_probe="H1"),
        _linked_item("b", answer_label="Unsupported", hallucination_probe="H2"),
        _linked_item("c", answer_label="No", image_path=None),
    ]

    model_inputs, eval_metadata, summary = build_prompt_layers(items)

    assert summary == {
        "input_items": 3,
        "model_input_records": 2,
        "eval_metadata_records": 2,
        "skipped_missing_image_path": 1,
        "skipped_missing_question": 0,
    }
    assert "answer_label" not in model_inputs[0]
    assert "answer_label" not in model_inputs[1]
    assert model_inputs[0]["prompt_template_id"] == YES_NO_UNCERTAIN_TEMPLATE_ID
    assert model_inputs[1]["prompt_template_id"] == SUPPORTED_UNSUPPORTED_TEMPLATE_ID
    assert eval_metadata[0]["answer_label"] == "Yes"
    assert eval_metadata[1]["answer_label"] == "Unsupported"


def test_mock_inference_parse_and_score_oracle_pipeline():
    items = [
        _linked_item("a", answer_label="Yes", granularity="G1_finding_existence"),
        _linked_item(
            "b",
            answer_label="Unsupported",
            granularity="G2_anatomical_localization",
            hallucination_probe="H2",
        ),
    ]
    model_inputs, eval_metadata, _ = build_prompt_layers(items)

    raw_outputs = build_mock_outputs(model_inputs, eval_metadata, model_name="mock", mode="oracle")
    parsed = parse_response_rows(raw_outputs)
    scored = score_closed_qa_rows(parsed)
    summary = summarize_scores(scored)

    assert len(raw_outputs) == 2
    assert {row["parse_status"] for row in parsed} == {"success"}
    assert {row["score"] for row in scored} == {"correct"}
    assert summary["overall"]["accuracy"] == 1.0
    assert summary["by_hallucination_probe"]["H1"]["items"] == 1
    assert summary["by_hallucination_probe"]["H2"]["items"] == 1
    assert summary["by_answer_label"]["Yes"]["items"] == 1
    assert summary["by_answer_label"]["Unsupported"]["items"] == 1


def test_oracle_mock_requires_eval_metadata():
    with pytest.raises(ValueError, match="requires eval metadata"):
        build_mock_outputs([{"item_id": "a", "image_path": "x", "prompt": "q"}], mode="oracle")


def test_parser_handles_all_closed_answer_spaces():
    assert parse_closed_answer("Yes") == ("Yes", "success")
    assert parse_closed_answer("No.") == ("No", "success")
    assert parse_closed_answer("Uncertain") == ("Uncertain", "success")
    assert parse_closed_answer("Supported") == ("Supported", "success")
    assert parse_closed_answer("Unsupported") == ("Unsupported", "success")
    assert parse_closed_answer("Yes or No") == (None, "multiple_answers")
    assert parse_closed_answer("maybe") == (None, "invalid_format")
    assert parse_closed_answer("") == (None, "empty_response")


def test_scoring_maps_h1_h2_and_invalid_cases():
    rows = [
        {
            "item_id": "a",
            "model_name": "mock",
            "granularity": "G1_finding_existence",
            "question_type": "h1_yes_no_qa",
            "hallucination_probe": "H1",
            "answer_label": "No",
            "parsed_answer": "Yes",
            "parse_status": "success",
        },
        {
            "item_id": "b",
            "model_name": "mock",
            "granularity": "G1_finding_existence",
            "question_type": "h1_yes_no_qa",
            "hallucination_probe": "H1",
            "answer_label": "Yes",
            "parsed_answer": "No",
            "parse_status": "success",
        },
        {
            "item_id": "c",
            "model_name": "mock",
            "granularity": "G2_anatomical_localization",
            "question_type": "h2_claim_support",
            "hallucination_probe": "H2",
            "answer_label": "Unsupported",
            "parsed_answer": "Supported",
            "parse_status": "success",
        },
        {
            "item_id": "d",
            "model_name": "mock",
            "granularity": "G2_anatomical_localization",
            "question_type": "h1_yes_no_qa",
            "hallucination_probe": "H1",
            "answer_label": "Yes",
            "parsed_answer": None,
            "parse_status": "invalid_format",
        },
    ]

    scored = score_closed_qa_rows(rows)
    summary = summarize_scores(scored)

    assert scored[0]["score"] == "H1_evidence_contradicted"
    assert scored[0]["hallucination_type"] == "H1"
    assert scored[0]["h1_error_direction"] == "false_positive"
    assert scored[1]["score"] == "H1_evidence_contradicted"
    assert scored[1]["hallucination_type"] == "H1"
    assert scored[1]["h1_error_direction"] == "false_negative"
    assert scored[2]["score"] == "H2_evidence_unsupported"
    assert scored[2]["hallucination_type"] == "H2"
    assert scored[3]["score"] == "invalid_response"
    assert summary["overall"]["h1_count"] == 2
    assert summary["overall"]["h1_false_positive_count"] == 1
    assert summary["overall"]["h1_false_negative_count"] == 1
    assert summary["overall"]["h2_count"] == 1
    assert summary["overall"]["invalid_count"] == 1


def test_jsonl_pipeline_helpers_and_file_summary(tmp_path):
    items = [_linked_item("a", answer_label="Yes")]
    model_inputs, eval_metadata, _ = build_prompt_layers(items)
    raw_outputs = build_mock_outputs(model_inputs, eval_metadata)
    parsed = parse_response_rows(raw_outputs)
    scored = score_closed_qa_rows(parsed)
    scored_path = tmp_path / "scored.jsonl"

    write_jsonl(scored, scored_path)
    loaded = read_jsonl(scored_path)
    summary = summarize_scored_files([scored_path])

    assert loaded == scored
    assert summary["overall"]["items"] == 1
    assert summary["overall"]["accuracy"] == 1.0


def _linked_item(
    item_id,
    *,
    answer_label,
    granularity="G1_finding_existence",
    hallucination_probe="H1",
    image_path="data/files/p10/p10000032/s50000001/dicom-a.jpg",
):
    question = "Is there evidence of opacity in this chest X-ray?"
    if hallucination_probe == "H2":
        question = (
            "Is the following claim supported by the chest X-ray? "
            "Claim: There is evidence of opacity."
        )
    return {
        "item_id": item_id,
        "image_path": image_path,
        "question": question,
        "answer_label": answer_label,
        "granularity": granularity,
        "question_type": "h2_claim_support"
        if hallucination_probe == "H2"
        else "h1_yes_no_qa",
        "hallucination_probe": hallucination_probe,
        "target_finding": "opacity",
        "target_anatomy": None,
        "evidence_sources": ["E2_structured_label"],
    }
