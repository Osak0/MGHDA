from pathlib import Path

import pytest

from ghm.evaluation.metrics import summarize_scored_files
from ghm.evaluation.parse_answer import parse_closed_answer, parse_response_rows
from ghm.evaluation.score_closed_qa import score_closed_qa_rows, summarize_scores
from ghm.granularity.common import read_jsonl, write_jsonl
from ghm.granularity.study2 import (
    ANSWER_CONTRADICTED,
    ANSWER_NOT_ENOUGH,
    ANSWER_SUPPORTED,
    CLAIM_NEGATIVE,
    CLAIM_POSITIVE,
    EVIDENCE_AFFIRMED,
    EVIDENCE_NOT_ENOUGH,
    PROMPT_TEMPLATE_ID,
    QUESTION_TYPE,
)
from ghm.inference.batch_infer import build_mock_outputs
from ghm.prompts.build_prompts import build_prompt_layers


def test_prompt_builder_splits_model_inputs_and_eval_metadata():
    items = [
        _linked_item("a", answer_label=ANSWER_SUPPORTED, claim_polarity=CLAIM_POSITIVE),
        _linked_item("b", answer_label=ANSWER_NOT_ENOUGH, evidence_state=EVIDENCE_NOT_ENOUGH),
        _linked_item("c", answer_label=ANSWER_CONTRADICTED, image_path=None),
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
    assert model_inputs[0]["prompt_template_id"] == PROMPT_TEMPLATE_ID
    assert "A. Supported" in model_inputs[0]["prompt"]
    assert eval_metadata[0]["answer_label"] == ANSWER_SUPPORTED
    assert eval_metadata[0]["claim_polarity"] == CLAIM_POSITIVE
    assert eval_metadata[1]["evidence_state"] == EVIDENCE_NOT_ENOUGH


def test_mock_inference_parse_and_score_oracle_pipeline():
    items = [
        _linked_item(
            "a",
            answer_label=ANSWER_SUPPORTED,
            granularity="G1_finding_existence",
            claim_polarity=CLAIM_POSITIVE,
        ),
        _linked_item(
            "b",
            answer_label=ANSWER_CONTRADICTED,
            granularity="G2_anatomical_localization",
            claim_polarity=CLAIM_NEGATIVE,
        ),
        _linked_item("c", answer_label=ANSWER_NOT_ENOUGH, evidence_state=EVIDENCE_NOT_ENOUGH),
    ]
    model_inputs, eval_metadata, _ = build_prompt_layers(items)

    raw_outputs = build_mock_outputs(model_inputs, eval_metadata, model_name="mock", mode="oracle")
    parsed = parse_response_rows(raw_outputs)
    scored = score_closed_qa_rows(parsed)
    summary = summarize_scores(scored)

    assert len(raw_outputs) == 3
    assert {row["parse_status"] for row in parsed} == {"success"}
    assert {row["score"] for row in scored} == {"correct"}
    assert summary["overall"]["accuracy"] == 1.0
    assert summary["by_claim_polarity"][CLAIM_POSITIVE]["items"] == 2
    assert summary["by_claim_polarity"][CLAIM_NEGATIVE]["items"] == 1
    assert summary["by_evidence_state_and_claim_polarity"][EVIDENCE_AFFIRMED][
        CLAIM_POSITIVE
    ]["items"] == 1
    assert summary["by_evidence_state_and_claim_polarity"][EVIDENCE_AFFIRMED][
        CLAIM_NEGATIVE
    ]["items"] == 1
    assert summary["by_evidence_state_and_claim_polarity"][EVIDENCE_NOT_ENOUGH][
        CLAIM_POSITIVE
    ]["items"] == 1
    assert summary["by_granularity_evidence_state_and_claim_polarity"][
        "G1_finding_existence"
    ][EVIDENCE_AFFIRMED][CLAIM_POSITIVE]["items"] == 1
    assert summary["by_answer_label"][ANSWER_NOT_ENOUGH]["items"] == 1
    assert summary["classification_metrics"]["macro_f1"] == 1.0
    assert summary["confusion_matrix"][ANSWER_SUPPORTED][ANSWER_SUPPORTED] == 1
    assert summary["confusion_matrix"][ANSWER_CONTRADICTED][ANSWER_CONTRADICTED] == 1
    assert summary["confusion_matrix"][ANSWER_NOT_ENOUGH][ANSWER_NOT_ENOUGH] == 1


def test_oracle_mock_requires_eval_metadata():
    with pytest.raises(ValueError, match="requires eval metadata"):
        build_mock_outputs([{"item_id": "a", "image_path": "x", "prompt": "q"}], mode="oracle")


def test_parser_handles_study2_answer_space():
    assert parse_closed_answer("A") == (ANSWER_SUPPORTED, "success")
    assert parse_closed_answer("A. Supported") == (ANSWER_SUPPORTED, "success")
    assert parse_closed_answer("B") == (ANSWER_CONTRADICTED, "success")
    assert parse_closed_answer("Contradicted.") == (ANSWER_CONTRADICTED, "success")
    assert parse_closed_answer("C") == (ANSWER_NOT_ENOUGH, "success")
    assert parse_closed_answer("Not enough evidence") == (ANSWER_NOT_ENOUGH, "success")
    assert parse_closed_answer("A or B") == (None, "multiple_answers")
    assert parse_closed_answer("maybe") == (None, "invalid_format")
    assert parse_closed_answer("") == (None, "empty_response")


def test_scoring_maps_study2_errors_and_invalid_cases():
    rows = [
        _parsed_row("a", answer_label=ANSWER_CONTRADICTED, parsed_answer=ANSWER_SUPPORTED),
        _parsed_row("b", answer_label=ANSWER_NOT_ENOUGH, parsed_answer=ANSWER_SUPPORTED),
        _parsed_row("c", answer_label=ANSWER_SUPPORTED, parsed_answer=ANSWER_NOT_ENOUGH),
        _parsed_row("d", answer_label=ANSWER_SUPPORTED, parsed_answer=None, parse_status="invalid_format"),
    ]

    scored = score_closed_qa_rows(rows)
    summary = summarize_scores(scored)

    assert scored[0]["score"] == "H1_evidence_contradicted"
    assert scored[0]["hallucination_type"] == "H1"
    assert scored[1]["score"] == "H2_evidence_unsupported"
    assert scored[1]["hallucination_type"] == "H2"
    assert scored[2]["score"] == "incorrect_non_hallucination"
    assert scored[3]["score"] == "invalid_response"
    assert summary["overall"]["h1_count"] == 1
    assert summary["overall"]["h2_count"] == 1
    assert summary["overall"]["invalid_count"] == 1
    assert summary["classification_metrics"]["per_label"][ANSWER_CONTRADICTED][
        "recall"
    ] == 0.0
    assert summary["confusion_matrix"][ANSWER_SUPPORTED]["INVALID_OR_UNPARSED"] == 1


def test_jsonl_pipeline_helpers_and_file_summary(tmp_path):
    items = [_linked_item("a", answer_label=ANSWER_SUPPORTED)]
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
    assert summary["data_quality"] == {
        "input_files": 1,
        "records": 1,
        "unique_item_ids": 1,
        "missing_item_id_count": 0,
        "duplicate_item_id_count": 0,
        "null_field_counts": {
            "item_id": 0,
            "granularity": 0,
            "claim_polarity": 0,
            "evidence_state": 0,
            "answer_label": 0,
            "parsed_answer": 0,
            "score": 0,
            "is_correct": 0,
        },
        "unexpected_answer_label_count": 0,
        "unexpected_parsed_answer_count": 0,
    }


def _linked_item(
    item_id,
    *,
    answer_label,
    granularity="G1_finding_existence",
    claim_polarity=CLAIM_POSITIVE,
    evidence_state=EVIDENCE_AFFIRMED,
    image_path="files/p10/p10000032/s50000001/dicom-a.jpg",
):
    return {
        "item_id": item_id,
        "image_path": image_path,
        "question": (
            "Evaluate the following medical claim based only on the visible "
            "radiographic evidence. Claim: There is evidence of opacity in this chest X-ray."
        ),
        "answer_label": answer_label,
        "granularity": granularity,
        "question_type": QUESTION_TYPE,
        "hallucination_probe": None,
        "claim_polarity": claim_polarity,
        "evidence_state": evidence_state,
        "target_finding": "opacity",
        "target_anatomy": None,
        "evidence_sources": ["E2_structured_label"],
    }


def _parsed_row(
    item_id,
    *,
    answer_label,
    parsed_answer,
    parse_status="success",
):
    return {
        "item_id": item_id,
        "model_name": "mock",
        "granularity": "G1_finding_existence",
        "question_type": QUESTION_TYPE,
        "claim_polarity": CLAIM_POSITIVE,
        "evidence_state": EVIDENCE_AFFIRMED,
        "answer_label": answer_label,
        "parsed_answer": parsed_answer,
        "parse_status": parse_status,
    }
