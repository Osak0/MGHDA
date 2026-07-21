from pathlib import Path

import pytest

from ghm.granularity.study2 import (
    ANSWER_SUPPORTED,
    CLAIM_POSITIVE,
    EVIDENCE_AFFIRMED,
    PROMPT_TEMPLATE_ID,
    QUESTION_TYPE,
)
from ghm.inference.medgemma_runner import (
    _validate_resumed_row,
    build_error_row,
    dry_run_rows,
    metadata_by_item_id,
    resolve_image_path,
)


def test_resolve_image_path_maps_relative_prompt_path_to_data_root(tmp_path):
    data_root = tmp_path / "data"

    resolved = resolve_image_path("files/p10/p100/s500/dicom.jpg", data_root)

    assert resolved == data_root / "files/p10/p100/s500/dicom.jpg"


def test_resolve_image_path_keeps_absolute_path(tmp_path):
    absolute = tmp_path / "image.jpg"

    assert resolve_image_path(absolute, tmp_path / "root") == absolute


def test_dry_run_counts_existing_images_and_metadata(tmp_path):
    data_root = tmp_path / "data"
    image_path = data_root / "files/p10/p100/s500/dicom.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"synthetic")
    prompts = [
        {
            "item_id": "a",
            "image_path": "files/p10/p100/s500/dicom.jpg",
            "prompt_template_id": PROMPT_TEMPLATE_ID,
            "prompt": "Question",
        },
        {
            "item_id": "b",
            "image_path": "files/missing.jpg",
            "prompt_template_id": PROMPT_TEMPLATE_ID,
            "prompt": "Question",
        },
    ]
    metadata = [
        {
            "item_id": "a",
            "answer_label": ANSWER_SUPPORTED,
            "granularity": "G1_finding_existence",
            "question_type": QUESTION_TYPE,
            "hallucination_probe": None,
            "claim_polarity": CLAIM_POSITIVE,
            "evidence_state": EVIDENCE_AFFIRMED,
        }
    ]

    rows, summary = dry_run_rows(prompts, metadata, data_root=data_root)

    assert summary == {
        "prompt_records": 2,
        "eval_metadata_records": 1,
        "duplicate_prompt_item_ids": 0,
        "duplicate_metadata_item_ids": 0,
        "unexpected_metadata": 0,
        "missing_metadata": 1,
        "missing_image_path": 0,
        "existing_images": 1,
        "missing_images": 1,
    }
    assert rows[0]["image_exists"] is True
    assert rows[0]["has_eval_metadata"] is True
    assert rows[1]["image_exists"] is False
    assert rows[1]["has_eval_metadata"] is False


def test_dry_run_reports_duplicate_and_unexpected_metadata_ids(tmp_path):
    image = tmp_path / "files/image.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"synthetic")
    prompts = [
        {"item_id": "a", "image_path": "files/image.jpg"},
        {"item_id": "a", "image_path": "files/image.jpg"},
    ]
    metadata = [{"item_id": "a"}, {"item_id": "b"}, {"item_id": "b"}]

    _, summary = dry_run_rows(prompts, metadata, data_root=tmp_path)

    assert summary["duplicate_prompt_item_ids"] == 1
    assert summary["duplicate_metadata_item_ids"] == 1
    assert summary["unexpected_metadata"] == 1


def test_error_row_preserves_eval_metadata_but_not_model_answer_in_prompt():
    record = {
        "item_id": "a",
        "image_path": "files/missing.jpg",
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "prompt": "Question",
    }
    metadata = {
        "answer_label": ANSWER_SUPPORTED,
        "granularity": "G1_finding_existence",
        "question_type": QUESTION_TYPE,
        "hallucination_probe": None,
        "claim_polarity": CLAIM_POSITIVE,
        "evidence_state": EVIDENCE_AFFIRMED,
    }

    row = build_error_row(
        record,
        metadata,
        model_name="medgemma",
        model_version="local",
        generation_config={"device": "cuda"},
        error_type="image_not_found",
        error_message="missing",
    )

    assert "answer_label" not in record
    assert row["answer_label"] == ANSWER_SUPPORTED
    assert row["model_id"] == "medgemma"
    assert row["claim_polarity"] == CLAIM_POSITIVE
    assert row["evidence_state"] == EVIDENCE_AFFIRMED
    assert row["raw_response"] == ""
    assert row["runtime"]["status"] == "error"


def test_metadata_by_item_id_ignores_rows_without_item_id():
    rows = [{"item_id": "a", "answer_label": "Yes"}, {"answer_label": "No"}]

    assert metadata_by_item_id(rows) == {"a": {"item_id": "a", "answer_label": "Yes"}}


def test_runner_requires_local_only_model_loading():
    source = (Path(__file__).parents[1] / "src/ghm/inference/medgemma_runner.py").read_text(
        encoding="utf-8"
    )

    assert source.count("local_files_only=True") == 1
    assert '"local_files_only": True' in source


def test_resume_rejects_another_model_or_prompt_template():
    config = {"seed": 42, "temperature": 0.0}
    previous = {
        "model_id": "google/medgemma-4b-it",
        "prompt_template_id": "claim_verification_abc_v1",
        "generation_config": config,
    }
    current = {"prompt_template_id": "claim_verification_abc_v1"}

    _validate_resumed_row(
        previous,
        current,
        model_name="google/medgemma-4b-it",
        generation_config=config,
    )
    with pytest.raises(ValueError, match="model identity"):
        _validate_resumed_row(
            previous,
            current,
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            generation_config=config,
        )
    with pytest.raises(ValueError, match="prompt template"):
        _validate_resumed_row(
            previous,
            {"prompt_template_id": "claim_verification_abc_definitions_v2"},
            model_name="google/medgemma-4b-it",
            generation_config=config,
        )
