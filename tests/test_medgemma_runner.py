from pathlib import Path

from ghm.inference.medgemma_runner import (
    build_error_row,
    dry_run_rows,
    metadata_by_item_id,
    resolve_image_path,
)


def test_resolve_image_path_maps_relative_prompt_path_to_data_root(tmp_path):
    data_root = tmp_path / "MGHDA-data"

    resolved = resolve_image_path("data/files/p10/p100/s500/dicom.jpg", data_root)

    assert resolved == data_root / "data/files/p10/p100/s500/dicom.jpg"


def test_resolve_image_path_keeps_absolute_path(tmp_path):
    absolute = tmp_path / "image.jpg"

    assert resolve_image_path(absolute, tmp_path / "root") == absolute


def test_dry_run_counts_existing_images_and_metadata(tmp_path):
    data_root = tmp_path / "MGHDA-data"
    image_path = data_root / "data/files/p10/p100/s500/dicom.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"synthetic")
    prompts = [
        {
            "item_id": "a",
            "image_path": "data/files/p10/p100/s500/dicom.jpg",
            "prompt_template_id": "yes_no_uncertain_v1",
            "prompt": "Question",
        },
        {
            "item_id": "b",
            "image_path": "data/files/missing.jpg",
            "prompt_template_id": "yes_no_uncertain_v1",
            "prompt": "Question",
        },
    ]
    metadata = [
        {
            "item_id": "a",
            "answer_label": "Yes",
            "granularity": "G1_finding_existence",
            "question_type": "h1_yes_no_qa",
            "hallucination_probe": "H1",
        }
    ]

    rows, summary = dry_run_rows(prompts, metadata, data_root=data_root)

    assert summary == {
        "prompt_records": 2,
        "missing_metadata": 1,
        "missing_image_path": 0,
        "existing_images": 1,
        "missing_images": 1,
    }
    assert rows[0]["image_exists"] is True
    assert rows[0]["has_eval_metadata"] is True
    assert rows[1]["image_exists"] is False
    assert rows[1]["has_eval_metadata"] is False


def test_error_row_preserves_eval_metadata_but_not_model_answer_in_prompt():
    record = {
        "item_id": "a",
        "image_path": "data/files/missing.jpg",
        "prompt_template_id": "yes_no_uncertain_v1",
        "prompt": "Question",
    }
    metadata = {
        "answer_label": "No",
        "granularity": "G1_finding_existence",
        "question_type": "h1_yes_no_qa",
        "hallucination_probe": "H1",
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
    assert row["answer_label"] == "No"
    assert row["raw_response"] == ""
    assert row["runtime"]["status"] == "error"


def test_metadata_by_item_id_ignores_rows_without_item_id():
    rows = [{"item_id": "a", "answer_label": "Yes"}, {"answer_label": "No"}]

    assert metadata_by_item_id(rows) == {"a": {"item_id": "a", "answer_label": "Yes"}}
