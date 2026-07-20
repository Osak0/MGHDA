import json

import pytest

from ghm.granularity.common import write_jsonl
from ghm.study3.linking import build_mimic_relative_path, normalize_id
from ghm.study3.transfer import (
    CHECKSUM_NAME,
    SUMMARY_NAME,
    create_transfer_manifest,
    verify_transfer,
)
from ghm.study3.validation import validate_study3_run


def test_transfer_manifest_is_study3_scoped_and_verifiable(tmp_path):
    data_root = tmp_path / "data"
    prompt_dir = data_root / "processed" / "study3" / "prompts"
    image = data_root / "files" / "p10" / "s1" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"synthetic-image")
    model_path = prompt_dir / "study3_g1_multiselect_model_inputs.jsonl"
    metadata_path = prompt_dir / "study3_g1_multiselect_eval_metadata.jsonl"
    write_jsonl([_model_input()], model_path)
    write_jsonl([_metadata()], metadata_path)
    output_dir = data_root / "outputs" / "study3" / "transfer"

    summary = create_transfer_manifest(
        data_root=data_root,
        output_dir=output_dir,
        model_input_paths=[model_path],
        eval_metadata_paths=[metadata_path],
        git_commit="abc123",
    )
    verified = verify_transfer(
        data_root=data_root,
        checksum_path=output_dir / CHECKSUM_NAME,
        summary_path=output_dir / SUMMARY_NAME,
        expected_git_commit="abc123",
    )

    assert summary["experiment_id"] == "study3_multiselect_v1"
    assert summary["model_input_records"] == 1
    assert verified["status"] == "pass"


def test_study3_linking_uses_official_portable_hierarchy():
    path = build_mimic_relative_path(
        subject_id="10000032",
        study_id="50414267",
        dicom_id="abc",
    )

    assert path.as_posix() == "p10/p10000032/s50414267/abc.jpg"
    assert normalize_id(10000032.0) == "10000032"


def test_transfer_rejects_study2_record(tmp_path):
    data_root = tmp_path / "data"
    prompt_dir = data_root / "processed" / "study3" / "prompts"
    image = data_root / "files" / "p10" / "s1" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"synthetic-image")
    model = _model_input()
    model["item_id"] = "study2_bad"
    model_path = prompt_dir / "study3_model_inputs.jsonl"
    metadata_path = prompt_dir / "study3_eval_metadata.jsonl"
    write_jsonl([model], model_path)
    write_jsonl([_metadata()], metadata_path)

    with pytest.raises(ValueError, match="study3_"):
        create_transfer_manifest(
            data_root=data_root,
            output_dir=data_root / "outputs" / "study3" / "transfer",
            model_input_paths=[model_path],
            eval_metadata_paths=[metadata_path],
            git_commit=None,
        )


def test_run_validation_detects_gold_leakage_and_id_alignment(tmp_path):
    image = tmp_path / "files" / "p10" / "s1" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"x")
    model = _model_input()
    metadata = _metadata()

    passed = validate_study3_run([model], [metadata], data_root=tmp_path)
    leaked = validate_study3_run(
        [{**model, "gold_selected_options": ["A"]}],
        [metadata],
        data_root=tmp_path,
    )

    assert passed["status"] == "pass"
    assert leaked["status"] == "fail"
    assert leaked["leaked_model_rows"] == 1


def _model_input():
    return {
        "item_id": "study3_g1_ms_test",
        "experiment_id": "study3_multiselect_v1",
        "image_path": "files/p10/s1/a.jpg",
        "prompt_template_id": "study3_multiselect_present_v1",
        "prompt": "synthetic",
    }


def _metadata():
    return {
        "item_id": "study3_g1_ms_test",
        "experiment_id": "study3_multiselect_v1",
        "granularity": "G1_finding_existence",
        "question_type": "anatomicalfinding_multiselect_v1",
        "query_relation": "present",
        "variant": "natural",
        "controlled_k": None,
        "anchor_id": "study3_g1_anchor_test",
        "option_set_id": "study3_option_set_test",
        "option_count": 2,
        "natural_option_count": 2,
        "answer_composition": "mixed",
        "options": [
            {"option_id": "A", "label_name": "lung opacity"},
            {"option_id": "B", "label_name": "pneumothorax"},
        ],
        "gold_selected_options": ["A"],
        "target_anatomy": None,
        "source_assertions": [],
        "evidence_sources": ["E2_structured_label"],
    }
