import json
from pathlib import Path

from ghm.evaluation.validate_run import validate_run_layers
from ghm.granularity.common import read_jsonl, write_jsonl
from ghm.inference.medgemma_runner import _append_checkpoint, _latest_rows_by_item_id
from ghm.migration.bundle import create_bundle, verify_bundle
from ghm.migration.preflight import collect_preflight


def test_root_data_ignore_does_not_hide_source_package():
    ignore_text = (Path(__file__).parents[1] / ".gitignore").read_text(encoding="utf-8")

    assert "/data/" in ignore_text.splitlines()
    assert "data/" not in ignore_text.splitlines()


def test_bundle_copies_deduplicated_images_and_verifies_sha256(tmp_path):
    data_root = tmp_path / "private-data"
    image = data_root / "data/files/p10/p100/s500/image.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"synthetic-image")
    prompt_path = data_root / "prompts/study2_g1_model_inputs.jsonl"
    metadata_path = data_root / "prompts/study2_g1_eval_metadata.jsonl"
    write_jsonl(
        [
            _prompt("a", "data/files/p10/p100/s500/image.jpg"),
            _prompt("b", "data/files/p10/p100/s500/image.jpg"),
        ],
        prompt_path,
    )
    write_jsonl([_metadata("a"), _metadata("b")], metadata_path)
    bundle = tmp_path / "bundle"

    summary = create_bundle(
        data_root=data_root,
        output_dir=bundle,
        model_input_paths=[prompt_path],
        eval_metadata_paths=[metadata_path],
        git_commit="abc123",
    )

    assert summary["model_input_records"] == 2
    assert summary["unique_images"] == 1
    assert (bundle / "data/files/p10/p100/s500/image.jpg").read_bytes() == b"synthetic-image"
    assert verify_bundle(bundle)["failed_files"] == 0
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert "data_root" not in manifest
    assert "image_paths" not in manifest


def test_bundle_verification_detects_corruption(tmp_path):
    data_root = tmp_path / "private-data"
    image = data_root / "data/files/image.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"original")
    prompt_path = tmp_path / "inputs.jsonl"
    metadata_path = tmp_path / "metadata.jsonl"
    write_jsonl([_prompt("a", "data/files/image.jpg")], prompt_path)
    write_jsonl([_metadata("a")], metadata_path)
    bundle = tmp_path / "bundle"
    create_bundle(
        data_root=data_root,
        output_dir=bundle,
        model_input_paths=[prompt_path],
        eval_metadata_paths=[metadata_path],
        git_commit=None,
    )

    (bundle / "data/files/image.jpg").write_bytes(b"corrupt")

    assert verify_bundle(bundle)["failed_files"] == 1


def test_checkpoint_keeps_latest_attempt_and_is_resume_ready(tmp_path):
    checkpoint = tmp_path / "run.checkpoint.jsonl"
    _append_checkpoint(checkpoint, _raw("a", status="error"))
    _append_checkpoint(checkpoint, _raw("a", status="success"))
    _append_checkpoint(checkpoint, _raw("b", status="success"))

    latest = _latest_rows_by_item_id(checkpoint)

    assert set(latest) == {"a", "b"}
    assert latest["a"]["runtime"]["status"] == "success"
    assert len(read_jsonl(checkpoint)) == 3


def test_run_validation_separates_model_invalids_from_infrastructure_failures():
    prompts = [_prompt("a", "data/files/image.jpg")]
    metadata = [_metadata("a")]
    raw = [_raw("a", status="success")]
    parsed = [{**raw[0], "parse_status": "invalid_format", "parsed_answer": None}]
    scored = [{"item_id": "a", "score": "invalid_response"}]

    result = validate_run_layers(
        model_inputs=prompts,
        eval_metadata=metadata,
        raw=raw,
        parsed=parsed,
        scored=scored,
    )

    assert result["valid"] is True
    assert result["runtime_failures"] == 0
    assert result["invalid_model_responses"] == 1


def test_run_validation_rejects_missing_and_duplicate_rows():
    prompts = [_prompt("a", "data/files/image.jpg")]
    metadata = [_metadata("a")]
    raw = [_raw("a", status="success"), _raw("a", status="success")]

    result = validate_run_layers(
        model_inputs=prompts,
        eval_metadata=metadata,
        raw=raw,
        parsed=[raw[0]],
        scored=[],
    )

    assert result["valid"] is False
    assert "duplicate_item_id:raw" in result["failures"]
    assert "item_id_set_mismatch:scored" in result["failures"]


def test_preflight_fails_closed_when_model_is_absent(tmp_path):
    data_root = tmp_path / "bundle"
    data_root.mkdir()

    result = collect_preflight(data_root=data_root, model_path=tmp_path / "missing-model")

    assert result["ready"] is False
    assert "model_directory_missing" in result["failures"]
    assert "data_root" not in result
    assert "model_path" not in result


def _prompt(item_id, image_path):
    return {
        "item_id": item_id,
        "image_path": image_path,
        "prompt_template_id": "claim_verification_abc_v1",
        "prompt": "Return only A, B, or C.",
    }


def _metadata(item_id):
    return {
        "item_id": item_id,
        "answer_label": "Not enough evidence",
        "granularity": "G1_finding_existence",
    }


def _raw(item_id, *, status):
    return {
        "item_id": item_id,
        "generation_config": {"seed": 42, "temperature": 0.0},
        "runtime": {"status": status},
    }
