import tarfile

from ghm.granularity.common import write_jsonl
from ghm.migration.archive import inspect_archive, verify_outer_checksum
from ghm.migration.bundle import verify_checksum_file
from ghm.migration.joint_bundle import create_private_archive
from ghm.study3.smoke import select_smoke_rows


def test_combined_private_archive_synthetic_end_to_end(tmp_path):
    data_root = tmp_path / "data"
    image = data_root / "files" / "synthetic.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"synthetic-image")

    study2_metadata = []
    for granularity, short in (
        ("G1_finding_existence", "g1"),
        ("G2_anatomical_localization", "g2"),
    ):
        metadata_rows = []
        for index in range(480):
            item_id = f"study2_{short}_{index:03d}"
            evidence = ("affirmed", "negated", "not_enough_evidence")[
                (index // 20) % 3
            ]
            polarity = ("positive", "negative")[(index // 10) % 2]
            metadata_rows.append(
                {
                    "item_id": item_id,
                    "answer_label": "A",
                    "granularity": granularity,
                    "evidence_state": evidence,
                    "claim_polarity": polarity,
                }
            )
        study2_metadata.extend(metadata_rows)
        for version, template in (
            ("v1", "claim_verification_abc_v1"),
            ("v2", "claim_verification_abc_definitions_v2"),
        ):
            root = data_root / "processed" / "study2" / version
            write_jsonl(
                [_model(row["item_id"], template) for row in metadata_rows],
                root / f"study2_{short}_model_inputs.jsonl",
            )
            write_jsonl(metadata_rows, root / f"study2_{short}_eval_metadata.jsonl")

    ablation_metadata = []
    for granularity in ("G1_finding_existence", "G2_anatomical_localization"):
        for evidence in ("affirmed", "negated", "not_enough_evidence"):
            for polarity in ("positive", "negative"):
                candidates = [
                    row
                    for row in study2_metadata
                    if row["granularity"] == granularity
                    and row["evidence_state"] == evidence
                    and row["claim_polarity"] == polarity
                ]
                ablation_metadata.extend(candidates[:10])
    for version, template in (
        ("v1", "claim_verification_abc_v1"),
        ("v2", "claim_verification_abc_definitions_v2"),
    ):
        root = data_root / "processed" / "study2" / "ablation" / version
        write_jsonl(
            [_model(row["item_id"], template) for row in ablation_metadata],
            root / "model_inputs.jsonl",
        )
        write_jsonl(ablation_metadata, root / "eval_metadata.jsonl")
        for short, granularity in (
            ("g1", "G1_finding_existence"),
            ("g2", "G2_anatomical_localization"),
        ):
            split_metadata = [
                row for row in ablation_metadata if row["granularity"] == granularity
            ]
            write_jsonl(
                [_model(row["item_id"], template) for row in split_metadata],
                root / f"{short}_model_inputs.jsonl",
            )
            write_jsonl(split_metadata, root / f"{short}_eval_metadata.jsonl")

    all_study3_models = []
    all_study3_metadata = []
    all_study3_items = []
    for granularity, short in (
        ("G1_finding_existence", "g1"),
        ("G2_anatomical_localization", "g2"),
    ):
        models = []
        metadata = []
        items = []
        for set_index in range(450):
            variant = "natural" if set_index < 250 else "controlled"
            controlled_k = None if variant == "natural" else 2 + (set_index % 4)
            option_set_id = f"study3_{short}_set_{set_index:03d}"
            for framing in ("state", "evidence"):
                for relation in ("present", "absent"):
                    item_id = f"{option_set_id}_{framing}_{relation}"
                    model = _model(
                        item_id,
                        f"study3_multiselect_{framing}_{relation}_v2",
                        experiment_id="study3_multiselect_v2",
                    )
                    row = {
                        "item_id": item_id,
                        "experiment_id": "study3_multiselect_v2",
                        "granularity": granularity,
                        "prompt_framing": framing,
                        "query_relation": relation,
                        "variant": variant,
                        "controlled_k": controlled_k,
                        "option_set_id": option_set_id,
                        "options": [
                            {"option_id": "A", "label_name": "finding a"},
                            {"option_id": "B", "label_name": "finding b"},
                        ],
                        "gold_selected_options": ["A"] if relation == "present" else ["B"],
                    }
                    models.append(model)
                    metadata.append(row)
                    items.append(row)
        root = data_root / "processed" / "study3" / "v2"
        write_jsonl(
            models,
            root / "prompts" / f"study3_{short}_multiselect_model_inputs.jsonl",
        )
        write_jsonl(
            metadata,
            root / "prompts" / f"study3_{short}_multiselect_eval_metadata.jsonl",
        )
        write_jsonl(
            items,
            root / "items" / f"study3_{short}_multiselect_items.jsonl",
        )
        all_study3_models.extend(models)
        all_study3_metadata.extend(metadata)
        all_study3_items.extend(items)
    smoke_inputs, smoke_metadata, _ = select_smoke_rows(
        all_study3_models,
        all_study3_metadata,
    )
    write_jsonl(
        smoke_inputs,
        data_root / "processed/study3/v2/smoke/model_inputs.jsonl",
    )
    write_jsonl(
        smoke_metadata,
        data_root / "processed/study3/v2/smoke/eval_metadata.jsonl",
    )

    output_dir = tmp_path / "release"
    summary = create_private_archive(
        data_root=data_root,
        output_dir=output_dir,
        git_commit="abc123",
    )
    archive = output_dir / "MGHDA-private-study2v1v2-study3v2-abc123.tar.gz"
    sidecar = output_dir / f"{archive.name}.sha256"

    assert summary["unique_images"] == 1
    assert inspect_archive(archive)["files"] == summary["files"]
    assert verify_outer_checksum(archive, sidecar)
    extracted = tmp_path / "extracted"
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(extracted, filter="data")
    assert verify_checksum_file(
        data_root=extracted,
        checksum_path=extracted / "outputs/transfer/private_files.sha256",
    )["failed_files"] == 0


def _model(item_id, template_id, *, experiment_id=None):
    row = {
        "item_id": item_id,
        "image_path": "files/synthetic.jpg",
        "prompt_template_id": template_id,
        "prompt": f"synthetic prompt {template_id}",
    }
    if experiment_id:
        row["experiment_id"] = experiment_id
    return row
