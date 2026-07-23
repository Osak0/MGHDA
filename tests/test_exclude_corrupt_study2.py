from pathlib import Path

from ghm.data.exclude_corrupt_study2 import exclude_corrupt_study2
from ghm.granularity.common import read_jsonl, write_jsonl


def _study2_rows(image_path: str, split: str):
    inputs = []
    metadata = []
    for index, polarity in enumerate(("positive", "negative")):
        item_id = f"{split}-{index}"
        inputs.append(
            {
                "item_id": item_id,
                "image_path": image_path,
                "prompt_template_id": "synthetic",
                "prompt": "synthetic",
            }
        )
        metadata.append(
            {
                "item_id": item_id,
                "granularity": split.upper(),
                "target_finding": "finding",
                "target_anatomy": None,
                "evidence_state": "affirmed",
                "claim_polarity": polarity,
            }
        )
    return inputs, metadata


def test_exclusion_removes_same_complete_pair_from_v1_v2(tmp_path):
    root = tmp_path / "data"
    bad_image = "files/p00/p00000001/s00000001/bad.jpg"
    good_image = "files/p00/p00000002/s00000002/good.jpg"
    bad_list = tmp_path / "bad.txt"
    bad_list.write_text(bad_image + "\n", encoding="utf-8")

    for version in ("v1", "v2"):
        ablation = root / f"processed/study2/ablation/{version}"
        ablation.mkdir(parents=True, exist_ok=True)
        for split in ("g1", "g2"):
            inputs, metadata = _study2_rows(good_image, split)
            write_jsonl(inputs, ablation / f"{split}_model_inputs.jsonl")
            write_jsonl(metadata, ablation / f"{split}_eval_metadata.jsonl")

            full = root / f"processed/study2/{version}"
            full.mkdir(parents=True, exist_ok=True)
            image = bad_image if split == "g1" else good_image
            inputs, metadata = _study2_rows(image, split)
            write_jsonl(inputs, full / f"study2_{split}_model_inputs.jsonl")
            write_jsonl(metadata, full / f"study2_{split}_eval_metadata.jsonl")

    prompts = root / "processed/study3/v2/prompts"
    smoke = root / "processed/study3/v2/smoke"
    prompts.mkdir(parents=True)
    smoke.mkdir(parents=True)
    for split in ("g1", "g2"):
        write_jsonl(
            [{"item_id": split, "image_path": good_image}],
            prompts / f"study3_{split}_multiselect_model_inputs.jsonl",
        )
    write_jsonl(
        [{"item_id": "smoke", "image_path": good_image}],
        smoke / "model_inputs.jsonl",
    )

    summary = exclude_corrupt_study2(
        data_root=root,
        repair_list=bad_list,
        apply=True,
    )

    assert summary["study2"]["v1_g1"]["removed"] == 2
    assert summary["study2"]["v2_g1"]["removed"] == 2
    assert summary["study2"]["v1_g2"]["removed"] == 0
    assert summary["study3_bad_image_records"] == {
        "full_g1": 0,
        "full_g2": 0,
        "smoke": 0,
    }
    assert read_jsonl(
        root / "processed/study2/v1/study2_g1_model_inputs.jsonl"
    ) == []
    assert (
        root / "outputs/audits/corrupt_image_exclusion_summary.json"
    ).is_file()
