from itertools import product
from pathlib import Path

from ghm.data.exclude_corrupt_study3 import exclude_corrupt_study3
from ghm.granularity.common import read_jsonl, write_jsonl


def _option_set(prefix: str, image_path: str):
    inputs = []
    metadata = []
    items = []
    for framing, relation in product(("state", "evidence"), ("present", "absent")):
        item_id = f"{prefix}-{framing}-{relation}"
        model_row = {"item_id": item_id, "image_path": image_path}
        metadata_row = {
            "item_id": item_id,
            "option_set_id": prefix,
            "prompt_framing": framing,
            "query_relation": relation,
            "variant": "natural",
            "controlled_k": None,
        }
        inputs.append(model_row)
        metadata.append(metadata_row)
        items.append({**model_row, **metadata_row})
    return inputs, metadata, items


def test_exclusion_removes_complete_four_question_set(tmp_path):
    root = tmp_path / "data"
    bad_image = "files/p00/p00000001/s00000001/bad.jpg"
    good_image = "files/p00/p00000002/s00000002/good.jpg"
    bad_list = tmp_path / "bad.txt"
    bad_list.write_text(bad_image + "\n", encoding="utf-8")
    prompts = root / "processed/study3/v2/prompts"
    items_dir = root / "processed/study3/v2/items"
    smoke = root / "processed/study3/v2/smoke"
    prompts.mkdir(parents=True)
    items_dir.mkdir(parents=True)
    smoke.mkdir(parents=True)

    smoke_inputs = []
    smoke_metadata = []
    for split, image in (("g1", good_image), ("g2", bad_image)):
        good = _option_set(f"{split}-good", good_image)
        bad = _option_set(f"{split}-bad", image)
        inputs = good[0] + bad[0]
        metadata = good[1] + bad[1]
        items = good[2] + bad[2]
        write_jsonl(
            inputs,
            prompts / f"study3_{split}_multiselect_model_inputs.jsonl",
        )
        write_jsonl(
            metadata,
            prompts / f"study3_{split}_multiselect_eval_metadata.jsonl",
        )
        write_jsonl(
            items,
            items_dir / f"study3_{split}_multiselect_items.jsonl",
        )
        smoke_inputs.extend(good[0][:1])
        smoke_metadata.extend(good[1][:1])
    write_jsonl(smoke_inputs, smoke / "model_inputs.jsonl")
    write_jsonl(smoke_metadata, smoke / "eval_metadata.jsonl")

    summary = exclude_corrupt_study3(
        data_root=root,
        repair_list=bad_list,
        apply=True,
    )

    assert summary["study3"]["g1"]["removed"] == 0
    assert summary["study3"]["g2"]["removed"] == 4
    assert summary["study3"]["g2"]["option_sets_removed"] == 1
    remaining = read_jsonl(
        prompts / "study3_g2_multiselect_model_inputs.jsonl"
    )
    assert len(remaining) == 4
    assert all(row["image_path"] == good_image for row in remaining)
