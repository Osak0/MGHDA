from pathlib import Path

from ghm.data.link_mimic_images import build_image_index_rows
from ghm.data.parse_chest_imagenome import parse_scene_graph_file
from ghm.granularity.build_g1_items import build_g1_h1_items, build_g1_h2_items
from ghm.granularity.build_g2_items import build_g2_h1_items, build_g2_h2_items
from ghm.granularity.common import read_jsonl, write_jsonl


FIXTURE = Path(
    "data/fixtures/0000d3be-591ae3b7-b03a7497-8319c02b-650bb4ab_SceneGraph.json"
)


def _fixture_rows():
    parsed = parse_scene_graph_file(FIXTURE)
    image_index = build_image_index_rows(parsed.object_rows, parsed.assertion_rows)
    return parsed.object_rows, parsed.assertion_rows, image_index


def test_image_index_preserves_identifiers():
    object_rows, assertion_rows, image_index = _fixture_rows()

    assert len(object_rows) == 36
    assert len(assertion_rows) == 22
    assert len(image_index) == 1
    row = image_index[0]
    assert row["patient_id"] == "19298916"
    assert row["study_id"] == "50189753"
    assert row["image_id"] == "0000d3be-591ae3b7-b03a7497-8319c02b-650bb4ab"
    assert row["dicom_id"] == row["image_id"]
    assert row["image_path"] is None


def test_g1_h1_items_from_fixture_preserve_ids_and_exclude_phrase_text():
    _, assertion_rows, image_index = _fixture_rows()

    items, summary = build_g1_h1_items(assertion_rows, image_index)

    assert summary == {
        "candidate_groups": 4,
        "items_written": 4,
        "yes_items": 1,
        "no_items": 3,
        "excluded_conflict_groups": 0,
    }
    pneumothorax = _find_item(items, target_finding="pneumothorax")
    assert pneumothorax["answer_label"] == "No"
    assert pneumothorax["granularity"] == "G1_finding_existence"
    assert pneumothorax["patient_id"] == "19298916"
    assert pneumothorax["study_id"] == "50189753"
    assert pneumothorax["dicom_id"] == pneumothorax["image_id"]
    assert pneumothorax["target_anatomy"] is None
    assert pneumothorax["bbox"] is None
    for assertion in pneumothorax["source_assertions"]:
        assert "phrase" not in assertion

    enlarged = _find_item(items, target_finding="enlarged cardiac silhouette")
    assert enlarged["answer_label"] == "Yes"
    assert enlarged["hallucination_probe"] == "H1"


def test_g1_h2_items_include_supported_and_unsupported_claims():
    _, assertion_rows, image_index = _fixture_rows()

    items, summary = build_g1_h2_items(
        assertion_rows,
        image_index,
        unsupported_per_image=1,
    )

    assert summary["candidate_supported_groups"] == 4
    assert summary["items_written"] == 4
    assert summary["supported_items"] == 4
    assert summary["unsupported_items"] == 0
    assert {item["answer_label"] for item in items} == {"Supported"}
    assert {item["hallucination_probe"] for item in items} == {"H2"}
    assert all("phrase" not in assertion for item in items for assertion in item["source_assertions"])


def test_g1_h2_constructs_unsupported_from_global_vocab_when_finding_unmentioned():
    rows = [
        _assertion("a1", image_id="i1", label_name="opacity", polarity="yes"),
        _assertion("a2", image_id="i2", label_name="effusion", polarity="yes"),
    ]
    image_index = build_image_index_rows([], rows)

    items, summary = build_g1_h2_items(rows, image_index, unsupported_per_image=1)

    assert summary["supported_items"] == 1
    assert summary["unsupported_items"] == 2
    assert {item["answer_label"] for item in items} == {"Supported", "Unsupported"}
    assert summary["h2_balance"]["balance_applied"] is True
    assert summary["h2_balance"]["target_negative_fraction"] == 0.8


def test_g1_h2_balancing_can_be_disabled():
    rows = [
        _assertion("a1", image_id="i1", label_name="opacity", polarity="yes"),
        _assertion("a2", image_id="i2", label_name="effusion", polarity="yes"),
    ]
    image_index = build_image_index_rows([], rows)

    items, summary = build_g1_h2_items(
        rows,
        image_index,
        unsupported_per_image=1,
        balance_h2=False,
    )

    assert summary["supported_items"] == 2
    assert summary["unsupported_items"] == 2
    assert summary["items_written"] == 4
    assert summary["h2_balance"]["balance_applied"] is False


def test_g2_h1_items_from_fixture_attach_anatomical_bbox():
    object_rows, assertion_rows, image_index = _fixture_rows()

    items, summary = build_g2_h1_items(assertion_rows, object_rows, image_index)

    assert summary == {
        "candidate_groups": 11,
        "items_written": 11,
        "yes_items": 1,
        "no_items": 10,
        "excluded_conflict_groups": 0,
        "excluded_missing_bbox_groups": 0,
    }
    item = _find_item(
        items,
        target_finding="pneumothorax",
        target_anatomy="right lung",
    )
    assert item["answer_label"] == "No"
    assert item["granularity"] == "G2_anatomical_localization"
    assert item["patient_id"] == "19298916"
    assert item["study_id"] == "50189753"
    assert item["dicom_id"] == item["image_id"]
    assert item["bbox"]["bbox_name"] == "right lung"
    assert item["bbox"]["coordinate_space"] == "original_image"
    assert item["valid_for_roi_mask"] is True
    assert item["valid_for_roi_only"] is True
    assert item["hallucination_probe"] == "H1"
    for assertion in item["source_assertions"]:
        assert "phrase" not in assertion


def test_g2_h2_items_include_supported_and_unsupported_claims_with_bbox():
    object_rows, assertion_rows, image_index = _fixture_rows()

    items, summary = build_g2_h2_items(
        assertion_rows,
        object_rows,
        image_index,
        unsupported_per_anatomy=1,
    )

    assert summary["candidate_supported_groups"] == 11
    assert summary["supported_items"] == 9
    assert summary["unsupported_items"] == 36
    assert summary["items_written"] == 45
    assert summary["supported_items_before_balance"] == 11
    assert summary["unsupported_items_before_balance"] == 36
    assert {item["answer_label"] for item in items} == {"Supported", "Unsupported"}
    assert {item["hallucination_probe"] for item in items} == {"H2"}
    assert all(item["bbox"] is not None for item in items)


def test_g2_h2_balancing_is_deterministic_with_same_seed():
    object_rows, assertion_rows, image_index = _fixture_rows()

    items_a, _ = build_g2_h2_items(
        assertion_rows,
        object_rows,
        image_index,
        unsupported_per_anatomy=1,
        h2_sampling_seed=7,
    )
    items_b, _ = build_g2_h2_items(
        assertion_rows,
        object_rows,
        image_index,
        unsupported_per_anatomy=1,
        h2_sampling_seed=7,
    )

    assert [item["item_id"] for item in items_a] == [item["item_id"] for item in items_b]


def test_conflicts_are_excluded_and_counted():
    rows = [
        {
            "assertion_id": "a1",
            "patient_id": "p1",
            "study_id": "s1",
            "image_id": "i1",
            "bbox_name": "right lung",
            "anatomy_bound": True,
            "raw_label": "anatomicalfinding|yes|opacity",
            "category": "anatomicalfinding",
            "polarity": "yes",
            "label_name": "opacity",
            "phrase_id": "s1|1",
            "phrase_index": 0,
            "source_quality": "silver",
        },
        {
            "assertion_id": "a2",
            "patient_id": "p1",
            "study_id": "s1",
            "image_id": "i1",
            "bbox_name": "right lung",
            "anatomy_bound": True,
            "raw_label": "anatomicalfinding|no|opacity",
            "category": "anatomicalfinding",
            "polarity": "no",
            "label_name": "opacity",
            "phrase_id": "s1|2",
            "phrase_index": 1,
            "source_quality": "silver",
        },
    ]
    objects = [
        {
            "patient_id": "p1",
            "study_id": "s1",
            "image_id": "i1",
            "bbox_name": "right lung",
            "x1": 0.0,
            "y1": 1.0,
            "x2": 2.0,
            "y2": 3.0,
            "original_x1": 0.0,
            "original_y1": 10.0,
            "original_x2": 20.0,
            "original_y2": 30.0,
            "source_quality": "silver",
        }
    ]
    image_index = build_image_index_rows(objects, rows)

    g1_items, g1_summary = build_g1_h1_items(rows, image_index)
    g2_items, g2_summary = build_g2_h1_items(rows, objects, image_index)

    assert g1_items == []
    assert g1_summary["excluded_conflict_groups"] == 1
    assert g2_items == []
    assert g2_summary["excluded_conflict_groups"] == 1


def test_h2_conflicts_are_excluded_from_supported_items_but_unsupported_still_counted():
    rows = [
        {
            "assertion_id": "a1",
            "patient_id": "p1",
            "study_id": "s1",
            "image_id": "i1",
            "bbox_name": "right lung",
            "anatomy_bound": True,
            "raw_label": "anatomicalfinding|yes|opacity",
            "category": "anatomicalfinding",
            "polarity": "yes",
            "label_name": "opacity",
            "phrase_id": "s1|1",
            "phrase_index": 0,
            "source_quality": "silver",
        },
        {
            "assertion_id": "a2",
            "patient_id": "p1",
            "study_id": "s1",
            "image_id": "i1",
            "bbox_name": "right lung",
            "anatomy_bound": True,
            "raw_label": "anatomicalfinding|no|opacity",
            "category": "anatomicalfinding",
            "polarity": "no",
            "label_name": "opacity",
            "phrase_id": "s1|2",
            "phrase_index": 1,
            "source_quality": "silver",
        },
    ]
    objects = [
        {
            "patient_id": "p1",
            "study_id": "s1",
            "image_id": "i1",
            "bbox_name": "right lung",
            "x1": 0.0,
            "y1": 1.0,
            "x2": 2.0,
            "y2": 3.0,
            "original_x1": 0.0,
            "original_y1": 10.0,
            "original_x2": 20.0,
            "original_y2": 30.0,
            "source_quality": "silver",
        }
    ]
    image_index = build_image_index_rows(objects, rows)

    g1_items, g1_summary = build_g1_h2_items(rows, image_index)
    g2_items, g2_summary = build_g2_h2_items(rows, objects, image_index)

    assert g1_summary["excluded_conflict_groups"] == 1
    assert g2_summary["excluded_conflict_groups"] == 1
    assert all(item["answer_label"] != "Supported" for item in g1_items + g2_items)


def test_jsonl_roundtrip_for_generated_items(tmp_path):
    _, assertion_rows, image_index = _fixture_rows()
    items, _ = build_g1_h1_items(assertion_rows, image_index)
    output = tmp_path / "g1_items.jsonl"

    write_jsonl(items, output)
    loaded = read_jsonl(output)

    assert loaded == items


def _find_item(items, *, target_finding, target_anatomy=None):
    matches = [
        item
        for item in items
        if item["target_finding"] == target_finding
        and item.get("target_anatomy") == target_anatomy
    ]
    assert len(matches) == 1
    return matches[0]


def _assertion(
    assertion_id,
    *,
    image_id,
    label_name,
    polarity,
    patient_id="p1",
    study_id="s1",
    bbox_name="right lung",
):
    return {
        "assertion_id": assertion_id,
        "patient_id": patient_id,
        "study_id": study_id,
        "image_id": image_id,
        "bbox_name": bbox_name,
        "anatomy_bound": True,
        "raw_label": f"anatomicalfinding|{polarity}|{label_name}",
        "category": "anatomicalfinding",
        "polarity": polarity,
        "label_name": label_name,
        "phrase_id": f"{study_id}|1",
        "phrase_index": 0,
        "source_quality": "silver",
    }
