from ghm.study3.constants import CONTROLLED, NATURAL
from ghm.study3.construction import (
    build_g1_anchors,
    build_g2_anchors,
    build_multiselect_items,
    sample_study3_anchors,
)


def test_g1_two_no_findings_build_none_and_all_selected_answers():
    rows = [
        _assertion("a1", "lung opacity", "no"),
        _assertion("a2", "pneumothorax", "no"),
    ]

    anchors, summary = build_g1_anchors(rows)
    anchors[0]["image_path"] = "files/p10/s1/a.jpg"
    items, item_summary = build_multiselect_items(
        anchors,
        set(),
        seed=42,
    )

    present = next(row for row in items if row["query_relation"] == "present")
    absent = next(row for row in items if row["query_relation"] == "absent")
    assert present["gold_selected_options"] == []
    assert absent["gold_selected_options"] == ["A", "B"]
    assert summary["excluded_conflict_findings"] == 0
    assert item_summary["natural_items"] == 2


def test_conflicts_are_removed_without_converting_missing_to_no():
    rows = [
        _assertion("a1", "pneumothorax", "yes"),
        _assertion("a2", "pneumothorax", "no"),
        _assertion("a3", "lung opacity", "no"),
        _assertion("a4", "atelectasis", "yes"),
    ]

    anchors, summary = build_g1_anchors(rows)

    assert [row["label_name"] for row in anchors[0]["options"]] == [
        "atelectasis",
        "lung opacity",
    ]
    assert summary["excluded_conflict_findings"] == 1


def test_g2_requires_anatomy_binding_and_complete_bbox():
    rows = [
        _assertion("a1", "lung opacity", "yes", bbox_name="right lung"),
        _assertion("a2", "pneumothorax", "no", bbox_name="right lung"),
        _assertion(
            "a3",
            "atelectasis",
            "yes",
            bbox_name=None,
            anatomy_bound=False,
        ),
    ]

    anchors, _ = build_g2_anchors(rows, [_object()])
    no_bbox_anchors, summary = build_g2_anchors(
        rows,
        [
            {
                **_object(),
                "x1": None,
                "y1": None,
                "x2": None,
                "y2": None,
                "original_x1": None,
                "original_y1": None,
                "original_x2": None,
                "original_y2": None,
            }
        ],
    )

    assert len(anchors) == 1
    assert anchors[0]["target_anatomy"] == "right lung"
    assert no_bbox_anchors == []
    assert summary["excluded_findings_without_valid_bbox"] == 2


def test_natural_and_controlled_nested_variants_are_deterministic():
    rows = [_assertion(f"a{i}", f"finding {i}", "yes" if i % 2 else "no") for i in range(6)]
    anchors, _ = build_g1_anchors(rows)
    anchors[0]["image_path"] = "files/p10/s1/a.jpg"

    selected_a, controlled_a, _ = sample_study3_anchors(
        anchors,
        natural_count=1,
        controlled_count=1,
        seed=42,
    )
    selected_b, controlled_b, _ = sample_study3_anchors(
        anchors,
        natural_count=1,
        controlled_count=1,
        seed=42,
    )
    items_a, _ = build_multiselect_items(selected_a, controlled_a, seed=42)
    items_b, _ = build_multiselect_items(selected_b, controlled_b, seed=42)

    assert items_a == items_b
    assert len([row for row in items_a if row["variant"] == NATURAL]) == 2
    controlled = [row for row in items_a if row["variant"] == CONTROLLED]
    assert len(controlled) == 8
    option_sets = {}
    for row in controlled:
        option_sets.setdefault(row["option_count"], row["options"])
    assert [row["label_name"] for row in option_sets[2]] == [
        row["label_name"] for row in option_sets[5][:2]
    ]


def test_target_budget_is_1800_items_per_granularity():
    anchors = [_synthetic_anchor(index) for index in range(500)]
    controlled_ids = {row["anchor_id"] for row in anchors[:100]}

    items, summary = build_multiselect_items(
        anchors,
        controlled_ids,
        seed=42,
    )

    assert len(items) == 1800
    assert summary["natural_items"] == 1000
    assert summary["controlled_items"] == 800


def _assertion(
    assertion_id,
    label,
    polarity,
    *,
    bbox_name=None,
    anatomy_bound=None,
):
    return {
        "assertion_id": assertion_id,
        "patient_id": "10000032",
        "study_id": "50414267",
        "image_id": "dicom-a",
        "bbox_name": bbox_name,
        "anatomy_bound": anatomy_bound if anatomy_bound is not None else bbox_name is not None,
        "raw_label": f"anatomicalfinding|{polarity}|{label}",
        "category": "anatomicalfinding",
        "polarity": polarity,
        "label_name": label,
        "phrase_id": None,
        "phrase_index": None,
        "source_quality": "synthetic",
    }


def _object():
    return {
        "patient_id": "10000032",
        "study_id": "50414267",
        "image_id": "dicom-a",
        "bbox_name": "right lung",
        "x1": 1.0,
        "y1": 2.0,
        "x2": 3.0,
        "y2": 4.0,
        "original_x1": 10.0,
        "original_y1": 20.0,
        "original_x2": 30.0,
        "original_y2": 40.0,
        "source_quality": "synthetic",
    }


def _synthetic_anchor(index):
    return {
        "anchor_id": f"study3_g1_anchor_{index:04d}",
        "experiment_id": "study3_multiselect_v1",
        "source_dataset": "ChestImaGenome",
        "patient_id": f"p{index}",
        "study_id": f"s{index}",
        "image_id": f"d{index}",
        "dicom_id": f"d{index}",
        "image_path": f"files/p{index}/s{index}/d{index}.jpg",
        "granularity": "G1_finding_existence",
        "target_anatomy": None,
        "bbox": None,
        "options": [
            {
                "label_name": f"finding {option}",
                "polarity": "yes" if option % 2 else "no",
                "source_assertions": [],
            }
            for option in range(5)
        ],
        "natural_option_count": 5,
        "answer_composition": "mixed",
        "source_quality": "synthetic",
    }
