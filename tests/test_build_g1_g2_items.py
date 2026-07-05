from ghm.granularity.build_g1_items import build_g1_claim_verification_items
from ghm.granularity.build_g2_items import build_g2_claim_verification_items
from ghm.granularity.common import read_jsonl, write_jsonl
from ghm.granularity.study2 import (
    ANSWER_CONTRADICTED,
    ANSWER_NOT_ENOUGH,
    ANSWER_SUPPORTED,
    CLAIM_NEGATIVE,
    CLAIM_POSITIVE,
    EVIDENCE_AFFIRMED,
    EVIDENCE_NEGATED,
    EVIDENCE_NOT_ENOUGH,
    QUESTION_TYPE,
)


def test_g1_claim_verification_maps_yes_no_and_missing_evidence():
    rows = [
        _assertion("a1", label_name="pneumothorax", polarity="yes"),
        _assertion("a2", label_name="lung opacity", polarity="no"),
    ]
    items, summary = build_g1_claim_verification_items(
        rows,
        [_image_index()],
        missing_finding_sample_size=2,
        missing_finding_seed=7,
    )

    pneumothorax_positive = _find_item(
        items, target_finding="pneumothorax", claim_polarity=CLAIM_POSITIVE
    )
    pneumothorax_negative = _find_item(
        items, target_finding="pneumothorax", claim_polarity=CLAIM_NEGATIVE
    )
    opacity_positive = _find_item(
        items, target_finding="lung opacity", claim_polarity=CLAIM_POSITIVE
    )
    opacity_negative = _find_item(
        items, target_finding="lung opacity", claim_polarity=CLAIM_NEGATIVE
    )
    missing_items = [item for item in items if item["evidence_state"] == EVIDENCE_NOT_ENOUGH]

    assert pneumothorax_positive["answer_label"] == ANSWER_SUPPORTED
    assert pneumothorax_positive["evidence_state"] == EVIDENCE_AFFIRMED
    assert pneumothorax_negative["answer_label"] == ANSWER_CONTRADICTED
    assert opacity_positive["answer_label"] == ANSWER_CONTRADICTED
    assert opacity_positive["evidence_state"] == EVIDENCE_NEGATED
    assert opacity_negative["answer_label"] == ANSWER_SUPPORTED
    assert {item["answer_label"] for item in missing_items} == {ANSWER_NOT_ENOUGH}
    assert len(missing_items) == 4
    assert summary["missing_findings_sampled"] == 2
    assert summary["items_written"] == 8
    assert all(item["question_type"] == QUESTION_TYPE for item in items)
    assert all("phrase" not in item["source_assertions"][0] for item in items)


def test_g1_missing_finding_sampling_is_deterministic():
    rows = [_assertion("a1", label_name="pneumothorax", polarity="yes")]
    items_a, _ = build_g1_claim_verification_items(
        rows,
        [_image_index()],
        missing_finding_sample_size=2,
        missing_finding_seed=11,
    )
    items_b, _ = build_g1_claim_verification_items(
        rows,
        [_image_index()],
        missing_finding_sample_size=2,
        missing_finding_seed=11,
    )

    missing_a = [
        item["target_finding"] for item in items_a if item["evidence_state"] == EVIDENCE_NOT_ENOUGH
    ]
    missing_b = [
        item["target_finding"] for item in items_b if item["evidence_state"] == EVIDENCE_NOT_ENOUGH
    ]
    assert missing_a == missing_b


def test_g2_claim_verification_requires_bbox_and_samples_missing_per_anatomy():
    rows = [
        _assertion("a1", label_name="pneumothorax", polarity="no", bbox_name="right lung"),
        _assertion(
            "a2",
            label_name="atelectasis",
            polarity="no",
            bbox_name="right lung",
            image_id="dicom-b",
        ),
        _assertion(
            "a3",
            label_name="lung opacity",
            polarity="yes",
            bbox_name="right lung",
            image_id="dicom-c",
        ),
        _assertion(
            "a4",
            label_name="enlarged cardiac silhouette",
            polarity="yes",
            bbox_name="cardiac silhouette",
            image_id="dicom-d",
        ),
    ]
    objects = [_object_row()]
    items, summary = build_g2_claim_verification_items(
        rows,
        objects,
        [_image_index()],
        missing_finding_sample_size=2,
        missing_finding_seed=3,
    )

    positive = _find_item(
        items,
        target_finding="pneumothorax",
        target_anatomy="right lung",
        claim_polarity=CLAIM_POSITIVE,
    )
    negative = _find_item(
        items,
        target_finding="pneumothorax",
        target_anatomy="right lung",
        claim_polarity=CLAIM_NEGATIVE,
    )
    missing_items = [item for item in items if item["evidence_state"] == EVIDENCE_NOT_ENOUGH]

    assert positive["answer_label"] == ANSWER_CONTRADICTED
    assert negative["answer_label"] == ANSWER_SUPPORTED
    assert positive["bbox"]["bbox_name"] == "right lung"
    assert len(missing_items) == 4
    assert {item["target_finding"] for item in missing_items} == {
        "atelectasis",
        "lung opacity",
    }
    assert summary["missing_findings_sampled"] == 2
    assert summary["items_written"] == 6
    assert all(item["target_anatomy"] == "right lung" for item in items)
    assert summary["missing_source"] == "same_bbox_vocabulary"


def test_g2_missing_excludes_positive_findings_elsewhere_in_current_image():
    rows = [
        _assertion("a1", label_name="pneumothorax", polarity="no", bbox_name="right lung"),
        _assertion(
            "a2",
            label_name="atelectasis",
            polarity="no",
            bbox_name="right lung",
            image_id="dicom-b",
        ),
        _assertion(
            "a3",
            label_name="lung opacity",
            polarity="yes",
            bbox_name="right lung",
            image_id="dicom-c",
        ),
        _assertion(
            "a4",
            label_name="atelectasis",
            polarity="yes",
            bbox_name="left lung",
        ),
    ]

    items, summary = build_g2_claim_verification_items(
        rows,
        [_object_row()],
        [_image_index()],
        missing_finding_sample_size=2,
        missing_finding_seed=3,
    )

    missing_items = [item for item in items if item["evidence_state"] == EVIDENCE_NOT_ENOUGH]

    assert {item["target_finding"] for item in missing_items} == {"lung opacity"}
    assert summary["missing_findings_sampled"] == 1
    assert summary["missing_candidate_shortage"] == 1


def test_conflicts_are_excluded_and_counted():
    rows = [
        _assertion("a1", label_name="pneumothorax", polarity="yes", bbox_name="right lung"),
        _assertion("a2", label_name="pneumothorax", polarity="no", bbox_name="right lung"),
    ]

    g1_items, g1_summary = build_g1_claim_verification_items(
        rows,
        [_image_index()],
        missing_finding_sample_size=0,
    )
    g2_items, g2_summary = build_g2_claim_verification_items(
        rows,
        [_object_row()],
        [_image_index()],
        missing_finding_sample_size=0,
    )

    assert g1_items == []
    assert g2_items == []
    assert g1_summary["excluded_conflict_groups"] == 1
    assert g2_summary["excluded_conflict_groups"] == 1


def test_jsonl_roundtrip_for_generated_items(tmp_path):
    items, _ = build_g1_claim_verification_items(
        [_assertion("a1", label_name="pneumothorax", polarity="yes")],
        [_image_index()],
        missing_finding_sample_size=0,
    )
    path = tmp_path / "items.jsonl"

    write_jsonl(items, path)

    assert read_jsonl(path) == items


def _find_item(
    items,
    *,
    target_finding,
    claim_polarity,
    target_anatomy=None,
):
    matches = [
        item
        for item in items
        if item["target_finding"] == target_finding
        and item["target_anatomy"] == target_anatomy
        and item["claim_polarity"] == claim_polarity
    ]
    assert len(matches) == 1
    return matches[0]


def _assertion(
    assertion_id,
    *,
    label_name,
    polarity,
    bbox_name=None,
    image_id="dicom-a",
):
    return {
        "assertion_id": assertion_id,
        "patient_id": "10000032",
        "study_id": "50414267",
        "image_id": image_id,
        "bbox_name": bbox_name,
        "anatomy_bound": bbox_name is not None,
        "raw_label": f"anatomicalfinding|{polarity}|{label_name}",
        "category": "anatomicalfinding",
        "polarity": polarity,
        "label_name": label_name,
        "phrase_id": None,
        "phrase_index": None,
        "phrase": "synthetic phrase should not be copied",
        "source_quality": "synthetic",
    }


def _image_index():
    return {
        "patient_id": "10000032",
        "study_id": "50414267",
        "image_id": "dicom-a",
        "dicom_id": "dicom-a",
        "image_path": None,
    }


def _object_row():
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
