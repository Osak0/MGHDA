from ghm.data.audit_chest_imagenome import (
    BBOX_FINDING_SUMMARY_COLUMNS,
    BBOX_QUALITY_SUMMARY_COLUMNS,
    COORDINATE_INCOMPLETE,
    REVIEW,
    USE_EXPLICIT_ONLY,
    USE_FOR_G2,
    build_bbox_finding_reference_tables,
)


def test_bbox_finding_reference_tables_are_aggregate_and_same_bbox_scoped():
    finding_rows, quality_rows, summary = build_bbox_finding_reference_tables(
        object_rows=[
            _object_row("img-a", "right lung"),
            _object_row("img-b", "right lung", original_x1=None),
            _object_row("img-a", "left lung"),
            _object_row("img-b", "left lung"),
            _object_row("img-a", "cardiac silhouette"),
        ],
        assertion_rows=[
            _assertion("img-a", "right lung", "pneumothorax", "yes"),
            _assertion("img-a", "right lung", "pneumothorax", "no"),
            _assertion("img-b", "right lung", "atelectasis", "no"),
            _assertion("img-a", "left lung", "atelectasis", "yes"),
            _assertion("img-b", "left lung", "lung opacity", "no"),
            _assertion("img-a", "cardiac silhouette", "enlarged cardiac silhouette", "yes"),
            _assertion("img-a", "right lung", "ignored support device", "yes", category="device"),
            _assertion("img-a", None, "pleural effusion", "yes"),
        ],
    )

    forbidden_columns = {"patient_id", "study_id", "image_id", "phrase", "image_path"}
    assert forbidden_columns.isdisjoint(BBOX_FINDING_SUMMARY_COLUMNS)
    assert forbidden_columns.isdisjoint(BBOX_QUALITY_SUMMARY_COLUMNS)
    assert all(forbidden_columns.isdisjoint(row) for row in finding_rows)
    assert all(forbidden_columns.isdisjoint(row) for row in quality_rows)

    right_lung_findings = {
        row["label_name"] for row in finding_rows if row["bbox_name"] == "right lung"
    }
    left_lung_findings = {
        row["label_name"] for row in finding_rows if row["bbox_name"] == "left lung"
    }
    assert right_lung_findings == {"atelectasis", "pneumothorax"}
    assert left_lung_findings == {"atelectasis", "lung opacity"}

    right_lung_quality = _find_quality(quality_rows, "right lung")
    left_lung_quality = _find_quality(quality_rows, "left lung")
    cardiac_quality = _find_quality(quality_rows, "cardiac silhouette")
    pneumothorax_row = _find_finding(finding_rows, "right lung", "pneumothorax")

    assert right_lung_quality["bbox_object_rows"] == 2
    assert right_lung_quality["original_bbox_complete_rate"] == 0.5
    assert right_lung_quality["n_conflict_image_bbox_finding"] == 1
    assert right_lung_quality["recommended_g2_use"] == REVIEW
    assert left_lung_quality["recommended_g2_use"] == USE_FOR_G2
    assert cardiac_quality["recommended_g2_use"] == USE_EXPLICIT_ONLY
    assert pneumothorax_row["n_conflict_image_bbox_finding"] == 1
    assert pneumothorax_row["g2_candidate_status"] != COORDINATE_INCOMPLETE
    assert summary["conflict_image_bbox_finding_groups"] == 1


def test_bbox_finding_status_marks_coordinate_incomplete_without_conflict():
    finding_rows, quality_rows, _ = build_bbox_finding_reference_tables(
        object_rows=[
            _object_row("img-a", "right lung"),
            _object_row("img-b", "right lung", original_y1=None),
        ],
        assertion_rows=[
            _assertion("img-a", "right lung", "atelectasis", "yes"),
            _assertion("img-b", "right lung", "atelectasis", "no"),
        ],
    )

    row = _find_finding(finding_rows, "right lung", "atelectasis")
    quality = _find_quality(quality_rows, "right lung")

    assert row["g2_candidate_status"] == COORDINATE_INCOMPLETE
    assert quality["recommended_g2_use"] == REVIEW


def _find_finding(rows, bbox_name, label_name):
    matches = [
        row
        for row in rows
        if row["bbox_name"] == bbox_name and row["label_name"] == label_name
    ]
    assert len(matches) == 1
    return matches[0]


def _find_quality(rows, bbox_name):
    matches = [row for row in rows if row["bbox_name"] == bbox_name]
    assert len(matches) == 1
    return matches[0]


def _object_row(image_id, bbox_name, **overrides):
    row = {
        "image_id": image_id,
        "bbox_name": bbox_name,
        "x1": 1.0,
        "y1": 2.0,
        "x2": 3.0,
        "y2": 4.0,
        "original_x1": 10.0,
        "original_y1": 20.0,
        "original_x2": 30.0,
        "original_y2": 40.0,
    }
    row.update(overrides)
    return row


def _assertion(
    image_id,
    bbox_name,
    label_name,
    polarity,
    *,
    category="anatomicalfinding",
    anatomy_bound=True,
):
    return {
        "image_id": image_id,
        "bbox_name": bbox_name,
        "anatomy_bound": anatomy_bound,
        "category": category,
        "polarity": polarity,
        "label_name": label_name,
    }
