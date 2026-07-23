from pathlib import Path

import pytest

from ghm.data.parse_chest_imagenome import (
    discover_scene_graphs,
    parse_label,
    parse_scene_graph_file,
    write_parquet_outputs,
    write_summary,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURE_DIR / "synthetic_chest_imagenome_SceneGraph.json"


def test_parse_fixture_object_and_assertion_counts():
    result = parse_scene_graph_file(FIXTURE)

    assert result.files_seen == 1
    assert len(result.object_rows) == 3
    assert len(result.assertion_rows) == 4
    assert result.warnings == {}


def test_parse_label_splits_category_polarity_and_name():
    assert parse_label("anatomicalfinding|no|pneumothorax") == (
        "anatomicalfinding",
        "no",
        "pneumothorax",
    )
    assert parse_label("malformed") == (None, None, None)


def test_pneumothorax_assertion_is_flattened_with_anatomy_binding():
    result = parse_scene_graph_file(FIXTURE)

    matches = [
        row
        for row in result.assertion_rows
        if row["raw_label"] == "anatomicalfinding|no|pneumothorax"
        and row["bbox_name"] == "right lung"
    ]

    assert len(matches) == 1
    row = matches[0]
    assert row["category"] == "anatomicalfinding"
    assert row["polarity"] == "no"
    assert row["label_name"] == "pneumothorax"
    assert row["anatomy_bound"] is True


def test_modifier_cues_remain_phrase_aligned():
    result = parse_scene_graph_file(FIXTURE)

    rows = [
        row
        for row in result.assertion_rows
        if row["raw_label"] == "anatomicalfinding|yes|enlarged cardiac silhouette"
    ]

    assert len(rows) == 2
    assert {row["phrase_index"] for row in rows} == {0, 1}
    for row in rows:
        assert row["bbox_name"] == "cardiac silhouette"
        assert row["severity_cues"] == ["severity|yes|mild"]
        assert row["comparison_cues"] == ["comparison|yes|no change"]


def test_discover_scene_graphs_is_deterministic():
    paths = discover_scene_graphs(FIXTURE_DIR)

    assert paths == [FIXTURE]


def test_write_parquet_outputs_and_summary_roundtrip(tmp_path):
    pa = pytest.importorskip("pyarrow.parquet")
    result = parse_scene_graph_file(FIXTURE)

    write_parquet_outputs(result, tmp_path)
    write_summary(result, tmp_path)

    objects = pa.read_table(tmp_path / "ci_objects.parquet")
    assertions = pa.read_table(tmp_path / "ci_attribute_assertions.parquet")

    assert objects.num_rows == 3
    assert assertions.num_rows == 4
    assert (tmp_path / "parse_chest_imagenome_summary.json").exists()
