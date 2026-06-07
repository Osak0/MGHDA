import csv
from pathlib import Path

from ghm.data.link_and_download_mimic_jpg import (
    available_links_from_manifest,
    build_link_rows,
    build_mimic_relative_path,
    collect_needed_images,
    download_manifest_rows,
    load_mimic_metadata,
    update_items_with_links,
    write_csv_rows,
    write_url_list,
)
from ghm.granularity.common import read_jsonl, write_jsonl


def test_mimic_relative_path_uses_official_hierarchy():
    path = build_mimic_relative_path(
        subject_id="10000032",
        study_id="50414267",
        dicom_id="02aa804e-bde0afdd-112c0b34-7bc16630-4e384014",
    )

    assert path == Path(
        "p10/p10000032/s50414267/"
        "02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"
    )


def test_linking_deduplicates_items_and_updates_image_paths(tmp_path):
    g1_items, g2_items = _toy_items()
    metadata = {
        ("10000032", "50414267", "dicom-a"): {
            "subject_id": "10000032",
            "study_id": "50414267",
            "dicom_id": "dicom-a",
            "split": "train",
        }
    }
    needed = collect_needed_images({"g1": g1_items, "g2": g2_items})

    link_rows, summary = build_link_rows(
        needed,
        metadata,
        files_root=tmp_path / "files",
        base_url="https://example.test/files",
    )
    link_by_key = {
        (row["patient_id"], row["study_id"], row["dicom_id"]): row
        for row in link_rows
        if row["link_status"] == "matched"
    }
    g1_linked, g1_summary = update_items_with_links(g1_items, link_by_key)
    g2_linked, g2_summary = update_items_with_links(g2_items, link_by_key)

    assert summary["unique_requested_images"] == 1
    assert summary["matched_metadata_rows"] == 1
    assert len(g1_linked) == 1
    assert len(g2_linked) == 1
    assert g1_summary["excluded_missing_link"] == 0
    assert g2_summary["excluded_missing_link"] == 0
    assert g1_linked[0]["image_path"] == g2_linked[0]["image_path"]
    assert g1_linked[0]["image_path"].endswith("dicom-a.jpg")


def test_missing_metadata_and_id_mismatch_are_counted(tmp_path):
    items = [
        _toy_item("a", patient_id="10000032", study_id="50414267", dicom_id="dicom-a"),
        _toy_item("b", patient_id="10000032", study_id="99999999", dicom_id="dicom-b"),
        _toy_item("c", patient_id="10000032", study_id="50414267", dicom_id="dicom-c"),
    ]
    metadata = {
        ("10000032", "50414267", "dicom-a"): {
            "subject_id": "10000032",
            "study_id": "50414267",
            "dicom_id": "dicom-a",
            "split": "train",
        },
        ("10000032", "11111111", "dicom-b"): {
            "subject_id": "10000032",
            "study_id": "11111111",
            "dicom_id": "dicom-b",
            "split": "validate",
        },
    }

    rows, summary = build_link_rows(
        collect_needed_images({"g1": items}),
        metadata,
        files_root=tmp_path / "files",
        base_url="https://example.test/files",
    )

    assert summary["matched_metadata_rows"] == 1
    assert summary["id_mismatches"] == 1
    assert summary["missing_metadata"] == 1
    assert sorted(row["link_status"] for row in rows) == [
        "id_mismatch",
        "matched",
        "missing_metadata",
    ]


def test_dry_run_manifest_marks_files_without_network(tmp_path):
    rows, _ = build_link_rows(
        collect_needed_images({"g1": [_toy_item("a")]}),
        {
            ("10000032", "50414267", "dicom-a"): {
                "subject_id": "10000032",
                "study_id": "50414267",
                "dicom_id": "dicom-a",
                "split": "train",
            }
        },
        files_root=tmp_path / "files",
        base_url="https://example.test/files",
    )

    manifest, summary = download_manifest_rows(
        rows,
        dry_run=True,
        link_only_existing=False,
        overwrite=False,
        timeout_seconds=1.0,
        download_limit=None,
        username_env=None,
        password_env=None,
    )

    assert summary["dry_run_files"] == 1
    assert summary["downloaded_files"] == 0
    assert manifest[0]["download_status"] == "dry_run"
    assert not Path(str(manifest[0]["image_path"])).exists()


def test_failed_download_status_excludes_linked_items():
    items = [_toy_item("a")]
    manifest = [
        {
            "patient_id": "10000032",
            "study_id": "50414267",
            "dicom_id": "dicom-a",
            "image_id": "dicom-a",
            "image_path": "data/files/p10/p10000032/s50414267/dicom-a.jpg",
            "download_status": "failed",
        }
    ]

    linked, summary = update_items_with_links(items, available_links_from_manifest(manifest))

    assert linked == []
    assert summary["excluded_missing_link"] == 1


def test_csv_and_jsonl_helpers_for_synthetic_inputs(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    split_path = tmp_path / "split.csv"
    _write_csv(
        metadata_path,
        ["subject_id", "study_id", "dicom_id"],
        [{"subject_id": "10000032", "study_id": "50414267", "dicom_id": "dicom-a"}],
    )
    _write_csv(
        split_path,
        ["subject_id", "study_id", "dicom_id", "split"],
        [
            {
                "subject_id": "10000032",
                "study_id": "50414267",
                "dicom_id": "dicom-a",
                "split": "train",
            }
        ],
    )
    metadata = load_mimic_metadata(metadata_path, split_path)
    manifest_path = tmp_path / "manifest.csv"
    item_path = tmp_path / "items.jsonl"

    assert metadata[("10000032", "50414267", "dicom-a")]["split"] == "train"
    write_csv_rows([], manifest_path, ["a", "b"])
    url_path = tmp_path / "urls.txt"
    write_url_list(
        [{"link_status": "matched", "source_url": "https://example.test/a.jpg"}],
        url_path,
    )
    write_jsonl([_toy_item("a")], item_path)
    assert manifest_path.exists()
    assert url_path.read_text(encoding="utf-8").strip() == "https://example.test/a.jpg"
    assert read_jsonl(item_path)[0]["dicom_id"] == "dicom-a"


def _toy_items():
    return [_toy_item("g1")], [_toy_item("g2")]


def _toy_item(
    item_id,
    *,
    patient_id="10000032",
    study_id="50414267",
    dicom_id="dicom-a",
):
    return {
        "item_id": item_id,
        "patient_id": patient_id,
        "study_id": study_id,
        "image_id": dicom_id,
        "dicom_id": dicom_id,
        "image_path": None,
        "question": "Is there evidence of opacity in this chest X-ray?",
        "answer_label": "Yes",
        "source_assertions": [],
    }


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
