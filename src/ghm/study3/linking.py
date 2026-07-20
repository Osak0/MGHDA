"""Study 3-only linking to already downloaded MIMIC-CXR-JPG images."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.0.0/files"


def load_mimic_metadata(
    metadata_csv: Path,
    split_csv: Path,
) -> dict[tuple[str, str, str], dict[str, str | None]]:
    """Load MIMIC metadata and split values without reading reports or images."""

    split_by_key: dict[tuple[str, str, str], str | None] = {}
    for row in _iter_csv_rows(split_csv):
        key = _metadata_key(row)
        if key is not None:
            split_by_key[key] = row.get("split")

    metadata: dict[tuple[str, str, str], dict[str, str | None]] = {}
    for row in _iter_csv_rows(metadata_csv):
        key = _metadata_key(row)
        if key is None:
            continue
        subject_id, study_id, dicom_id = key
        metadata[key] = {
            "subject_id": subject_id,
            "study_id": study_id,
            "dicom_id": dicom_id,
            "split": split_by_key.get(key),
        }
    return metadata


def collect_needed_images(
    anchors: Iterable[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, str]]:
    """Collect unique image identifiers required by Study 3 anchors."""

    needed: dict[tuple[str, str, str], dict[str, str]] = {}
    for anchor in anchors:
        patient_id = normalize_id(anchor.get("patient_id"))
        study_id = normalize_id(anchor.get("study_id"))
        dicom_id = normalize_id(anchor.get("dicom_id") or anchor.get("image_id"))
        image_id = normalize_id(anchor.get("image_id") or dicom_id)
        if patient_id and study_id and dicom_id and image_id:
            needed[(patient_id, study_id, dicom_id)] = {
                "patient_id": patient_id,
                "study_id": study_id,
                "dicom_id": dicom_id,
                "image_id": image_id,
            }
    return needed


def build_link_rows(
    needed: dict[tuple[str, str, str], dict[str, str]],
    metadata: dict[tuple[str, str, str], dict[str, str | None]],
    *,
    files_root: Path,
    base_url: str,
    image_path_root: Path = Path("files"),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build portable/local paths without downloading or opening any image."""

    rows: list[dict[str, Any]] = []
    missing_metadata = 0
    id_mismatches = 0
    metadata_dicoms = {key[2] for key in metadata}
    for key in sorted(needed):
        patient_id, study_id, dicom_id = key
        request = needed[key]
        metadata_row = metadata.get(key)
        if metadata_row is None:
            link_status = "id_mismatch" if dicom_id in metadata_dicoms else "missing_metadata"
            id_mismatches += int(link_status == "id_mismatch")
            missing_metadata += int(link_status == "missing_metadata")
            rows.append(
                {
                    **request,
                    "image_path": None,
                    "_local_image_path": None,
                    "link_status": link_status,
                }
            )
            continue
        relative = build_mimic_relative_path(
            subject_id=metadata_row["subject_id"],
            study_id=metadata_row["study_id"],
            dicom_id=metadata_row["dicom_id"],
        )
        rows.append(
            {
                **request,
                "subject_id": metadata_row["subject_id"],
                "split": metadata_row.get("split"),
                "image_path": str(image_path_root / relative),
                "_local_image_path": str(files_root / relative),
                "relative_path": relative.as_posix(),
                "source_url": join_url(base_url, relative.as_posix()),
                "link_status": "matched",
            }
        )
    return rows, {
        "unique_requested_images": len(needed),
        "matched_metadata_rows": len(needed) - missing_metadata - id_mismatches,
        "missing_metadata": missing_metadata,
        "id_mismatches": id_mismatches,
    }


def build_mimic_relative_path(
    *,
    subject_id: str | None,
    study_id: str | None,
    dicom_id: str | None,
) -> Path:
    """Build the canonical path below the MIMIC-CXR-JPG files root."""

    if not subject_id or not study_id or not dicom_id:
        raise ValueError("subject_id, study_id, and dicom_id are required")
    subject = subject_id if subject_id.startswith("p") else f"p{subject_id}"
    study = study_id if study_id.startswith("s") else f"s{study_id}"
    subject_prefix = f"p{subject[1:3]}"
    filename = dicom_id if dicom_id.endswith(".jpg") else f"{dicom_id}.jpg"
    return Path(subject_prefix) / subject / study / filename


def normalize_id(value: Any) -> str | None:
    """Normalize numeric/string identifiers from JSON and CSV."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def join_url(base_url: str, relative_path: str) -> str:
    return f"{base_url.rstrip('/')}/{relative_path.lstrip('/')}"


def _metadata_key(row: dict[str, Any]) -> tuple[str, str, str] | None:
    subject_id = normalize_id(row.get("subject_id"))
    study_id = normalize_id(row.get("study_id"))
    dicom_id = normalize_id(row.get("dicom_id"))
    if not (subject_id and study_id and dicom_id):
        return None
    return subject_id, study_id, dicom_id


def _iter_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        yield from csv.DictReader(file)
