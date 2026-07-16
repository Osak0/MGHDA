"""Link G1/G2 items to MIMIC-CXR-JPG files and download needed images.

Codex tests this module with synthetic metadata/items only. Real restricted
metadata, item files, and image downloads should be run locally by the user.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import shutil
import socket
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from ghm.granularity.common import read_jsonl, update_summary, write_jsonl, write_parquet_rows


DEFAULT_BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.0.0/files"

DEFAULT_ITEM_FILES = {
    "g1_h1": Path("data/processed/items/g1_h1_items.jsonl"),
    "g1_h2": Path("data/processed/items/g1_h2_items.jsonl"),
    "g2_h1": Path("data/processed/items/g2_h1_items.jsonl"),
    "g2_h2": Path("data/processed/items/g2_h2_items.jsonl"),
}

DEFAULT_LINKED_ITEM_FILES = {
    "g1_h1": Path("data/processed/items/g1_h1_items_linked.jsonl"),
    "g1_h2": Path("data/processed/items/g1_h2_items_linked.jsonl"),
    "g2_h1": Path("data/processed/items/g2_h1_items_linked.jsonl"),
    "g2_h2": Path("data/processed/items/g2_h2_items_linked.jsonl"),
}

DEFAULT_STUDY2_ITEM_FILES = {
    "study2_g1": Path("data/processed/items/study2_g1_claim_verification_items.jsonl"),
    "study2_g2": Path("data/processed/items/study2_g2_claim_verification_items.jsonl"),
}

DEFAULT_STUDY2_LINKED_ITEM_FILES = {
    "study2_g1": Path(
        "data/processed/items/study2_g1_claim_verification_items_linked.jsonl"
    ),
    "study2_g2": Path(
        "data/processed/items/study2_g2_claim_verification_items_linked.jsonl"
    ),
}

NEEDED_INDEX_COLUMNS = [
    "patient_id",
    "study_id",
    "image_id",
    "dicom_id",
    "subject_id",
    "split",
    "image_path",
    "relative_path",
    "source_url",
    "link_status",
]

MANIFEST_COLUMNS = [
    "patient_id",
    "study_id",
    "dicom_id",
    "subject_id",
    "split",
    "image_path",
    "relative_path",
    "source_url",
    "download_status",
]


def load_mimic_metadata(
    metadata_csv: Path, split_csv: Path
) -> dict[tuple[str, str, str], dict[str, str | None]]:
    """Load MIMIC metadata and split rows indexed by subject/study/dicom."""

    split_by_key: dict[tuple[str, str, str], str | None] = {}
    for row in _read_csv_rows(split_csv):
        subject_id = normalize_id(row.get("subject_id"))
        study_id = normalize_id(row.get("study_id"))
        dicom_id = normalize_id(row.get("dicom_id"))
        if subject_id and study_id and dicom_id:
            split_by_key[(subject_id, study_id, dicom_id)] = row.get("split")

    metadata: dict[tuple[str, str, str], dict[str, str | None]] = {}
    for row in _read_csv_rows(metadata_csv):
        subject_id = normalize_id(row.get("subject_id"))
        study_id = normalize_id(row.get("study_id"))
        dicom_id = normalize_id(row.get("dicom_id"))
        if not (subject_id and study_id and dicom_id):
            continue
        key = (subject_id, study_id, dicom_id)
        metadata[key] = {
            "subject_id": subject_id,
            "study_id": study_id,
            "dicom_id": dicom_id,
            "split": split_by_key.get(key),
        }
    return metadata


def collect_needed_images(
    item_groups: dict[str, list[dict[str, Any]]],
) -> dict[tuple[str | None, str | None, str | None], dict[str, str | None]]:
    """Collect unique images required by item groups."""

    needed: dict[tuple[str | None, str | None, str | None], dict[str, str | None]] = {}
    for items in item_groups.values():
        for item in items:
            patient_id = normalize_id(item.get("patient_id"))
            study_id = normalize_id(item.get("study_id"))
            dicom_id = normalize_id(item.get("dicom_id") or item.get("image_id"))
            image_id = normalize_id(item.get("image_id") or dicom_id)
            if not (patient_id and study_id and dicom_id):
                continue
            key = (patient_id, study_id, dicom_id)
            needed[key] = {
                "patient_id": patient_id,
                "study_id": study_id,
                "dicom_id": dicom_id,
                "image_id": image_id,
            }
    return needed


def build_link_rows(
    needed: dict[tuple[str | None, str | None, str | None], dict[str, str | None]],
    metadata: dict[tuple[str, str, str], dict[str, str | None]],
    *,
    files_root: Path,
    base_url: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build local image paths and source URLs for needed images."""

    rows: list[dict[str, Any]] = []
    missing_metadata = 0
    id_mismatches = 0
    matched_metadata = 0
    metadata_by_dicom: dict[str, list[dict[str, str | None]]] = {}
    for row in metadata.values():
        dicom_id = row.get("dicom_id")
        if dicom_id:
            metadata_by_dicom.setdefault(dicom_id, []).append(row)

    for key in sorted(needed, key=_sort_key):
        patient_id, study_id, dicom_id = key
        request = needed[key]
        metadata_row = metadata.get((patient_id or "", study_id or "", dicom_id or ""))
        if metadata_row is None:
            link_status = "id_mismatch" if dicom_id in metadata_by_dicom else "missing_metadata"
            if link_status == "id_mismatch":
                id_mismatches += 1
            else:
                missing_metadata += 1
            rows.append(
                {
                    "patient_id": patient_id,
                    "study_id": study_id,
                    "image_id": request.get("image_id"),
                    "dicom_id": dicom_id,
                    "subject_id": None,
                    "split": None,
                    "image_path": None,
                    "relative_path": None,
                    "source_url": None,
                    "link_status": link_status,
                }
            )
            continue

        matched_metadata += 1
        relative_path = build_mimic_relative_path(
            subject_id=metadata_row["subject_id"],
            study_id=metadata_row["study_id"],
            dicom_id=metadata_row["dicom_id"],
        )
        image_path = files_root / relative_path
        source_url = join_url(base_url, relative_path.as_posix())
        rows.append(
            {
                "patient_id": patient_id,
                "study_id": study_id,
                "image_id": request.get("image_id"),
                "dicom_id": dicom_id,
                "subject_id": metadata_row["subject_id"],
                "split": metadata_row.get("split"),
                "image_path": str(image_path),
                "relative_path": relative_path.as_posix(),
                "source_url": source_url,
                "link_status": "matched",
            }
        )

    summary = {
        "unique_requested_images": len(needed),
        "matched_metadata_rows": matched_metadata,
        "missing_metadata": missing_metadata,
        "id_mismatches": id_mismatches,
    }
    return rows, summary


def update_items_with_links(
    items: list[dict[str, Any]], link_by_key: dict[tuple[str, str, str], dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return linked items with image_path filled, excluding unlinked items."""

    linked: list[dict[str, Any]] = []
    excluded_missing_link = 0
    for item in items:
        patient_id = normalize_id(item.get("patient_id"))
        study_id = normalize_id(item.get("study_id"))
        dicom_id = normalize_id(item.get("dicom_id") or item.get("image_id"))
        if not (patient_id and study_id and dicom_id):
            excluded_missing_link += 1
            continue
        link = link_by_key.get((patient_id, study_id, dicom_id))
        if not link or link.get("link_status") != "matched" or not link.get("image_path"):
            excluded_missing_link += 1
            continue
        updated = dict(item)
        updated["patient_id"] = patient_id
        updated["study_id"] = study_id
        updated["image_id"] = link.get("image_id") or dicom_id
        updated["dicom_id"] = dicom_id
        updated["image_path"] = link["image_path"]
        linked.append(updated)

    return linked, {
        "input_items": len(items),
        "linked_items": len(linked),
        "excluded_missing_link": excluded_missing_link,
    }


def download_manifest_rows(
    link_rows: list[dict[str, Any]],
    *,
    dry_run: bool,
    link_only_existing: bool,
    overwrite: bool,
    timeout_seconds: float,
    download_limit: int | None,
    username_env: str | None,
    password_env: str | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Download matched image rows or mark what would happen in dry-run mode."""

    manifest_rows: list[dict[str, Any]] = []
    summary = {
        "downloaded_files": 0,
        "skipped_existing_files": 0,
        "dry_run_files": 0,
        "failed_downloads": 0,
        "unlinked_files": 0,
        "not_attempted_due_to_limit": 0,
        "failure_reasons": {},
    }
    failure_reasons: Counter[str] = Counter()
    headers = build_auth_headers(username_env=username_env, password_env=password_env)
    attempted_downloads = 0

    for row in link_rows:
        manifest_row = {column: row.get(column) for column in MANIFEST_COLUMNS}
        if row.get("link_status") != "matched":
            manifest_row["download_status"] = "unlinked"
            summary["unlinked_files"] += 1
            manifest_rows.append(manifest_row)
            continue

        image_path = Path(str(row["image_path"]))
        source_url = str(row["source_url"])
        if dry_run:
            manifest_row["download_status"] = "dry_run"
            summary["dry_run_files"] += 1
        elif link_only_existing:
            if image_path.exists():
                manifest_row["download_status"] = "skipped_existing"
                summary["skipped_existing_files"] += 1
            else:
                manifest_row["download_status"] = "not_attempted_link_only"
                summary["not_attempted_due_to_limit"] += 1
        elif image_path.exists() and not overwrite:
            manifest_row["download_status"] = "skipped_existing"
            summary["skipped_existing_files"] += 1
        elif download_limit is not None and attempted_downloads >= download_limit:
            manifest_row["download_status"] = "not_attempted_due_to_limit"
            summary["not_attempted_due_to_limit"] += 1
        else:
            attempted_downloads += 1
            try:
                download_file(
                    source_url,
                    image_path,
                    headers=headers,
                    timeout_seconds=timeout_seconds,
                )
                manifest_row["download_status"] = "downloaded"
                summary["downloaded_files"] += 1
            except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
                manifest_row["download_status"] = "failed"
                failure_reasons[classify_download_error(exc)] += 1
                summary["failed_downloads"] += 1
        manifest_rows.append(manifest_row)

    summary["failure_reasons"] = dict(sorted(failure_reasons.items()))
    return manifest_rows, summary


def write_csv_rows(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    """Write CSV rows with a stable column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def write_url_list(rows: list[dict[str, Any]], path: Path) -> None:
    """Write matched source URLs for external tools such as wget."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            if row.get("link_status") == "matched" and row.get("source_url"):
                file.write(str(row["source_url"]))
                file.write("\n")


def build_mimic_relative_path(
    *, subject_id: str | None, study_id: str | None, dicom_id: str | None
) -> Path:
    """Build the canonical MIMIC-CXR-JPG file path below the files root."""

    if not subject_id or not study_id or not dicom_id:
        raise ValueError("subject_id, study_id, and dicom_id are required")
    subject = subject_id if subject_id.startswith("p") else f"p{subject_id}"
    study = study_id if study_id.startswith("s") else f"s{study_id}"
    numeric_subject = subject[1:]
    subject_prefix = f"p{numeric_subject[:2]}"
    filename = dicom_id if dicom_id.endswith(".jpg") else f"{dicom_id}.jpg"
    return Path(subject_prefix) / subject / study / filename


def join_url(base_url: str, relative_path: str) -> str:
    """Join a base URL and POSIX relative path."""

    return f"{base_url.rstrip('/')}/{relative_path.lstrip('/')}"


def download_file(
    source_url: str,
    destination: Path,
    *,
    headers: dict[str, str],
    timeout_seconds: float,
) -> None:
    """Download one file to the destination path."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(source_url, headers=headers)
    with urllib.request.urlopen(
        request, timeout=timeout_seconds
    ) as response, destination.open("wb") as file:
        shutil.copyfileobj(response, file)


def build_auth_headers(
    *, username_env: str | None, password_env: str | None
) -> dict[str, str]:
    """Build optional HTTP Basic auth headers from environment variables."""

    if not username_env and not password_env:
        return {}
    username = os.environ.get(username_env or "")
    password = os.environ.get(password_env or "")
    if not username or not password:
        return {}
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def normalize_id(value: Any) -> str | None:
    """Normalize numeric/string IDs from JSON and CSV."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def classify_download_error(exc: BaseException) -> str:
    """Classify download failures without exposing URLs or identifiers."""

    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP_{exc.code}"
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, TimeoutError | socket.timeout):
            return "timeout"
        return reason.__class__.__name__
    if isinstance(exc, TimeoutError | socket.timeout):
        return "timeout"
    return exc.__class__.__name__


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="Link current G1/G2 items to MIMIC-CXR-JPG image paths."
    )
    parser.add_argument("--g1-h1-items", type=Path, default=DEFAULT_ITEM_FILES["g1_h1"])
    parser.add_argument("--g1-h2-items", type=Path, default=DEFAULT_ITEM_FILES["g1_h2"])
    parser.add_argument("--g2-h1-items", type=Path, default=DEFAULT_ITEM_FILES["g2_h1"])
    parser.add_argument("--g2-h2-items", type=Path, default=DEFAULT_ITEM_FILES["g2_h2"])
    parser.add_argument("--study2-g1-items", type=Path, default=DEFAULT_STUDY2_ITEM_FILES["study2_g1"])
    parser.add_argument("--study2-g2-items", type=Path, default=DEFAULT_STUDY2_ITEM_FILES["study2_g2"])
    parser.add_argument(
        "--study2-only",
        action="store_true",
        help="Link Study 2 G1/G2 claim-verification item files instead of legacy H1/H2 files.",
    )
    parser.add_argument("--metadata", type=Path, required=True, help="MIMIC metadata CSV.")
    parser.add_argument("--split", type=Path, required=True, help="MIMIC split CSV.")
    parser.add_argument(
        "--files-root",
        type=Path,
        default=Path("data/files"),
        help="Local root for downloaded MIMIC-CXR-JPG images.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL corresponding to the MIMIC-CXR-JPG files directory.",
    )
    parser.add_argument(
        "--needed-index",
        type=Path,
        default=Path("data/interim/needed_mimic_jpg_index.parquet"),
        help="Output Parquet index of needed images.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/interim/needed_mimic_jpg_download_manifest.csv"),
        help="Output CSV download manifest.",
    )
    parser.add_argument(
        "--url-list",
        type=Path,
        default=Path("data/interim/needed_mimic_jpg_urls.txt"),
        help="Output URL list for wget -i partial downloads.",
    )
    parser.add_argument("--g1-h1-output", type=Path, default=DEFAULT_LINKED_ITEM_FILES["g1_h1"])
    parser.add_argument("--g1-h2-output", type=Path, default=DEFAULT_LINKED_ITEM_FILES["g1_h2"])
    parser.add_argument("--g2-h1-output", type=Path, default=DEFAULT_LINKED_ITEM_FILES["g2_h1"])
    parser.add_argument("--g2-h2-output", type=Path, default=DEFAULT_LINKED_ITEM_FILES["g2_h2"])
    parser.add_argument(
        "--study2-g1-output",
        type=Path,
        default=DEFAULT_STUDY2_LINKED_ITEM_FILES["study2_g1"],
    )
    parser.add_argument(
        "--study2-g2-output",
        type=Path,
        default=DEFAULT_STUDY2_LINKED_ITEM_FILES["study2_g2"],
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/audits/mimic_jpg_link_summary.json"),
        help="Output aggregate summary JSON.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not download files.")
    parser.add_argument(
        "--link-only-existing",
        action="store_true",
        help="Do not download; link only files already present under --files-root.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing files.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="Per-image HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--download-limit",
        type=int,
        default=None,
        help="Optional maximum number of missing images to attempt downloading.",
    )
    parser.add_argument(
        "--username-env",
        default=None,
        help="Environment variable containing an optional HTTP username.",
    )
    parser.add_argument(
        "--password-env",
        default=None,
        help="Environment variable containing an optional HTTP password.",
    )
    args = parser.parse_args(argv)
    if args.dry_run and args.link_only_existing:
        parser.error("--dry-run and --link-only-existing cannot be used together.")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive.")
    if args.download_limit is not None and args.download_limit < 1:
        parser.error("--download-limit must be positive when provided.")

    if args.study2_only:
        item_paths = {
            "study2_g1": args.study2_g1_items,
            "study2_g2": args.study2_g2_items,
        }
        output_paths = {
            "study2_g1": args.study2_g1_output,
            "study2_g2": args.study2_g2_output,
        }
    else:
        item_paths = {
            "g1_h1": args.g1_h1_items,
            "g1_h2": args.g1_h2_items,
            "g2_h1": args.g2_h1_items,
            "g2_h2": args.g2_h2_items,
        }
        output_paths = {
            "g1_h1": args.g1_h1_output,
            "g1_h2": args.g1_h2_output,
            "g2_h1": args.g2_h1_output,
            "g2_h2": args.g2_h2_output,
        }
    item_groups = {name: read_jsonl(path) for name, path in item_paths.items()}
    metadata = load_mimic_metadata(args.metadata, args.split)
    needed = collect_needed_images(item_groups)
    link_rows, link_summary = build_link_rows(
        needed,
        metadata,
        files_root=args.files_root,
        base_url=args.base_url,
    )
    try:
        manifest_rows, download_summary = download_manifest_rows(
            link_rows,
            dry_run=args.dry_run,
            link_only_existing=args.link_only_existing,
            overwrite=args.overwrite,
            timeout_seconds=args.timeout_seconds,
            download_limit=args.download_limit,
            username_env=args.username_env,
            password_env=args.password_env,
        )
    except KeyboardInterrupt:
        parser.exit(status=130, message="error: download interrupted by user.\n")
    available_links = available_links_from_manifest(manifest_rows)
    linked_groups: dict[str, list[dict[str, Any]]] = {}
    item_summaries: dict[str, dict[str, int]] = {}
    for name, items in item_groups.items():
        linked_groups[name], item_summaries[name] = update_items_with_links(items, available_links)

    try:
        write_parquet_rows(link_rows, args.needed_index, NEEDED_INDEX_COLUMNS)
    except RuntimeError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")
    write_csv_rows(manifest_rows, args.manifest, MANIFEST_COLUMNS)
    write_url_list(link_rows, args.url_list)
    for name, linked_items in linked_groups.items():
        write_jsonl(linked_items, output_paths[name])

    summary = {
        "linking": link_summary,
        "download": download_summary,
        "items": item_summaries,
        "dry_run": args.dry_run,
    }
    update_summary(args.summary, "mimic_jpg_linking", summary)
    print(
        "Linked MIMIC-CXR-JPG images: "
        f"requested={link_summary['unique_requested_images']}, "
        f"matched={link_summary['matched_metadata_rows']}, "
        f"missing_metadata={link_summary['missing_metadata']}, "
        f"downloaded={download_summary['downloaded_files']}, "
        f"skipped_existing={download_summary['skipped_existing_files']}, "
        f"failed={download_summary['failed_downloads']}, "
        f"failure_reasons={download_summary['failure_reasons']}, "
        f"item_summaries={item_summaries}"
    )
    return 0


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def available_links_from_manifest(
    manifest_rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Return links whose files are available or intentionally dry-run linked."""

    available_statuses = {"downloaded", "skipped_existing", "dry_run"}
    return {
        (
            str(row["patient_id"]),
            str(row["study_id"]),
            str(row["dicom_id"]),
        ): {**row, "link_status": "matched"}
        for row in manifest_rows
        if row.get("download_status") in available_statuses
    }


def _sort_key(key: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple("" if value is None else str(value) for value in key)


if __name__ == "__main__":
    raise SystemExit(main())
