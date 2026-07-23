"""Safely replace explicitly listed, corrupted MIMIC-CXR-JPG files.

This utility is intentionally narrow: it only downloads paths present in a
private operator-provided list, verifies each replacement with Pillow before
an atomic replace, and keeps the original file in a private audit directory.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Callable

from PIL import Image

from ghm.data.link_and_download_mimic_jpg import (
    DEFAULT_BASE_URL,
    build_auth_headers,
    classify_download_error,
    download_file,
    join_url,
)


DownloadFunction = Callable[..., None]


def load_safe_relative_paths(path: Path) -> list[PurePosixPath]:
    """Load unique ``files/...`` paths without exposing them in logs."""

    values = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not values:
        raise ValueError("the private repair list is empty")

    result: list[PurePosixPath] = []
    for value in sorted(values):
        relative = PurePosixPath(value)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) < 2
            or relative.parts[0] != "files"
        ):
            raise ValueError("repair list contains an unsafe path")
        result.append(relative)
    return result


def strict_image_decode(path: Path) -> None:
    """Require Pillow to decode the complete JPEG without truncation mode."""

    with Image.open(path) as image:
        image.load()
        if image.format != "JPEG":
            raise ValueError("downloaded replacement is not a JPEG")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repair_images(
    *,
    data_root: Path,
    repair_list: Path,
    headers: dict[str, str],
    base_url: str = DEFAULT_BASE_URL,
    timeout_seconds: float = 180.0,
    downloader: DownloadFunction = download_file,
) -> dict[str, object]:
    """Repair listed images and return an aggregate-safe summary."""

    root = data_root.resolve()
    relative_paths = load_safe_relative_paths(repair_list)
    backup_root = root / "outputs/audits/pre_repair_local_truncated_images"
    private_audit_path = (
        root / "outputs/audits/study2_image_repair_local_private.json"
    )

    repaired: list[dict[str, object]] = []
    already_valid = 0
    for relative in relative_paths:
        destination = (root / Path(*relative.parts)).resolve()
        if root not in destination.parents:
            raise ValueError("resolved repair path escaped the data root")
        if not destination.is_file():
            raise FileNotFoundError("a listed local image is missing")

        try:
            strict_image_decode(destination)
        except Exception:  # noqa: BLE001 - a corrupt image is the expected input.
            pass
        else:
            already_valid += 1
            continue

        before_sha256 = sha256_file(destination)
        source_relative = PurePosixPath(*relative.parts[1:])
        source_url = join_url(base_url, source_relative.as_posix())
        temporary = destination.with_name(destination.name + ".repair-download.tmp")
        backup = backup_root / Path(*relative.parts)

        try:
            temporary.unlink(missing_ok=True)
            downloader(
                source_url,
                temporary,
                headers=headers,
                timeout_seconds=timeout_seconds,
            )
            strict_image_decode(temporary)
            after_sha256 = sha256_file(temporary)
            if after_sha256 == before_sha256:
                raise ValueError("replacement is identical to the corrupted file")

            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

        repaired.append(
            {
                "relative_path": relative.as_posix(),
                "before_sha256": before_sha256,
                "after_sha256": after_sha256,
                "source_base_url": base_url,
                "strict_decode_after_repair": True,
            }
        )

    private_audit_path.parent.mkdir(parents=True, exist_ok=True)
    private_audit_path.write_text(
        json.dumps(
            {
                "listed_files": len(relative_paths),
                "repaired_files": repaired,
                "already_valid_files": already_valid,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "listed_files": len(relative_paths),
        "repaired_files": len(repaired),
        "already_valid_files": already_valid,
        "strict_decode_failures": 0,
    }


def _credentials(
    username_env: str,
    password_env: str,
) -> tuple[str, str]:
    username = os.environ.get(username_env)
    password = os.environ.get(password_env)
    if bool(username) != bool(password):
        raise ValueError("both PhysioNet credential variables must be set")
    if username and password:
        return username, password
    if not sys.stdin.isatty():
        raise RuntimeError(
            "credentials are unavailable because standard input is not a terminal"
        )
    username = input("PhysioNet username: ").strip()
    password = getpass.getpass("PhysioNet password: ")
    if not username or not password:
        raise ValueError("PhysioNet credentials are required")
    return username, password


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repair-list", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--username-env",
        default="PHYSIONET_USERNAME",
        help="Optional environment variable containing the PhysioNet username.",
    )
    parser.add_argument(
        "--password-env",
        default="PHYSIONET_PASSWORD",
        help="Optional environment variable containing the PhysioNet password.",
    )
    args = parser.parse_args(argv)

    try:
        username, password = _credentials(args.username_env, args.password_env)
        os.environ["_MGHDA_REPAIR_USERNAME"] = username
        os.environ["_MGHDA_REPAIR_PASSWORD"] = password
        try:
            summary = repair_images(
                data_root=args.data_root,
                repair_list=args.repair_list,
                headers=build_auth_headers(
                    username_env="_MGHDA_REPAIR_USERNAME",
                    password_env="_MGHDA_REPAIR_PASSWORD",
                ),
                base_url=args.base_url,
                timeout_seconds=args.timeout_seconds,
            )
        finally:
            os.environ.pop("_MGHDA_REPAIR_USERNAME", None)
            os.environ.pop("_MGHDA_REPAIR_PASSWORD", None)
            password = ""
    except Exception as exc:  # noqa: BLE001 - keep private paths out of tracebacks.
        print(
            "MIMIC JPG repair failed: "
            f"error_type={classify_download_error(exc)}",
            file=sys.stderr,
        )
        print(
            "The original image was not replaced. "
            "Private paths, URLs, credentials, and hashes were not printed.",
            file=sys.stderr,
        )
        return 1

    print(
        "MIMIC JPG repair completed: "
        f"listed={summary['listed_files']}, "
        f"repaired={summary['repaired_files']}, "
        f"already_valid={summary['already_valid_files']}, "
        f"strict_decode_failures={summary['strict_decode_failures']}"
    )
    print("Private paths, URLs, credentials, and hashes were not printed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
