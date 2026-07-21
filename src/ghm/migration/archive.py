"""Inspect archives and verify an extracted private bundle safely."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath

from ghm.migration.bundle import sha256_file, verify_checksum_file


def inspect_archive(path: Path) -> dict[str, int]:
    """Reject traversal paths, links, devices, and duplicate members."""

    names: set[str] = set()
    files = 0
    directories = 0
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                raise ValueError("archive contains an unsafe path")
            normalized = pure.as_posix()
            if normalized in names:
                raise ValueError("archive contains duplicate member paths")
            names.add(normalized)
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError("archive links and device nodes are forbidden")
            if member.isdir():
                directories += 1
            elif member.isfile():
                files += 1
            else:
                raise ValueError("archive contains an unsupported member type")
    return {"files": files, "directories": directories}


def verify_outer_checksum(archive: Path, sidecar: Path) -> bool:
    line = sidecar.read_text(encoding="utf-8").strip()
    expected, filename = line.split("  ", 1)
    return filename == archive.name and sha256_file(archive) == expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--archive", type=Path, required=True)
    outer = subparsers.add_parser("verify-outer")
    outer.add_argument("--archive", type=Path, required=True)
    outer.add_argument("--sidecar", type=Path, required=True)
    internal = subparsers.add_parser("verify-internal")
    internal.add_argument("--data-root", type=Path, required=True)
    internal.add_argument(
        "--checksum",
        type=Path,
        default=Path("outputs/transfer/private_files.sha256"),
    )
    commit = subparsers.add_parser("verify-commit")
    commit.add_argument("--data-root", type=Path, required=True)
    commit.add_argument("--expected", required=True)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        result = inspect_archive(args.archive)
        print(
            f"Archive inspection passed: files={result['files']}, "
            f"directories={result['directories']}"
        )
        return 0
    if args.command == "verify-outer":
        passed = verify_outer_checksum(args.archive, args.sidecar)
        print(f"Outer checksum: status={'pass' if passed else 'fail'}")
        return 0 if passed else 1
    if args.command == "verify-commit":
        manifest = args.data_root / "outputs/transfer/private_bundle_manifest.json"
        with manifest.open("r", encoding="utf-8") as file:
            actual = json.load(file).get("git_commit")
        passed = actual == args.expected
        print(f"Bundle commit identity: status={'pass' if passed else 'fail'}")
        return 0 if passed else 1
    checksum = args.checksum
    if not checksum.is_absolute():
        checksum = args.data_root / checksum
    result = verify_checksum_file(data_root=args.data_root, checksum_path=checksum)
    print(
        "Internal checksum: "
        f"checked={result['checked_files']}, failed={result['failed_files']}"
    )
    return 0 if result["failed_files"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
