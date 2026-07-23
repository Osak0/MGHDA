"""Exclude complete Study 2 claim pairs that reference corrupted images.

The command operates only on separated model-input/eval-metadata JSONL layers.
It requires v1/v2 to remove identical item IDs, refuses to alter the fixed
ablation subset, backs up every changed file, and prints aggregate counts only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from ghm.data.repair_mimic_jpg import load_safe_relative_paths
from ghm.granularity.common import read_jsonl
from ghm.granularity.study2 import CLAIM_NEGATIVE, CLAIM_POSITIVE


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    values = [str(row.get("item_id")) for row in rows if row.get("item_id") is not None]
    if len(values) != len(rows) or len(values) != len(set(values)):
        raise ValueError("input layer contains missing or duplicate item IDs")
    return values


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".corrupt-exclusion.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _pair_counts(
    metadata_rows: list[dict[str, Any]],
    removed_ids: set[str],
) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    groups: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in metadata_rows:
        if str(row.get("item_id")) not in removed_ids:
            continue
        evidence_state = str(row.get("evidence_state"))
        polarity = str(row.get("claim_polarity"))
        counts[(evidence_state, polarity)] += 1
        groups[
            (
                str(row.get("granularity")),
                evidence_state,
                str(row.get("target_finding")),
                str(row.get("target_anatomy")),
            )
        ].add(polarity)
    expected = {CLAIM_POSITIVE, CLAIM_NEGATIVE}
    if not groups or any(polarities != expected for polarities in groups.values()):
        raise ValueError("corrupt-image exclusions are not complete claim pairs")
    return counts


def _dataset_paths(data_root: Path, version: str, split: str) -> tuple[Path, Path]:
    directory = data_root / f"processed/study2/{version}"
    return (
        directory / f"study2_{split}_model_inputs.jsonl",
        directory / f"study2_{split}_eval_metadata.jsonl",
    )


def _study3_impact(data_root: Path, bad_paths: set[str]) -> dict[str, int]:
    candidates = {
        "full_g1": (
            data_root
            / "processed/study3/v2/prompts/"
            "study3_g1_multiselect_model_inputs.jsonl"
        ),
        "full_g2": (
            data_root
            / "processed/study3/v2/prompts/"
            "study3_g2_multiselect_model_inputs.jsonl"
        ),
        "smoke": data_root / "processed/study3/v2/smoke/model_inputs.jsonl",
    }
    result: dict[str, int] = {}
    for name, path in candidates.items():
        if not path.is_file():
            result[name] = -1
            continue
        result[name] = sum(
            str(row.get("image_path")) in bad_paths for row in read_jsonl(path)
        )
    return result


def exclude_corrupt_study2(
    *,
    data_root: Path,
    repair_list: Path,
    apply: bool,
) -> dict[str, Any]:
    root = data_root.resolve()
    bad_paths = {path.as_posix() for path in load_safe_relative_paths(repair_list)}

    ablation_matches: dict[str, int] = {}
    for version in ("v1", "v2"):
        for split in ("g1", "g2"):
            path = (
                root
                / f"processed/study2/ablation/{version}/{split}_model_inputs.jsonl"
            )
            rows = read_jsonl(path)
            ablation_matches[f"{version}_{split}"] = sum(
                str(row.get("image_path")) in bad_paths for row in rows
            )
    if any(ablation_matches.values()):
        raise ValueError("a corrupt image occurs in the completed ablation subset")

    layers: dict[tuple[str, str], dict[str, Any]] = {}
    removed_by_dataset: dict[tuple[str, str], set[str]] = {}
    for version in ("v1", "v2"):
        for split in ("g1", "g2"):
            model_path, metadata_path = _dataset_paths(root, version, split)
            model_rows = read_jsonl(model_path)
            metadata_rows = read_jsonl(metadata_path)
            model_ids = set(_ids(model_rows))
            metadata_ids = set(_ids(metadata_rows))
            if model_ids != metadata_ids:
                raise ValueError("Study 2 model-input and metadata IDs differ")
            removed_ids = {
                str(row["item_id"])
                for row in model_rows
                if str(row.get("image_path")) in bad_paths
            }
            if removed_ids:
                pair_counts = _pair_counts(metadata_rows, removed_ids)
            else:
                pair_counts = Counter()
            layers[(version, split)] = {
                "model_path": model_path,
                "metadata_path": metadata_path,
                "model_rows": model_rows,
                "metadata_rows": metadata_rows,
                "pair_counts": pair_counts,
            }
            removed_by_dataset[(version, split)] = removed_ids

    for split in ("g1", "g2"):
        if removed_by_dataset[("v1", split)] != removed_by_dataset[("v2", split)]:
            raise ValueError("Study 2 v1/v2 corrupt-image exclusion IDs differ")
    if not any(removed_by_dataset.values()):
        raise ValueError("the listed corrupt image is not present in Study 2 full inputs")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "applied": apply,
        "bad_unique_images": len(bad_paths),
        "ablation_matches": ablation_matches,
        "study2": {},
        "study3_bad_image_records": _study3_impact(root, bad_paths),
    }
    changed_paths: list[Path] = []
    backup_root = root / "outputs/audits/pre_corrupt_image_exclusion"
    for (version, split), layer in layers.items():
        removed_ids = removed_by_dataset[(version, split)]
        model_rows = layer["model_rows"]
        metadata_rows = layer["metadata_rows"]
        kept_model = [
            row for row in model_rows if str(row.get("item_id")) not in removed_ids
        ]
        kept_metadata = [
            row for row in metadata_rows if str(row.get("item_id")) not in removed_ids
        ]
        if set(_ids(kept_model)) != set(_ids(kept_metadata)):
            raise ValueError("filtered Study 2 input and metadata IDs differ")
        key = f"{version}_{split}"
        summary["study2"][key] = {
            "before": len(model_rows),
            "removed": len(removed_ids),
            "after": len(kept_model),
            "removed_strata": {
                f"{evidence_state}|{polarity}": count
                for (evidence_state, polarity), count in sorted(
                    layer["pair_counts"].items()
                )
            },
        }
        if apply and removed_ids:
            for path, rows in (
                (layer["model_path"], kept_model),
                (layer["metadata_path"], kept_metadata),
            ):
                relative = path.relative_to(root)
                backup = backup_root / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup)
                _atomic_write_jsonl(path, rows)
                changed_paths.append(relative)

    audit_path = root / "outputs/audits/corrupt_image_exclusion_summary.json"
    private_manifest = (
        root / "outputs/audits/corrupt_image_exclusion_changed_files_private.txt"
    )
    if apply:
        summary["applied"] = True
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        changed_paths.append(audit_path.relative_to(root))
        manifest_relative = private_manifest.relative_to(root)
        private_manifest.write_text(
            "\n".join(
                path.as_posix()
                for path in sorted([*changed_paths, manifest_relative])
            )
            + "\n",
            encoding="utf-8",
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repair-list", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    try:
        summary = exclude_corrupt_study2(
            data_root=args.data_root,
            repair_list=args.repair_list,
            apply=args.apply,
        )
    except Exception as exc:  # noqa: BLE001 - do not expose private row values.
        print(
            f"Study 2 corrupt-image exclusion failed: {exc.__class__.__name__}",
            file=os.sys.stderr,
        )
        return 1

    print(
        "Study 2 corrupt-image exclusion: "
        f"applied={summary['applied']}, "
        f"bad_unique_images={summary['bad_unique_images']}"
    )
    for key, values in sorted(summary["study2"].items()):
        print(
            f"{key}: before={values['before']}, "
            f"removed={values['removed']}, after={values['after']}"
        )
    impact = summary["study3_bad_image_records"]
    print(
        "Study 3 bad-image records: "
        f"full_g1={impact['full_g1']}, "
        f"full_g2={impact['full_g2']}, smoke={impact['smoke']}"
    )
    print("Private paths and item IDs were not printed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
