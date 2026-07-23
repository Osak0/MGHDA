"""Exclude complete Study 3 four-question sets for corrupted images."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from ghm.data.exclude_corrupt_study2 import _atomic_write_jsonl, _ids
from ghm.data.repair_mimic_jpg import load_safe_relative_paths
from ghm.granularity.common import read_jsonl


EXPECTED_QUESTIONS = {
    ("state", "present"),
    ("state", "absent"),
    ("evidence", "present"),
    ("evidence", "absent"),
}


def _paths(root: Path, split: str) -> tuple[Path, Path, Path]:
    return (
        root
        / "processed/study3/v2/prompts"
        / f"study3_{split}_multiselect_model_inputs.jsonl",
        root
        / "processed/study3/v2/prompts"
        / f"study3_{split}_multiselect_eval_metadata.jsonl",
        root
        / "processed/study3/v2/items"
        / f"study3_{split}_multiselect_items.jsonl",
    )


def _validate_option_sets(
    metadata_rows: list[dict[str, Any]],
    affected_sets: set[str],
) -> set[str]:
    removed_ids: set[str] = set()
    for option_set_id in affected_sets:
        rows = [
            row
            for row in metadata_rows
            if str(row.get("option_set_id")) == option_set_id
        ]
        questions = {
            (
                str(row.get("prompt_framing")),
                str(row.get("query_relation")),
            )
            for row in rows
        }
        if len(rows) != 4 or questions != EXPECTED_QUESTIONS:
            raise ValueError("affected Study 3 option set is not a complete four-question set")
        removed_ids.update(str(row["item_id"]) for row in rows)
    return removed_ids


def exclude_corrupt_study3(
    *,
    data_root: Path,
    repair_list: Path,
    apply: bool,
) -> dict[str, Any]:
    root = data_root.resolve()
    bad_paths = {path.as_posix() for path in load_safe_relative_paths(repair_list)}
    smoke_inputs_path = root / "processed/study3/v2/smoke/model_inputs.jsonl"
    smoke_metadata_path = root / "processed/study3/v2/smoke/eval_metadata.jsonl"
    smoke_inputs = read_jsonl(smoke_inputs_path)
    smoke_metadata = read_jsonl(smoke_metadata_path)
    if set(_ids(smoke_inputs)) != set(_ids(smoke_metadata)):
        raise ValueError("Study 3 smoke input and metadata IDs differ")
    direct_smoke_matches = {
        str(row["item_id"])
        for row in smoke_inputs
        if str(row.get("image_path")) in bad_paths
    }
    if direct_smoke_matches:
        raise ValueError("the corrupt image occurs in the fixed Study 3 smoke subset")

    layers: dict[str, dict[str, Any]] = {}
    all_removed_ids: set[str] = set()
    for split in ("g1", "g2"):
        model_path, metadata_path, items_path = _paths(root, split)
        model_rows = read_jsonl(model_path)
        metadata_rows = read_jsonl(metadata_path)
        item_rows = read_jsonl(items_path)
        model_ids = set(_ids(model_rows))
        metadata_ids = set(_ids(metadata_rows))
        item_ids = set(_ids(item_rows))
        if model_ids != metadata_ids or model_ids != item_ids:
            raise ValueError("Study 3 item, model-input, and metadata IDs differ")
        metadata_by_id = {str(row["item_id"]): row for row in metadata_rows}
        direct_ids = {
            str(row["item_id"])
            for row in model_rows
            if str(row.get("image_path")) in bad_paths
        }
        affected_sets = {
            str(metadata_by_id[item_id].get("option_set_id"))
            for item_id in direct_ids
        }
        if "None" in affected_sets:
            raise ValueError("affected Study 3 row is missing option_set_id")
        removed_ids = (
            _validate_option_sets(metadata_rows, affected_sets)
            if affected_sets
            else set()
        )
        if direct_ids != removed_ids:
            raise ValueError("not every row in an affected option set uses the corrupt image")
        if all_removed_ids.intersection(removed_ids):
            raise ValueError("Study 3 item IDs overlap between granularities")
        all_removed_ids.update(removed_ids)
        removed_metadata = [
            row for row in metadata_rows if str(row["item_id"]) in removed_ids
        ]
        strata = Counter(
            (
                str(row.get("prompt_framing")),
                str(row.get("query_relation")),
                str(row.get("variant")),
                str(row.get("controlled_k")),
            )
            for row in removed_metadata
        )
        layers[split] = {
            "model_path": model_path,
            "metadata_path": metadata_path,
            "items_path": items_path,
            "model_rows": model_rows,
            "metadata_rows": metadata_rows,
            "item_rows": item_rows,
            "removed_ids": removed_ids,
            "affected_sets": affected_sets,
            "strata": strata,
        }

    if not all_removed_ids:
        raise ValueError("the listed corrupt image is not present in Study 3 full inputs")
    if set(_ids(smoke_inputs)).intersection(all_removed_ids):
        raise ValueError("an affected Study 3 option set occurs in smoke")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "applied": apply,
        "bad_unique_images": len(bad_paths),
        "smoke_records": len(smoke_inputs),
        "smoke_removed": 0,
        "study3": {},
    }
    changed_paths: list[Path] = []
    backup_root = root / "outputs/audits/pre_corrupt_image_exclusion"
    for split, layer in layers.items():
        removed_ids = layer["removed_ids"]
        kept_model = [
            row
            for row in layer["model_rows"]
            if str(row["item_id"]) not in removed_ids
        ]
        kept_metadata = [
            row
            for row in layer["metadata_rows"]
            if str(row["item_id"]) not in removed_ids
        ]
        kept_items = [
            row
            for row in layer["item_rows"]
            if str(row["item_id"]) not in removed_ids
        ]
        if not (
            set(_ids(kept_model))
            == set(_ids(kept_metadata))
            == set(_ids(kept_items))
        ):
            raise ValueError("filtered Study 3 layers have different ID sets")
        summary["study3"][split] = {
            "before": len(layer["model_rows"]),
            "removed": len(removed_ids),
            "after": len(kept_model),
            "option_sets_removed": len(layer["affected_sets"]),
            "removed_strata": {
                "|".join(key): value
                for key, value in sorted(layer["strata"].items())
            },
        }
        if apply and removed_ids:
            for path, rows in (
                (layer["model_path"], kept_model),
                (layer["metadata_path"], kept_metadata),
                (layer["items_path"], kept_items),
            ):
                relative = path.relative_to(root)
                backup = backup_root / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup)
                _atomic_write_jsonl(path, rows)
                changed_paths.append(relative)

    audit_path = root / "outputs/audits/corrupt_image_exclusion_study3_summary.json"
    private_manifest = (
        root / "outputs/audits/corrupt_image_exclusion_study3_changed_files_private.txt"
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
        summary = exclude_corrupt_study3(
            data_root=args.data_root,
            repair_list=args.repair_list,
            apply=args.apply,
        )
    except Exception as exc:  # noqa: BLE001 - keep private row values out of logs.
        print(
            f"Study 3 corrupt-image exclusion failed: {exc.__class__.__name__}",
            file=os.sys.stderr,
        )
        return 1
    print(
        "Study 3 corrupt-image exclusion: "
        f"applied={summary['applied']}, "
        f"bad_unique_images={summary['bad_unique_images']}, "
        f"smoke_records={summary['smoke_records']}, smoke_removed=0"
    )
    for split, values in sorted(summary["study3"].items()):
        print(
            f"{split}: before={values['before']}, removed={values['removed']}, "
            f"after={values['after']}, "
            f"option_sets_removed={values['option_sets_removed']}"
        )
    print("Private paths, option-set IDs, and item IDs were not printed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
