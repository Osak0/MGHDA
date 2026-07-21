"""Create the combined Study 2 v1/v2 and Study 3 v2 private archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from ghm.granularity.common import read_jsonl
from ghm.migration.bundle import sha256_file


INTERNAL_CHECKSUM = Path("outputs/transfer/private_files.sha256")
INTERNAL_MANIFEST = Path("outputs/transfer/private_bundle_manifest.json")


def create_private_archive(
    *,
    data_root: Path,
    output_dir: Path,
    git_commit: str,
) -> dict[str, Any]:
    """Validate fixed versioned layers and stream only required files into tar.gz."""

    data_root = data_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = _dataset_paths(data_root)
    archive_sources: dict[Path, Path] = {}
    image_paths: set[Path] = set()
    dataset_ids: dict[str, set[str]] = {}
    template_ids: set[str] = set()
    study3_counts: Counter[str] = Counter()

    for name, (model_path, metadata_path) in datasets.items():
        model_rows = read_jsonl(model_path)
        metadata_rows = read_jsonl(metadata_path)
        model_ids = _validated_rows(model_rows, model_inputs=True)
        metadata_ids = _validated_rows(metadata_rows, model_inputs=False)
        if model_ids != metadata_ids:
            raise ValueError(f"{name}: model-input and metadata item IDs differ")
        dataset_ids[name] = model_ids
        _validate_template_ids(name, model_rows)
        _add_source(archive_sources, data_root, model_path)
        _add_source(archive_sources, data_root, metadata_path)
        for row in model_rows:
            relative = _portable_image_path(row.get("image_path"))
            image_path = (data_root / relative).resolve()
            if not image_path.is_relative_to(data_root) or not image_path.is_file():
                raise FileNotFoundError(f"{name}: one or more images are missing")
            image_paths.add(image_path)
            template_ids.add(str(row.get("prompt_template_id")))
        if name.startswith("study3_full"):
            for row in metadata_rows:
                study3_counts[
                    "|".join(
                        str(row.get(field))
                        for field in (
                            "granularity",
                            "prompt_framing",
                            "query_relation",
                            "variant",
                            "controlled_k",
                        )
                    )
                ] += 1

    _validate_cross_dataset_contracts(dataset_ids, datasets)
    for version in ("v1", "v2"):
        split_ids: set[str] = set()
        for split in ("g1", "g2"):
            model_path = (
                data_root
                / f"processed/study2/ablation/{version}/{split}_model_inputs.jsonl"
            )
            metadata_path = (
                data_root
                / f"processed/study2/ablation/{version}/{split}_eval_metadata.jsonl"
            )
            model_ids = _validated_rows(read_jsonl(model_path), model_inputs=True)
            metadata_ids = _validated_rows(read_jsonl(metadata_path), model_inputs=False)
            if model_ids != metadata_ids or split_ids.intersection(model_ids):
                raise ValueError("Study 2 ablation split layers are invalid")
            split_ids.update(model_ids)
            _add_source(archive_sources, data_root, model_path)
            _add_source(archive_sources, data_root, metadata_path)
        if split_ids != dataset_ids[f"study2_ablation_{version}"]:
            raise ValueError("Study 2 ablation split IDs differ from combined layer")
    _validate_study3_four_question_contract(
        [
            row
            for name in ("study3_full_g1", "study3_full_g2")
            for row in read_jsonl(datasets[name][1])
        ]
    )
    _validate_study3_smoke_contract(read_jsonl(datasets["study3_smoke"][1]))
    for path in image_paths:
        _add_source(archive_sources, data_root, path)
    study3_item_ids: set[str] = set()
    for path in (
        data_root / "processed/study3/v2/items/study3_g1_multiselect_items.jsonl",
        data_root / "processed/study3/v2/items/study3_g2_multiselect_items.jsonl",
    ):
        study3_item_ids.update(_validated_rows(read_jsonl(path), model_inputs=False))
        _add_source(archive_sources, data_root, path)
    full_study3_ids = dataset_ids["study3_full_g1"] | dataset_ids["study3_full_g2"]
    if study3_item_ids != full_study3_ids:
        raise ValueError("Study 3 item and prompt IDs must be identical")

    checksums = {
        relative.as_posix(): sha256_file(source)
        for relative, source in sorted(
            archive_sources.items(),
            key=lambda pair: pair[0].as_posix(),
        )
    }
    manifest = {
        "bundle_schema_version": 2,
        "experiment_ids": [
            "study2_g1_g2_claim_verification",
            "study3_multiselect_v2",
        ],
        "git_commit": git_commit,
        "datasets": {name: len(ids) for name, ids in sorted(dataset_ids.items())},
        "unique_images": len(image_paths),
        "files": len(archive_sources) + 2,
        "prompt_template_ids": sorted(template_ids),
        "study3_counts": dict(sorted(study3_counts.items())),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    checksum_bytes = "".join(
        f"{digest}  {relative}\n"
        for relative, digest in {
            **checksums,
            INTERNAL_MANIFEST.as_posix(): hashlib.sha256(manifest_bytes).hexdigest(),
        }.items()
    ).encode("utf-8")
    commit_token = git_commit.strip()
    if not commit_token or any(
        character not in "0123456789abcdef"
        for character in commit_token.lower()
    ):
        raise ValueError("git_commit must be a hexadecimal commit ID")
    archive_path = output_dir / (
        f"MGHDA-private-study2v1v2-study3v2-{commit_token}.tar.gz"
    )
    if archive_path.exists():
        raise FileExistsError("private archive already exists; use a clean output directory")
    with tarfile.open(archive_path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for relative, source in sorted(
            archive_sources.items(),
            key=lambda pair: pair[0].as_posix(),
        ):
            archive.add(source, arcname=relative.as_posix(), recursive=False)
        _add_bytes(archive, INTERNAL_MANIFEST, manifest_bytes)
        _add_bytes(archive, INTERNAL_CHECKSUM, checksum_bytes)
    _write_outer_checksum(archive_path)
    return {**manifest, "archive_bytes": archive_path.stat().st_size}


def _dataset_paths(data_root: Path) -> dict[str, tuple[Path, Path]]:
    return {
        "study2_full_v1_g1": (
            data_root / "processed/study2/v1/study2_g1_model_inputs.jsonl",
            data_root / "processed/study2/v1/study2_g1_eval_metadata.jsonl",
        ),
        "study2_full_v1_g2": (
            data_root / "processed/study2/v1/study2_g2_model_inputs.jsonl",
            data_root / "processed/study2/v1/study2_g2_eval_metadata.jsonl",
        ),
        "study2_full_v2_g1": (
            data_root / "processed/study2/v2/study2_g1_model_inputs.jsonl",
            data_root / "processed/study2/v2/study2_g1_eval_metadata.jsonl",
        ),
        "study2_full_v2_g2": (
            data_root / "processed/study2/v2/study2_g2_model_inputs.jsonl",
            data_root / "processed/study2/v2/study2_g2_eval_metadata.jsonl",
        ),
        "study2_ablation_v1": (
            data_root / "processed/study2/ablation/v1/model_inputs.jsonl",
            data_root / "processed/study2/ablation/v1/eval_metadata.jsonl",
        ),
        "study2_ablation_v2": (
            data_root / "processed/study2/ablation/v2/model_inputs.jsonl",
            data_root / "processed/study2/ablation/v2/eval_metadata.jsonl",
        ),
        "study3_full_g1": (
            data_root
            / "processed/study3/v2/prompts/study3_g1_multiselect_model_inputs.jsonl",
            data_root
            / "processed/study3/v2/prompts/study3_g1_multiselect_eval_metadata.jsonl",
        ),
        "study3_full_g2": (
            data_root
            / "processed/study3/v2/prompts/study3_g2_multiselect_model_inputs.jsonl",
            data_root
            / "processed/study3/v2/prompts/study3_g2_multiselect_eval_metadata.jsonl",
        ),
        "study3_smoke": (
            data_root / "processed/study3/v2/smoke/model_inputs.jsonl",
            data_root / "processed/study3/v2/smoke/eval_metadata.jsonl",
        ),
    }


def _validated_rows(
    rows: list[dict[str, Any]],
    *,
    model_inputs: bool,
) -> set[str]:
    ids: list[str] = []
    forbidden = {
        "answer_label",
        "gold_selected_options",
        "source_assertions",
        "polarity",
        "evidence_state",
    }
    for row in rows:
        if row.get("item_id") is None:
            raise ValueError("transfer layer contains a row without item_id")
        ids.append(str(row["item_id"]))
        if model_inputs and forbidden.intersection(row):
            raise ValueError("model input contains answer leakage")
    if len(ids) != len(set(ids)):
        raise ValueError("transfer layer contains duplicate item_id values")
    return set(ids)


def _validate_template_ids(name: str, rows: list[dict[str, Any]]) -> None:
    template_ids = {str(row.get("prompt_template_id")) for row in rows}
    if name.startswith("study2_full_v1") or name == "study2_ablation_v1":
        expected = {"claim_verification_abc_v1"}
    elif name.startswith("study2_full_v2") or name == "study2_ablation_v2":
        expected = {"claim_verification_abc_definitions_v2"}
    elif name.startswith("study3"):
        expected = {
            f"study3_multiselect_{framing}_{relation}_v2"
            for framing in ("state", "evidence")
            for relation in ("present", "absent")
        }
        if name == "study3_smoke" and template_ids != expected:
            raise ValueError("Study 3 smoke must cover all four prompt templates")
        if not template_ids.issubset(expected):
            raise ValueError(f"{name} contains an unexpected prompt template")
        return
    else:
        return
    if template_ids != expected:
        raise ValueError(f"{name} contains an unexpected prompt template")


def _validate_cross_dataset_contracts(
    dataset_ids: dict[str, set[str]],
    datasets: dict[str, tuple[Path, Path]],
) -> None:
    if dataset_ids["study2_ablation_v1"] != dataset_ids["study2_ablation_v2"]:
        raise ValueError("Study 2 v1/v2 ablation item IDs must be identical")
    full_study2_v1 = (
        dataset_ids["study2_full_v1_g1"] | dataset_ids["study2_full_v1_g2"]
    )
    full_study2_v2 = (
        dataset_ids["study2_full_v2_g1"] | dataset_ids["study2_full_v2_g2"]
    )
    if any(
        len(dataset_ids[name]) != 480
        for name in (
            "study2_full_v1_g1",
            "study2_full_v1_g2",
            "study2_full_v2_g1",
            "study2_full_v2_g2",
        )
    ):
        raise ValueError("Study 2 full v1/v2 must contain 480 records per granularity")
    if len(full_study2_v1) != 960 or len(full_study2_v2) != 960:
        raise ValueError("Study 2 full v1/v2 must each contain 960 unique items")
    if full_study2_v1 != full_study2_v2:
        raise ValueError("Study 2 full v1/v2 item IDs must be identical")
    _validate_paired_prompt_sets(
        datasets,
        left_names=("study2_full_v1_g1", "study2_full_v1_g2"),
        right_names=("study2_full_v2_g1", "study2_full_v2_g2"),
        label="Study 2 full v1/v2",
    )
    if len(dataset_ids["study2_ablation_v1"]) != 120:
        raise ValueError("Study 2 paired ablation must contain exactly 120 unique items")
    if not dataset_ids["study2_ablation_v1"].issubset(full_study2_v1):
        raise ValueError("Study 2 ablation IDs must be a subset of full v1/v2")
    v1_metadata = read_jsonl(datasets["study2_ablation_v1"][1])
    v2_metadata = read_jsonl(datasets["study2_ablation_v2"][1])
    if v1_metadata != v2_metadata:
        raise ValueError("Study 2 paired ablation metadata must be identical")
    ablation_counts = Counter(
        (
            str(row.get("granularity")),
            str(row.get("evidence_state")),
            str(row.get("claim_polarity")),
        )
        for row in v1_metadata
    )
    if len(ablation_counts) != 12 or set(ablation_counts.values()) != {10}:
        raise ValueError("Study 2 ablation must contain 10 records in each of 12 strata")
    full_study3 = dataset_ids["study3_full_g1"] | dataset_ids["study3_full_g2"]
    if any(
        len(dataset_ids[name]) != 1800
        for name in ("study3_full_g1", "study3_full_g2")
    ):
        raise ValueError("Study 3 full inputs must contain 1800 records per granularity")
    if len(full_study3) != 3600:
        raise ValueError("Study 3 v2 full inputs must contain exactly 3600 unique items")
    if len(dataset_ids["study3_smoke"]) != 40:
        raise ValueError("Study 3 smoke inputs must contain exactly 40 unique items")
    if not dataset_ids["study3_smoke"].issubset(full_study3):
        raise ValueError("Study 3 smoke IDs must be a subset of full inputs")
    _validate_paired_prompt_sets(
        datasets,
        left_names=("study2_ablation_v1",),
        right_names=("study2_ablation_v2",),
        label="Study 2 ablation v1/v2",
    )


def _validate_paired_prompt_sets(
    datasets: dict[str, tuple[Path, Path]],
    *,
    left_names: tuple[str, ...],
    right_names: tuple[str, ...],
    label: str,
) -> None:
    left_inputs = [row for name in left_names for row in read_jsonl(datasets[name][0])]
    right_inputs = [row for name in right_names for row in read_jsonl(datasets[name][0])]
    left_metadata = [row for name in left_names for row in read_jsonl(datasets[name][1])]
    right_metadata = [row for name in right_names for row in read_jsonl(datasets[name][1])]
    if left_metadata != right_metadata:
        raise ValueError(f"{label} metadata must be identical")
    left_by_id = {str(row["item_id"]): row for row in left_inputs}
    right_by_id = {str(row["item_id"]): row for row in right_inputs}
    if set(left_by_id) != set(right_by_id):
        raise ValueError(f"{label} input IDs differ")
    for item_id in left_by_id:
        left_base = {
            key: value
            for key, value in left_by_id[item_id].items()
            if key not in {"prompt", "prompt_template_id"}
        }
        right_base = {
            key: value
            for key, value in right_by_id[item_id].items()
            if key not in {"prompt", "prompt_template_id"}
        }
        if left_base != right_base:
            raise ValueError(f"{label} differ beyond prompt text/template ID")


def _validate_study3_four_question_contract(rows: list[dict[str, Any]]) -> None:
    by_set: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_set.setdefault(str(row.get("option_set_id")), []).append(row)
    expected = {
        ("state", "present"),
        ("state", "absent"),
        ("evidence", "present"),
        ("evidence", "absent"),
    }
    for set_rows in by_set.values():
        keys = {
            (str(row.get("prompt_framing")), str(row.get("query_relation")))
            for row in set_rows
        }
        if len(set_rows) != 4 or keys != expected:
            raise ValueError("each Study 3 option set must have exactly four questions")
        options = {
            json.dumps(row.get("options"), sort_keys=True) for row in set_rows
        }
        if len(options) != 1:
            raise ValueError("Study 3 four-question option order differs")
        for relation in ("present", "absent"):
            gold = {
                json.dumps(row.get("gold_selected_options"), sort_keys=True)
                for row in set_rows
                if row.get("query_relation") == relation
            }
            if len(gold) != 1:
                raise ValueError("Study 3 state/evidence gold differs for one relation")
    for granularity in ("G1_finding_existence", "G2_anatomical_localization"):
        granularity_rows = [
            row for row in rows if row.get("granularity") == granularity
        ]
        natural = sum(row.get("variant") == "natural" for row in granularity_rows)
        controlled = sum(
            row.get("variant") == "controlled" for row in granularity_rows
        )
        if natural != 1000 or controlled != 800:
            raise ValueError(
                "Study 3 must contain 1000 natural and 800 controlled records "
                "per granularity"
            )


def _validate_study3_smoke_contract(rows: list[dict[str, Any]]) -> None:
    cells = {
        (
            str(row.get("granularity")),
            str(row.get("prompt_framing")),
            str(row.get("query_relation")),
            (
                "natural"
                if row.get("variant") == "natural"
                else f"controlled_k{row.get('controlled_k')}"
            ),
        )
        for row in rows
    }
    expected = {
        (granularity, framing, relation, variant)
        for granularity in ("G1_finding_existence", "G2_anatomical_localization")
        for framing in ("state", "evidence")
        for relation in ("present", "absent")
        for variant in (
            "natural",
            "controlled_k2",
            "controlled_k3",
            "controlled_k4",
            "controlled_k5",
        )
    }
    if len(rows) != 40 or cells != expected:
        raise ValueError("Study 3 smoke must contain the fixed 40 coverage cells")


def _add_source(sources: dict[Path, Path], data_root: Path, source: Path) -> None:
    source = source.resolve()
    if not source.is_relative_to(data_root) or not source.is_file() or source.is_symlink():
        raise ValueError("bundle source must be a regular file inside data root")
    relative = source.relative_to(data_root)
    previous = sources.get(relative)
    if previous is not None and previous != source:
        raise ValueError("bundle contains conflicting destination paths")
    sources[relative] = source


def _portable_image_path(value: Any) -> Path:
    if not value:
        raise ValueError("model input contains an empty image_path")
    pure = PurePosixPath(str(value).replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or pure.parts[:1] != ("files",):
        raise ValueError("image_path must be a relative path under files/")
    return Path(*pure.parts)


def _add_bytes(archive: tarfile.TarFile, relative: Path, content: bytes) -> None:
    info = tarfile.TarInfo(relative.as_posix())
    info.size = len(content)
    info.mode = 0o600
    archive.addfile(info, io.BytesIO(content))


def _write_outer_checksum(path: Path) -> None:
    sidecar = Path(f"{path}.sha256")
    sidecar.write_text(
        f"{sha256_file(path)}  {path.name}\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args(argv)
    summary = create_private_archive(
        data_root=args.data_root,
        output_dir=args.output_dir,
        git_commit=args.git_commit,
    )
    print(
        "Created combined private archive: "
        f"datasets={len(summary['datasets'])}, "
        f"unique_images={summary['unique_images']}, files={summary['files']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
