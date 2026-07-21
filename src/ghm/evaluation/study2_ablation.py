"""Build and compare the fixed paired Study 2 prompt ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from ghm.evaluation.score_closed_qa import (
    ANSWER_SUPPORTED,
    summarize_scores,
)
from ghm.granularity.common import read_jsonl, write_jsonl
from ghm.prompts.build_prompts import build_prompt_layers


STRATUM_FIELDS = ("granularity", "evidence_state", "claim_polarity")


def select_paired_ablation_items(
    items: Iterable[dict[str, Any]],
    *,
    per_stratum: int = 10,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select a deterministic balanced 12-stratum subset without exposing IDs."""

    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        key = tuple(str(item.get(field)) for field in STRATUM_FIELDS)
        groups[key].append(item)
    expected_granularities = {"G1_finding_existence", "G2_anatomical_localization"}
    expected_evidence = {"affirmed", "negated", "not_enough_evidence"}
    expected_polarities = {"positive", "negative"}
    expected = {
        (granularity, evidence, polarity)
        for granularity in expected_granularities
        for evidence in expected_evidence
        for polarity in expected_polarities
    }
    missing = sorted(key for key in expected if len(groups[key]) < per_stratum)
    if missing:
        raise ValueError(
            "insufficient records in one or more ablation strata: "
            f"strata={len(missing)}"
        )

    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for key in sorted(expected):
        ranked = sorted(
            groups[key],
            key=lambda row: _stable_rank(str(row.get("item_id")), seed),
        )
        chosen = ranked[:per_stratum]
        selected.extend(chosen)
        counts["|".join(key)] = len(chosen)
    selected.sort(key=lambda row: str(row.get("item_id")))
    ids = [str(row.get("item_id")) for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("ablation selection contains duplicate item_id values")
    return selected, counts


def build_ablation_layers(
    items: Iterable[dict[str, Any]],
    *,
    per_stratum: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Create v1/v2 layers over the exact same selected items."""

    selected, counts = select_paired_ablation_items(
        items,
        per_stratum=per_stratum,
        seed=seed,
    )
    v1_inputs, v1_metadata, _ = build_prompt_layers(selected, template_version="v1")
    v2_inputs, v2_metadata, _ = build_prompt_layers(selected, template_version="v2")
    if [row["item_id"] for row in v1_inputs] != [row["item_id"] for row in v2_inputs]:
        raise AssertionError("paired prompt layers lost item alignment")
    if v1_metadata != v2_metadata:
        raise AssertionError("v1/v2 evaluation metadata must be identical")
    return {
        "selected_items": selected,
        "v1_model_inputs": v1_inputs,
        "v2_model_inputs": v2_inputs,
        "eval_metadata": v1_metadata,
        "strata_counts": counts,
    }


def compare_paired_scores(
    v1_rows: list[dict[str, Any]],
    v2_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare aligned v1/v2 scored rows with paired uncertainty tests."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    v1 = _unique_index(v1_rows, "v1")
    v2 = _unique_index(v2_rows, "v2")
    if set(v1) != set(v2):
        raise ValueError("v1 and v2 item_id sets differ")
    ids = sorted(v1)
    deltas = [
        float(v2[item_id].get("is_correct") is True)
        - float(v1[item_id].get("is_correct") is True)
        for item_id in ids
    ]
    rng = random.Random(seed)
    bootstrap = sorted(
        mean(rng.choice(deltas) for _ in deltas)
        for _ in range(bootstrap_samples)
    )
    v1_only = sum(
        v1[item_id].get("is_correct") is True
        and v2[item_id].get("is_correct") is not True
        for item_id in ids
    )
    v2_only = sum(
        v2[item_id].get("is_correct") is True
        and v1[item_id].get("is_correct") is not True
        for item_id in ids
    )
    return {
        "paired_items": len(ids),
        "v1": _study2_diagnostics(v1_rows),
        "v2": _study2_diagnostics(v2_rows),
        "accuracy_difference_v2_minus_v1": mean(deltas),
        "paired_bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "ci95_low": _percentile(bootstrap, 0.025),
            "ci95_high": _percentile(bootstrap, 0.975),
        },
        "mcnemar_exact": {
            "v1_correct_v2_wrong": v1_only,
            "v1_wrong_v2_correct": v2_only,
            "two_sided_p": _exact_mcnemar_p(v1_only, v2_only),
        },
    }


def _study2_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_scores(rows)
    predictions = [row.get("parsed_answer") for row in rows]
    polarity = summary["by_claim_polarity"]
    positive = polarity.get("positive", {}).get("accuracy")
    negative = polarity.get("negative", {}).get("accuracy")
    return {
        **summary,
        "supported_prediction_rate": (
            sum(value == ANSWER_SUPPORTED for value in predictions) / len(rows)
            if rows
            else None
        ),
        "polarity_gap_positive_minus_negative": (
            positive - negative
            if positive is not None and negative is not None
            else None
        ),
    }


def _stable_rank(item_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()


def _unique_index(
    rows: Iterable[dict[str, Any]],
    label: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = row.get("item_id")
        if item_id is None or str(item_id) in result:
            raise ValueError(f"{label} rows contain missing or duplicate item_id")
        result[str(item_id)] = row
    return result


def _percentile(values: list[float], fraction: float) -> float:
    index = fraction * (len(values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _exact_mcnemar_p(left: int, right: int) -> float:
    discordant = left + right
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) * (0.5**discordant)
        for k in range(0, min(left, right) + 1)
    )
    return min(1.0, 2 * tail)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--inputs", type=Path, nargs="+", required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--per-stratum", type=int, default=10)
    build.add_argument("--seed", type=int, default=42)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--v1-scored", type=Path, nargs="+", required=True)
    compare.add_argument("--v2-scored", type=Path, nargs="+", required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--bootstrap-samples", type=int, default=10_000)
    compare.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.command == "build":
        result = build_ablation_layers(
            (row for path in args.inputs for row in read_jsonl(path)),
            per_stratum=args.per_stratum,
            seed=args.seed,
        )
        for version in ("v1", "v2"):
            directory = args.output_root / version
            write_jsonl(result[f"{version}_model_inputs"], directory / "model_inputs.jsonl")
            write_jsonl(result["eval_metadata"], directory / "eval_metadata.jsonl")
            metadata_by_id = {
                str(row["item_id"]): row for row in result["eval_metadata"]
            }
            for short_name, granularity in (
                ("g1", "G1_finding_existence"),
                ("g2", "G2_anatomical_localization"),
            ):
                split_inputs = [
                    row
                    for row in result[f"{version}_model_inputs"]
                    if metadata_by_id[str(row["item_id"])]["granularity"]
                    == granularity
                ]
                split_ids = {str(row["item_id"]) for row in split_inputs}
                split_metadata = [
                    row
                    for row in result["eval_metadata"]
                    if str(row["item_id"]) in split_ids
                ]
                write_jsonl(split_inputs, directory / f"{short_name}_model_inputs.jsonl")
                write_jsonl(
                    split_metadata,
                    directory / f"{short_name}_eval_metadata.jsonl",
                )
        summary = {
            "records": len(result["selected_items"]),
            "per_stratum": args.per_stratum,
            "seed": args.seed,
            "strata_counts": result["strata_counts"],
        }
        args.output_root.mkdir(parents=True, exist_ok=True)
        with (args.output_root / "ablation_summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, sort_keys=True)
            file.write("\n")
        print(
            "Built Study 2 paired prompt ablation: "
            f"records={summary['records']}, strata={len(summary['strata_counts'])}"
        )
        return 0

    result = compare_paired_scores(
        [row for path in args.v1_scored for row in read_jsonl(path)],
        [row for path in args.v2_scored for row in read_jsonl(path)],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
        file.write("\n")
    print(f"Compared Study 2 paired prompts: items={result['paired_items']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
