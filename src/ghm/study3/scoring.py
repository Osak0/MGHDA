"""Score Study 3 multi-select responses and create aggregate summaries."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from ghm.granularity.common import read_jsonl, write_jsonl
from ghm.study3.answers import parse_multiselect_answer
from ghm.study3.constants import (
    CONTROLLED,
    EXPERIMENT_ID,
    require_study3_experiment,
    require_study3_item_id,
    require_study3_output_path,
)


SUMMARY_DIMENSIONS = (
    "granularity",
    "prompt_framing",
    "query_relation",
    "variant",
    "option_count",
    "answer_composition",
)


def score_multiselect_rows(
    raw_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join raw outputs to gold metadata and score each option set."""

    metadata_index = _unique_index(metadata_rows, label="eval metadata")
    raw_index = _unique_index(raw_rows, label="raw responses")
    if set(metadata_index) != set(raw_index):
        missing_raw = len(set(metadata_index) - set(raw_index))
        unexpected_raw = len(set(raw_index) - set(metadata_index))
        raise ValueError(
            "raw response and eval metadata item IDs differ: "
            f"missing_raw={missing_raw}, unexpected_raw={unexpected_raw}"
        )

    scored: list[dict[str, Any]] = []
    for item_id in sorted(metadata_index):
        metadata = metadata_index[item_id]
        raw = raw_index[item_id]
        require_study3_experiment(metadata.get("experiment_id"))
        require_study3_item_id(item_id)
        options = metadata.get("options")
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError(f"invalid Study 3 options for item {item_id}")
        option_ids = [str(option["option_id"]) for option in options]
        gold = {str(value) for value in metadata.get("gold_selected_options", [])}
        runtime_status = raw.get("runtime", {}).get("status")
        if runtime_status not in {None, "success"}:
            parsed, parse_status = None, "infrastructure_error"
        else:
            parsed, parse_status = parse_multiselect_answer(
                raw.get("raw_response"),
                option_ids,
            )
        row = {
            field: metadata.get(field)
            for field in (
                "item_id",
                "experiment_id",
                "granularity",
                "question_type",
                "prompt_framing",
                "query_relation",
                "variant",
                "controlled_k",
                "anchor_id",
                "option_set_id",
                "option_count",
                "natural_option_count",
                "answer_composition",
                "target_anatomy",
                "options",
                "gold_selected_options",
            )
        }
        row.update(
            {
                "model_id": raw.get("model_id") or raw.get("model_name"),
                "model_name": raw.get("model_name"),
                "model_version": raw.get("model_version"),
                "raw_response": raw.get("raw_response"),
                "parsed_selected_options": parsed,
                "parse_status": parse_status,
                "runtime_status": runtime_status,
            }
        )
        if parsed is None:
            row.update(_invalid_metrics(option_ids, gold))
        else:
            row.update(_set_metrics(option_ids, gold, set(parsed)))
        scored.append(row)
    return scored


def summarize_scores(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Build aggregate-only metrics and controlled-K comparisons."""

    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "overall": _summarize_group(rows),
    }
    for dimension in SUMMARY_DIMENSIONS:
        summary[f"by_{dimension}"] = _summarize_by(rows, dimension)
    summary["by_prompt_framing_and_query_relation"] = _summarize_cross(
        rows,
        "prompt_framing",
        "query_relation",
    )
    summary["by_option_position"] = _summarize_option_positions(rows)
    summary["complement_consistency"] = _complement_consistency(rows)
    summary["framing_agreement"] = _framing_agreement(rows)
    summary["controlled_k_paired_bootstrap"] = {
        framing: _controlled_k_bootstrap(
            [row for row in rows if row.get("prompt_framing") == framing],
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        for framing in ("state", "evidence")
    }
    return summary


def _set_metrics(
    option_ids: list[str],
    gold: set[str],
    predicted: set[str],
) -> dict[str, Any]:
    tp = len(gold & predicted)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    tn = len(set(option_ids) - gold - predicted)
    option_count = len(option_ids)
    union = gold | predicted
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn
        else 1.0
    )
    option_results = [
        {
            "option_id": option_id,
            "position": index + 1,
            "gold_selected": option_id in gold,
            "predicted_selected": option_id in predicted,
            "correct": (option_id in gold) == (option_id in predicted),
        }
        for index, option_id in enumerate(option_ids)
    ]
    return {
        "valid_response": True,
        "exact_set_correct": predicted == gold,
        "hamming_accuracy": (tp + tn) / option_count,
        "jaccard": len(gold & predicted) / len(union) if union else 1.0,
        "set_precision": precision,
        "set_recall": recall,
        "set_f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "gold_is_none": not gold,
        "predicted_is_none": not predicted,
        "none_correct": (not predicted) if not gold else None,
        "option_results": option_results,
    }


def _invalid_metrics(option_ids: list[str], gold: set[str]) -> dict[str, Any]:
    return {
        "valid_response": False,
        "exact_set_correct": False,
        "hamming_accuracy": None,
        "jaccard": None,
        "set_precision": None,
        "set_recall": None,
        "set_f1": None,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 0,
        "gold_is_none": not gold,
        "predicted_is_none": None,
        "none_correct": None,
        "option_results": [
            {
                "option_id": option_id,
                "position": index + 1,
                "gold_selected": option_id in gold,
                "predicted_selected": None,
                "correct": None,
            }
            for index, option_id in enumerate(option_ids)
        ],
    }


def _summarize_by(
    rows: list[dict[str, Any]],
    dimension: str,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(dimension))].append(row)
    return {
        key: _summarize_group(group_rows)
        for key, group_rows in sorted(groups.items())
    }


def _summarize_cross(
    rows: list[dict[str, Any]],
    first: str,
    second: str,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(first))].append(row)
    return {
        key: _summarize_by(group_rows, second)
        for key, group_rows in sorted(groups.items())
    }


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("valid_response") is True]
    tp = sum(int(row.get("true_positive", 0)) for row in valid)
    fp = sum(int(row.get("false_positive", 0)) for row in valid)
    fn = sum(int(row.get("false_negative", 0)) for row in valid)
    tn = sum(int(row.get("true_negative", 0)) for row in valid)
    none_rows = [row for row in valid if row.get("gold_is_none") is True]
    return {
        "items": len(rows),
        "valid_items": len(valid),
        "invalid_items": len(rows) - len(valid),
        "invalid_rate": (len(rows) - len(valid)) / len(rows) if rows else None,
        "exact_set_accuracy": _mean_field(valid, "exact_set_correct"),
        "mean_hamming_accuracy": _mean_field(valid, "hamming_accuracy"),
        "mean_jaccard": _mean_field(valid, "jaccard"),
        "mean_set_f1": _mean_field(valid, "set_f1"),
        "micro_precision": tp / (tp + fp) if tp + fp else None,
        "micro_recall": tp / (tp + fn) if tp + fn else None,
        "micro_f1": 2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn
        else None,
        "micro_hamming_accuracy": (tp + tn) / (tp + fp + fn + tn)
        if tp + fp + fn + tn
        else None,
        "none_items": len(none_rows),
        "none_accuracy": _mean_field(none_rows, "none_correct"),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def _summarize_option_positions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("valid_response") is not True:
            continue
        for result in row.get("option_results", []):
            positions[int(result["position"])].append(result)
    return {
        str(position): {
            "option_decisions": len(results),
            "accuracy": mean(float(result["correct"]) for result in results),
            "selection_rate": mean(
                float(result["predicted_selected"]) for result in results
            ),
            "gold_selection_rate": mean(
                float(result["gold_selected"]) for result in results
            ),
        }
        for position, results in sorted(positions.items())
    }


def _complement_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"by_prompt_framing": {}}
    for framing in ("state", "evidence"):
        framing_rows = [
            row for row in rows if row.get("prompt_framing") == framing
        ]
        framing_result = _complement_group(framing_rows)
        for dimension in ("granularity", "variant", "option_count"):
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in framing_rows:
                groups[str(row.get(dimension))].append(row)
            framing_result[f"by_{dimension}"] = {
                key: _complement_group(group_rows)
                for key, group_rows in sorted(groups.items())
            }
        result["by_prompt_framing"][framing] = framing_result
    return result


def _framing_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure state/evidence prediction agreement for the same set and relation."""

    pairs: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("valid_response") is True:
            key = (str(row.get("option_set_id")), str(row.get("query_relation")))
            pairs[key][str(row.get("prompt_framing"))] = row
    paired = 0
    exact = 0
    decisions = 0
    agreeing = 0
    for framing_rows in pairs.values():
        if set(framing_rows) != {"state", "evidence"}:
            continue
        paired += 1
        state = set(framing_rows["state"]["parsed_selected_options"])
        evidence = set(framing_rows["evidence"]["parsed_selected_options"])
        option_ids = {
            str(option["option_id"]) for option in framing_rows["state"]["options"]
        }
        exact += int(state == evidence)
        decisions += len(option_ids)
        agreeing += sum((option in state) == (option in evidence) for option in option_ids)
    return {
        "valid_paired_questions": paired,
        "exact_set_agreement": exact / paired if paired else None,
        "option_decisions": decisions,
        "option_agreement": agreeing / decisions if decisions else None,
    }


def _complement_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_set: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("valid_response") is True:
            by_set[str(row.get("option_set_id"))][str(row.get("query_relation"))] = row
    paired_sets = 0
    option_decisions = 0
    consistent = 0
    exact_pairs = 0
    for relation_rows in by_set.values():
        if set(relation_rows) != {"present", "absent"}:
            continue
        paired_sets += 1
        present = set(relation_rows["present"]["parsed_selected_options"])
        absent = set(relation_rows["absent"]["parsed_selected_options"])
        option_ids = {
            str(option["option_id"])
            for option in relation_rows["present"]["options"]
        }
        per_option = [
            (option_id in present) != (option_id in absent)
            for option_id in option_ids
        ]
        option_decisions += len(per_option)
        consistent += sum(per_option)
        exact_pairs += int(all(per_option))
    return {
        "valid_paired_option_sets": paired_sets,
        "option_decisions": option_decisions,
        "option_complement_accuracy": consistent / option_decisions
        if option_decisions
        else None,
        "exact_pair_consistency": exact_pairs / paired_sets if paired_sets else None,
    }


def _controlled_k_bootstrap(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    anchor_k_values: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if (
            row.get("variant") == CONTROLLED
            and row.get("valid_response") is True
            and row.get("option_count") is not None
        ):
            anchor_k_values[str(row["anchor_id"])][int(row["option_count"])].append(
                float(row["hamming_accuracy"])
            )
    per_anchor = {
        anchor: {k: mean(values) for k, values in by_k.items()}
        for anchor, by_k in anchor_k_values.items()
    }
    all_k = sorted({k for by_k in per_anchor.values() for k in by_k})
    if not all_k:
        return {"baseline_k": None, "comparisons": {}}
    baseline = min(all_k)
    rng = random.Random(seed)
    comparisons: dict[str, Any] = {}
    for target in all_k:
        if target == baseline:
            continue
        differences = [
            values[target] - values[baseline]
            for values in per_anchor.values()
            if baseline in values and target in values
        ]
        if not differences:
            continue
        bootstrap_means = [
            mean(rng.choice(differences) for _ in differences)
            for _ in range(samples)
        ]
        bootstrap_means.sort()
        comparisons[str(target)] = {
            "paired_anchors": len(differences),
            "mean_hamming_difference_vs_baseline": mean(differences),
            "ci95_low": _percentile(bootstrap_means, 0.025),
            "ci95_high": _percentile(bootstrap_means, 0.975),
        }
    return {
        "baseline_k": baseline,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "comparisons": comparisons,
    }


def _unique_index(
    rows: Iterable[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = row.get("item_id")
        if item_id is None:
            raise ValueError(f"{label} contains a row without item_id")
        key = str(item_id)
        if key in index:
            raise ValueError(f"{label} contains duplicate item_id {key}")
        index[key] = row
    return index


def _mean_field(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [row.get(field) for row in rows if row.get(field) is not None]
    return mean(float(value) for value in values) if values else None


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute percentile of an empty sequence")
    index = fraction * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def main(argv: list[str] | None = None) -> int:
    """Parse, score, and summarize one Study 3 model run."""

    parser = argparse.ArgumentParser(description="Score Study 3 multi-select outputs.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--eval-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args(argv)
    for path in (args.output, args.summary):
        require_study3_output_path(path)
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")

    raw_rows = read_jsonl(args.raw)
    metadata_rows = read_jsonl(args.eval_metadata)
    scored = score_multiselect_rows(raw_rows, metadata_rows)
    summary = summarize_scores(
        scored,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_jsonl(scored, args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    print(
        "Scored Study 3 outputs: "
        f"items={summary['overall']['items']}, "
        f"valid={summary['overall']['valid_items']}, "
        f"exact={summary['overall']['exact_set_accuracy']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
