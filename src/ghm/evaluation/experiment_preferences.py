"""Build aggregate-only diagnostics for Study 2 and Study 3 result folders.

The module intentionally consumes row-level Study 3 scored/raw files locally but
never copies response text, prompts, identifiers, option text, or image paths to
its outputs.  This makes the generated CSV/JSON files suitable for sharing.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


STUDY2_LABELS = ("Supported", "Contradicted", "Not enough evidence")
STUDY3_GROUP_FIELDS = (
    "granularity",
    "prompt_framing",
    "query_relation",
    "variant",
    "option_count",
)
FORBIDDEN_OUTPUT_KEYS = {
    "raw_response",
    "prompt",
    "item_id",
    "anchor_id",
    "option_set_id",
    "image_path",
    "target_anatomy",
    "options",
    "source_assertions",
    "evidence_sources",
    "patient_id",
    "study_id",
    "image_id",
}
SAFE_JSON_SUFFIXES = (
    "_score_summary.json",
    "_combined_score_summary.json",
    "_validation.json",
    "_transfer_summary.json",
)


def analyze_result_directories(
    *,
    study2_dir: Path,
    study3_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Analyze two result directories and write aggregate-only artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = inventory_result_files(study2_dir, study3_dir)
    study2_rows: list[dict[str, Any]] = []
    study3_summary_rows: list[dict[str, Any]] = []
    study3_group_rows: list[dict[str, Any]] = []
    study3_parse_rows: list[dict[str, Any]] = []
    study3_set_size_rows: list[dict[str, Any]] = []
    study3_position_rows: list[dict[str, Any]] = []
    study3_pair_rows: list[dict[str, Any]] = []
    study3_invalid_rows: list[dict[str, Any]] = []
    study3_runtime_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for path in sorted(study2_dir.glob("*_score_summary.json")):
        if path.stat().st_size == 0:
            continue
        try:
            summary = _read_json_object(path)
            study2_rows.extend(study2_answer_distribution(summary, path.name))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"file_name": path.name, "error_type": type(exc).__name__})

    for path in sorted(study3_dir.glob("*_g1_g2_score_summary.json")):
        if path.stat().st_size == 0:
            continue
        try:
            summary = _read_json_object(path)
            study3_summary_rows.extend(
                study3_aggregate_summary_metrics(summary, path.name)
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"file_name": path.name, "error_type": type(exc).__name__})

    for path in sorted(study3_dir.glob("*_scored.jsonl")):
        if path.stat().st_size == 0:
            continue
        try:
            result = study3_preference_diagnostics(path)
            study3_group_rows.extend(result["group_metrics"])
            study3_parse_rows.extend(result["parse_distribution"])
            study3_set_size_rows.extend(result["set_size_distribution"])
            study3_position_rows.extend(result["position_bias"])
            study3_pair_rows.extend(result["pair_consistency"])
            study3_invalid_rows.extend(result["invalid_shape_distribution"])
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"file_name": path.name, "error_type": type(exc).__name__})

    for path in sorted(study3_dir.glob("*_raw.jsonl")):
        if path.stat().st_size == 0:
            continue
        try:
            study3_runtime_rows.extend(study3_runtime_diagnostics(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"file_name": path.name, "error_type": type(exc).__name__})

    artifacts: dict[str, list[dict[str, Any]]] = {
        "file_inventory.csv": inventory,
        "study2_answer_distribution.csv": study2_rows,
        "study3_summary_metrics.csv": study3_summary_rows,
        "study3_group_metrics.csv": study3_group_rows,
        "study3_parse_distribution.csv": study3_parse_rows,
        "study3_set_size_distribution.csv": study3_set_size_rows,
        "study3_position_bias.csv": study3_position_rows,
        "study3_pair_consistency.csv": study3_pair_rows,
        "study3_invalid_shape_distribution.csv": study3_invalid_rows,
        "study3_runtime_diagnostics.csv": study3_runtime_rows,
    }
    for file_name, rows in artifacts.items():
        _write_csv(output_dir / file_name, rows)

    report = {
        "schema_version": "aggregate_experiment_preferences_v1",
        "privacy": {
            "aggregate_only": True,
            "contains_row_level_identifiers": False,
            "contains_response_or_prompt_text": False,
        },
        "source_status": _source_status(inventory),
        "artifact_rows": {
            file_name: len(rows) for file_name, rows in sorted(artifacts.items())
        },
        "analysis_errors": errors,
    }
    assert_aggregate_safe(report)
    report_path = output_dir / "sanitized_experiment_analysis.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    return report


def inventory_result_files(
    study2_dir: Path,
    study3_dir: Path,
) -> list[dict[str, Any]]:
    """Return a filename-only inventory without exposing absolute paths."""

    rows: list[dict[str, Any]] = []
    for experiment, directory in (("study2", study2_dir), ("study3", study3_dir)):
        if not directory.is_dir():
            rows.append(
                {
                    "experiment": experiment,
                    "file_name": "",
                    "bytes": 0,
                    "file_kind": "missing_directory",
                    "status": "missing",
                }
            )
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            size = path.stat().st_size
            rows.append(
                {
                    "experiment": experiment,
                    "file_name": path.name,
                    "bytes": size,
                    "file_kind": _file_kind(path),
                    "status": "empty" if size == 0 else "available",
                }
            )
    return rows


def study2_answer_distribution(
    summary: Mapping[str, Any],
    file_name: str,
) -> list[dict[str, Any]]:
    """Extract all six Study 2 evidence-state by claim-polarity cells."""

    cross = summary.get("by_evidence_state_and_claim_polarity")
    if not isinstance(cross, Mapping):
        raise ValueError("Study 2 summary lacks evidence/polarity cross-tab")
    rows: list[dict[str, Any]] = []
    for evidence_state, by_polarity in sorted(cross.items()):
        if not isinstance(by_polarity, Mapping):
            continue
        for claim_polarity, cell in sorted(by_polarity.items()):
            if not isinstance(cell, Mapping):
                continue
            distribution = cell.get("answer_distribution", {})
            if not isinstance(distribution, Mapping):
                distribution = {}
            items = _as_int(cell.get("items"))
            label_counts = {
                label: _as_int(distribution.get(label)) for label in STUDY2_LABELS
            }
            no_count = _as_int(distribution.get("No"))
            null_count = _as_int(distribution.get("None"))
            canonical_total = sum(label_counts.values())
            other_out_of_schema = max(
                0, items - canonical_total - no_count - null_count
            )
            noncanonical_count = items - canonical_total
            row: dict[str, Any] = {
                "source_file": file_name,
                "evidence_state": str(evidence_state),
                "claim_polarity": str(claim_polarity),
                "question_wording": (
                    "There is" if str(claim_polarity) == "positive" else "There isn't"
                ),
                "gold_label": _study2_gold(
                    str(evidence_state), str(claim_polarity)
                ),
                "items": items,
                "scorer_accuracy": cell.get("accuracy"),
                "end_to_end_accuracy": _ratio(_as_int(cell.get("correct")), items),
            }
            for label in STUDY2_LABELS:
                slug = _slug(label)
                row[f"{slug}_count"] = label_counts[label]
                row[f"{slug}_rate"] = _ratio(label_counts[label], items)
            row["no_count"] = no_count
            row["no_rate"] = _ratio(no_count, items)
            row["null_or_unparsed_count"] = null_count
            row["null_or_unparsed_rate"] = _ratio(null_count, items)
            row["other_out_of_schema_count"] = other_out_of_schema
            row["other_out_of_schema_rate"] = _ratio(other_out_of_schema, items)
            row["noncanonical_count"] = noncanonical_count
            row["noncanonical_rate"] = _ratio(noncanonical_count, items)
            row["scorer_invalid_count"] = _as_int(cell.get("invalid_count"))
            row["manual_review_count"] = _as_int(cell.get("manual_review_count"))
            rows.append(row)
    return rows


def study3_preference_diagnostics(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Aggregate one Study 3 scored JSONL without retaining sensitive fields."""

    run_name = _run_name(path.name, "_scored.jsonl")
    rows = list(_read_jsonl(path))
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    parse_groups: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    set_size_groups: dict[tuple[str, ...], Counter[tuple[str, int]]] = defaultdict(
        Counter
    )
    invalid_shapes: dict[tuple[str, str, str], int] = Counter()
    positions: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"decisions": 0, "gold": 0, "predicted": 0, "correct": 0}
    )
    pair_cache: dict[
        tuple[str, str], dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    seen_ids: set[str] = set()
    duplicate_ids = 0

    for row in rows:
        item_id = str(row.get("item_id", ""))
        if item_id:
            duplicate_ids += int(item_id in seen_ids)
            seen_ids.add(item_id)
        group_key = tuple(str(row.get(field)) for field in STUDY3_GROUP_FIELDS)
        groups[group_key].append(row)
        parse_status = str(row.get("parse_status") or "missing")
        parse_groups[("overall", "all")][parse_status] += 1
        for field in STUDY3_GROUP_FIELDS:
            parse_groups[(field, str(row.get(field)))][parse_status] += 1

        valid = row.get("valid_response") is True
        if valid:
            predicted = _string_set(row.get("parsed_selected_options"))
            gold = _string_set(row.get("gold_selected_options"))
            for field, value in (
                ("overall", "all"),
                ("query_relation", str(row.get("query_relation"))),
                ("prompt_framing", str(row.get("prompt_framing"))),
                ("option_count", str(row.get("option_count"))),
                ("granularity", str(row.get("granularity"))),
            ):
                key = (field, value)
                set_size_groups[key][("gold", len(gold))] += 1
                set_size_groups[key][("predicted", len(predicted))] += 1
            for result in row.get("option_results") or []:
                if not isinstance(result, Mapping):
                    continue
                position = _as_int(result.get("position"))
                if position < 1:
                    continue
                bucket = positions[(str(row.get("granularity")), position)]
                bucket["decisions"] += 1
                bucket["gold"] += int(result.get("gold_selected") is True)
                bucket["predicted"] += int(result.get("predicted_selected") is True)
                bucket["correct"] += int(result.get("correct") is True)
            option_set_id = str(row.get("option_set_id", ""))
            framing = str(row.get("prompt_framing"))
            relation = str(row.get("query_relation"))
            if option_set_id and relation in {"present", "absent"}:
                pair_cache[(framing, option_set_id)][relation] = {
                    "predicted": predicted,
                    "option_ids": {
                        str(option.get("option_id"))
                        for option in row.get("options") or []
                        if isinstance(option, Mapping)
                    },
                }
        else:
            shape = _response_shape(row.get("raw_response"), parse_status)
            invalid_shapes[
                (
                    parse_status,
                    shape,
                    _length_bucket(row.get("raw_response")),
                )
            ] += 1

    group_rows = [
        _study3_group_row(run_name, key, group_rows)
        for key, group_rows in sorted(groups.items())
    ]
    parse_rows: list[dict[str, Any]] = []
    for (dimension, value), counts in sorted(parse_groups.items()):
        total = sum(counts.values())
        for status, count in sorted(counts.items()):
            parse_rows.append(
                {
                    "run_name": run_name,
                    "dimension": dimension,
                    "value": value,
                    "parse_status": status,
                    "count": count,
                    "rate": _ratio(count, total),
                }
            )
    size_rows = [
        {
            "run_name": run_name,
            "dimension": dimension,
            "value": value,
            "set_kind": kind,
            "set_size": size,
            "count": count,
            "rate": _ratio(
                count,
                sum(
                    other_count
                    for (other_kind, _), other_count in counts.items()
                    if other_kind == kind
                ),
            ),
        }
        for (dimension, value), counts in sorted(set_size_groups.items())
        for (kind, size), count in sorted(counts.items())
    ]
    position_rows = [
        {
            "run_name": run_name,
            "granularity": granularity,
            "position": position,
            "option_decisions": bucket["decisions"],
            "gold_selection_rate": _ratio(bucket["gold"], bucket["decisions"]),
            "predicted_selection_rate": _ratio(
                bucket["predicted"], bucket["decisions"]
            ),
            "selection_gap": _ratio(
                bucket["predicted"] - bucket["gold"], bucket["decisions"]
            ),
            "option_accuracy": _ratio(bucket["correct"], bucket["decisions"]),
        }
        for (granularity, position), bucket in sorted(positions.items())
    ]
    pair_rows = _pair_consistency_rows(run_name, pair_cache)
    invalid_rows = [
        {
            "run_name": run_name,
            "parse_status": status,
            "response_shape": shape,
            "length_bucket": length_bucket,
            "count": count,
            "rate_among_invalid": _ratio(count, sum(invalid_shapes.values())),
        }
        for (status, shape, length_bucket), count in sorted(invalid_shapes.items())
    ]
    quality_row = {
        "run_name": run_name,
        "granularity": "__data_quality__",
        "prompt_framing": "all",
        "query_relation": "all",
        "variant": "all",
        "option_count": "all",
        "items": len(rows),
        "duplicate_record_count": duplicate_ids,
    }
    group_rows.append(quality_row)
    return {
        "group_metrics": group_rows,
        "parse_distribution": parse_rows,
        "set_size_distribution": size_rows,
        "position_bias": position_rows,
        "pair_consistency": pair_rows,
        "invalid_shape_distribution": invalid_rows,
    }


def study3_aggregate_summary_metrics(
    summary: Mapping[str, Any],
    file_name: str,
) -> list[dict[str, Any]]:
    """Flatten safe Study 3 summary metrics for cross-model comparison."""

    sections = {
        "overall": "overall",
        "granularity": "by_granularity",
        "prompt_framing": "by_prompt_framing",
        "query_relation": "by_query_relation",
        "variant": "by_variant",
        "option_count": "by_option_count",
        "answer_composition": "by_answer_composition",
    }
    rows: list[dict[str, Any]] = []
    for dimension, source_key in sections.items():
        source = summary.get(source_key)
        groups = {"all": source} if dimension == "overall" else source
        if not isinstance(groups, Mapping):
            continue
        for value, metrics in sorted(groups.items(), key=lambda item: str(item[0])):
            if not isinstance(metrics, Mapping):
                continue
            tp = _as_int(metrics.get("true_positive"))
            fp = _as_int(metrics.get("false_positive"))
            fn = _as_int(metrics.get("false_negative"))
            tn = _as_int(metrics.get("true_negative"))
            decisions = tp + fp + fn + tn
            items = _as_int(metrics.get("items"))
            valid_items = _as_int(metrics.get("valid_items"))
            valid_exact = metrics.get("exact_set_accuracy")
            rows.append(
                {
                    "source_file": file_name,
                    "dimension": dimension,
                    "value": str(value),
                    "items": items,
                    "valid_items": valid_items,
                    "invalid_items": _as_int(metrics.get("invalid_items")),
                    "invalid_rate": metrics.get("invalid_rate"),
                    "valid_only_exact_set_accuracy": valid_exact,
                    "end_to_end_exact_set_accuracy": (
                        float(valid_exact) * valid_items / items
                        if isinstance(valid_exact, (int, float)) and items
                        else None
                    ),
                    "valid_only_mean_hamming_accuracy": metrics.get(
                        "mean_hamming_accuracy"
                    ),
                    "micro_hamming_accuracy": metrics.get(
                        "micro_hamming_accuracy"
                    ),
                    "micro_precision": metrics.get("micro_precision"),
                    "micro_recall": metrics.get("micro_recall"),
                    "micro_f1": metrics.get("micro_f1"),
                    "none_accuracy": metrics.get("none_accuracy"),
                    "gold_selection_rate": _ratio(tp + fn, decisions),
                    "predicted_selection_rate": _ratio(tp + fp, decisions),
                    "selection_gap": _ratio(fp - fn, decisions),
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                    "true_negative": tn,
                }
            )
    return rows


def study3_runtime_diagnostics(path: Path) -> list[dict[str, Any]]:
    """Summarize runtime status and latency from raw outputs without text."""

    run_name = _run_name(path.name, "_raw.jsonl")
    status_counts: Counter[str] = Counter()
    latencies: list[float] = []
    generation_configs: Counter[tuple[Any, ...]] = Counter()
    config_fields = ("max_new_tokens", "temperature", "do_sample", "seed")
    for row in _read_jsonl(path):
        runtime = row.get("runtime") if isinstance(row.get("runtime"), Mapping) else {}
        status_counts[str(runtime.get("status") or "missing")] += 1
        latency = runtime.get("latency_seconds")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
        config = (
            row.get("generation_config")
            if isinstance(row.get("generation_config"), Mapping)
            else {}
        )
        generation_configs[tuple(config.get(field) for field in config_fields)] += 1
    total = sum(status_counts.values())
    rows = [
        {
            "run_name": run_name,
            "record_type": "runtime_status",
            "status": status,
            "count": count,
            "rate": _ratio(count, total),
        }
        for status, count in sorted(status_counts.items())
    ]
    if latencies:
        ordered = sorted(latencies)
        rows.append(
            {
                "run_name": run_name,
                "record_type": "latency_seconds",
                "count": len(ordered),
                "mean": mean(ordered),
                "p50": _percentile(ordered, 0.50),
                "p95": _percentile(ordered, 0.95),
            }
        )
    for values, count in sorted(
        generation_configs.items(), key=lambda item: tuple(str(v) for v in item[0])
    ):
        row: dict[str, Any] = {
            "run_name": run_name,
            "record_type": "generation_config",
            "count": count,
        }
        row.update(dict(zip(config_fields, values)))
        rows.append(row)
    return rows


def assert_aggregate_safe(value: Any) -> None:
    """Reject accidental inclusion of sensitive row-level keys."""

    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_OUTPUT_KEYS & {str(key) for key in value}
        if forbidden:
            raise ValueError(
                "aggregate output contains forbidden keys: "
                + ", ".join(sorted(forbidden))
            )
        for nested in value.values():
            assert_aggregate_safe(nested)
    elif isinstance(value, list):
        for nested in value:
            assert_aggregate_safe(nested)


def _study3_group_row(
    run_name: str,
    key: tuple[str, ...],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    valid = [row for row in rows if row.get("valid_response") is True]
    tp = sum(_as_int(row.get("true_positive")) for row in valid)
    fp = sum(_as_int(row.get("false_positive")) for row in valid)
    fn = sum(_as_int(row.get("false_negative")) for row in valid)
    tn = sum(_as_int(row.get("true_negative")) for row in valid)
    gold_selected = tp + fn
    predicted_selected = tp + fp
    decisions = tp + fp + fn + tn
    gold_none = [row for row in valid if row.get("gold_is_none") is True]
    result = {
        "run_name": run_name,
        **dict(zip(STUDY3_GROUP_FIELDS, key)),
        "items": len(rows),
        "valid_items": len(valid),
        "invalid_items": len(rows) - len(valid),
        "invalid_rate": _ratio(len(rows) - len(valid), len(rows)),
        "end_to_end_exact_set_accuracy": _ratio(
            sum(int(row.get("exact_set_correct") is True) for row in valid),
            len(rows),
        ),
        "valid_only_exact_set_accuracy": _mean_field(valid, "exact_set_correct"),
        "valid_only_mean_hamming_accuracy": _mean_field(
            valid, "hamming_accuracy"
        ),
        "valid_only_mean_jaccard": _mean_field(valid, "jaccard"),
        "valid_only_mean_set_f1": _mean_field(valid, "set_f1"),
        "micro_precision": _ratio(tp, tp + fp),
        "micro_recall": _ratio(tp, tp + fn),
        "micro_f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "micro_hamming_accuracy": _ratio(tp + tn, decisions),
        "gold_selection_rate": _ratio(gold_selected, decisions),
        "predicted_selection_rate": _ratio(predicted_selected, decisions),
        "selection_gap": _ratio(predicted_selected - gold_selected, decisions),
        "gold_none_items": len(gold_none),
        "predicted_none_rate": _ratio(
            sum(int(row.get("predicted_is_none") is True) for row in valid),
            len(valid),
        ),
        "none_accuracy": _mean_field(gold_none, "none_correct"),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }
    return result


def _pair_consistency_rows(
    run_name: str,
    cache: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"pairs": 0, "exact": 0, "decisions": 0, "consistent": 0}
    )
    for (framing, _), relations in cache.items():
        if set(relations) != {"present", "absent"}:
            continue
        present = relations["present"]["predicted"]
        absent = relations["absent"]["predicted"]
        option_ids = relations["present"]["option_ids"]
        decisions = [
            (option_id in present) != (option_id in absent)
            for option_id in option_ids
        ]
        totals[framing]["pairs"] += 1
        totals[framing]["exact"] += int(bool(decisions) and all(decisions))
        totals[framing]["decisions"] += len(decisions)
        totals[framing]["consistent"] += sum(decisions)
    return [
        {
            "run_name": run_name,
            "prompt_framing": framing,
            "valid_paired_option_sets": values["pairs"],
            "exact_pair_consistency": _ratio(values["exact"], values["pairs"]),
            "option_decisions": values["decisions"],
            "option_complement_accuracy": _ratio(
                values["consistent"], values["decisions"]
            ),
        }
        for framing, values in sorted(totals.items())
    ]


def _response_shape(value: Any, parse_status: str) -> str:
    if parse_status == "infrastructure_error":
        return "infrastructure_error"
    if not isinstance(value, str):
        return "non_string"
    text = value.strip()
    if not text:
        return "empty"
    upper = text.upper()
    if re.fullmatch(r"NONE[.!]?", upper):
        return "none_only"
    if re.search(r"\bNONE\b", upper):
        return "contains_none_plus_other"
    if re.fullmatch(r"[A-Z]", upper):
        return "single_letter"
    if re.fullmatch(r"[\[\]()A-Z,;\s]+", upper):
        return "letter_sequence"
    if re.fullmatch(r"(YES|NO)[.!]?", upper):
        return "yes_no_only"
    return "prose_or_other"


def _length_bucket(value: Any) -> str:
    if not isinstance(value, str):
        return "non_string"
    length = len(value.strip())
    if length == 0:
        return "0"
    if length == 1:
        return "1"
    if length <= 5:
        return "2-5"
    if length <= 20:
        return "6-20"
    if length <= 100:
        return "21-100"
    return ">100"


def _source_status(inventory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = [row for row in inventory if row.get("status") == "available"]
    empty = [row for row in inventory if row.get("status") == "empty"]
    return {
        "files_seen": len(inventory),
        "available_files": len(available),
        "empty_files": len(empty),
        "empty_file_names": [str(row.get("file_name")) for row in empty],
        "row_level_files_available": sum(
            row.get("file_kind") in {"raw_jsonl", "scored_jsonl"}
            and row.get("status") == "available"
            for row in inventory
        ),
    }


def _file_kind(path: Path) -> str:
    name = path.name.lower()
    if name.endswith("_raw.jsonl"):
        return "raw_jsonl"
    if name.endswith("_scored.jsonl"):
        return "scored_jsonl"
    if name.endswith(SAFE_JSON_SUFFIXES):
        return "aggregate_json"
    if name.endswith(".jsonl"):
        return "other_jsonl"
    return "other_json"


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path.name} line {line_number} is not a JSON object"
                )
            yield value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        forbidden = FORBIDDEN_OUTPUT_KEYS & {str(key) for key in row}
        if forbidden:
            raise ValueError(
                f"{path.name} contains forbidden columns: "
                + ", ".join(sorted(forbidden))
            )
        for key in row:
            if str(key) not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        if not fieldnames:
            return
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _study2_gold(evidence_state: str, claim_polarity: str) -> str:
    if evidence_state == "not_enough_evidence":
        return "Not enough evidence"
    if evidence_state == "affirmed":
        return "Supported" if claim_polarity == "positive" else "Contradicted"
    if evidence_state == "negated":
        return "Contradicted" if claim_polarity == "positive" else "Supported"
    return "unknown"


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value}


def _mean_field(rows: Iterable[Mapping[str, Any]], field: str) -> float | None:
    values = [row.get(field) for row in rows if row.get(field) is not None]
    return mean(float(value) for value in values) if values else None


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    index = fraction * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _run_name(file_name: str, suffix: str) -> str:
    return file_name[: -len(suffix)] if file_name.endswith(suffix) else file_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create aggregate-only answer preference diagnostics for Study 2/3."
        )
    )
    parser.add_argument("--study2-dir", type=Path, required=True)
    parser.add_argument("--study3-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_result_directories(
        study2_dir=args.study2_dir,
        study3_dir=args.study3_dir,
        output_dir=args.output_dir,
    )
    status = report["source_status"]
    print(
        "Aggregate-only analysis complete: "
        f"available_files={status['available_files']}, "
        f"empty_files={status['empty_files']}, "
        f"output_files={len(report['artifact_rows']) + 1}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
