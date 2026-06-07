"""Create tables and figures from aggregate score summary JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_GROUPS = ["overall", "by_granularity", "by_hallucination_probe", "by_answer_label"]
METRIC_FIELDS = [
    "items",
    "scored_items",
    "correct",
    "accuracy",
    "h1_count",
    "h1_false_positive_count",
    "h1_false_negative_count",
    "h2_count",
    "invalid_count",
    "manual_review_count",
    "uncertain_or_abstention_count",
]


def load_summary(path: Path) -> dict[str, Any]:
    """Load one aggregate summary JSON."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def flatten_summaries(paths: list[Path], run_names: list[str] | None = None) -> list[dict[str, Any]]:
    """Flatten summary JSON files into table rows."""

    rows: list[dict[str, Any]] = []
    names = run_names or [path.stem for path in paths]
    if len(names) != len(paths):
        raise ValueError("--run-names must have the same length as --inputs")

    for run_name, path in zip(names, paths, strict=True):
        summary = load_summary(path)
        rows.extend(_flatten_one_summary(summary, run_name=run_name, source_path=path))
    return rows


def write_tables(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    """Write overall and grouped CSV tables."""

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    overall_path = tables_dir / "overall_metrics.csv"
    group_path = tables_dir / "group_metrics.csv"
    _write_csv(
        [row for row in rows if row["group_type"] == "overall"],
        overall_path,
    )
    _write_csv(
        [row for row in rows if row["group_type"] != "overall"],
        group_path,
    )
    return overall_path, group_path


def render_figures(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    """Render aggregate metric figures."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Figure rendering requires matplotlib. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []

    by_probe = [row for row in rows if row["group_type"] == "by_hallucination_probe"]
    if by_probe:
        path = figures_dir / "accuracy_by_probe.png"
        _bar_chart(
            plt,
            by_probe,
            path=path,
            title="Accuracy by Hallucination Probe",
            y_field="accuracy",
            ylabel="Accuracy",
        )
        figure_paths.append(path)

        path = figures_dir / "h_counts_by_probe.png"
        _grouped_count_chart(
            plt,
            by_probe,
            path=path,
            title="H1/H2 Counts by Probe",
            fields=["h1_count", "h2_count"],
            ylabel="Count",
        )
        figure_paths.append(path)

    overall = [row for row in rows if row["group_type"] == "overall"]
    if overall:
        path = figures_dir / "overall_accuracy.png"
        if len(overall) == 1:
            _overall_single_chart(plt, overall[0], path=path)
        else:
            _line_chart(
                plt,
                overall,
                path=path,
                title="Overall Accuracy Across Runs",
                y_field="accuracy",
                ylabel="Accuracy",
            )
        figure_paths.append(path)

        path = figures_dir / "overall_h_rates.png"
        _line_or_bar_rates(plt, overall, path=path)
        figure_paths.append(path)

    h1_rows = [row for row in rows if row["group_type"] == "by_hallucination_probe" and row["group"] == "H1"]
    if h1_rows:
        path = figures_dir / "h1_direction_counts.png"
        _grouped_count_chart(
            plt,
            h1_rows,
            path=path,
            title="H1 Direction Counts",
            fields=["h1_false_positive_count", "h1_false_negative_count"],
            ylabel="Count",
        )
        figure_paths.append(path)

    return figure_paths


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Visualize aggregate evaluation summaries.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument(
        "--run-names",
        nargs="+",
        default=None,
        help="Optional display names matching --inputs order.",
    )
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    try:
        rows = flatten_summaries(args.inputs, args.run_names)
        overall_path, group_path = write_tables(rows, args.output_dir)
        figure_paths = [] if args.no_figures else render_figures(rows, args.output_dir)
    except (RuntimeError, ValueError) as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    print(
        "Wrote evaluation report artifacts: "
        f"overall_table={overall_path}, group_table={group_path}, figures={len(figure_paths)}"
    )
    return 0


def _flatten_one_summary(
    summary: dict[str, Any], *, run_name: str, source_path: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_type in SUMMARY_GROUPS:
        payload = summary.get(group_type)
        if payload is None:
            continue
        if group_type == "overall":
            rows.append(_metric_row(run_name, source_path, group_type, "overall", payload))
        else:
            for group_name, metrics in sorted(payload.items()):
                rows.append(_metric_row(run_name, source_path, group_type, group_name, metrics))
    return rows


def _metric_row(
    run_name: str,
    source_path: Path,
    group_type: str,
    group_name: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "run_name": run_name,
        "source_file": str(source_path),
        "group_type": group_type,
        "group": group_name,
    }
    for field in METRIC_FIELDS:
        row[field] = metrics.get(field, 0 if field.endswith("_count") else None)
    row["h1_rate"] = _rate(row.get("h1_count"), row.get("items"))
    row["h2_rate"] = _rate(row.get("h2_count"), row.get("items"))
    row["invalid_rate"] = _rate(row.get("invalid_count"), row.get("items"))
    row["uncertain_or_abstention_rate"] = _rate(
        row.get("uncertain_or_abstention_count"), row.get("items")
    )
    return row


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "run_name",
        "source_file",
        "group_type",
        "group",
        *METRIC_FIELDS,
        "h1_rate",
        "h2_rate",
        "invalid_rate",
        "uncertain_or_abstention_rate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _bar_chart(plt: Any, rows: list[dict[str, Any]], *, path: Path, title: str, y_field: str, ylabel: str) -> None:
    labels = [_label(row) for row in rows]
    values = [_number(row.get(y_field)) for row in rows]
    plt.figure(figsize=(max(7, len(labels) * 1.4), 4.5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _grouped_count_chart(
    plt: Any,
    rows: list[dict[str, Any]],
    *,
    path: Path,
    title: str,
    fields: list[str],
    ylabel: str,
) -> None:
    labels = [_label(row) for row in rows]
    x_positions = list(range(len(labels)))
    width = 0.8 / len(fields)
    plt.figure(figsize=(max(7, len(labels) * 1.6), 4.5))
    for index, field in enumerate(fields):
        offsets = [x + (index - (len(fields) - 1) / 2) * width for x in x_positions]
        plt.bar(offsets, [_number(row.get(field)) for row in rows], width=width, label=field)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x_positions, labels, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _line_chart(
    plt: Any,
    rows: list[dict[str, Any]],
    *,
    path: Path,
    title: str,
    y_field: str,
    ylabel: str,
) -> None:
    labels = [row["run_name"] for row in rows]
    values = [_number(row.get(y_field)) for row in rows]
    plt.figure(figsize=(max(7, len(labels) * 1.4), 4.5))
    plt.plot(labels, values, marker="o")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _overall_single_chart(plt: Any, row: dict[str, Any], *, path: Path) -> None:
    labels = ["accuracy", "h1_rate", "h2_rate", "invalid_rate", "uncertain_rate"]
    values = [
        _number(row.get("accuracy")),
        _number(row.get("h1_rate")),
        _number(row.get("h2_rate")),
        _number(row.get("invalid_rate")),
        _number(row.get("uncertain_or_abstention_rate")),
    ]
    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, values)
    plt.title(f"Overall Metrics: {row['run_name']}")
    plt.ylabel("Rate")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _line_or_bar_rates(plt: Any, rows: list[dict[str, Any]], *, path: Path) -> None:
    if len(rows) == 1:
        _overall_single_chart(plt, rows[0], path=path)
        return
    labels = [row["run_name"] for row in rows]
    plt.figure(figsize=(max(7, len(labels) * 1.4), 4.5))
    for field in ["h1_rate", "h2_rate", "invalid_rate", "uncertain_or_abstention_rate"]:
        plt.plot(labels, [_number(row.get(field)) for row in rows], marker="o", label=field)
    plt.title("Overall Error/Abstention Rates Across Runs")
    plt.ylabel("Rate")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _label(row: dict[str, Any]) -> str:
    if row["run_name"] == row["group"]:
        return str(row["group"])
    return f"{row['run_name']}:{row['group']}"


def _rate(numerator: Any, denominator: Any) -> float | None:
    denominator_number = _number(denominator)
    if denominator_number == 0:
        return None
    return round(_number(numerator) / denominator_number, 6)


def _number(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
