"""Combine Study 3 scored splits and render aggregate figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ghm.granularity.common import read_jsonl
from ghm.study3.constants import require_study3_output_path
from ghm.study3.scoring import summarize_scores


def render_figures(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    """Render compact K and polarity charts from an aggregate summary."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("Study 3 figures require matplotlib") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    by_k = summary.get("by_option_count", {})
    k_values = sorted(
        (int(key), value)
        for key, value in by_k.items()
        if str(key).isdigit()
    )
    if k_values:
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(
            [key for key, _ in k_values],
            [value.get("mean_hamming_accuracy") for _, value in k_values],
            marker="o",
            label="Hamming accuracy",
        )
        axis.plot(
            [key for key, _ in k_values],
            [value.get("exact_set_accuracy") for _, value in k_values],
            marker="s",
            label="Exact-set accuracy",
        )
        axis.set_xlabel("Option count (K)")
        axis.set_ylabel("Accuracy")
        axis.set_ylim(0, 1)
        axis.set_title("Study 3 performance by option count")
        axis.legend()
        figure.tight_layout()
        path = output_dir / "study3_accuracy_by_option_count.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)

    by_relation = summary.get("by_query_relation", {})
    relations = [name for name in ("present", "absent") if name in by_relation]
    if relations:
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.bar(
            relations,
            [by_relation[name].get("mean_hamming_accuracy") for name in relations],
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel("Mean Hamming accuracy")
        axis.set_title("Study 3 performance by question relation")
        figure.tight_layout()
        path = output_dir / "study3_accuracy_by_query_relation.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)

    framing_relation = summary.get("by_prompt_framing_and_query_relation", {})
    if all(framing in framing_relation for framing in ("state", "evidence")):
        relations = ("present", "absent")
        positions = list(range(len(relations)))
        width = 0.36
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar(
            [position - width / 2 for position in positions],
            [
                framing_relation["state"].get(relation, {}).get(
                    "mean_hamming_accuracy"
                )
                for relation in relations
            ],
            width,
            label="state",
        )
        axis.bar(
            [position + width / 2 for position in positions],
            [
                framing_relation["evidence"].get(relation, {}).get(
                    "mean_hamming_accuracy"
                )
                for relation in relations
            ],
            width,
            label="evidence",
        )
        axis.set_xticks(positions, relations)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Mean Hamming accuracy")
        axis.set_title("Study 3 framing by query relation")
        axis.legend()
        figure.tight_layout()
        path = output_dir / "study3_accuracy_by_framing_relation.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    """Combine scored JSONL files, summarize them, and optionally make figures."""

    parser = argparse.ArgumentParser(description="Report aggregate Study 3 results.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args(argv)
    require_study3_output_path(args.summary)
    if args.figures_dir is not None:
        require_study3_output_path(args.figures_dir)

    rows = [row for path in args.inputs for row in read_jsonl(path)]
    summary = summarize_scores(
        rows,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    figures = render_figures(summary, args.figures_dir) if args.figures_dir else []
    print(
        "Reported Study 3 results: "
        f"items={summary['overall']['items']}, figures={len(figures)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
