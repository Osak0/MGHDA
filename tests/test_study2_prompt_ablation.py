from ghm.evaluation.study2_ablation import (
    build_ablation_layers,
    compare_paired_scores,
    main,
)
from ghm.granularity.common import read_jsonl, write_jsonl
from pathlib import Path


def test_ablation_is_balanced_paired_and_definition_only():
    items = []
    index = 0
    for granularity in (
        "G1_finding_existence",
        "G2_anatomical_localization",
    ):
        for evidence_state in ("affirmed", "negated", "not_enough_evidence"):
            for claim_polarity in ("positive", "negative"):
                for _ in range(12):
                    index += 1
                    items.append(
                        {
                            "item_id": f"item-{index:04d}",
                            "image_path": f"files/image-{index % 3}.jpg",
                            "question": f"Synthetic claim {index}.",
                            "answer_label": "A",
                            "granularity": granularity,
                            "question_type": "claim_verification_abc",
                            "hallucination_probe": None,
                            "target_finding": "synthetic",
                            "target_anatomy": None,
                            "claim_polarity": claim_polarity,
                            "evidence_state": evidence_state,
                            "evidence_sources": [],
                        }
                    )

    result = build_ablation_layers(items, per_stratum=10, seed=42)

    assert len(result["v1_model_inputs"]) == 120
    assert result["eval_metadata"] == result["eval_metadata"]
    assert [row["item_id"] for row in result["v1_model_inputs"]] == [
        row["item_id"] for row in result["v2_model_inputs"]
    ]
    assert set(result["strata_counts"].values()) == {10}
    v2_prompt = result["v2_model_inputs"][0]["prompt"]
    assert "affirms the exact claim" in v2_prompt
    assert "supports the logical opposite of the claim" in v2_prompt
    assert "establishes neither the claim nor its logical opposite" in v2_prompt
    assert "merely unmentioned" not in v2_prompt


def test_paired_comparison_reports_bootstrap_and_exact_mcnemar():
    v1 = [_scored("a", True), _scored("b", False), _scored("c", False)]
    v2 = [_scored("a", True), _scored("b", True), _scored("c", False)]

    result = compare_paired_scores(v1, v2, bootstrap_samples=100, seed=42)

    assert result["paired_items"] == 3
    assert result["accuracy_difference_v2_minus_v1"] == 1 / 3
    assert result["mcnemar_exact"]["v1_wrong_v2_correct"] == 1
    assert result["mcnemar_exact"]["two_sided_p"] == 1.0


def test_ablation_cli_writes_split_layers_for_checkpoint_reuse(tmp_path):
    items = []
    index = 0
    for granularity in ("G1_finding_existence", "G2_anatomical_localization"):
        for evidence in ("affirmed", "negated", "not_enough_evidence"):
            for polarity in ("positive", "negative"):
                for _ in range(10):
                    index += 1
                    items.append(
                        {
                            "item_id": f"item-{index}",
                            "image_path": "files/synthetic.jpg",
                            "question": "Synthetic claim.",
                            "answer_label": "A",
                            "granularity": granularity,
                            "question_type": "claim_verification_abc",
                            "hallucination_probe": None,
                            "target_finding": "synthetic",
                            "target_anatomy": None,
                            "claim_polarity": polarity,
                            "evidence_state": evidence,
                            "evidence_sources": [],
                        }
                    )
    input_path = tmp_path / "linked.jsonl"
    output_root = tmp_path / "ablation"
    write_jsonl(items, input_path)

    assert main(
        [
            "build",
            "--inputs",
            str(input_path),
            "--output-root",
            str(output_root),
        ]
    ) == 0

    for version in ("v1", "v2"):
        assert len(read_jsonl(output_root / version / "g1_model_inputs.jsonl")) == 60
        assert len(read_jsonl(output_root / version / "g2_model_inputs.jsonl")) == 60


def test_study2_runner_uses_one_checkpoint_per_split_across_phases():
    source = (
        Path(__file__).parents[1] / "scripts/study2/run.sh"
    ).read_text(encoding="utf-8")

    assert '"$run_root/checkpoints/${split}.jsonl"' in source
    assert '"$phase_root/checkpoints' not in source


def _scored(item_id, correct):
    return {
        "item_id": item_id,
        "answer_label": "A",
        "parsed_answer": "A" if correct else "B",
        "is_correct": correct,
        "score": "correct" if correct else "H1_evidence_contradicted",
        "granularity": "G1_finding_existence",
        "claim_polarity": "positive",
        "evidence_state": "affirmed",
    }
