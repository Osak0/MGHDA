from ghm.study3.answers import parse_multiselect_answer
from ghm.study3.prompts import build_prompt_layers
from ghm.study3.scoring import score_multiselect_rows, summarize_scores
from ghm.study3.smoke import select_smoke_rows


def test_parser_accepts_sets_and_none_and_rejects_ambiguous_answers():
    valid = ["A", "B", "C"]

    assert parse_multiselect_answer("A, C", valid) == (["A", "C"], "valid_options")
    assert parse_multiselect_answer("c and a", valid) == (["A", "C"], "valid_options")
    assert parse_multiselect_answer("NONE", valid) == ([], "valid_none")
    assert parse_multiselect_answer("NONE, A", valid)[0] is None
    assert parse_multiselect_answer("D", valid)[0] is None


def test_prompt_layer_contains_no_gold_or_polarity():
    item = _item(query_relation="absent", gold=["A", "B"], prompt_framing="evidence")

    model_inputs, metadata, summary = build_prompt_layers([item])

    assert set(model_inputs[0]) == {
        "item_id",
        "experiment_id",
        "image_path",
        "prompt_template_id",
        "prompt",
    }
    assert "gold_selected_options" not in model_inputs[0]
    assert metadata[0]["gold_selected_options"] == ["A", "B"]
    assert "radiographic evidence explicitly supports absence in the right lung" in model_inputs[0]["prompt"]
    assert "merely unmentioned" in model_inputs[0]["prompt"]
    assert summary["absent_prompts"] == 1


def test_scoring_reports_option_and_exact_metrics():
    metadata = [
        _metadata("study3_g2_ms_a", "present", [], "set-a", "state"),
        _metadata("study3_g2_ms_b", "absent", ["A", "B"], "set-a", "state"),
    ]
    raw = [
        _raw("study3_g2_ms_a", "NONE"),
        _raw("study3_g2_ms_b", "A"),
    ]

    scored = score_multiselect_rows(raw, metadata)
    summary = summarize_scores(scored, bootstrap_samples=20)

    assert scored[0]["exact_set_correct"] is True
    assert scored[1]["hamming_accuracy"] == 0.5
    assert summary["overall"]["exact_set_accuracy"] == 0.5
    assert summary["overall"]["micro_hamming_accuracy"] == 0.75
    assert summary["complement_consistency"]["by_prompt_framing"]["state"][
        "option_complement_accuracy"
    ] == 0.5


def test_fixed_smoke_has_40_coverage_cells():
    metadata = []
    model_inputs = []
    index = 0
    for granularity in ("G1_finding_existence", "G2_anatomical_localization"):
        for framing in ("state", "evidence"):
            for relation in ("present", "absent"):
                for variant_key in ("natural", 2, 3, 4, 5):
                    index += 1
                    item_id = f"study3_smoke_{index}"
                    variant = "natural" if variant_key == "natural" else "controlled"
                    metadata.append(
                        {
                            "item_id": item_id,
                            "granularity": granularity,
                            "prompt_framing": framing,
                            "query_relation": relation,
                            "variant": variant,
                            "controlled_k": None if variant == "natural" else variant_key,
                        }
                    )
                    model_inputs.append({"item_id": item_id, "prompt": "synthetic"})

    selected_inputs, selected_metadata, summary = select_smoke_rows(
        model_inputs,
        metadata,
    )

    assert len(selected_inputs) == len(selected_metadata) == 40
    assert summary["coverage_cells"] == 40


def _item(*, query_relation, gold, prompt_framing="state"):
    return {
        "item_id": "study3_g2_ms_test",
        "experiment_id": "study3_multiselect_v2",
        "image_path": "files/p10/s1/a.jpg",
        "granularity": "G2_anatomical_localization",
        "question_type": "anatomicalfinding_multiselect_v2",
        "prompt_framing": prompt_framing,
        "query_relation": query_relation,
        "variant": "natural",
        "controlled_k": None,
        "anchor_id": "study3_g2_anchor_test",
        "option_set_id": "study3_option_set_test",
        "option_count": 2,
        "natural_option_count": 2,
        "answer_composition": "all_no",
        "options": [
            {"option_id": "A", "label_name": "lung opacity"},
            {"option_id": "B", "label_name": "pneumothorax"},
        ],
        "gold_selected_options": gold,
        "target_anatomy": "right lung",
        "bbox": {"bbox_name": "right lung"},
        "source_assertions": [],
        "evidence_sources": ["E1_bbox", "E2_anatomy_finding_pair"],
        "source_quality": "synthetic",
    }


def _metadata(item_id, relation, gold, option_set_id, prompt_framing):
    item = _item(
        query_relation=relation,
        gold=gold,
        prompt_framing=prompt_framing,
    )
    item["item_id"] = item_id
    item["option_set_id"] = option_set_id
    return item


def _raw(item_id, response):
    return {
        "item_id": item_id,
        "raw_response": response,
        "model_name": "mock",
        "runtime": {"status": "success"},
    }
