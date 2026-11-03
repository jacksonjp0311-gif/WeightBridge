import json
from pathlib import Path

import pytest

from experiments.phase_iiid.analysis import analyze_split, select_practice
from experiments.phase_iiid.describe_grounding import describe_split


CONFIG = json.loads((Path(__file__).parents[1] / "experiments/phase_iiid/protocol.json").read_bytes())


def records(values, order):
    return [{"sum_log_probability": values[i], "mean_log_probability": values[i] / (i + 1),
             "terminated_sum_log_probability": values[i] - 1, "answer_tokens": i + 1,
             "answer_token_ids": [101] if i == 0 else [102, 103]} for i in order]


def split(*, signal):
    rows = []
    for family in ("word-order", "date-order", "fraction-order"):
        for paraphrase in (0, 1):
            for order in ([0, 1], [1, 0]):
                # A strong shared prior always favors the first canonical answer.
                # Only the informative question determines the weaker residual.
                contexts = {name: records(values, order) for name, values in {
                    "task": (-2, -9) if signal else (-3, -9),
                    "counterfactual": (-4, -8) if signal else (-3, -9),
                    "neutral_absent": (-3, -9), "neutral_generic": (-3, -9), "irrelevant": (-3, -9),
                }.items()}
                correct, changed = order.index(0), order.index(1)
                displays = {}
                for display in ("labeled", "bullets"):
                    displays[display] = {"contexts": contexts, "letter": records((-2, -3), order),
                                         "generation": {context: {
                                             "strict_position": position if display == "bullets" else None,
                                             "semantic_position": position, "stop_reason": "eos",
                                             "semantic_kind": "answer" if display == "bullets" else "label",
                                         } for context, position in (("task", correct), ("counterfactual", changed))}}
                rows.append({"id": f"{family}-{paraphrase}-{correct}", "item_id": family, "family": family,
                             "order": order, "paraphrase": paraphrase, "labels": ["A", "B"], "correct_position": correct,
                             "counterfactual_position": changed, "displays": displays})
    return rows


def test_practice_selects_weak_question_evidence_over_the_shared_prior():
    rows = split(signal=True)
    decision = select_practice(rows, CONFIG)
    assert decision["selected_ranker"]["rule"] == {
        "display": "labeled", "kind": "prior_sum", "alpha": 1.0, "neutral": "neutral_absent",
    }
    analysis = analyze_split(rows, CONFIG, decision)
    assert analysis["ranker_gate"]["passed"]
    assert analysis["selected_ranker"]["both_correct_fraction"] == 1
    assert analysis["selected_ranker"]["mean_directional_margin_advantage"] == 3
    assert analysis["generation_gate"]["passed"]
    assert analysis["generation_by_display"]["labeled"]["strict"]["generated_accuracy"] == 0
    assert analysis["generation_by_display"]["labeled"]["semantic"]["generated_accuracy"] == 1
    assert analysis["generation_by_display"]["labeled"]["strict"]["parser_disagreements"] == 12
    assert analysis["prior_minus_raw"]["labeled"]["question_interval"] == [0, 0]
    assert analysis["answer_lengths"]["labeled"]["prior_sum"]["unequal"] == {"rows": 12, "correct": 12}


def test_locked_rule_does_not_qualify_a_prior_only_split():
    decision = select_practice(split(signal=True), CONFIG)
    analysis = analyze_split(split(signal=False), CONFIG, decision)
    assert not analysis["ranker_gate"]["passed"]
    assert analysis["selected_ranker"]["task_accuracy"] == 0.5
    assert analysis["selected_ranker"]["both_correct_fraction"] == 0
    assert analysis["selected_ranker"]["mean_directional_margin_advantage"] == pytest.approx(0)


def test_description_separates_display_compliance_from_semantic_accuracy():
    rows = split(signal=True)
    result = describe_split(rows, select_practice(rows, CONFIG), CONFIG)
    display = result["bullets_minus_labeled_generation"]
    assert display["strict"]["correct"] == {"estimate": 1.0, "question_interval": [1.0, 1.0]}
    assert display["semantic"]["correct"] == {"estimate": 0.0, "question_interval": [0.0, 0.0]}
    assert result["label_strata"]["AB"]["letter_first_position"] == 6
    assert result["raw_context_preferences"]["labeled"]["neutral_absent"] == {
        "left_preferences": 6, "right_preferences": 6, "ties": 0, "mean_absolute_margin": 6.0,
    }
