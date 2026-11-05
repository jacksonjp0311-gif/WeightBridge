"""Deterministic fake observations validate gates, not model efficacy."""
from copy import deepcopy

import pytest

from experiments.phase_iiiq_baseline.data import generate, PopulationSpec
from experiments.phase_iiiq_baseline.measurement import endpoint, question
from experiments.phase_iiiq_baseline.smoke_checks import exact_predictions, validate_smoke
from weightbridge.adaptation.behavior_endpoint import _observation


@pytest.fixture
def observations():
    settings = {"max_input_tokens": 128, "max_new_tokens": 16, "batch_size": 8}
    contract = endpoint(settings)
    rows = generate("smoke", PopulationSpec(target_families=4), "synthetic-smoke-check-fixture", set(), set())
    def measurement(correct_count):
        measured = []
        for index, row in enumerate(rows):
            answer = row["answer"] if index < correct_count else next(x for x in row["operands"] if x != row["oracle_answer"])
            output = {**row, "text": " "+answer+"\n", "stop_reason": "eos",
                "generated_token_ids": list(map(ord, " "+answer+"\n"))}
            measured.append({**output, **_observation(output, question(output), contract)})
        return {"endpoint": contract.manifest(), "rows": measured}
    parent, learned = measurement(0), measurement(32)
    before = {"mean_token_nll": 1., "row_ids": [row["id"] for row in rows],
        "tokenization_sha256": "a"*64, "supervised_tokens": 320, "prompt_representation": "bare", "forward_calls": 8}
    receipt = {"status": "COMPLETED", "base_unchanged": True, "all_factors_changed": True,
        "parameter_changes": [{"name": "module.A", "changed_entries": 2, "norm": .01},
                              {"name": "module.B", "changed_entries": 2, "norm": .01}],
        "matched_loss_before": before, "matched_loss_after": {**before, "mean_token_nll": .4},
        "zero_init_measurement": deepcopy(parent)}
    return parent, learned, receipt, measurement


def test_full_smoke_requirements_pass_on_deterministic_fixture_only(observations):
    parent, learned, receipt, _ = observations
    gate = validate_smoke(parent, learned, deepcopy(learned), deepcopy(parent), receipt)
    assert gate["passed"] and gate["matched_loss_ratio"] == .4
    assert gate["strict_comparison_gain"] == 8 and gate["complete_family_gain"] == 4
    assert gate["correct_completed_presentations"] == 32
    assert "no independent improvement" in gate["scope"]


def test_shuffled_training_trace_loss_is_not_a_matched_population_loss(observations):
    parent, learned, receipt, _ = observations
    receipt.pop("matched_loss_before")
    receipt.pop("matched_loss_after")
    receipt.update(first_step_loss=1., last_step_loss=.1)
    gate = validate_smoke(parent, learned, learned, parent, receipt)
    assert not gate["passed"] and not gate["checks"]["matched_loss_finite"]
    assert gate["matched_loss_ratio"] is None


@pytest.mark.parametrize("field,value", [("row_ids", ["another-population"]),
    ("tokenization_sha256", "b"*64), ("supervised_tokens", 319), ("prompt_representation", "native")])
def test_matched_loss_identity_cannot_change(observations, field, value):
    parent, learned, receipt, _ = observations
    receipt["matched_loss_after"][field] = value
    with pytest.raises(ValueError, match="same complete tokenized population"):
        validate_smoke(parent, learned, learned, parent, receipt)


@pytest.mark.parametrize("after", [.50001, float("nan"), -1.])
def test_nonfinite_or_insufficient_loss_reduction_refuses_smoke(observations, after):
    parent, learned, receipt, _ = observations
    receipt["matched_loss_after"]["mean_token_nll"] = after
    assert not validate_smoke(parent, learned, learned, parent, receipt)["passed"]


def test_zero_or_unchanged_factors_fail_even_if_metadata_claims_all_changed(observations):
    parent, learned, receipt, _ = observations
    receipt["parameter_changes"][0]["changed_entries"] = 0
    gate = validate_smoke(parent, learned, learned, parent, receipt)
    assert not gate["checks"]["all_adapter_factors_changed"]


def test_fresh_reload_requires_native_tokens_not_only_same_semantic_answer(observations):
    parent, learned, receipt, _ = observations
    reloaded = deepcopy(learned)
    reloaded["rows"][0]["generated_token_ids"].append(999)
    gate = validate_smoke(parent, learned, reloaded, parent, receipt)
    assert not gate["checks"]["fresh_reload_predictions_identical"]


def test_disabled_and_zero_adapters_must_reproduce_parent(observations):
    parent, learned, receipt, _ = observations
    receipt["zero_init_measurement"] = deepcopy(learned)
    gate = validate_smoke(parent, learned, learned, learned, receipt)
    assert not gate["checks"]["zero_init_parent_identical"]
    assert not gate["checks"]["disabled_parent_identical"]


def test_partial_accuracy_below_ninety_percent_cannot_pass(observations):
    parent, _, receipt, measurement = observations
    learned = measurement(28)
    gate = validate_smoke(parent, learned, learned, parent, receipt)
    assert not gate["checks"]["completed_correct_rate"]


def test_parent_correct_metadata_or_semantic_family_tampering_is_rejected(observations):
    parent, learned, receipt, _ = observations
    corrupt = deepcopy(learned)
    corrupt["rows"][0]["comparison_id"] = "invented"
    with pytest.raises(ValueError, match="identities|different questions"):
        validate_smoke(parent, learned, corrupt, parent, receipt)
    corrupt = deepcopy(parent)
    corrupt["rows"][0]["correct"] = True
    with pytest.raises(ValueError, match="semantic outcome"):
        exact_predictions(parent, corrupt)


def test_sealed_or_development_rows_are_not_smoke_evidence(observations):
    parent, learned, _, _ = observations
    another = deepcopy(learned)
    another["rows"][0]["id"] = "baseline-decimal/hidden/target/0000"
    with pytest.raises(ValueError, match="another study partition"):
        exact_predictions(parent, another)
