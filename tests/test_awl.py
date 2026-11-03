from copy import deepcopy

import pytest

from weightbridge.adaptation.awl import AdaptationIntent, OptimizationSpec, parse_awl
from weightbridge.adaptation.synthesis import plan_candidates


_FIRST = "model.language_model.layers.19.mlp.down_proj.weight"
_SECOND = "model.language_model.layers.23.mlp.down_proj.weight"
_UNREQUESTED = "model.language_model.layers.21.mlp.down_proj.weight"
_DOCUMENT = {
    "schema": "weightbridge.awl.v1",
    "observe": "threshold-reversal",
    "diagnose": "The answer direction has weakened for target prompts.",
    "target": [_FIRST, _SECOND],
    "optimize": {"method": "activation_ridge", "strengths": [0.03, 0.05]},
}


def test_parse_awl_returns_immutable_intent_with_independent_lists():
    document = deepcopy(_DOCUMENT)
    intent = parse_awl(document)
    document["target"].clear()
    document["optimize"]["strengths"].clear()
    assert intent == AdaptationIntent(
        "threshold-reversal",
        "The answer direction has weakened for target prompts.",
        (_FIRST, _SECOND),
        OptimizationSpec("activation_ridge", (0.03, 0.05)),
    )


@pytest.mark.parametrize("change", [
    {"schema": "weightbridge.awl.v2"},
    {"observe": " "},
    {"diagnose": "a" * 513},
    {"target": []},
    {"target": [_FIRST, _FIRST]},
    {"optimize": {"method": "unknown", "strengths": [0.03]}},
    {"optimize": {"method": "activation_ridge", "strengths": [True]}},
    {"optimize": {"method": "activation_ridge", "strengths": [float("nan")]}},
    {"optimize": {"method": "activation_ridge", "strengths": [float("inf")]}},
    {"optimize": {"method": "activation_ridge", "strengths": [1.01]}},
    {"optimize": {"method": "activation_ridge", "strengths": [0.03, 0.03]}},
    {"optimize": {"method": "activation_ridge", "strengths": [0.03], "max_candidates": 1000}},
    {"preserve": False},
    {"retain": True},
    {"holdout": "hidden.json"},
])
def test_parse_awl_rejects_invalid_values_and_host_actions(change):
    with pytest.raises(ValueError, match="AWL"):
        parse_awl({**_DOCUMENT, **change})


def test_plan_stays_within_intent_and_host_budget_despite_preferences():
    intent = parse_awl(_DOCUMENT)
    allowed = [_FIRST, _SECOND, _UNREQUESTED]
    preferred = [_UNREQUESTED, _SECOND, _SECOND, "missing.weight"]
    assert plan_candidates(intent, allowed, 3, preferred) == (
        (_SECOND, 0.03), (_FIRST, 0.03), (_SECOND, 0.05),
    )
    assert plan_candidates(intent, allowed, 0, preferred) == ()
    with pytest.raises(ValueError, match="allowlist"):
        plan_candidates(intent, [_FIRST], 3)
    with pytest.raises(ValueError, match="budget"):
        plan_candidates(intent, allowed, True)
