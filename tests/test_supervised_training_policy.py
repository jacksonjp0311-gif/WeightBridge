from dataclasses import replace

import pytest

from weightbridge.adaptation.training_policy import TrainingAuthorization
from experiments.phase_iiiq_baseline.measurement import compare, question, summarize


def authorization():
    return TrainingAuthorization("weightbridge.supervised_training_authorization.v1", "a" * 64,
        ("model.layers.0.mlp.down_proj",), 1000, 8, 16, 0, (1e-4, 3e-4), "AdamW", 2, 400,
        100000, 1000000, 1000000, 3600, 18000, 3, True, "c" * 64)


def test_training_authorization_is_separate_and_bounded():
    auth = authorization()
    auth.validate_plan(parent_sha256="a" * 64, module_names=auth.allowed_modules,
        trainable_parameters=900, rank=8, alpha=16, dropout=0, learning_rate=1e-4, steps=384, run_index=1)
    with pytest.raises(ValueError):
        auth.validate_plan(parent_sha256="b" * 64, module_names=auth.allowed_modules,
            trainable_parameters=900, rank=8, alpha=16, dropout=0, learning_rate=1e-4, steps=384, run_index=1)
    with pytest.raises(RuntimeError):
        auth.check_usage(steps=401, answer_tokens=1, input_tokens=1, wall_seconds=1, cpu_seconds=1, rss_bytes=1)
    with pytest.raises(ValueError):
        replace(auth, allowed_modules=("model.visual.proj",))
    assert TrainingAuthorization.from_dict(auth.to_dict()) == auth


def test_general_control_semantics_and_tampering():
    row = {"family": "integer-add", "group": "preserve", "operands": ["5", "-2"],
           "answer": "3", "prompt": "Add the integers.\nIntegers: 5 and -2\nAnswer:"}
    assert question(row).answer == "3"
    with pytest.raises(ValueError):
        question({**row, "answer": "4"})
    with pytest.raises(ValueError):
        question({**row, "operands": ["5", "-1"]})


def measurement(correct):
    rows = []
    for variant in (0, 1):
        for order in (0, 1):
            for wording in (0, 1):
                index = variant * 4 + order * 2 + wording
                rows.append({"id": str(index), "item": f"{variant}/{order}",
                    "comparison_id": str(variant), "pair_id": "f", "wording": wording,
                    "order": order, "group": "target", "family": "larger-decimal",
                    "prompt": f"question {index}", "answer": "-1", "tier": "negative-close",
                    "correct": index in correct, "valid": True, "completed": True,
                    "status": "correct" if index in correct else "incorrect", "stop_reason": "token_budget"})
    return {"endpoint": {}, "rows": rows}


def test_complete_family_requires_eight_responses_and_preservation():
    parent, partial, full = measurement({0}), measurement(set(range(7))), measurement(set(range(8)))
    assert summarize(partial)["complete_families"] == 0
    assert compare(parent, full)["passed"]
    assert not compare(parent, partial)["passed"]
    regressed = measurement(set(range(1, 8)))
    assert compare(parent, regressed)["prohibited_regressions"] == ["0"]
    assert summarize(full)["decode_termination"] == {"token_budget": 8}
