from collections import Counter
import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def cohort(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "experiments/original_weight_edits"))
    return importlib.import_module("run_qwen_repair_cohort")


def test_frozen_cases_and_prompts_are_balanced_and_disjoint(cohort):
    protocol = cohort.load_protocol(cohort.CONFIG)
    assert Counter((case["layer"], case["fault"]) for case in protocol["cases"]) == {
        (layer, fault): 2 for layer in cohort.LAYERS for fault in ("reversal", "one_position")
    }
    seen = set()
    for spec in protocol["cases"]:
        splits = cohort.build_splits(spec["slot"])
        for groups in splits.values():
            for rows in groups.values():
                assert Counter(row["label"] for row in rows) == {"A": len(rows) // 2, "B": len(rows) // 2}
                for row in rows:
                    assert row["text"] not in seen
                    seen.add(row["text"])
        assert Counter((row["form"], row["label"]) for row in splits["holdout"]["target"]) == {
            (form, label): 6 for form in ("same", "paraphrase") for label in ("A", "B")
        }


def test_signed_fit_and_random_control_norm(cohort):
    x = np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0]])
    columns, vector = cohort.activation_gate(x, np.array([1.0, -1.0]), sentinel_weight=256)
    margins = x[:, columns] @ vector
    assert margins[0] > 0 > margins[1]
    assert margins[2] == 0
    delta = np.outer([0.5, -0.5, 0.25], vector)
    random = cohort.random_delta(np.random.default_rng(1), delta)
    assert random.shape == delta.shape
    assert np.linalg.matrix_rank(random) == 1
    assert np.linalg.norm(random) == pytest.approx(np.linalg.norm(delta))


def test_gate_keeps_pristine_locality_and_damaged_changes_separate(cohort):
    def scores(target, sentinel):
        return {
            "target": [{"id": f"t{i}", "correct": value} for i, value in enumerate(target)],
            "sentinel": [{"id": "s0", "prediction": sentinel, "correct": sentinel == "A"}],
        }

    holdout = {"target": [{"form": form} for form in ("same", "same", "paraphrase", "paraphrase")]}
    arms = {"guided": 0, "random": None}
    gate = cohort.protocol()["gate"]
    values = {
        "pristine": scores([True, True, True, True], "A"),
        "damaged": scores([True, False, False, False], "B"),
        "guided": scores([True, True, True, False], "A"),
        "random": scores([True, False, False, False], "B"),
    }
    result = cohort.outcome(values, holdout, arms, gate)
    assert result["disposition"] == "PASS"
    assert result["sentinel_flips_vs_pristine"]["guided"] == 0
    assert result["sentinel_flips_vs_damaged"]["guided"] == 1
    values["random"] = values["guided"]
    assert cohort.outcome(values, holdout, arms, gate)["disposition"] == "FAIL"
