from collections import Counter
import importlib
from pathlib import Path
import re

import pytest


@pytest.fixture
def wording(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "experiments/original_weight_edits"))
    return importlib.import_module("run_qwen_wording_ablation")


def test_paired_arms_share_items_and_budget_with_fresh_holdout_wordings(wording):
    protocol = wording.load_protocol(wording.CONFIG)
    previous = wording.prior.protocol()
    for key in ("gate", "calibration", "candidate_evaluations_per_arm", "policy", "strengths", "splits"):
        assert protocol[key] == previous[key]

    seen = set()
    for spec in protocol["cases"]:
        groups = wording.build_splits(spec["slot"])
        single, multi, holdout = (groups[key] for key in ("single", "multi", "holdout"))
        assert single["sentinel"] == multi["sentinel"]
        assert len(multi["sentinel"]) == 16
        assert len(single["target"]) == len(multi["target"]) == 8
        for a, b in zip(single["target"], multi["target"]):
            assert (a["id"], a["label"]) == (b["id"], b["label"])
            assert re.findall(r"\d+", a["text"]) == re.findall(r"\d+", b["text"])[:2]
        assert Counter(row["wording"] for row in multi["target"]) == {"standard": 4, "inline": 2, "brackets": 2}
        assert {row["wording"] for row in holdout["target"] if row["form"] == "paraphrase"} == {
            "comparison", "maximum"
        }
        for group in groups.values():
            for rows in group.values():
                assert Counter(row["label"] for row in rows) == {"A": len(rows) // 2, "B": len(rows) // 2}
        for name in ("construction", "single", "holdout"):
            for row in groups[name]["target"]:
                pair = tuple(sorted(re.findall(r"\d+", row["text"])))
                assert pair not in seen
                seen.add(pair)


def test_each_guided_arm_uses_its_own_random_baseline_and_the_same_locality_gate(wording):
    def score(values, prediction):
        return {"target": [{"correct": value} for value in values],
                "sentinel": [{"id": "s0", "correct": prediction == "A", "prediction": prediction}]}

    holdout = {"target": [{"form": "same"}, {"form": "same"}, {"form": "paraphrase"}, {"form": "paraphrase"}]}
    scores = {
        "pristine": score([True, True, True, True], "A"),
        "damaged": score([True, False, False, False], "A"),
        "single": score([True, True, True, False], "A"),
        "multi": score([True, True, True, False], "A"),
        "random_single": score([True, False, False, False], "A"),
        "random_multi": score([True, True, True, False], "A"),
    }
    selected = {name: 0 for name in ("single", "multi", "random_single", "random_multi")}
    gate = wording.protocol()["gate"]
    result = wording.outcomes(scores, holdout, selected, gate)
    assert result["single"]["disposition"] == "PASS"
    assert result["multi"]["disposition"] == "FAIL"
    scores["random_multi"] = scores["damaged"]
    scores["multi"] = score([True, True, True, True], "B")
    assert wording.outcomes(scores, holdout, selected, gate)["multi"]["disposition"] == "FAIL"
