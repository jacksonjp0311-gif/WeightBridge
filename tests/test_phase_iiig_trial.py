"""Exercise the admission and exact behavioral acceptance gates."""

import pytest

from experiments.phase_iiig import run_trial
from experiments.phase_iiig.run_trial import compare, qualify
from weightbridge.core import file_hash


def item(name, family, group, answer, predictions):
    return [
        {"item": name, "id": f"{name}-{wording}", "wording": wording, "family": family,
         "group": group, "prompt": f"{name} wording {wording}", "answer": answer, "exact": prediction}
        for wording, prediction in enumerate(predictions)
    ]


def test_qualification_distinguishes_repeatable_semantic_error_from_bad_format_and_instability():
    rows = [
        *item("solved", "addition", "target", "17", ["17", "17"]),
        *item("error", "addition", "target", "17", ["18", "18"]),
        *item("unstable", "addition", "target", "17", ["18", "19"]),
        *item("unparsed", "addition", "target", "17", [None, None]),
        *item("easy", "addition", "positive", "7", ["7", "7"]),
        *item("copy", "copy", "preserve", "5", ["5", "5"]),
    ]
    thresholds = {"min_parse_rate": .8, "min_positive_solved": 1, "min_preserve_solved": 1,
                  "min_target_solved": 1, "min_repeatable_errors": 1}
    result = qualify(rows, "addition", thresholds)
    assert result["disposition"] == "PASS"
    assert result["repeatable_errors"] == {"error": "18"}
    assert result["parsed"] == [10, 12]
    assert qualify(rows, "addition", {**thresholds, "min_repeatable_errors": 2})["disposition"] == "HOLD"


def test_gain_requires_both_wordings_and_refuses_presentation_level_regression():
    base = [
        *item("target", "addition", "target", "17", ["18", "18"]),
        *item("partial", "addition", "target", "19", ["19", "20"]),
        *item("copy", "copy", "preserve", "7", ["7", "7"]),
    ]
    gate = {"min_target_gain": 1, "max_regressions": 0}
    candidate = [{**row, "exact": "17"} if row["item"] == "target" else row for row in base]
    assert compare(base, candidate, "addition", gate, 1.)["disposition"] == "PASS"
    candidate[1] = {**candidate[1], "exact": "18"}
    assert compare(base, candidate, "addition", gate, 1.)["target_item_gain"] == 0
    candidate[1] = {**candidate[1], "exact": "17"}
    candidate[2] = {**candidate[2], "exact": "20"}
    outcome = compare(base, candidate, "addition", gate, 1.)
    assert outcome["target_item_gain"] == 1
    assert outcome["regressions"] == {"target": ["partial-0"]}
    assert outcome["disposition"] == "FAIL"


def test_hidden_worker_refuses_missing_joint_lock_before_reading_roster(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run_trial, "_material", lambda material: {})

    def reject_read(*args):
        pytest.fail("Hidden roster was read before verifying the joint lock")

    monkeypatch.setattr(run_trial, "load_split", reject_read)
    with pytest.raises(ValueError, match="requires an immutable candidate lock"):
        run_trial._worker(tmp_path, "base", tmp_path, "hidden", "addition", tmp_path / "out.json",
                          file_hash(manifest), None, None)
