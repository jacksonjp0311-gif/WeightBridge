"""Exercise the admission and exact behavioral acceptance gates."""

import json
from types import SimpleNamespace

import pytest

from experiments.phase_iiig import run_replication, run_trial
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


@pytest.mark.parametrize("winner_qualifies", [True, False])
def test_finish_only_accepts_preselected_winner_and_replays_rollback(tmp_path, monkeypatch, winner_qualifies):
    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    out, peer, material = (tmp_path / name for name in ("local", "peer", "material"))
    for directory in (out, peer, material):
        directory.mkdir()
    write(material / "manifest.json", {})
    manifest_sha = file_hash(material / "manifest.json")
    config = run_trial._json(run_trial.PROTOCOL)
    qwen_sha = config["checkpoints"]["qwen"]
    sources = {"script": "digest"}
    monkeypatch.setattr(run_trial, "_sources", lambda: sources)
    monkeypatch.setattr(run_trial, "load_checkpoint", lambda _: SimpleNamespace(checkpoint_sha256=lambda: qwen_sha))
    candidates = {"constrained": {}, "gradient": {}}
    common = {"manifest_sha256": manifest_sha, "source_sha256": sources}
    lock = {**common, "model": "qwen", "base_sha256": qwen_sha, "family": "addition",
            "winner": "constrained", "eligible": {"constrained": "c", "gradient": "g"},
            "candidates": candidates}
    write(out / "candidate-lock.json", lock)
    write(peer / "candidate-lock.json", {**common, "model": "smollm2", "base_sha256": config["checkpoints"]["smollm2"]})
    write(out / "preparation.json", {"model": "qwen", "winner": "constrained", "candidates": candidates,
                                     "selected_family": "addition", "single_cycle": "FAIL",
                                     "resources": {"wall_seconds": 0, "host_cpu_seconds": 0, "peak_rss_bytes": 0}})
    base = [
        *item("a", "addition", "target", "17", ["18", "18"]),
        *item("b", "addition", "target", "19", ["20", "20"]),
        *item("copy", "copy", "preserve", "7", ["7", "7"]),
    ]
    winner = [{**row, "exact": row["answer"]} if winner_qualifies or row["item"] == "a" else row for row in base]
    alternative = [{**row, "exact": row["answer"]} for row in base]
    calls = []

    def verifier(path, expected, material, split, family, output, manifest_sha, timeout, seal):
        assert split == "hidden" and family == "addition"
        assert run_trial._json(seal)["both_candidate_locks"] == {
            "qwen": file_hash(out / "candidate-lock.json"), "smollm2": file_hash(peer / "candidate-lock.json")}
        calls.append(output.stem)
        rows = {qwen_sha: base, "c": winner, "g": alternative}[expected]
        return {"measurement": {"rows": rows}}

    monkeypatch.setattr(run_trial, "_independent", verifier)
    report = run_trial.finish(tmp_path, material, out, peer)
    assert report["candidates"]["gradient"]["hidden"]["disposition"] == "PASS"
    assert report["single_cycle"] == ("PASS" if winner_qualifies else "FAIL")
    assert ("restart" in calls) == ("rollback" in calls) == winner_qualifies


def test_replication_fit_caps_practice_items_without_dropping_controls(monkeypatch):
    qualification = {"solved": {"target": list(range(20)), "positive": list(range(4)),
                                "preserve": list(range(4))},
                     "repeatable_errors": {f"error-{index}": "1" for index in range(12)}}

    def fit(runner, measured, family, scoped, settings):
        assert scoped["solved"]["positive"] == list(range(4))
        assert scoped["solved"]["preserve"] == list(range(4))
        assert scoped["solved"]["target"] == list(range(16))
        assert list(scoped["repeatable_errors"]) == [f"error-{index}" for index in range(8)]
        return [None] * 64

    monkeypatch.setattr(run_replication, "_FIT_ROWS", fit)
    assert len(run_replication._fit_rows(None, [], "multiplication", qualification, {})) == 64


def test_decimal_fit_uses_observed_wrong_answer_and_exact_decimal_preservation_rival(monkeypatch):
    from experiments.phase_iiig import run_decimal

    measured = [
        *item("target", "larger-decimal", "target", "-0.9", ["-0.91", "-0.91"]),
        *item("copy", "copy", "preserve", "-1.005", ["-1.005", "-1.005"]),
    ]
    qualification = {"solved": {"target": [], "positive": [], "preserve": ["copy"]},
                     "repeatable_errors": {"target": "-0.91"}}
    monkeypatch.setattr(run_decimal, "answer_pair", lambda runner, prompt, answer, rival, **_: (answer, rival))
    fit = run_decimal._fit_rows(None, measured, "larger-decimal", qualification, {"max_input_tokens": 64})
    assert [row["answers"] for row in fit] == [("-0.9", "-0.91")] * 2 + [("-1.005", "-0.005")] * 2
    assert [row["preserve"] for row in fit] == [False, False, True, True]
