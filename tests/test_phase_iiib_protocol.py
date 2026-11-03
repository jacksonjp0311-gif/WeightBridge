import json
from collections import defaultdict
from types import SimpleNamespace

import pytest

from experiments.phase_iiib import qwen_material, run_qwen_ablation
from weightbridge.adaptation.awl_v2 import parse_awl_v2
from weightbridge.core import file_hash
from weightbridge.evaluation.hf_edit import _load_suite


def test_frozen_qwen_material_balances_formats_and_separates_items(tmp_path, monkeypatch):
    shard = tmp_path / "test.safetensors"
    shard.write_bytes(b"fixture identity")
    design = json.loads(qwen_material.DESIGN.read_bytes())
    design.update(shard_sha256=file_hash(shard))
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design), encoding="utf-8")
    monkeypatch.setattr(qwen_material, "DESIGN", design_path)
    checkpoint = SimpleNamespace(checkpoint_sha256=lambda: design["model_sha256"], shards=[shard])
    monkeypatch.setattr(qwen_material, "load_checkpoint", lambda _: checkpoint)
    seen_rosters = set()

    for index, is_pilot in enumerate((True, False, False)):
        output = tmp_path / f"material-{index}"
        frozen = qwen_material.materialize(tmp_path, output, pilot=is_pilot, roster_index=max(0, index - 1))
        practice_keys, holdout_keys = set(), set()

        def read(relative):
            assert file_hash(output / relative) == frozen["artifacts"][relative]
            return _load_suite((output / relative).read_bytes())

        def keys(suite, group):
            by_item = defaultdict(list)
            for row in suite[group]:
                by_item[row["item_key"]].append(row)
            for paired in by_item.values():
                assert {row["answer_position"] for row in paired} == {0, 1}
                assert {row["label"] for row in paired} == set(suite["choices"])
                assert len(paired) == 2
            return set(by_item)

        for cycle in frozen["cycles"]:
            generic, aware, screen = (read(cycle["practice"][method]) for method in ("generic", "aware", "screen"))
            assert generic["target"] == aware["target"] == screen["target"]
            assert len(generic["sentinel"]) == len(aware["sentinel"])
            assert set(row["family"] for row in generic["sentinel"]) == {"generic"}
            assert set(row["family"] for row in aware["sentinel"]) == {"generic", *(
                family for family in design["families"] if family != cycle["family"]
            )}
            current_practice = keys(screen, "target") | keys(screen, "sentinel")
            assert not practice_keys & current_practice
            practice_keys.update(current_practice)
            for arm, relative in cycle["intents"].items():
                intent = parse_awl_v2(json.loads((output / relative).read_bytes()))
                fitting = generic if arm == "A" else aware
                assert set(intent.preserve) == {row["family"] for row in fitting["sentinel"]}

            ab, xy = (read(cycle["holdouts"][name]) for name in ("ab", "xy"))
            assert ab["choices"] != xy["choices"]
            assert keys(ab, "target") == keys(xy, "target")
            assert keys(ab, "sentinel") == keys(xy, "sentinel")
            current_holdout = keys(ab, "target") | keys(ab, "sentinel")
            assert not holdout_keys & current_holdout
            holdout_keys.update(current_holdout)

        final_ab, final_xy = (read(frozen["final"][name]) for name in ("ab", "xy"))
        assert keys(final_ab, "target") == keys(final_xy, "target")
        final_keys = keys(final_ab, "target") | keys(final_ab, "sentinel")
        assert not final_keys & (practice_keys | holdout_keys)
        assert not practice_keys & holdout_keys
        roster_keys = practice_keys | holdout_keys | final_keys
        assert len(roster_keys) == frozen["item_count"]
        assert not seen_rosters & roster_keys
        seen_rosters.update(roster_keys)


@pytest.mark.parametrize("failure", (None, "prior_flip", "prior_mismatch", "alternate_regression"))
def test_cycle_gate_checks_both_choice_formats_and_prior_predictions(tmp_path, monkeypatch, failure):
    design = json.loads(qwen_material.DESIGN.read_bytes())
    sources, prior = {}, {}
    for fmt, choices in design["choice_formats"].items():
        target = {"id": fmt + "-target", "text": "current target", "label": choices[0]}
        sentinel = {"id": fmt + "-sentinel", "text": "current sentinel", "label": choices[1]}
        earlier = {"id": fmt + "-prior", "text": "earlier target", "label": choices[0]}
        source = tmp_path / f"source-{fmt}.json"
        source.write_text(json.dumps({"choices": choices, "target": [target], "sentinel": [sentinel]}),
                          encoding="utf-8")
        sources[fmt] = source
        prior[fmt] = {earlier["id"]: (earlier, choices[0])}

    def fake_evaluate(_base, _candidate, suite_path, _report_path, **_inference):
        suite = json.loads(suite_path.read_bytes())
        choices = suite["choices"]
        alternate = choices == design["choice_formats"]["xy"]

        def result(row, parent, candidate):
            return {**row, "base_prediction": parent, "candidate_prediction": candidate}

        target = result(suite["target"][0], choices[1], choices[0])
        if alternate and failure == "alternate_regression":
            target = result(suite["target"][0], choices[0], choices[1])
        sentinel, retained = suite["sentinel"]
        if alternate and failure == "prior_flip":
            preservation = result(retained, choices[0], choices[1])
        elif alternate and failure == "prior_mismatch":
            preservation = result(retained, choices[1], choices[1])
        else:
            preservation = result(retained, choices[0], choices[0])
        return {"target": [target], "sentinel": [result(sentinel, choices[1], choices[1]), preservation]}

    monkeypatch.setattr(run_qwen_ablation, "evaluate_hf_edit", fake_evaluate)
    _, observed = run_qwen_ablation._verify(tmp_path, tmp_path, tmp_path, sources, prior, design)
    assert observed["accepted_by_verifier"] is (failure is None)
    assert observed["formats"]["ab"]["target_gain"] == 1
    assert observed["formats"]["xy"]["prior_count"] == 1
