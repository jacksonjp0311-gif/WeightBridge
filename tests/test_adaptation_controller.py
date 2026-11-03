import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from weightbridge.adaptation import controller
from weightbridge.agents import hf_session
from weightbridge.core import file_hash
from weightbridge.evaluation import hf_edit
from weightbridge.inspection.safetensors import load_checkpoint


_TENSORS = {
    family: f"model.language_model.layers.{layer}.mlp.down_proj.weight"
    for family, layer in (("a", 19), ("b", 21), ("c", 23))
}


def _write(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _suite(kind, families):
    return {
        "schema": "weightbridge.hf_edit_suite.v1",
        "choices": ["A", "B"],
        "target": [
            {"id": f"{family}:{kind}:target", "text": f"{kind} target for {family}", "label": "A"}
            for family in families
        ],
        "sentinel": [{"id": f"sentinel:{kind}", "text": f"{kind} unrelated prompt", "label": "B"}],
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    safetensors = pytest.importorskip("safetensors.numpy")
    base = tmp_path / "original"
    base.mkdir()
    _write(base / "config.json", {"model_type": "qwen3_5"})
    safetensors.save_file(
        {name: np.zeros((1, 2), dtype=np.float32) for name in _TENSORS.values()},
        str(base / "model.safetensors"),
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    root = tmp_path / "research"

    def score(path, choices, rows, **_):
        weights = safetensors.load_file(str(path / "model.safetensors"))
        edited = {family: weights[name][0, 0] > 0 for family, name in _TENSORS.items()}
        # The third edit repairs its own family but reverses the first repair.
        edited["a"] = edited["a"] and not edited["c"]
        return ["A" if edited.get(row["id"].split(":")[0], False) else "B" for row in rows]

    def propose(path, practice, tensor, strength, inference):
        checkpoint = load_checkpoint(path)
        assert set(practice) == {"schema", "choices", "target", "sentinel"}
        assert all(":practice-" in row["id"] for row in practice["target"])
        return {
            "schema": "weightbridge.patch.v1",
            "base_sha256": checkpoint.checkpoint_sha256(),
            "tensor_sha256": {tensor: checkpoint.tensor_sha256(tensor)},
            "reason": "Exercise the adaptation controller with a known numeric intervention",
            "changes": [{"tensor": tensor, "index": [0, 0], "op": "add", "value": strength}],
        }

    for module in (controller, hf_edit, hf_session):
        monkeypatch.setattr(module, "_score_checkpoint", score)
    monkeypatch.setattr(hf_session, "propose_direction", propose)

    def freeze(*, overlap=False):
        cycles = []
        for number, (family, tensor) in enumerate(_TENSORS.items(), 1):
            cycle_id = f"cycle-{number:02d}"
            intent = {
                "schema": "weightbridge.awl.v1", "observe": f"Errors on family {family}",
                "diagnose": "A small numeric edit can repair this family", "target": [tensor],
                "optimize": {"method": "activation_ridge", "strengths": [0.25]},
            }
            practice = _suite(f"practice-{family}", [family])
            holdout = _suite(f"holdout-{family}", [family])
            if overlap and number == 1:
                holdout["target"][0]["text"] = practice["target"][0]["text"].upper() + " "
            paths = {name: _write(inputs / f"{cycle_id}-{name}.json", document)
                     for name, document in (("intent", intent), ("practice", practice), ("holdout", holdout))}
            cycles.append({"id": cycle_id, "family": family,
                           **{f"{name}_sha256": file_hash(path) for name, path in paths.items()}})
        final = _write(inputs / "final.json", _suite("final", list(_TENSORS)))
        protocol = _write(inputs / "protocol.json", {
            "schema": "weightbridge.phase_iii.protocol.v1",
            "base_sha256": load_checkpoint(base).checkpoint_sha256(),
            "policy": {"schema": "weightbridge.policy.v1", "allow_tensors": list(_TENSORS.values()),
                       "max_entries": 1, "max_fraction": 0.5, "max_abs_delta": 0.5},
            "inference": {"device": "cpu", "dtype": "auto", "cpu_threads": 1,
                          "batch_size": 1, "max_input_tokens": 32},
            "max_candidates": 1, "min_target_gain": 1, "max_sentinel_flips": 1,
            "max_target_regressions": 0, "preservation_penalty": 1.0, "edit_cost_penalty": 1.0,
            "cycles": cycles, "final_sha256": file_hash(final),
        })
        return controller.init_research(root, base, protocol)

    return root, inputs, freeze


def test_two_retained_repairs_and_rejection_of_a_third(setup):
    root, inputs, freeze = setup
    initial = freeze()
    final = inputs / "final.json"
    with pytest.raises(ValueError, match="Complete every frozen cycle"):
        controller.score_final(root, final)
    with pytest.raises(ValueError, match="Earlier cycles"):
        controller.search_cycle(root, "cycle-02", inputs / "cycle-02-intent.json", inputs / "cycle-02-practice.json")

    retention = []
    accepted = []
    for number in range(1, 4):
        cycle_id = f"cycle-{number:02d}"
        selection = controller.search_cycle(
            root, cycle_id, inputs / f"{cycle_id}-intent.json", inputs / f"{cycle_id}-practice.json",
        )
        assert selection["finalist"]["tensor"] == list(_TENSORS.values())[number - 1]
        holdout = inputs / f"{cycle_id}-holdout.json"
        if number == 1:
            original = holdout.read_bytes()
            holdout.write_bytes(original + b" ")
            with pytest.raises(ValueError, match="differs from the frozen protocol"):
                controller.verify_cycle(root, cycle_id, holdout)
            holdout.write_bytes(original)
        review = controller.verify_cycle(root, cycle_id, holdout, retention)
        assert review["target_gain"] == 1
        assert review["sentinel_prediction_flips"] == 0
        if number < 3:
            assert review["disposition"] == "ACCEPT"
            assert review["retention_flips"] == 0
            accepted.append(controller.research_status(root))
            retention.append(holdout)
        else:
            assert review["disposition"] == "REJECT"
            assert review["retention_count"] == 4
            assert review["retention_flips"] == 1
            report = json.loads((root / "evaluation" / cycle_id / "report.json").read_bytes())
            assert report["disposition"] == "PASS"
            assert controller.research_status(root)["head_sha256"] == accepted[-1]["head_sha256"]

    resumed = subprocess.run(
        [sys.executable, "-m", "weightbridge", "phase-iii", "status", "--root", str(root)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True,
    )
    assert json.loads(resumed.stdout)["head_sha256"] == accepted[-1]["head_sha256"]
    score = controller.score_final(root, final)
    assert score["base_target_correct"] == 0
    assert score["head_target_correct"] == score["target_gain"] == 2
    assert score["target_regressions"] == score["sentinel_prediction_flips"] == 0
    assert score["accepted_cycles"] == 2
    assert [row["head_prediction"] for row in score["target"]] == ["A", "A", "B"]
    with pytest.raises(ValueError, match="already evaluated"):
        controller.score_final(root, final)

    rollback = controller.rollback_research(root, accepted[0]["head_sha256"], reason="Restore the first repair")
    assert rollback["head_sha256"] == accepted[0]["head_sha256"]
    assert [entry["cycle_id"] for entry in rollback["ancestry"]] == [None, "cycle-01"]
    for state in (initial, *accepted):
        assert load_checkpoint(state["head_path"]).checkpoint_sha256() == state["head_sha256"]


def test_independent_verification_rejects_practice_overlap(setup):
    root, inputs, freeze = setup
    initial = freeze(overlap=True)
    cycle_id = "cycle-01"
    controller.search_cycle(root, cycle_id, inputs / f"{cycle_id}-intent.json", inputs / f"{cycle_id}-practice.json")
    with pytest.raises(ValueError, match="overlap cycle practice"):
        controller.verify_cycle(root, cycle_id, inputs / f"{cycle_id}-holdout.json")
    assert controller.research_status(root) == initial
