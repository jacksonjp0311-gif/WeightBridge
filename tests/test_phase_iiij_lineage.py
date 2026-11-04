"""Exercise Phase III-J host closure with real checkpoint and lineage artifacts."""

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.phase_iiij import decimal_roster, verification
from weightbridge.adaptation.lineage import audit_lineage, rollback_lineage
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches import apply_safetensors_patch


@pytest.fixture
def closure_fixture(tmp_path, monkeypatch):
    save = pytest.importorskip("safetensors.numpy").save_file
    base = tmp_path / "parent"
    base.mkdir()
    json_dump_atomic(base / "config.json", {})
    tensor = base / "model.safetensors"
    save({"weight": np.zeros(2, dtype=np.float32)}, str(tensor))
    parent_bytes = tensor.read_bytes()
    checkpoint = load_checkpoint(base)
    base_sha = checkpoint.checkpoint_sha256()

    out = tmp_path / "private-trial"
    candidate = out / "checkpoints" / "A"
    receipt = apply_safetensors_patch(base, {
        "schema": "weightbridge.patch.v1", "base_sha256": base_sha,
        "tensor_sha256": {"weight": checkpoint.tensor_sha256("weight")}, "reason": "Phase III-J fixture",
        "changes": [{"tensor": "weight", "index": [0], "op": "add", "value": .25}],
    }, {"schema": "weightbridge.policy.v1", "allow_tensors": ["weight"], "max_entries": 1,
        "max_fraction": .5, "max_abs_delta": .5}, candidate)
    candidate_sha = receipt["candidate_sha256"]
    material, hidden = tmp_path / "material", tmp_path / "sealed-fixture.json"
    decimal_roster.freeze(material, hidden, hidden_seed="synthetic-phase-iiij-closure")

    config_path = tmp_path / "fixture-config.json"
    config = {**verification._read(verification.PROTOCOL), "checkpoint_sha256": base_sha}
    json_dump_atomic(config_path, config)
    monkeypatch.setattr(verification, "PROTOCOL", config_path)
    state = SimpleNamespace(current=None, change_replay=None, launches=[])

    @contextmanager
    def open_checkpoint(path, config, expected_sha):
        assert load_checkpoint(path).checkpoint_sha256() == expected_sha
        yield None, expected_sha, SimpleNamespace(architecture="synthetic")

    def measure(sha, rows, contract):
        # Choose the first two frozen semantic items for repeatable parent errors.
        failing = tuple(dict.fromkeys(row["item"] for row in rows if row["group"] == "target"))[:2]
        responses = [{**row, "text": next(option for option in row["operands"] if option != row["answer"])
                      if sha == base_sha and row["item"] in failing else row["answer"], "stop_reason": "eos"}
                     for row in rows]
        if state.current == state.change_replay:
            # The parsed decimal is unchanged. Exact worker replay still has to fail.
            responses[0]["text"] = " " + responses[0]["text"]
        return {"endpoint": contract.manifest(), "resources": {}, "rows": responses}

    def launch(args, *, cwd, check, timeout):
        assert args[1:4] == ["-m", "experiments.phase_iiij.verification", "worker"]
        assert cwd == verification.ROOT and check and timeout > 0
        values = dict(zip(args[4::2], args[5::2], strict=True))
        output = Path(values["--out"])
        state.current = output.stem
        state.launches.append(state.current)
        verification.worker(Path(values["--base"]), values["--sha256"], Path(values["--material"]),
                            values["--split"], output, values["--manifest-sha256"],
                            lock=Path(values["--lock"]) if "--lock" in values else None,
                            lock_sha=values.get("--lock-sha256"),
                            hidden=Path(values["--hidden"]) if "--hidden" in values else None)

    monkeypatch.setattr(verification, "_open", open_checkpoint)
    monkeypatch.setattr(verification, "measure_endpoint", measure)
    monkeypatch.setattr(verification.subprocess, "run", launch)
    return SimpleNamespace(base=base, base_sha=base_sha, parent_bytes=parent_bytes, tensor=tensor,
                           out=out, hidden=hidden, material=material, candidate=candidate,
                           candidate_sha=candidate_sha, receipt=receipt, state=state,
                           candidates={"A": {"checkpoint_sha256": candidate_sha, "realized_update_norm": .25}})


@pytest.mark.parametrize("change_replay", [None, "hidden-replay", "hidden-rollback"])
def test_selection_and_hidden_closure_replay_exactly_before_lineage(closure_fixture, change_replay):
    run = closure_fixture
    run.state.change_replay = change_replay
    selected = verification.assess(run.base, run.material, run.out, run.candidates)
    assert selected["winner"] == "A"
    assert run.state.launches == ["selection-base", "selection-A"]
    result = verification.finish(run.base, run.material, run.out, run.hidden, selected["candidate_lock_sha256"])
    assert run.state.launches == ["selection-base", "selection-A", "hidden-base", "hidden-candidate",
                                  "hidden-replay", "hidden-rollback"]
    assert result["hidden_gate_passed"] is True
    assert result["target_item_gain"] == 2
    assert result["restart_exact"] is (change_replay != "hidden-replay")
    assert result["rollback_exact"] is (change_replay != "hidden-rollback")
    assert run.tensor.read_bytes() == run.parent_bytes
    assert load_checkpoint(run.base).checkpoint_sha256() == run.base_sha
    workspace = run.out / "checkpoints" / "lineage"
    if change_replay is not None:
        assert result["disposition"] == "FAIL" and result["meta"]["action"] == "hold"
        assert not workspace.exists()
        return
    assert result["disposition"] == "PASS" and result["meta"]["action"] == "record_verified"
    audit = audit_lineage(workspace)
    assert audit["head_sha256"] == result["accepted_candidate_sha256"] == run.candidate_sha
    assert audit["last_event_sha256"] == result["lineage_event_sha256"]
    rolled_back = rollback_lineage(workspace, run.base_sha, reason="Restore synthetic parent")
    assert rolled_back["head_sha256"] == run.base_sha
    assert [event["kind"] for event in rolled_back["events"]] == ["ACCEPT", "ROLLBACK"]


def test_receipt_identity_key_is_required_before_candidate_measurement(closure_fixture):
    run = closure_fixture
    bad_receipt = dict(run.receipt)
    bad_receipt["checkpoint_sha256"] = bad_receipt.pop("candidate_sha256")
    (run.candidate / ".weightbridge_receipt.json").write_text(json.dumps(bad_receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt does not bind"):
        verification.assess(run.base, run.material, run.out, run.candidates)
    assert run.state.launches == ["selection-base"]
    assert not (run.out / "candidate-lock.json").exists()
