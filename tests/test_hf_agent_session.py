import json
from pathlib import Path

import numpy as np
import pytest

from weightbridge.agents import hf_direction, hf_session
from weightbridge.agents.hf_session_cli import call_hf_session_tool, hf_session_tools
from weightbridge.evaluation import hf_edit
from weightbridge.inspection.safetensors import load_checkpoint


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def host_inputs(tmp_path, monkeypatch):
    safetensors = pytest.importorskip("safetensors.numpy")
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    safetensors.save_file(
        {"weight": np.zeros(2, dtype=np.float32), "forbidden": np.zeros(1, dtype=np.float32)},
        str(base / "model.safetensors"),
    )
    policy = tmp_path / "policy.json"
    write_json(policy, {"schema": "weightbridge.policy.v1", "allow_tensors": ["weight"],
                        "max_entries": 1, "max_fraction": 0.5, "max_abs_delta": 0.5})
    practice = tmp_path / "practice.json"
    write_json(practice, {
        "schema": "weightbridge.hf_edit_suite.v1", "choices": ["A", "B"],
        "target": [{"id": "practice-t", "text": "Target practice question", "label": "A"}],
        "sentinel": [{"id": "practice-s", "text": "Unrelated practice question", "label": "B"}],
    })

    def score(path, choices, rows, **_):
        assert choices == ["A", "B"]
        assert [row["id"] for row in rows] == ["practice-t", "practice-s"]
        value = float(safetensors.load_file(str(path / "model.safetensors"))["weight"][0])
        return ["A" if value else "B", "A" if value == 0.125 else "B"]

    monkeypatch.setattr(hf_edit, "_score_checkpoint", score)
    monkeypatch.setattr(hf_session, "_score_checkpoint", score)
    return base, policy, practice


def make_patch(base, *, tensor="weight", value=0.25):
    checkpoint = load_checkpoint(base)
    return {
        "schema": "weightbridge.patch.v1",
        "base_sha256": checkpoint.checkpoint_sha256(),
        "tensor_sha256": {tensor: checkpoint.tensor_sha256(tensor)},
        "reason": "Test a bounded scalar edit.",
        "changes": [{"tensor": tensor, "index": [0], "op": "add", "value": value}],
    }


def test_session_charges_rejected_proposal_and_keeps_original_host_inputs(tmp_path, host_inputs):
    base, policy, practice = host_inputs
    session = tmp_path / "session"
    base_hash = load_checkpoint(base).checkpoint_sha256()
    hf_session.init_hf_session(session, base, policy, practice, max_candidates=2)

    changed_policy = json.loads(policy.read_text(encoding="utf-8"))
    changed_policy["allow_tensors"].append("forbidden")
    write_json(policy, changed_policy)
    changed_practice = json.loads(practice.read_text(encoding="utf-8"))
    changed_practice["target"][0]["id"] = "outside-session"
    write_json(practice, changed_practice)

    rejected = hf_session.try_hf_patch(session, make_patch(base, tensor="forbidden"))
    assert rejected["disposition"] == "REJECTED"
    assert rejected["remaining_candidates"] == 1
    assert "candidate_path" not in rejected

    accepted = hf_session.try_hf_patch(session, make_patch(base))
    assert accepted["disposition"] == "PASS"
    assert accepted["remaining_candidates"] == 0
    assert accepted["candidate_id"] == "candidate-0002"
    assert Path(accepted["candidate_path"]).is_dir()
    assert load_checkpoint(base).checkpoint_sha256() == base_hash

    status = call_hf_session_tool(session, {"tool": "describe", "arguments": {}})
    assert status["used_candidates"] == 2
    assert status["remaining_candidates"] == 0
    with pytest.raises(ValueError):
        hf_session.try_hf_patch(session, make_patch(base))


def test_session_can_finalize_only_a_practice_pass(tmp_path, host_inputs):
    base, policy, practice = host_inputs
    session = tmp_path / "session"
    hf_session.init_hf_session(session, base, policy, practice, max_candidates=3)

    failed = hf_session.try_hf_patch(session, make_patch(base, value=0.125))
    assert failed["disposition"] == "FAIL"
    report = json.loads(Path(failed["practice_report_path"]).read_text(encoding="utf-8"))
    assert report["target_gain"] == 1
    assert report["sentinel_prediction_flips"] == 1
    with pytest.raises(ValueError):
        hf_session.finalize_hf_candidate(session, failed["candidate_id"])

    passed = hf_session.try_hf_patch(session, make_patch(base))
    assert passed["disposition"] == "PASS"
    with pytest.raises(ValueError):
        hf_session.finalize_hf_candidate(session, "../" + passed["candidate_id"])

    hf_session.finalize_hf_candidate(session, passed["candidate_id"])
    status = hf_session.describe_hf_session(session)
    assert passed["candidate_id"] in json.dumps(status["decision"])
    assert status["remaining_candidates"] == 1
    with pytest.raises(ValueError):
        hf_session.try_hf_patch(session, make_patch(base))
    with pytest.raises(ValueError):
        hf_session.stop_hf_session(session, "A candidate was already finalized.")


def test_session_rejects_precision_noop(tmp_path, host_inputs):
    base, policy, practice = host_inputs
    session = tmp_path / "session"
    hf_session.init_hf_session(session, base, policy, practice, max_candidates=1)

    result = hf_session.try_hf_patch(session, make_patch(base, value=0.0))

    assert result["disposition"] == "REJECTED"
    assert result["changed_entries"] == 0
    assert result["remaining_candidates"] == 0
    assert "stored precision" in result["reason"]


def test_session_rejects_holdout_fields_and_checkpoint_nested_storage(tmp_path, host_inputs):
    base, policy, practice = host_inputs
    with pytest.raises(ValueError):
        hf_session.init_hf_session(base / "session", base, policy, practice)

    supplied = json.loads(practice.read_text(encoding="utf-8"))
    supplied["holdout"] = [{"id": "secret", "text": "Hidden test question", "label": "A"}]
    write_json(practice, supplied)
    with pytest.raises(ValueError):
        hf_session.init_hf_session(tmp_path / "session", base, policy, practice)


def test_session_rechecks_finalist_bytes_before_finalization(tmp_path, host_inputs):
    base, policy, practice = host_inputs
    session = tmp_path / "session"
    hf_session.init_hf_session(session, base, policy, practice)
    result = hf_session.try_hf_patch(session, make_patch(base))
    assert result["disposition"] == "PASS"

    candidate = Path(result["candidate_path"])
    (candidate / "config.json").write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(ValueError):
        hf_session.finalize_hf_candidate(session, result["candidate_id"])
    assert hf_session.describe_hf_session(session)["decision"] is None


def test_session_rejects_base_checkpoint_drift(tmp_path, host_inputs):
    base, policy, practice = host_inputs
    session = tmp_path / "session"
    hf_session.init_hf_session(session, base, policy, practice)
    (base / "config.json").write_text('{"tampered": true}', encoding="utf-8")

    with pytest.raises(ValueError, match="Base checkpoint changed"):
        hf_session.try_hf_patch(session, make_patch(base))


def test_interrupted_proposal_keeps_its_reserved_budget_slot(tmp_path, host_inputs, monkeypatch):
    base, policy, practice = host_inputs
    session = tmp_path / "session"
    hf_session.init_hf_session(session, base, policy, practice, max_candidates=2)
    patch = make_patch(base)

    def interrupt(*args, **kwargs):
        raise SystemExit(1)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(hf_session, "apply_safetensors_patch", interrupt)
        with pytest.raises(SystemExit):
            hf_session.try_hf_patch(session, patch)

    status = hf_session.describe_hf_session(session)
    assert status["used_candidates"] == 1
    assert status["remaining_candidates"] == 1
    assert status["attempts"] == [{"candidate_id": "candidate-0001", "disposition": "INCOMPLETE"}]
    with pytest.raises(ValueError, match="no completed practice evaluation"):
        hf_session.finalize_hf_candidate(session, "candidate-0001")
    result = hf_session.try_hf_patch(session, patch)
    assert result["candidate_id"] == "candidate-0002"
    assert result["remaining_candidates"] == 0


def test_stop_closes_session_without_selecting_an_eligible_candidate(tmp_path, host_inputs):
    base, policy, practice = host_inputs
    session = tmp_path / "session"
    hf_session.init_hf_session(session, base, policy, practice, max_candidates=2)
    result = hf_session.try_hf_patch(session, make_patch(base))
    assert result["disposition"] == "PASS"

    decision = call_hf_session_tool(session, {"tool": "stop", "arguments": {"reason": "Finish this trial."}})
    assert decision["disposition"] == "STOPPED"
    assert "candidate_id" not in decision
    assert hf_session.describe_hf_session(session)["decision"] == decision
    with pytest.raises(ValueError, match="already has a decision"):
        hf_session.finalize_hf_candidate(session, result["candidate_id"])
    with pytest.raises(ValueError, match="already has a decision"):
        hf_session.try_hf_patch(session, make_patch(base))


def test_session_rejects_changed_frozen_practice(tmp_path, host_inputs):
    base, policy, practice = host_inputs
    session = tmp_path / "session"
    hf_session.init_hf_session(session, base, policy, practice)
    frozen = session / "practice.json"
    supplied = json.loads(frozen.read_text(encoding="utf-8"))
    supplied["target"][0]["label"] = "B"
    write_json(frozen, supplied)

    with pytest.raises(ValueError, match="Frozen practice changed"):
        hf_session.describe_hf_session(session)
    with pytest.raises(ValueError, match="Frozen practice changed"):
        hf_session.try_hf_patch(session, make_patch(base))


def test_tool_interface_excludes_host_paths_and_rejects_extra_arguments(tmp_path):
    session = tmp_path / "host-session"
    forbidden = {"session", "base", "policy", "practice", "holdout", "candidate_path"}
    schemas = {tool["name"]: tool["inputSchema"] for tool in hf_session_tools()}
    for schema in schemas.values():
        assert schema["additionalProperties"] is False
        assert forbidden.isdisjoint(schema["properties"])

    with pytest.raises(ValueError, match="Unknown HF edit session tool"):
        call_hf_session_tool(session, {"tool": "evaluate_holdout", "arguments": {}})
    with pytest.raises(ValueError, match="exactly tool and arguments"):
        call_hf_session_tool(session, {"tool": "describe", "arguments": {}, "session": "alternate"})
    for argument in ("session", "policy", "holdout"):
        with pytest.raises(ValueError, match="try_patch expects arguments"):
            call_hf_session_tool(session, {"tool": "try_patch", "arguments": {"patch": {}, argument: "alternate"}})


def test_tool_rejects_unauthorized_direction_before_compilation(tmp_path, host_inputs, monkeypatch):
    base, policy, practice = host_inputs
    session = tmp_path / "session"
    hf_session.init_hf_session(session, base, policy, practice, max_candidates=1)

    def unexpected_compilation(*args, **kwargs):
        pytest.fail("Out-of-scope direction reached the compiler")

    monkeypatch.setattr(hf_direction, "propose_direction", unexpected_compilation)
    request = {"tool": "try_direction", "arguments": {"tensor": "weight", "strength": 0.25}}
    result = call_hf_session_tool(session, request)
    assert result["disposition"] == "REJECTED"
    assert "direction allowlist" in result["reason"]
    assert result["remaining_candidates"] == 0
    with pytest.raises(ValueError, match="candidate budget is exhausted"):
        call_hf_session_tool(session, request)
