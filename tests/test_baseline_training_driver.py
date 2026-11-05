import json
from pathlib import Path

import pytest

from experiments.phase_iiiq_baseline.integrity import PARENT, authorize_stage, lock_finalist
from experiments.phase_iiiq_baseline.train import data_order, learning_rate_factor
from weightbridge.core import EditRejected, json_dump_atomic


def test_epoch_order_is_deterministic_and_each_example_has_equal_exposure():
    first, second = data_order(32, 7, 2), data_order(32, 7, 2)
    assert first == second and sorted(first[:32]) == sorted(first[32:]) == list(range(32))
    assert first[:32] != first[32:]
    assert learning_rate_factor(0, 128, 7) == pytest.approx(1/7)
    assert learning_rate_factor(6, 128, 7) == 1
    assert learning_rate_factor(7, 128, 7) == 1
    assert learning_rate_factor(127, 128, 7) == pytest.approx(1/121)


def test_failed_or_incomplete_stage_cannot_open_independent_data(tmp_path):
    adapter = tmp_path / "adapter"; adapter.mkdir(); (adapter / "factors").write_bytes(b"weights")
    with pytest.raises(ValueError):
        lock_finalist(tmp_path, adapter_path=adapter, protocol_sha256="p", manifest_sha256="m",
                      practice_gate={"passed": True}, development_gate={"passed": False})
    lock_finalist(tmp_path, adapter_path=adapter, protocol_sha256="p", manifest_sha256="m",
                  practice_gate={"passed": True}, development_gate={"passed": True})
    with pytest.raises(FileNotFoundError):
        authorize_stage(tmp_path, "selection", representation_sha256="candidate", roster_sha256="rows")
    for name in ("merged-development", "preservation-validation"):
        json_dump_atomic(tmp_path / f"{name}-gate.json", {"passed": name != "preservation-validation",
                         "representation_sha256": "candidate", "protocol_sha256": "p", "manifest_sha256": "m"})
    with pytest.raises(ValueError):
        authorize_stage(tmp_path, "selection", representation_sha256="candidate", roster_sha256="rows")
    with pytest.raises(EditRejected):
        lock_finalist(tmp_path, adapter_path=adapter, protocol_sha256="p", manifest_sha256="m",
                      practice_gate={"passed": True}, development_gate={"passed": True})


def test_authorization_is_bound_to_stored_representation(tmp_path):
    adapter = tmp_path / "adapter"; adapter.mkdir(); (adapter / "factors").write_bytes(b"weights")
    lock_finalist(tmp_path, adapter_path=adapter, protocol_sha256="p", manifest_sha256="m",
                  practice_gate={"passed": True}, development_gate={"passed": True})
    json_dump_atomic(tmp_path / "merged-development-gate.json", {"passed": True,
        "representation_sha256": "candidate", "protocol_sha256": "p", "manifest_sha256": "m"})
    value = authorize_stage(tmp_path, "preservation-validation", representation_sha256="candidate", roster_sha256="rows")
    assert value["parent_sha256"] == PARENT and not value["production_authority"]
    with pytest.raises(ValueError):
        authorize_stage(tmp_path, "preservation-validation", representation_sha256="different", roster_sha256="rows")


def test_write_once_events_survive_interrupted_attempt_and_refuse_overwrite(tmp_path):
    json_dump_atomic(tmp_path / "started.json", {"status": "STARTED"})
    json_dump_atomic(tmp_path / "steps/0001.json", {"step": 1, "forward_calls": 4})
    # A restart can audit partial work, but cannot relabel it as completed or overwrite it.
    assert not (tmp_path / "completed.json").exists()
    assert json.loads((tmp_path / "steps/0001.json").read_bytes())["forward_calls"] == 4
    with pytest.raises(EditRejected):
        json_dump_atomic(tmp_path / "steps/0001.json", {"step": 1, "forward_calls": 0})
