import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.phase_iiik import verification
from weightbridge.core import json_dump_atomic


def test_validation_accepts_equivalent_surface_and_rejects_correctness_loss():
    config = json.loads(verification.PROTOCOL.read_bytes())
    contract = verification.endpoint(config)
    before = {"endpoint": contract.manifest(), "rows": [
        {"id": f"item/{wording}", "item": "item", "wording": wording, "group": "target",
         "family": "larger-decimal", "prompt": f"Wording {wording}\nValues: 6 or 5\nAnswer:",
         "operands": ["6", "5"], "answer": "6", "text": "6", "stop_reason": "eos", "generated_token_ids": [6]}
        for wording in contract.required_wordings]}
    changed = {**before, "rows": [dict(row, text="6.0", generated_token_ids=[6, 9, 0]) for row in before["rows"]]}
    gate = verification.preservation_gate(before, changed, config)
    assert gate["passed"]
    assert gate["semantic_matches_with_changed_tokens"] == [row["id"] for row in before["rows"]]
    changed["rows"][0] = {**changed["rows"][0], "text": "5"}
    assert verification.preservation_gate(before, changed, config)["passed"] is False


def test_validation_failure_never_launches_selection_candidate(tmp_path, monkeypatch):
    base = tmp_path / "base"
    material = tmp_path / "material"
    material.mkdir()
    json_dump_atomic(material / "manifest.json", {})
    run = tmp_path / "run"
    candidate = run / "checkpoints" / "E"
    candidate.mkdir(parents=True)
    json_dump_atomic(candidate / ".weightbridge_receipt.json", {"base_sha256": "parent", "candidate_sha256": "edited"})
    monkeypatch.setattr(verification, "_read", lambda path: (
        {"checkpoint_sha256": "parent"} if path == verification.PROTOCOL else json.loads(Path(path).read_bytes())))
    monkeypatch.setattr(verification, "read_manifest", lambda _material: {})
    monkeypatch.setattr(verification, "source_hashes", lambda: {})
    monkeypatch.setattr(verification, "load_checkpoint", lambda _base: SimpleNamespace(checkpoint_sha256=lambda: "parent"))
    calls = []

    def launch(_path, sha, _material, split, out, *_args):
        assert split == verification.VALIDATION_SPLIT
        calls.append(sha)
        record = {"measurement": {"id": sha}, "resources": {}}
        json_dump_atomic(out, record)
        return record

    monkeypatch.setattr(verification, "_launch", launch)
    monkeypatch.setattr(verification, "preservation_gate", lambda *_args: {"passed": False, "prohibited_regressions": ["row"]})
    outcome = verification.assess(base, material, run, {"E": {"checkpoint_sha256": "edited", "realized_update_norm": .01}})
    assert calls == ["parent", "edited"]
    assert outcome["winner"] is None and not outcome["selection"]
    assert outcome["validation"]["E"]["prohibited_regressions"] == ["row"]


def test_hidden_worker_checks_lock_before_any_roster_load(tmp_path, monkeypatch):
    material = tmp_path / "material"
    material.mkdir()
    json_dump_atomic(material / "manifest.json", {})
    monkeypatch.setattr(verification, "read_manifest", lambda _material: {})
    monkeypatch.setattr(verification, "load_split", lambda *_args, **_kwargs: pytest.fail("Unexpected sealed roster access"))
    with pytest.raises(ValueError, match="finalist lock"):
        verification.worker(tmp_path / "base", "unknown", material, "hidden", tmp_path / "out.json",
                            verification.file_hash(material / "manifest.json"))
