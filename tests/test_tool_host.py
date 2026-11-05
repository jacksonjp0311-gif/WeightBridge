"""CLI plumbing only; no model or protected bank is executed by these fixtures."""
from pathlib import Path

import pytest

from experiments.phase_iiis_tool_transfer import host
from weightbridge.core import file_hash
from test_tool_study import protocol, write


def initialized(tmp_path):
    p = protocol(tmp_path)
    config = host.read(p)
    config["resources"] = {"max_memory_bytes": 100, "evaluation_limits": {
        "max_wall_seconds": 10, "max_cpu_seconds": 10, "max_memory_bytes": 100}}
    write(p, config)
    paths = {}
    for task in ("tool", "decimal"):
        path = write(tmp_path / "material/development" / f"{task}.json", [])
        paths[task] = {"path": str(path), "sha256": file_hash(path)}
    manifest = write(tmp_path / "manifest.json", {"splits": {"development": {"files": paths}}})
    write(tmp_path / "initialized.json", {"schema": "weightbridge.phase_iiis.initialization.v1",
        "manifest_sha256": file_hash(manifest), "source_sha256": {},
        "arms": {"original-joint": {"protocol_sha256": file_hash(p)}}})
    return p, manifest


def test_parent_cli_refuses_redirected_development_before_model_launch(tmp_path, monkeypatch):
    p, manifest = initialized(tmp_path)
    value = host.read(manifest)
    value["splits"]["development"]["files"]["tool"]["path"] = str(tmp_path / "unopened-bank.json")
    write(manifest, value)
    init = host.read(tmp_path / "initialized.json")
    init["manifest_sha256"] = file_hash(manifest)
    write(tmp_path / "initialized.json", init)
    monkeypatch.setattr(host.execute, "run", lambda *a, **k: pytest.fail("Unexpected model launch"))
    with pytest.raises(ValueError, match="Development roster"):
        host.baselines(tmp_path, tmp_path / "original-model", tmp_path / "r8-model", tmp_path / "ledger")
    failed = host.read(tmp_path / "baselines/failed.json")
    assert not failed["passed"] and not failed["protected_banks_opened"]


def test_parent_cli_cannot_restart_an_incomplete_attempt(tmp_path, monkeypatch):
    initialized(tmp_path)
    (tmp_path / "baselines").mkdir()
    monkeypatch.setattr(host.execute, "run", lambda *a, **k: pytest.fail("Unexpected retry"))
    with pytest.raises(FileExistsError, match="repeated"):
        host.baselines(tmp_path, tmp_path / "original-model", tmp_path / "r8-model", tmp_path / "ledger")


def test_host_request_has_closed_operations_and_no_executable_content(tmp_path, monkeypatch):
    path = write(tmp_path / "request.json", {"schema": "weightbridge.phase_iiis.host_request.v1",
        "operation": "__import__", "arguments": {"out": str(tmp_path / "out.json")}})
    with pytest.raises(ValueError, match="Unknown"):
        host.request(path)
    calls = []
    monkeypatch.setitem(host.REQUESTS, "lock-arms", lambda **kwargs: calls.append(kwargs) or {"passed": True})
    write(path, {"schema": "weightbridge.phase_iiis.host_request.v1", "operation": "lock-arms",
                 "arguments": {"out": str(tmp_path / "out.json"), "locks": {}}})
    assert host.request(path)["passed"]
    assert calls == [{"out": str(tmp_path / "out.json"), "locks": {}}]


def test_host_request_cannot_publish_raw_evidence_into_checkout(tmp_path):
    path = write(tmp_path / "request.json", {"schema": "weightbridge.phase_iiis.host_request.v1",
        "operation": "lock-arms", "arguments": {"out": str(host.ROOT / "public-raw.json"), "locks": {}}})
    with pytest.raises(ValueError, match="public source"):
        host.request(path)
