"""Host initialization refuses mutable identities before any model execution."""
import json

import pytest

from experiments.phase_iiiq_baseline import initialize as host
from weightbridge.core import file_hash


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def config():
    return {"checkpoint_sha256": host.PARENT, "source_sha256": {},
            "adapter": {"rank": 8, "alpha": 16, "dropout": 0},
            "training": {"learning_rates": [0.0001, 0.0003], "optimizer_steps": 128},
            "smoke_training": {"learning_rates": [0.001], "optimizer_steps": 32}}


def test_smoke_and_main_authorizations_bind_distinct_bundles_and_gate(tmp_path):
    bundle = write(tmp_path / "bundle.json", {"rows": [1]})
    gate = write(tmp_path / "gate.json", {"passed": True})
    rows = [{"name": "model.language_model.layers.0.mlp.up_proj"}]
    smoke = host._authorization(config(), rows, bundle, smoke=True)
    main = host._authorization(config(), rows, bundle, smoke=False, smoke_gate=gate)
    assert smoke.training_bundle_sha256 == main.training_bundle_sha256 == file_hash(bundle)
    assert smoke.max_runs == 1 and smoke.max_steps_per_run == 32 and not smoke.allow_merge
    assert main.max_runs == 2 and main.max_steps_per_run == 128 and main.allow_merge
    assert main.smoke_gate_sha256 == file_hash(gate)
    assert main.max_answer_tokens_per_run == 40_000 and main.max_cpu_seconds_per_run == 30_000
    assert not main.production_deployment


def test_private_paths_refuse_checkout_and_immutable_base(tmp_path):
    with pytest.raises(ValueError):
        host._private(host.ROOT / "results/new")
    with pytest.raises(ValueError):
        host._private(tmp_path / "base/outputs", tmp_path / "base")


def test_populated_source_manifest_is_verified_and_never_rewritten(tmp_path, monkeypatch):
    path = write(tmp_path / "protocol.json", {**config(), "source_sha256": {"a.py": "f"*64}})
    original = path.read_bytes()
    checked = []
    monkeypatch.setattr(host, "verify_sources", lambda value: checked.append(value))
    assert host._freeze_sources(path)["source_sha256"] == {"a.py": "f"*64}
    assert checked == [{"a.py": "f"*64}] and path.read_bytes() == original


def test_empty_source_manifest_fills_once_with_lf(tmp_path, monkeypatch):
    path = write(tmp_path / "protocol.json", config())
    monkeypatch.setattr(host, "source_manifest", lambda: {"initialize.py": "a"*64})
    monkeypatch.setattr(host, "verify_sources", lambda value: None)
    assert host._freeze_sources(path)["source_sha256"] == {"initialize.py": "a"*64}
    before = path.read_bytes()
    assert b"\r\n" not in before
    host._freeze_sources(path)
    assert path.read_bytes() == before


def test_refuses_wrong_inventory_and_changed_training_settings(tmp_path):
    path = write(tmp_path / "inventory.json", {"checkpoint_sha256": host.PARENT,
                                              "allowed_count": 185, "allowed": []})
    with pytest.raises(ValueError):
        host._inventory(path)
    bundle = write(tmp_path / "bundle.json", {})
    changed = config()
    changed["training"]["learning_rates"] = [.01]
    with pytest.raises(ValueError):
        host._authorization(changed, [], bundle, smoke=False)
