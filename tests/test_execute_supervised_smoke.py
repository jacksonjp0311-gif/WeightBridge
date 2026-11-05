"""Disposable fake workers test orchestration and budget gates without models."""
from pathlib import Path
import sys
import time

import pytest

from experiments.phase_iiiq_baseline import execute_smoke as execution
from experiments.phase_iiiq_baseline.data import generate, PopulationSpec
from experiments.phase_iiiq_baseline.smoke_checks import _inputs
from weightbridge.core import file_hash, json_dump_atomic


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    base = tmp_path/"base"
    base.mkdir()
    rows = generate("smoke", PopulationSpec(target_families=4), "synthetic-orchestration-fixture", set(), set())
    protocol, roster, bundle, auth = (tmp_path/name for name in ("protocol.json", "rows.json", "bundle.json", "authorization.json"))
    json_dump_atomic(protocol, {"source_sha256": {"fixture": "a"*64}, "checkpoint_sha256": execution.PARENT,
        "gates": {"smoke": {}}, "resources": {"smoke_max_wall_seconds": 20, "smoke_max_cpu_seconds": 20,
                                            "max_memory_bytes": 100_000_000}})
    json_dump_atomic(roster, rows)
    json_dump_atomic(bundle, {"rows": rows, "split_ids": ["smoke"], "protocol_sha256": file_hash(protocol)})
    json_dump_atomic(auth, {})
    monkeypatch.setattr(execution, "verify_sources", lambda _: None)
    monkeypatch.setattr(execution, "exact_predictions", lambda first, second: first["identity"] == second["identity"])
    monkeypatch.setattr(execution, "validate_smoke", lambda *args, **kwargs: {"passed": True, "checks": {"quality": True}})
    return base, protocol, auth, bundle, roster, tmp_path/"run"


def fake_worker(argv, name, root, budget):
    record = {"worker": name, "exit_code": 0, "cpu_seconds": 1., "peak_rss_bytes": 1000,
              "cpu_measurement": "complete_windows_process_times"}
    budget["completed_cpu"] += 1.
    budget["workers"].append(record)
    root = Path(root)
    if name == "zero":
        json_dump_atomic(root/"zero/zero-init-check.json", {"zero_init_parent_identical": True,
            "parent_measurement": {"identity": "parent"}, "zero_init_measurement": {"identity": "parent"},
            "matched_loss_before": {"fixture_loss": 1.}})
    elif name == "training":
        directory = root/"training/checkpoints/step-0032"
        directory.mkdir(parents=True)
        (directory/"adapter.safetensors").write_bytes(b"synthetic artifact; no model weights")
        json_dump_atomic(root/"training/completed.json", {"status": "COMPLETED", "checkpoints": [str(directory)]})
    else:
        json_dump_atomic(root/name/"fresh-adapter-check.json", {"trained_measurement": {"identity": "trained"},
            "disabled_measurement": {"identity": "parent"}, "matched_loss_after": {"fixture_loss": .1}})
    return record


def test_sequential_smoke_checks_use_four_fresh_workers_and_pass_only_after_all(inputs, monkeypatch):
    seen = []
    def worker(*args):
        seen.append(args[1])
        return fake_worker(*args)
    monkeypatch.setattr(execution, "_worker", worker)
    gate = execution.run(*inputs)
    assert seen == ["zero", "training", "fresh-1", "fresh-2"]
    assert gate["passed"] and gate["checks"]["second_disabled_parent_identical"]
    assert gate["resources"]["worker_cpu_seconds"] == 4.
    assert gate["resources"]["cpu_seconds"] < 5.
    assert gate["resources"]["cpu_accounting_complete"]
    assert not gate["lineage_authority"] and gate["no_main_training_authority"]


def test_worker_failure_records_rejection_and_never_starts_later_workers(inputs, monkeypatch):
    seen = []
    def worker(*args):
        seen.append(args[1])
        if args[1] == "training":
            raise RuntimeError("Synthetic bounded failure")
        return fake_worker(*args)
    monkeypatch.setattr(execution, "_worker", worker)
    result = execution.run(*inputs)
    assert seen == ["zero", "training"] and not result["passed"]
    assert result["error"]["message"] == "Synthetic bounded failure"
    assert (inputs[-1]/"smoke-gate.json").is_file()


def test_output_inside_original_is_refused_before_any_write(inputs):
    base, protocol, auth, bundle, rows, _ = inputs
    with pytest.raises(ValueError, match="original checkpoint"):
        execution.run(base, protocol, auth, bundle, rows, base/"illegal")
    assert not (base/"illegal").exists()


def test_smoke_helper_also_structurally_protects_original_directory(inputs, monkeypatch):
    base, protocol, _, _, rows, _ = inputs
    from experiments.phase_iiiq_baseline import smoke_checks
    monkeypatch.setattr(smoke_checks, "verify_sources", lambda _: None)
    with pytest.raises(ValueError, match="original checkpoint"):
        _inputs(protocol, rows, base/"illegal", base=base)
    assert not (base/"illegal").exists()


def test_real_tiny_child_records_complete_process_cost_without_model_runtime(tmp_path):
    budget = {"wall_start": time.perf_counter(), "host_cpu_start": time.process_time(),
        "completed_cpu": 0., "workers": [], "max_wall": 10., "max_cpu": 5., "max_memory": 100_000_000}
    result = execution._worker([sys.executable, "-c", "print('synthetic smoke worker')"], "tiny", tmp_path, budget)
    assert result["exit_code"] == 0 and result["cpu_seconds"] >= 0
    assert len(budget["workers"]) == 1
    assert (tmp_path/"tiny-process.json").is_file()
    assert result["cpu_measurement"] in {"complete_windows_process_times", "sampled_process_lower_bound"}
