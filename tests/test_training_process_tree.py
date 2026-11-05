"""Host-only accounting regressions. No pretrained model is loaded."""
from pathlib import Path
import os
import sys

import pytest

from experiments.phase_iiiq_baseline import run_training as wrapper
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.adaptation.training_policy import TrainingAuthorization


def test_virtualenv_redirector_descendant_cpu_and_memory_are_included(tmp_path):
    # On Windows sys.executable may itself spawn the real interpreter. This
    # deliberate further child reproduces the launcher-accounting trap.
    child = "import time; payload=bytearray(24*1024**2); end=time.perf_counter()+.35\nwhile time.perf_counter()<end: sum(i*i for i in range(2000))\ntime.sleep(.25)"
    parent = "import subprocess,sys; subprocess.run([sys.executable,'-c',"+repr(child)+"],check=True)"
    result = wrapper.launch_owned_tree([sys.executable, "-c", parent], tmp_path/"tree.log",
        wall_limit=10., cpu_limit=5., memory_limit=512*1024**2)
    assert result["exit_code"] == 0 and result["budget_failure"] is None
    assert result["observed_process_count"] >= 2
    children = [row for row in result["processes"] if not row["is_root_launcher"]]
    assert sum(row["cpu_seconds"] for row in children) >= .2
    assert max(row["peak_rss_bytes"] for row in children) > 24*1024**2
    assert result["resources"]["observed_tree_cpu_seconds"] == pytest.approx(sum(row["cpu_seconds"] for row in result["processes"]))
    assert result["resources"]["peak_rss_bytes"] == sum(row["peak_rss_bytes"] for row in result["processes"])
    if os.name == "nt":
        assert result["resources"]["cpu_accounting_complete_for_observed_processes"]
        assert all(row["cpu_counter_complete"] for row in result["processes"])


def test_budget_refusal_terminates_only_the_owned_new_tree_and_retains_cost(tmp_path):
    child = "import time\nwhile True: sum(i*i for i in range(2000))"
    parent = "import subprocess,sys; subprocess.run([sys.executable,'-c',"+repr(child)+"] )"
    result = wrapper.launch_owned_tree([sys.executable, "-c", parent], tmp_path/"bounded.log",
        wall_limit=5., cpu_limit=.3, memory_limit=512*1024**2)
    assert result["budget_failure"] == "whole_tree_cpu_budget"
    assert result["resources"]["observed_tree_cpu_seconds"] >= .3
    assert result["resources"]["wall_seconds"] < 5.


@pytest.fixture
def bound_inputs(tmp_path, monkeypatch):
    base = tmp_path/"base"; base.mkdir()
    protocol, authorization, bundle, gate = (tmp_path/name for name in ("protocol.json", "authorization.json", "bundle.json", "gate.json"))
    json_dump_atomic(protocol, {"checkpoint_sha256": wrapper.PARENT, "source_sha256": {"fixture": "a"*64},
        "resources": {"max_wall_seconds_per_run": 20., "max_cpu_seconds_per_run": 20., "max_memory_bytes": 100_000_000}})
    json_dump_atomic(bundle, {"protocol_sha256": file_hash(protocol), "split_ids": ["train", "replay"]})
    json_dump_atomic(gate, {"passed": True, "protocol_sha256": file_hash(protocol)})
    auth = TrainingAuthorization(schema="weightbridge.supervised_training_authorization.v1", parent_sha256=wrapper.PARENT,
        allowed_modules=("model.language_model.layers.0.mlp.down_proj",), max_trainable_parameters=1000,
        rank=8, alpha=16., dropout=0., learning_rates=(.0001,), optimizer="AdamW", max_runs=1,
        max_steps_per_run=128, max_answer_tokens_per_run=1000, max_input_tokens_per_run=1000,
        max_memory_bytes=100_000_000, max_wall_seconds_per_run=20., max_cpu_seconds_per_run=20.,
        max_development_checkpoints_per_run=2, allow_merge=True, training_bundle_sha256=file_hash(bundle),
        smoke_gate_sha256=file_hash(gate))
    json_dump_atomic(authorization, auth.to_dict())
    monkeypatch.setattr(wrapper, "verify_sources", lambda _: None)
    return base, protocol, authorization, bundle, tmp_path/"runs/run0", gate


def test_wrapper_refuses_output_inside_original_before_writing(bound_inputs):
    base, protocol, authorization, bundle, _, gate = bound_inputs
    with pytest.raises(ValueError, match="outside original"):
        wrapper.run(base, protocol, authorization, bundle, base/"illegal", run_index=0, smoke_gate=gate)
    assert not (base/"illegal").exists()


def test_wrapper_requires_exact_admitted_smoke_before_launch(bound_inputs, monkeypatch):
    base, protocol, authorization, bundle, out, gate = bound_inputs
    gate.write_text('{"passed":false}', encoding="utf-8")
    monkeypatch.setattr(wrapper, "launch_owned_tree", lambda *args, **kwargs: pytest.fail("Unauthorized worker launched"))
    with pytest.raises(ValueError, match="exact admitted smoke"):
        wrapper.run(base, protocol, authorization, bundle, out, run_index=0, smoke_gate=gate)
    assert not out.exists()


def test_body_resources_exceeding_capture_refuse_complete_accounting(bound_inputs, monkeypatch):
    base, protocol, authorization, bundle, out, gate = bound_inputs
    def launch(argv, log, **_):
        out.mkdir(parents=True)
        Path(log).write_bytes(b"synthetic process fixture")
        completed = {"status": "COMPLETED", "smoke": False, "parent_sha256": wrapper.PARENT,
            "protocol_sha256": file_hash(protocol), "authorization_sha256": file_hash(authorization),
            "bundle_sha256": file_hash(bundle), "run_index": 0,
            "resources": {"cpu_seconds": 5., "peak_rss_bytes": 1000}}
        json_dump_atomic(out/"completed.json", completed)
        return {"exit_code": 0, "budget_failure": None, "resources": {"observed_tree_cpu_seconds": 1.,
            "cpu_seconds": 1., "wall_seconds": 1., "peak_rss_bytes": 10,
            "cpu_accounting_complete_for_observed_processes": True}}
    monkeypatch.setattr(wrapper, "launch_owned_tree", launch)
    result = wrapper.run(base, protocol, authorization, bundle, out, run_index=0, smoke_gate=gate)
    assert not result["passed"] and result["status"] == "HOLD"
    assert not result["checks"]["worker_cpu_covered"] and not result["checks"]["worker_peak_covered"]
    assert result["execution"]["resources"]["known_peak_rss_floor_bytes"] == 1000
    assert (out.parent/"run-0-process.json").is_file()
