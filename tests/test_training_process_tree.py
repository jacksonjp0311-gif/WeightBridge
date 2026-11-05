"""Host-only accounting regressions. No pretrained model is loaded."""
from pathlib import Path
import os
import subprocess
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
    assert (out.parent/f"{out.name}-process.json").is_file()


def test_real_child_receives_required_smoke_admission_args_and_retry_evidence_stays_separate(bound_inputs, monkeypatch):
    base, protocol, authorization, bundle, old_out, gate = bound_inputs
    out = old_out.parent/"run-0-retry1"
    old_out.parent.mkdir()
    (old_out.parent/"run-0.log").write_bytes(b"Preserved failed first-attempt log")
    (old_out.parent/"run-0-process.json").write_bytes(b"Preserved failed first-attempt receipt")
    previous = {path.name: path.read_bytes() for path in old_out.parent.iterdir()}
    received = []
    def launch(argv, log, **_):
        # A real isolated child parses the exact arguments intended for the
        # frozen trainer and verifies the transported admission identity. No
        # model is loaded. Missing --smoke-gate fails argparse before this check.
        checker = '''import argparse,json,hashlib,pathlib
p=argparse.ArgumentParser()
for name in ('base','protocol','authorization','bundle','out','run-index','smoke-gate'):
 p.add_argument('--'+name,required=True)
a=p.parse_args()
assert a.run_index=='0'
assert json.loads(pathlib.Path(a.smoke_gate).read_bytes())['protocol_sha256']==hashlib.sha256(pathlib.Path(a.protocol).read_bytes()).hexdigest()
'''
        subprocess.run([sys.executable, "-c", checker, *argv[3:]], check=True)
        received.append(argv)
        out.mkdir()
        Path(log).write_bytes(b"New retry fixture log")
        completed = {"status": "COMPLETED", "smoke": False, "parent_sha256": wrapper.PARENT,
            "protocol_sha256": file_hash(protocol), "authorization_sha256": file_hash(authorization),
            "bundle_sha256": file_hash(bundle), "run_index": 0,
            "resources": {"cpu_seconds": .5, "peak_rss_bytes": 1000}}
        json_dump_atomic(out/"completed.json", completed)
        return {"exit_code": 0, "budget_failure": None, "resources": {"observed_tree_cpu_seconds": 1.,
            "cpu_seconds": 1., "wall_seconds": 1., "peak_rss_bytes": 2000,
            "cpu_accounting_complete_for_observed_processes": True}}
    monkeypatch.setattr(wrapper, "launch_owned_tree", launch)
    result = wrapper.run(base, protocol, authorization, bundle, out, run_index=0, smoke_gate=gate)
    assert result["passed"] and result["attempt_directory"] == "run-0-retry1"
    assert len(received) == 1 and "--smoke-gate" in received[0]
    assert (out.parent/"run-0-retry1.log").is_file()
    assert (out.parent/"run-0-retry1-process.json").is_file()
    assert all((out.parent/name).read_bytes() == raw for name, raw in previous.items())


@pytest.mark.skipif(os.name != "nt", reason="Windows execution state API")
def test_temporary_request_is_cleared_when_the_worker_raises(monkeypatch):
    requested = []
    monkeypatch.setattr(wrapper, "_set_thread_execution_state", lambda flags: requested.append(flags) or 1)
    with pytest.raises(RuntimeError, match="worker failed"):
        with wrapper._temporary_sleep_request() as state:
            assert state["succeeded"] is True and state["released"] is None
            raise RuntimeError("worker failed")
    assert requested == [wrapper._ES_CONTINUOUS | wrapper._ES_SYSTEM_REQUIRED, wrapper._ES_CONTINUOUS]
    assert state["released"] is True


@pytest.mark.skipif(os.name != "nt", reason="Windows execution state API")
def test_declined_sleep_request_refuses_before_model_launch_and_retains_receipt(bound_inputs, monkeypatch):
    base, protocol, authorization, bundle, out, gate = bound_inputs
    monkeypatch.setattr(wrapper, "_set_thread_execution_state", lambda _: 0)
    monkeypatch.setattr(wrapper, "launch_owned_tree", lambda *args, **kwargs: pytest.fail("Worker launched after refusal"))
    result = wrapper.run(base, protocol, authorization, bundle, out, run_index=0, smoke_gate=gate)
    assert result["status"] == "HOLD"
    assert result["execution"]["observed_process_count"] == 0
    assert result["temporary_sleep_request"]["requested"] is True
    assert result["temporary_sleep_request"]["succeeded"] is False
    assert result["temporary_sleep_request"]["released"] is None
    assert (out.parent/f"{out.name}-process.json").is_file()


@pytest.mark.skipif(os.name != "nt", reason="Windows execution state API")
def test_real_windows_api_accepts_and_releases_short_system_request():
    with wrapper._temporary_sleep_request() as state:
        assert state["requested"] is True
        assert state["succeeded"] is True
    assert state["released"] is True
