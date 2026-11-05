"""New R admission boundaries; inherited Q numerical tests remain unchanged."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from experiments.phase_iiir_replication import training
from experiments.phase_iiiq_baseline.integrity import ROOT, read
from weightbridge.core import EditRejected, file_hash


def config(rank=8):
    value = read(ROOT/"experiments/phase_iiiq_baseline/protocol.json")
    value.update(schema="weightbridge.phase_iiir.protocol.v1", replication_arm=f"rank{rank}", source_sha256={})
    value["adapter"].update(rank=rank, alpha=rank*2)
    value["training"]["learning_rates"] = [.0003]
    return value


def inventory():
    return tuple(sorted(read(ROOT/"results/phase-iii-q-baseline/preflight/inventory-v1/architecture-inventory.json")["allowed"],
                        key=lambda row: row["name"]))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize("rank,count", [(8, 5411328), (4, 2705664)])
def test_rank_configuration_measures_exact_real_module_coverage(rank, count, tmp_path):
    cfg, rows = config(rank), inventory()
    bundle = write(tmp_path/"bundle.json", {})
    auth = training.authorization(cfg, rows, bundle)
    assert auth.rank == rank and auth.alpha == rank*2 and auth.max_trainable_parameters == count
    assert len(auth.allowed_modules) == 186 and auth.max_runs == 1
    assert auth.learning_rates == (.0003,) and auth.max_steps_per_run == 128
    assert training.verify_authorization(cfg, auth, smoke=False) == rank
    with pytest.raises(ValueError, match="186-module"):
        training.authorization(cfg, rows[:-1], bundle)
    with pytest.raises(ValueError, match="186-module"):
        training.authorization(cfg, [*rows[:-1], rows[0]], bundle)


@pytest.mark.parametrize("field,value", [("seed", 606118), ("learning_rates", [.0001, .0003]),
    ("optimizer_steps", 64), ("epochs", 2), ("checkpoint_steps", [32,64,128]), ("micro_batch_size", 8)])
def test_recipe_refuses_new_training_search_or_exposure(field, value):
    cfg = config()
    cfg["training"][field] = value
    with pytest.raises(ValueError, match="frozen replication recipe"):
        training.validate_recipe(cfg)


def test_resource_and_authorization_limits_cannot_be_extended(tmp_path):
    cfg = config()
    bundle = write(tmp_path/"bundle.json", {})
    auth = training.authorization(cfg, inventory(), bundle)
    for changed in (replace(auth, max_runs=2), replace(auth, max_cpu_seconds_per_run=30001),
                    replace(auth, max_answer_tokens_per_run=40001), replace(auth, max_trainable_parameters=5411329)):
        with pytest.raises(ValueError):
            training.verify_authorization(cfg, changed, smoke=False)
    cfg["resources"]["max_wall_seconds_per_run"] = 5401
    with pytest.raises(ValueError):
        training.validate_recipe(cfg)


def prepared(tmp_path, *, smoke=False):
    cfg = config()
    cfg["source_sha256"] = {"weightbridge/core.py": file_hash(ROOT/"weightbridge/core.py")}
    protocol = write(tmp_path/"protocol.json", cfg)
    bundle = write(tmp_path/"bundle.json", {"schema": "weightbridge.supervised_training_bundle.v1",
        "protocol_sha256": file_hash(protocol), "rows": [], "manifest_sha256": "a"*64,
        "split_ids": ["smoke"] if smoke else ["train", "replay"]})
    auth = training.authorization(cfg, inventory(), bundle, smoke=smoke)
    authorization = write(tmp_path/("smoke-authorization.json" if smoke else "main-authorization.json"), auth.to_dict())
    return protocol, authorization, bundle


@pytest.mark.parametrize("smoke", [False, True])
def test_launch_claim_remains_spent_after_interruption_or_changed_output(tmp_path, smoke):
    protocol, authorization, bundle = prepared(tmp_path, smoke=smoke)
    claim, receipt = training.claim_launch(protocol, authorization, bundle, tmp_path/"attempt", smoke=smoke)
    assert claim.is_file() and not receipt["retries_authorized"] and not receipt["exact_resume_implemented"]
    # No output was made: even interruption before the worker cannot create a
    # replacement attempt at another name without changing the frozen study.
    with pytest.raises(EditRejected):
        training.claim_launch(protocol, authorization, bundle, tmp_path/"replacement", smoke=smoke)


def test_training_claim_refuses_sealed_material_even_when_hash_authorized(tmp_path):
    protocol, authorization, bundle = prepared(tmp_path)
    material = read(bundle)
    material["split_ids"] = ["train", "replay", "hidden"]
    write(bundle, material)
    auth = read(authorization)
    auth["training_bundle_sha256"] = file_hash(bundle)
    write(authorization, auth)
    with pytest.raises(ValueError, match="authorized fitting"):
        training.claim_launch(protocol, authorization, bundle, tmp_path/"attempt", smoke=False)
    assert not (tmp_path/"main-launch-claim.json").exists()


def test_equal_order_and_exposure_do_not_depend_on_rank_initialization(tmp_path):
    torch = pytest.importorskip("torch")
    from weightbridge.adaptation.supervised_lora import TorchLoRA
    bundle = write(tmp_path/"bundle.json", {"rows": [{"id": f"row-{i}"} for i in range(2048)]})
    before = training.data_exposure_receipt(bundle, config(8))
    for rank in (8, 4):
        generator = torch.Generator(device="cpu").manual_seed(606117)
        layers = [TorchLoRA(9, 17, rank=rank, alpha=rank*2, generator=generator) for _ in range(3)]
        assert sum(p.numel() for layer in layers for p in layer.parameters()) == 3*rank*(9+17)
        assert training.data_exposure_receipt(bundle, config(rank)) == before
    assert before["examples_seen"] == 2048 and before["batches"] == 128
    assert before["cross_rank_initial_factors_identical"] is False


def test_q_initialization_keeps_zero_output_and_both_factor_gradient_flow_for_each_rank():
    torch = pytest.importorskip("torch")
    from weightbridge.adaptation.supervised_lora import TorchLoRA
    inputs = torch.tensor([[1.,2.,-1.,.5],[-1.,.2,.7,2.]])
    for rank in (8, 4):
        layer = TorchLoRA(4, 5, rank=rank, alpha=rank*2,
                         generator=torch.Generator().manual_seed(606117))
        assert torch.count_nonzero(layer(inputs)) == 0
        assert layer.A.dtype == layer.B.dtype == torch.float32
        initial_a = layer.A.detach().clone()
        optimizer = torch.optim.AdamW(layer.parameters(), lr=.001, weight_decay=.01)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            (layer(inputs)-1).square().mean().backward()
            assert torch.isfinite(layer.A.grad).all() and torch.isfinite(layer.B.grad).all()
            assert torch.count_nonzero(layer.B.grad) > 0
            if step:
                assert torch.count_nonzero(layer.A.grad) > 0
            optimizer.step()
        assert not torch.equal(layer.A, initial_a)
        assert torch.count_nonzero(layer.B) > 0


def test_smoke_rank_authorization_does_not_enable_main_training(tmp_path):
    cfg = config(4)
    auth = training.authorization(cfg, inventory(), write(tmp_path/"smoke.json", {}), smoke=True)
    assert not auth.allow_merge and auth.max_steps_per_run == 32
    assert auth.max_runs == 1 and auth.smoke_gate_sha256 is None
    with pytest.raises(ValueError):
        training.verify_authorization(cfg, auth, smoke=False)


def test_rank4_gate_precedes_launch_claim_and_rejects_missing_replication(tmp_path):
    path = write(tmp_path/"protocol.json", config(4))
    with pytest.raises(ValueError, match="independently verified"):
        training.run(tmp_path/"base", path, tmp_path/"auth", tmp_path/"bundle", tmp_path/"out",
                     smoke_gate=tmp_path/"smoke")
    assert not (tmp_path/"main-launch-claim.json").exists()


def test_rank4_replication_admission_rechecks_protocol_and_failed_exact_replay(tmp_path, monkeypatch):
    from experiments.phase_iiir_replication.study import seal_replication, validate_replication_gate
    from experiments.phase_iiir_replication import execute
    monkeypatch.setattr(execute, "validate_stage_execution", lambda *args, **kwargs: {
        "outcome": {"disposition": "VERIFIED_REPLICATION_PASS"}})
    config8 = config(8)
    config8.update(master_protocol_sha256="a"*64, global_data_manifest_sha256="b"*64)
    protocol8 = write(tmp_path/"rank8-protocol.json", config8)
    verified = read(ROOT/"results/phase-iii-q-baseline/baseline-v1/final-verification/final-verification.json")
    verified["protocol_sha256"] = file_hash(protocol8)
    final = write(tmp_path/"final.json", verified)
    gate_path = tmp_path/"replication.json"
    seal_replication(protocol8, final, gate_path)
    config4 = {**config(4), "master_protocol_sha256": "a"*64, "global_data_manifest_sha256": "b"*64}
    assert validate_replication_gate(config4, gate_path)["passed"]
    altered = {**config4, "master_protocol_sha256": "c"*64}
    with pytest.raises(ValueError, match="fully verified"):
        validate_replication_gate(altered, gate_path)
    verified["replays"]["hidden"]["reconstructed"]["exact_native_generation"] = False
    write(final, verified)
    with pytest.raises(ValueError, match="identity changed"):
        validate_replication_gate(config4, gate_path)
    # Updating the wrapper digest cannot turn a recorded replay failure into
    # passing independent evidence.
    wrapper = read(gate_path)
    wrapper["verification_sha256"] = file_hash(final)
    write(gate_path, wrapper)
    with pytest.raises(ValueError, match="exactly replay"):
        validate_replication_gate(config4, gate_path)


@pytest.fixture
def main_records(tmp_path, monkeypatch):
    from experiments.phase_iiir_replication import smoke
    monkeypatch.setattr(smoke, "validate_gate", lambda *args: {"passed": True})
    cfg = config()
    cfg["source_sha256"] = {"weightbridge/core.py": file_hash(ROOT/"weightbridge/core.py")}
    protocol = write(tmp_path/"protocol.json", cfg)
    bundle = write(tmp_path/"training-bundle.json", {"schema": "weightbridge.supervised_training_bundle.v1",
        "protocol_sha256": file_hash(protocol), "manifest_sha256": "a"*64, "split_ids": ["train", "replay"],
        "rows": [{"id": f"fixture-{i}"} for i in range(2048)]})
    smoke_gate = write(tmp_path/"smoke-gate.json", {"fixture": True})
    auth = training.authorization(cfg, inventory(), bundle, smoke_gate=smoke_gate)
    auth_path = write(tmp_path/"main-authorization.json", auth.to_dict())
    out = tmp_path/"run"
    claim, _ = training.claim_launch(protocol, auth_path, bundle, out, smoke=False)
    out.mkdir()
    exposure = training.data_exposure_receipt(bundle, cfg)
    for name in ("prospective-data-order.json", "main-data-order.json"):
        write(tmp_path/name, exposure)
    identity = {"protocol_sha256": file_hash(protocol), "authorization_sha256": file_hash(auth_path),
        "bundle_sha256": file_hash(bundle), "parent_sha256": training.PARENT, "run_index": 0}
    checkpoints = [out/"checkpoints"/f"step-{step:04d}" for step in (64,128)]
    for step, path in zip((64,128), checkpoints):
        path.mkdir(parents=True)
        (path/"adapter.safetensors").write_bytes(b"Synthetic evidence fixture, not model weights")
        write(path/"receipt.json", {"sha256": file_hash(path/"adapter.safetensors"), "step": step,
            "trainable_parameters": 5411328, "host_metadata": {**identity, "step": step, "smoke": False},
            "configuration": {"rank": 8, "alpha": 16, "seed": 606117, "checkpoint_sha256": training.PARENT,
                "factor_dtype": "F32", "base_dtype": "BF16", "modules": [{"module_name": n} for n in auth.allowed_modules]}})
    completed = {**identity, "status": "COMPLETED", "smoke": False, "manifest_sha256": "a"*64,
        "counts": {"optimizer_steps": 128, "examples_seen": 2048, "forward_calls": 512, "backward_calls": 512,
                   "answer_tokens": 20000, "input_tokens": 200000},
        "all_factors_changed": True, "all_factors_received_nonzero_gradient": True, "base_unchanged": True,
        "parameter_changes": [{"name": f"factor-{i}", "changed_entries": 1, "nonzero_gradient_steps": 127} for i in range(372)],
        "checkpoints": [str(path) for path in checkpoints],
        "resources": {"wall_seconds": 1., "cpu_seconds": .1, "peak_rss_bytes": 1000}}
    write(out/"completed.json", completed)
    inv = {"trainable_parameters": 5411328, "module_count": 186, "data_order_sha256": exposure["data_order_sha256"]}
    write(out/"inventory.json", inv)
    (tmp_path/"run.log").write_text("Synthetic test execution", encoding="utf-8")
    resources = {"cpu_seconds": .22, "host_cpu_seconds": .02, "observed_tree_cpu_seconds": .2,
        "wall_seconds": 1.1, "peak_rss_bytes": 2000, "cpu_accounting_complete_for_observed_processes": True}
    process = {**identity, "passed": True, "checks": {"fixture_complete": True},
        "training_completed_receipt_sha256": file_hash(out/"completed.json"), "log_sha256": file_hash(tmp_path/"run.log"),
        "smoke_gate_sha256": file_hash(smoke_gate), "limits": {"wall_seconds": 5400., "cpu_seconds": 30000., "memory_bytes": 24*1024**3},
        "temporary_sleep_request": {"requested": True, "succeeded": True, "released": True},
        "execution": {"exit_code": 0, "budget_failure": None, "resources": resources,
                      "processes": [{"cpu_seconds": .2, "peak_rss_bytes": 2000}]}}
    process_path = write(tmp_path/"run-process.json", process)
    receipt = {"schema": "weightbridge.phase_iiir.main_execution.v1", "rank": 8, "passed": True, "status": "COMPLETED",
        "protocol_sha256": file_hash(protocol), "launch_claim_sha256": file_hash(claim),
        "process_receipt_sha256": file_hash(process_path), "smoke_gate_path": str(smoke_gate),
        "data_order": exposure, "resources": resources,
        "checks": training._main_checks(process, completed, inv, exposure, 8)}
    write(tmp_path/"main-execution.json", receipt)
    return tmp_path


def test_completed_training_is_replayed_from_artifacts(main_records):
    assert training.validate_main_execution(main_records)["passed"]


@pytest.mark.parametrize("kind", ["order", "adapter", "process", "count"])
def test_completed_pass_flag_cannot_hide_altered_native_training_artifact(main_records, kind):
    root = main_records
    if kind in ("order", "count"):
        path = root/"run/inventory.json"
        value = read(path)
        value["data_order_sha256" if kind == "order" else "trainable_parameters"] = "f"*64 if kind == "order" else 1
        write(path, value)
    elif kind == "adapter":
        (root/"run/checkpoints/step-0064/adapter.safetensors").write_bytes(b"Different synthetic artifact")
    else:
        path = root/"run-process.json"
        value = read(path)
        value["execution"]["budget_failure"] = "whole_tree_wall_budget"
        write(path, value)
    with pytest.raises(ValueError):
        training.validate_main_execution(root)
