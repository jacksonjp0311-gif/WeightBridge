"""Fake-worker orchestration exercises host replay and resource scope, not efficacy."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.phase_iiir_replication import smoke, training
from experiments.phase_iiiq_baseline.data import generate, PopulationSpec
from experiments.phase_iiiq_baseline.measurement import endpoint, question
from experiments.phase_iiiq_baseline.integrity import ROOT, read
from weightbridge.adaptation.behavior_endpoint import _observation
from weightbridge.core import file_hash, json_dump_atomic


@pytest.fixture
def fake_run(tmp_path, monkeypatch):
    cfg = read(ROOT/"experiments/phase_iiiq_baseline/protocol.json")
    cfg.update(schema="weightbridge.phase_iiir.protocol.v1", replication_arm="rank4",
               source_sha256={"weightbridge/core.py": file_hash(ROOT/"weightbridge/core.py")})
    cfg["adapter"].update(rank=4, alpha=8)
    cfg["training"]["learning_rates"] = [.0003]
    protocol = tmp_path/"protocol.json"
    json_dump_atomic(protocol, cfg)
    rows = generate("smoke", PopulationSpec(target_families=4), "r-smoke-pure-host-fixture", set(), set())
    roster = tmp_path/"material/smoke.json"
    json_dump_atomic(roster, rows)
    bundle = tmp_path/"smoke-bundle.json"
    json_dump_atomic(bundle, {"schema": "weightbridge.supervised_training_bundle.v1",
        "protocol_sha256": file_hash(protocol), "manifest_sha256": "a"*64, "rows": rows, "split_ids": ["smoke"]})
    inventory = sorted(read(ROOT/"results/phase-iii-q-baseline/preflight/inventory-v1/architecture-inventory.json")["allowed"],
                       key=lambda row: row["name"])
    auth = tmp_path/"smoke-authorization.json"
    json_dump_atomic(auth, training.authorization(cfg, inventory, bundle, smoke=True).to_dict())
    contract = endpoint(cfg["inference"])

    def measure(correct):
        measured = []
        for row in rows:
            answer = row["answer"] if correct else next(x for x in row["operands"] if x != row["oracle_answer"])
            text = " "+answer+"\n"
            output = {**row, "text": text, "stop_reason": "token_budget", "generated_token_ids": list(map(ord, text))}
            measured.append({**output, **_observation(output, question(output), contract)})
        return {"endpoint": contract.manifest(), "rows": measured}
    parent, candidate = measure(False), measure(True)
    loss = {"mean_token_nll": 1., "row_ids": [r["id"] for r in rows], "tokenization_sha256": "d"*64,
            "supervised_tokens": 320, "prompt_representation": "bare", "forward_calls": 8}
    resource = {"wall_seconds": .001, "cpu_seconds": .0001, "peak_rss_bytes": 1000}
    root, seen = tmp_path/"smoke", []

    def launch(argv, log, **kwargs):
        name = Path(log).stem
        seen.append(name)
        Path(log).write_text("Synthetic host-worker fixture, no model inference", encoding="utf-8")
        identity = {"checkpoint_sha256": training.PARENT, "protocol_sha256": file_hash(protocol),
                    "roster_sha256": file_hash(roster), "base_unchanged": True, "resources": resource}
        if name == "zero":
            json_dump_atomic(root/"zero/zero-init-check.json", {**identity, "zero_init_parent_identical": True,
                "parent_measurement": parent, "zero_init_measurement": parent, "matched_loss_before": loss})
        elif name == "training":
            checkpoint = root/"training/checkpoints/step-0032"
            checkpoint.mkdir(parents=True)
            (checkpoint/"adapter.safetensors").write_bytes(b"Synthetic, not weights")
            json_dump_atomic(root/"training/completed.json", {"status": "COMPLETED", "smoke": True,
                "checkpoints": [str(checkpoint)], "protocol_sha256": file_hash(protocol),
                "authorization_sha256": file_hash(auth), "bundle_sha256": file_hash(bundle),
                "parent_sha256": training.PARENT, "base_unchanged": True, "run_index": 0,
                "all_factors_changed": True, "parameter_changes": [{"name": "factor", "changed_entries": 1, "norm": .1}],
                "resources": resource})
        else:
            json_dump_atomic(root/name/"fresh-adapter-check.json", {**identity,
                "adapter_sha256": file_hash(root/"training/checkpoints/step-0032/adapter.safetensors"),
                "trained_measurement": candidate, "disabled_measurement": parent,
                "matched_loss_after": {**loss, "mean_token_nll": .1}})
        return {"exit_code": 0, "budget_failure": None, "resources": {"wall_seconds": .002,
            "cpu_seconds": .0003, "host_cpu_seconds": .0001, "observed_tree_cpu_seconds": .0002,
            "peak_rss_bytes": 2000, "cpu_accounting_complete_for_observed_processes": True}}

    @contextmanager
    def sleep():
        state = {"requested": True, "succeeded": True, "released": None}
        try:
            yield state
        finally:
            state["released"] = True
    monkeypatch.setattr(smoke, "launch_owned_tree", launch)
    monkeypatch.setattr(smoke, "_temporary_sleep_request", sleep)
    return (tmp_path/"base", protocol, auth, bundle, roster, root), seen


def test_rank_specific_smoke_round_trip_replays_without_duplicate_cpu(fake_run):
    paths, seen = fake_run
    result = smoke.run(*paths)
    assert result["passed"] and result["rank"] == 4 and result["alpha"] == 8
    assert seen == ["zero", "training", "fresh-1", "fresh-2"]
    assert result["resources"]["observed_tree_cpu_seconds"] == .0008
    assert result["temporary_sleep_request"]["released"]
    _, protocol, auth, bundle, rows, root = paths
    assert smoke.validate_gate(protocol, auth, bundle, rows, root/"smoke-gate.json")["passed"]


def test_smoke_gate_rejects_worker_tampering_without_refitting(fake_run):
    paths, _ = fake_run
    smoke.run(*paths)
    _, protocol, auth, bundle, rows, root = paths
    artifact = root/"fresh-2/fresh-adapter-check.json"
    changed = read(artifact)
    changed["trained_measurement"]["rows"][0]["generated_token_ids"].append(999)
    artifact.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="worker receipt changed"):
        smoke.validate_gate(protocol, auth, bundle, rows, root/"smoke-gate.json")


def test_failed_owned_worker_stops_smoke_and_spends_single_attempt(fake_run, monkeypatch):
    paths, seen = fake_run
    original = smoke.launch_owned_tree
    def fail(argv, log, **kwargs):
        result = original(argv, log, **kwargs)
        if Path(log).stem == "training":
            result["budget_failure"] = "whole_tree_cpu_budget"
            result["exit_code"] = 1
        return result
    monkeypatch.setattr(smoke, "launch_owned_tree", fail)
    result = smoke.run(*paths)
    assert not result["passed"] and seen == ["zero", "training"]
    assert result["temporary_sleep_request"]["released"]
    assert (paths[1].parent/"smoke-launch-claim.json").exists()
    assert result["error"] is not None
