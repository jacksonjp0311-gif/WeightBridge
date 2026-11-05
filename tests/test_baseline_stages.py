"""Host-stage fixtures only. Mock oracle outputs are not neural measurements."""
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.phase_iiiq_baseline import data, study
from experiments.phase_iiiq_baseline.integrity import PARENT, authorize_stage
from weightbridge.adaptation.behavior_endpoint import _observation
from weightbridge.core import file_hash, json_dump_atomic


@pytest.fixture
def trial(tmp_path, monkeypatch):
    config = study.read(data.PROTOCOL)
    config["source_sha256"] = {}
    config["data"]["populations"] = {name: asdict(spec) for name, spec in data.DEFAULT_POPULATIONS.items()}
    config["data"]["populations"]["train"] = {"target_families": 16}
    config["data"]["populations"]["development"] = {"target_families": 4}
    config["data"]["populations"]["selection"] = {"target_families": 4}
    config["data"]["populations"]["hidden"] = {"target_families": 4}
    protocol = tmp_path / "protocol.json"
    json_dump_atomic(protocol, config)
    material, hidden = tmp_path / "private/material", tmp_path / "sealed/hidden.json"
    manifest = data.freeze(material, hidden, protocol=protocol, host_seeds={"selection": "s-secret", "hidden": "h-secret"})
    runs = []
    for index in (0, 1):
        root = tmp_path / f"run-{index}"
        checkpoints = []
        for step in config["training"]["checkpoint_steps"]:
            checkpoint = root / "checkpoints" / f"step-{step:04d}"
            checkpoint.mkdir(parents=True)
            adapter = checkpoint / "adapter.safetensors"
            adapter.write_bytes(f"synthetic-adapter-{index}-{step}".encode())
            digest = file_hash(adapter)
            receipt = {"sha256": digest, "factor_sha256": digest, "step": step, "base_unchanged": True,
                "configuration": {"checkpoint_sha256": PARENT},
                "effective_update": {"stored_update_norm": .4-index*.1},
                "host_metadata": {"protocol_sha256": file_hash(protocol), "manifest_sha256": file_hash(material / "manifest.json"),
                                  "run_index": index, "step": step}}
            json_dump_atomic(checkpoint / "receipt.json", receipt)
            checkpoints.append(str(checkpoint))
        json_dump_atomic(root / "completed.json", {"status": "COMPLETED", "base_unchanged": True,
            "smoke": False, "parent_sha256": PARENT, "protocol_sha256": file_hash(protocol),
            "manifest_sha256": file_hash(material / "manifest.json"), "run_index": index,
            "counts": {"optimizer_steps": config["training"]["optimizer_steps"]}, "checkpoints": checkpoints})
        runs.append(root)
    return SimpleNamespace(root=tmp_path, protocol=protocol, config=config, material=material, hidden=hidden,
        manifest=manifest, runs=runs, base=tmp_path / "base", out=tmp_path / "assessment")


def processes(monkeypatch, trial, *, development_family=True, regression=False, merged_fails=False):
    calls = []
    def launch(base, protocol_path, rows_path, out, *, checkpoint=PARENT, adapter=None, adapter_sha=None, stage, authorization=None):
        calls.append((stage, checkpoint, adapter_sha))
        rows = study.read(rows_path)
        contract = study.endpoint(trial.config["inference"])
        families = sorted({row["pair_id"] for row in rows if row["group"] == "target"})
        parent = adapter is None and checkpoint == PARENT
        rendered = []
        for index, row in enumerate(rows):
            target = row["group"] == "target"
            correct = not target or row["pair_id"] == families[-1]
            if not parent and target and row["pair_id"] == families[0] and not (merged_fails and stage.startswith("merged")):
                correct = not (not development_family and "development" in stage and row["counterfactual"] == 1)
            if regression and not parent and not target and index == next(i for i, r in enumerate(rows) if r["group"] != "target"):
                correct = False
            if correct:
                answer = row["answer"]
            elif row["family"] == "larger-decimal":
                answer = next(value for value in row["operands"] if Decimal(value) != Decimal(row["answer"]))
            else:
                answer = "0"
            output = {**row, "text": " "+answer+"\n", "stop_reason": "token_budget"}
            rendered.append({**output, **_observation(output, contract.question(row), contract)})
        measurement = {"endpoint": contract.manifest(), "rows": rendered, "resources": {}}
        result = {"schema": "weightbridge.supervised_baseline.evaluation.v1", "stage": stage,
            "checkpoint_sha256": checkpoint, "adapter_sha256": adapter_sha,
            "protocol_sha256": file_hash(protocol_path), "roster_sha256": file_hash(rows_path),
            "source_sha256": trial.config["source_sha256"], "measurement": measurement,
            "summary": study.summarize(measurement), "resources": {}}
        json_dump_atomic(out, result)
        return result
    monkeypatch.setattr(study, "_launch", launch)
    return calls


def test_probes_keep_full_families_and_eighty_replay_presentations(trial):
    inputs = {name: data.load_split(trial.material, name, trial.manifest, protocol=trial.protocol) for name in ("train", "replay", "development")}
    probes = study.probe_rosters(inputs["train"], inputs["replay"], inputs["development"], trial.config)
    assert len(probes["practice"]) == 208
    assert sum(row["group"] != "target" for row in probes["practice"]) == 80
    assert sum(row["group"] == "target" for row in probes["practice"]) == 128
    assert len(probes["development"]) == 112  # Four-family unit fixture, not the real 32-family cohort.


@pytest.mark.parametrize("location", ["checkout", "inside_parent", "contains_parent"])
def test_host_assessment_and_finish_refuse_public_or_overlapping_evidence_paths(trial, location):
    destination = {"checkout": data.ROOT / "never-create-host-evidence",
        "inside_parent": trial.base / "never-create-host-evidence", "contains_parent": trial.root}[location]
    with pytest.raises(ValueError, match="outside the public checkout"):
        study.assess(trial.base, trial.protocol, trial.material, destination, trial.runs)
    with pytest.raises(ValueError, match="outside the public checkout"):
        study.finish(trial.base, trial.root / "merged", trial.root / "receipt.json", trial.root / "policy.json",
                     trial.protocol, trial.material, destination, hidden_path=trial.hidden)


def test_development_strict_gain_without_complete_family_never_opens_independent_data(monkeypatch, trial):
    calls = processes(monkeypatch, trial, development_family=False)
    result = study.assess(trial.base, trial.protocol, trial.material, trial.out, trial.runs)
    assert result["reason"] == "no_practice_and_development_qualified_adapter"
    assert result["winner"] is None and result["selection"] is result["hidden"] is None
    assert all(record["practice_gate"]["passed"] for record in result["candidates"].values())
    assert all(record["development_gate"]["strict_comparison_gain"] == 1 for record in result["candidates"].values())
    assert {stage for stage, _, _ in calls} == {"practice", "development"}
    assert not (trial.out / "finalist-lock.json").exists()
    with pytest.raises(FileExistsError):
        study.assess(trial.base, trial.protocol, trial.material, trial.out, trial.runs)


def test_preservation_regression_blocks_adapter_even_when_targets_improve(monkeypatch, trial):
    processes(monkeypatch, trial, regression=True)
    result = study.assess(trial.base, trial.protocol, trial.material, trial.out, trial.runs)
    assert not result["winner"]
    assert all(record["practice_gate"]["strict_comparison_gain"] > 0
               and record["practice_gate"]["prohibited_regressions"] for record in result["candidates"].values())


def test_fixed_checkpoint_finalist_ranking_is_replayable_and_host_only(monkeypatch, trial):
    processes(monkeypatch, trial)
    result = study.assess(trial.base, trial.protocol, trial.material, trial.out, trial.runs)
    assert result["winner"] == "run-1-step-0064"  # Same gains: earlier checkpoint, then smaller measured norm.
    assert study._replay_screening(trial.out, trial.protocol) == result
    assert result["disposition"] == "HOLD" and not result["lineage_accepted"]
    assert result["preservation_validation"] is result["selection"] is result["hidden"] is None
    with pytest.raises(FileNotFoundError):
        authorize_stage(trial.out, "selection", representation_sha256="f"*64,
                        roster_sha256=trial.manifest["splits"]["selection"]["sha256"])


def test_incomplete_training_and_worker_interruption_record_hold(monkeypatch, trial):
    def fail(*args, **kwargs):
        raise RuntimeError("fixture worker interrupted")
    monkeypatch.setattr(study, "_launch", fail)
    result = study.assess(trial.base, trial.protocol, trial.material, trial.out, trial.runs)
    assert result["reason"] == "incomplete_behavioral_evaluation" and result["errors"]
    assert not (trial.out / "finalist-lock.json").exists() and result["hidden"] is None


def test_incomplete_training_is_refused_before_model_evaluation(monkeypatch, trial):
    path = trial.runs[1] / "completed.json"
    value = study.read(path); value["status"] = "FAILED"
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(study, "_launch", lambda *args, **kwargs: pytest.fail("No model after incomplete training"))
    result = study.assess(trial.base, trial.protocol, trial.material, trial.out, trial.runs)
    assert result["errors"] and result["winner"] is None


def test_edited_eligibility_cannot_authorize_a_later_stage(monkeypatch, trial):
    processes(monkeypatch, trial)
    study.assess(trial.base, trial.protocol, trial.material, trial.out, trial.runs)
    path = trial.out / "assessment.json"
    value = study.read(path); value["candidates"][value["winner"]]["eligible"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="eligibility"):
        study._replay_screening(trial.out, trial.protocol)


def merged_fixture(monkeypatch, trial):
    reference = study.read(trial.out / "finalist-reference.json")
    merged = trial.root / "merged"
    policy = trial.root / "merge-policy.json"
    json_dump_atomic(policy, {"schema": "weightbridge.supervised_lora_merge.v1", "base_checkpoint_sha256": PARENT,
        "authority": "unit-test-host", "stored_dtype": "BF16", "merge_math": "float64_factor_product_then_round_once_into_BF16",
        "production_authority": False, "lineage_authority": False})
    receipt = trial.root / "merge-receipt.json"
    json_dump_atomic(receipt, {"schema": "weightbridge.supervised_lora_merge_receipt.v1", "status": "ISOLATED_CANDIDATE",
        "base_sha256": PARENT, "candidate_sha256": "f"*64, "factor_sha256": reference["factor_sha256"],
        "authorization_sha256": file_hash(policy), "base_unchanged": True})
    monkeypatch.setattr(study, "load_checkpoint", lambda path: SimpleNamespace(checkpoint_sha256=lambda: PARENT if Path(path) == trial.base else "f"*64))
    return merged, receipt, policy


def test_merged_behavior_failure_keeps_all_independent_rosters_unopened(monkeypatch, trial):
    calls = processes(monkeypatch, trial, merged_fails=True)
    study.assess(trial.base, trial.protocol, trial.material, trial.out, trial.runs)
    merged, receipt, policy = merged_fixture(monkeypatch, trial)
    result = study.finish(trial.base, merged, receipt, policy, trial.protocol, trial.material, trial.out, hidden_path=trial.hidden)
    assert result["reason"] == "merged_representation_failed_practice_or_development"
    assert result["preservation_validation"] is result["selection"] is result["hidden"] is None
    assert not any(stage in ("selection", "hidden", "preservation-validation") for stage, _, _ in calls)


def test_independent_success_still_requires_exact_reconstruction_and_no_auto_promotion(monkeypatch, trial):
    processes(monkeypatch, trial)
    study.assess(trial.base, trial.protocol, trial.material, trial.out, trial.runs)
    merged, receipt, policy = merged_fixture(monkeypatch, trial)
    result = study.finish(trial.base, merged, receipt, policy, trial.protocol, trial.material, trial.out, hidden_path=trial.hidden)
    assert result["hidden"]["passed"]
    assert result["reason"] == "independent_behavior_qualified_exact_reconstruction_and_rollback_required"
    assert result["disposition"] == "HOLD" and result["exact_reconstruction"] is None
    assert not result["lineage_accepted"] and not result["production_authority"]
    with pytest.raises(FileExistsError):
        study.finish(trial.base, merged, receipt, policy, trial.protocol, trial.material, trial.out, hidden_path=trial.hidden)
