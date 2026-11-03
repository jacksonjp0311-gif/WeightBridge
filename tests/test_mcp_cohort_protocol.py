import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.original_weight_edits.run_mcp_agent_experiment import (
    make_blinded_task,
    make_random_control_patch,
)
from weightbridge.evaluation.mcp_cohort import (
    CohortProtocolError,
    MECHANISMS,
    _canonical_hash,
    load_cohort_task,
    summarize_cohort,
    validate_protocol,
)
from weightbridge.evaluation.freeze_cohort import freeze_protocol


def make_protocol(tmp_path):
    bound_files = {}
    for name, relative in {
        "task_generator": "task_generator.py",
        "mcp_tool_server": "mcp_server.py",
        "agent_prompt": "agent_prompt.md",
        "cohort_evaluator": "cohort_evaluator.py",
        "protocol_freezer": "protocol_freezer.py",
    }.items():
        path = tmp_path / relative
        path.write_text(name, encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        bound_files[name] = {"path": relative, "sha256": digest}
    policy = {
        "schema": "weightbridge.policy.v1",
        "allow_tensors": ["input_weight", "hidden_bias", "output_weight", "output_bias"],
        "max_entries": 1, "max_fraction": 1 / 12, "max_abs_delta": 8.0,
    }
    sampling = {"temperature": 0.0}
    identity_receipt = {
        "schema": "weightbridge.llm_identity_receipt.v1",
        "provider": "test-provider", "model_id": "test-model",
        "version": "2026-01", "sampling": sampling,
        "host_call_id": "test-call-001",
    }
    identity_path = tmp_path / "model_identity_receipt.json"
    identity_path.write_text(json.dumps(identity_receipt, sort_keys=True), encoding="utf-8")
    identity_sha256 = hashlib.sha256(identity_path.read_bytes()).hexdigest()
    protocol = {
        "schema": "weightbridge.mcp_llm_cohort_protocol.v2",
        "cohort_id": "test-cohort",
        "status": "FROZEN",
        "protocol_sha256": "",
        "model_binding": {
            "provider": "test-provider", "model_id": "test-model",
            "version": "2026-01", "identity_provenance": "test receipt",
            "identity_evidence_path": "model_identity_receipt.json",
            "identity_evidence_sha256": identity_sha256,
            "identity_verified": True, "sampling": sampling,
        },
        "prompt_path": bound_files["agent_prompt"]["path"],
        "prompt_sha256": bound_files["agent_prompt"]["sha256"],
        "source_hashes": bound_files,
        "host_policy": policy,
        "host_policy_sha256": hashlib.sha256(json.dumps(
            policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "max_candidates": 8,
        "max_failure_examples": 48,
        "tasks": [
            {"task_id": f"trial-{i:02d}", "seed": 70000 + i,
             "mechanism": MECHANISMS[(i - 1) // 5]}
            for i in range(1, 21)
        ],
        "execution_order": [f"trial-{i:02d}" for i in range(20, 0, -1)],
        "task_design": {
            "practice_and_holdout_per_class": 32,
            "fresh_agent_context_per_task": True,
            "hidden_labels_accessible_to_agent": False,
            "agent_surface": "MCP tools only; no filesystem, repository, or network tools",
            "tool_access_isolated": True,
            "random_control": "one-entry random edit; same tensor and absolute magnitude when a candidate exists; fallback magnitude 4.0 otherwise",
        },
        "decision_gate": {
            "repair_definition": "candidate achieves 100 percent holdout accuracy and baseline is below 100 percent",
            "min_repair_rate": 0.7, "max_regression_rate": 0.05,
            "min_candidate_minus_random": 0.2, "ci_lower_strictly_above": 0.0,
        },
        "analysis": {
            "bootstrap_seed": 918273,
            "confidence": 0.95,
            "method": "paired task-level accuracy differences; bootstrap seeds within each mechanism stratum",
            "resamples": 10_000,
        },
    }
    protocol["protocol_sha256"] = _canonical_hash(protocol)
    return protocol


def make_records(protocol, tmp_path, *, random_accuracy=0.5):
    records = []
    execution_positions = {task_id: index for index, task_id in
                           enumerate(protocol["execution_order"], 1)}
    baseline = [True] * 32 + [False] * 32
    candidate = [True] * 64
    random_n = int(round(random_accuracy * 64))
    random = [True] * random_n + [False] * (64 - random_n)
    for task in protocol["tasks"]:
        binding = {
            "cohort_id": protocol["cohort_id"],
            "protocol_sha256": protocol["protocol_sha256"],
            "task_id": task["task_id"], "task_seed": task["seed"],
            "execution_index": execution_positions[task["task_id"]],
            "mechanism": task["mechanism"],
            "agent_session_id": f"session-{task['task_id']}",
            "provider": protocol["model_binding"]["provider"],
            "model_id": protocol["model_binding"]["model_id"],
            "version": protocol["model_binding"]["version"],
            "identity_evidence_sha256": protocol["model_binding"]["identity_evidence_sha256"],
            "identity_verified": True,
            "prompt_sha256": protocol["prompt_sha256"],
        }
        artifact_root = tmp_path / f"artifact-{task['task_id']}"
        artifact_root.mkdir()
        transcript = artifact_root / "agent_mcp_transcript.jsonl"
        event = {"schema": "weightbridge.mcp_exchange.v1", "sequence": 1,
                 "request": {"tool": "get_diagnostic_report", "arguments": {}},
                 "response": {"ok": True}}
        transcript.write_text(json.dumps(event) + "\n", encoding="utf-8")
        identity_copy = artifact_root / "model_identity_receipt.json"
        identity_copy.write_bytes((tmp_path / "model_identity_receipt.json").read_bytes())
        records.append({
            "schema": "weightbridge.experiment.v1",
            "timestamp_utc": f"2026-01-01T00:00:{execution_positions[task['task_id']]:02d}+00:00",
            "protocol_binding": binding,
            "artifact_root": artifact_root.name,
            "artifacts": ["agent_mcp_transcript.jsonl", "model_identity_receipt.json"],
            "task_oracle": {"mechanism": task["mechanism"]},
            "resources": {"mcp_tool_calls": 1},
            "agent": {"mcp_transcript_sha256": hashlib.sha256(transcript.read_bytes()).hexdigest()},
            "affected_parameters": {"tensor": "hidden_bias", "count": 1,
                                     "max_abs_delta": 0.7, "candidate_tensor_index": [0]},
            "control_modification": {"tensor": "hidden_bias", "index": [1],
                                     "delta": -0.7, "parameter_count": 1,
                                     "magnitude_match_type": "same_tensor_and_absolute_magnitude"},
            "evaluation": {
                "seed": task["seed"],
                "holdout_accessed_by_agent": False,
                "metrics": {
                    "baseline": {"per_example_correct": baseline},
                    "ai_candidate": {"per_example_correct": candidate},
                    "random_control": {"per_example_correct": random},
                },
            },
        })
    return records


def test_forced_blinded_task_mechanisms_are_reproducible():
    for mechanism in MECHANISMS:
        first = make_blinded_task(70011, 16, mechanism=mechanism)
        second = make_blinded_task(70011, 16, mechanism=mechanism)
        assert first["oracle"]["mechanism"] == mechanism
        assert first["oracle"] == second["oracle"]
        assert np.array_equal(first["practice"]["features"], second["practice"]["features"])
        assert np.array_equal(first["holdout"]["labels"], second["holdout"]["labels"])
        assert not np.array_equal(first["practice"]["features"], first["holdout"]["features"])


def test_random_control_matches_tensor_and_magnitude_or_uses_frozen_fallback():
    task = make_blinded_task(70011, 16, mechanism="hidden_threshold_shift")
    matched, match_type = make_random_control_patch(
        task["model"], "a" * 64, 1234, target_tensor="hidden_bias",
        target_index=[0], magnitude=0.7)
    change = matched["changes"][0]
    assert match_type == "same_tensor_and_absolute_magnitude"
    assert change["tensor"] == "hidden_bias"
    assert change["index"] != [0]
    assert abs(change["value"]) == 0.7

    fallback, fallback_type = make_random_control_patch(task["model"], "a" * 64, 1234)
    assert fallback_type == "predeclared_fallback_magnitude"
    assert abs(fallback["changes"][0]["value"]) == 4.0


def test_manifest_rejects_unpinned_template_before_start(tmp_path):
    protocol = make_protocol(tmp_path)
    protocol["model_binding"]["identity_verified"] = False
    protocol["protocol_sha256"] = _canonical_hash(protocol)
    errors = validate_protocol(protocol, root=tmp_path)
    assert any("identity has no host verification" in error for error in errors)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(CohortProtocolError, match="identity has no host verification"):
        load_cohort_task(manifest, "trial-01", root=tmp_path)


def test_manifest_rejects_unregistered_analysis_plan(tmp_path):
    protocol = make_protocol(tmp_path)
    protocol["analysis"]["resamples"] = 500
    protocol["protocol_sha256"] = _canonical_hash(protocol)
    errors = validate_protocol(protocol, root=tmp_path)
    assert "analysis plan differs from the preregistered stratified bootstrap" in errors


def _write_template_for_freeze(tmp_path):
    protocol = make_protocol(tmp_path)
    protocol["status"] = "TEMPLATE_HOLD"
    protocol["model_binding"].update({
        "provider": None, "model_id": None, "version": None,
        "identity_provenance": None, "identity_verified": False,
        "sampling": None, "identity_evidence_path": None,
        "identity_evidence_sha256": None,
    })
    protocol["protocol_sha256"] = _canonical_hash(protocol)
    path = tmp_path / "template.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return path


def test_freeze_command_creates_valid_new_protocol_and_receipt_bundle(tmp_path):
    template = _write_template_for_freeze(tmp_path)
    receipt = tmp_path / "model_identity_receipt.json"
    output_dir = tmp_path / "frozen"
    output_dir.mkdir()
    output = output_dir / "cohort-a.json"
    result = freeze_protocol(
        template, receipt, output,
        cohort_id="cohort-a-2026-11-03",
        identity_provenance="test host receipt; unit test only",
        root=tmp_path,
    )
    frozen = json.loads(output.read_text(encoding="utf-8"))
    assert frozen["status"] == "FROZEN"
    assert frozen["cohort_id"] == "cohort-a-2026-11-03"
    assert frozen["model_binding"]["identity_verified"] is True
    assert result["identity_receipt_path"].is_file()
    assert validate_protocol(frozen, root=tmp_path) == []
    loaded, task = load_cohort_task(output, "trial-01", root=tmp_path)
    assert loaded["cohort_id"] == "cohort-a-2026-11-03"
    assert task["seed"] == frozen["tasks"][0]["seed"]
    with pytest.raises(FileExistsError, match="immutable"):
        freeze_protocol(
            template, receipt, output,
            cohort_id="cohort-a-2026-11-03",
            identity_provenance="test host receipt; unit test only",
            root=tmp_path,
        )


def test_freeze_command_rejects_incomplete_identity_receipt(tmp_path):
    template = _write_template_for_freeze(tmp_path)
    bad_receipt = tmp_path / "bad_identity_receipt.json"
    bad_receipt.write_text(json.dumps({
        "schema": "weightbridge.llm_identity_receipt.v1",
        "provider": "test-provider",
    }), encoding="utf-8")
    output_dir = tmp_path / "frozen"
    output_dir.mkdir()
    output = output_dir / "cohort-a.json"
    with pytest.raises(CohortProtocolError, match="identity receipt is incomplete"):
        freeze_protocol(
            template, bad_receipt, output,
            cohort_id="cohort-a",
            identity_provenance="unit test",
            root=tmp_path,
        )
    assert not output.exists()
    assert not output.with_name("cohort-a.identity_receipt.json").exists()


def test_checked_in_protocol_template_has_valid_hashes_but_is_hold_for_identity():
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / "experiments" / "original_weight_edits" / "mcp_llm_transfer_v2.template.json"
    protocol = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = validate_protocol(protocol, root=project_root)
    assert "protocol hash mismatch" not in errors
    assert not any("source hash changed" in error for error in errors)
    assert "host policy hash mismatch" not in errors
    assert any("LLM identity is not pinned" in error for error in errors)
    with pytest.raises(CohortProtocolError, match="LLM identity is not pinned"):
        load_cohort_task(manifest_path, "trial-01", root=project_root)


def test_runner_refuses_template_before_creating_a_task():
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "experiments" / "original_weight_edits" / "run_mcp_agent_experiment.py"
    manifest = project_root / "experiments" / "original_weight_edits" / "mcp_llm_transfer_v2.template.json"
    result = subprocess.run(
        [sys.executable, str(script), "--cohort-manifest", str(manifest), "--task-id", "trial-01"],
        cwd=project_root, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "cohort protocol is not runnable" in result.stderr
    assert "LLM identity is not pinned" in result.stderr


def test_stratified_cohort_gate_uses_holdout_arrays_and_matched_random(tmp_path):
    protocol = make_protocol(tmp_path)
    summary = summarize_cohort(protocol, make_records(protocol, tmp_path), root=tmp_path)
    assert summary["disposition"] == "PASS"
    assert summary["metrics"]["repairs"] == 20
    assert summary["metrics"]["repair_rate"] == 1.0
    assert summary["metrics"]["regression_rate"] == 0.0
    paired = summary["metrics"]["candidate_minus_matched_random"]
    assert paired["mean_candidate_minus_random"] == 0.5
    assert paired["ci95_stratified_by_mechanism"] == [0.5, 0.5]
    assert paired["bootstrap_seed"] == protocol["analysis"]["bootstrap_seed"]
    assert paired["resamples"] == protocol["analysis"]["resamples"]
    assert paired["confidence"] == protocol["analysis"]["confidence"]
    assert summary["analysis"] == protocol["analysis"]


def test_cohort_fails_when_matched_random_is_as_good_as_agent(tmp_path):
    protocol = make_protocol(tmp_path)
    summary = summarize_cohort(protocol, make_records(protocol, tmp_path, random_accuracy=1.0),
                               root=tmp_path)
    assert summary["disposition"] == "FAIL"
    assert summary["metrics"]["candidate_minus_matched_random"]["mean_candidate_minus_random"] == 0.0


def test_incomplete_cohort_is_inconclusive(tmp_path):
    protocol = make_protocol(tmp_path)
    summary = summarize_cohort(protocol, make_records(protocol, tmp_path)[:19], root=tmp_path)
    assert summary["disposition"] == "INCONCLUSIVE"
    assert summary["records_bound_to_plan"] == 19
    assert len(summary["missing_task_ids"]) == 1


def test_holdout_exposure_prevents_cohort_pass(tmp_path):
    protocol = make_protocol(tmp_path)
    records = make_records(protocol, tmp_path)
    records[0]["evaluation"]["holdout_accessed_by_agent"] = True
    summary = summarize_cohort(protocol, records, root=tmp_path)
    assert summary["disposition"] == "INCONCLUSIVE"
    assert any("holdout exposure" in reason for reason in summary["reasons"])


def test_unmatched_random_edit_prevents_cohort_pass(tmp_path):
    protocol = make_protocol(tmp_path)
    records = make_records(protocol, tmp_path)
    records[0]["control_modification"]["tensor"] = "output_bias"
    summary = summarize_cohort(protocol, records, root=tmp_path)
    assert summary["disposition"] == "INCONCLUSIVE"
    assert any("not matched to the AI edit" in reason for reason in summary["reasons"])


def test_transcript_hash_tampering_prevents_cohort_pass(tmp_path):
    protocol = make_protocol(tmp_path)
    records = make_records(protocol, tmp_path)
    transcript = tmp_path / records[0]["artifact_root"] / "agent_mcp_transcript.jsonl"
    transcript.write_text("tampered\n", encoding="utf-8")
    summary = summarize_cohort(protocol, records, root=tmp_path)
    assert summary["disposition"] == "INCONCLUSIVE"
    assert any("hashed MCP transcript is missing" in reason for reason in summary["reasons"])


def test_identity_receipt_tampering_prevents_cohort_pass(tmp_path):
    protocol = make_protocol(tmp_path)
    records = make_records(protocol, tmp_path)
    evidence = tmp_path / records[0]["artifact_root"] / "model_identity_receipt.json"
    evidence.write_text("tampered", encoding="utf-8")
    summary = summarize_cohort(protocol, records, root=tmp_path)
    assert summary["disposition"] == "INCONCLUSIVE"
    assert any("hashed LLM identity receipt is missing" in reason for reason in summary["reasons"])


def test_execution_order_mismatch_prevents_cohort_pass(tmp_path):
    protocol = make_protocol(tmp_path)
    records = make_records(protocol, tmp_path)
    records[0]["protocol_binding"]["execution_index"] = 1
    summary = summarize_cohort(protocol, records, root=tmp_path)
    assert summary["disposition"] == "INCONCLUSIVE"
    assert any("execution order differs" in reason for reason in summary["reasons"])
