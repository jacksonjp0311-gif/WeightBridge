"""Deterministic host-verification fixtures with no model execution."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.phase_iiiq_baseline import verify_final as verifier
from weightbridge.core import file_hash, json_dump_atomic


def independent(**updates):
    receipt = {"schema": "weightbridge.supervised_baseline.independent.v1",
        "disposition": "HOLD", "reason": verifier.PENDING_REASON, "exact_reconstruction": None,
        "lineage_accepted": False, "production_authority": False,
        **{name: {"passed": True} for name in ("merged", "preservation_validation", "selection", "hidden")}}
    return {**receipt, **updates}


def test_unqualified_verification_writes_hold_without_opening_rosters(tmp_path, monkeypatch):
    assessment = tmp_path / "assessment"
    receipt = assessment / "independent.json"
    json_dump_atomic(receipt, independent(hidden=None))
    original_hash = file_hash(receipt)
    opened = []
    original_read = verifier.read

    def read_only_gate(path):
        opened.append(path)
        assert path == receipt
        return original_read(path)

    monkeypatch.setattr(verifier, "read", read_only_gate)
    out = tmp_path / "verification"
    result = verifier.run(tmp_path / "parent", tmp_path / "merged", tmp_path / "merge.json", tmp_path / "auth.json",
                          tmp_path / "protocol.json", tmp_path / "material", assessment, out)
    assert result["disposition"] == "HOLD" and result["reason"] == "independent_gate_refused"
    assert opened == [receipt] and result["worker_attempts"] == []
    assert result["replays"] == dict.fromkeys(verifier.STAGES)
    assert file_hash(receipt) == original_hash and (out / "final-verification.json").is_file()


def generation():
    return {"endpoint": {"format": "decimal"}, "rows": [
        {"id": "fixture", "generated_token_ids": [4, 5, 6], "text": " -3\n", "stop_reason": "eos"}]}


@pytest.mark.parametrize("field,value", [("generated_token_ids", [4, 7, 6]), ("text", " -3.0\n"),
                                          ("stop_reason", "length"), ("id", "other")])
def test_exact_replay_requires_native_tokens_text_and_stop_reason(field, value):
    prior = generation()
    replay = deepcopy(prior)
    assert verifier._generation_equal(prior, replay)
    replay["rows"][0][field] = value
    assert not verifier._generation_equal(prior, replay)


def test_merge_reconstruction_requires_checkpoint_and_numerical_receipt_identity(tmp_path, monkeypatch):
    receipt = tmp_path / "merge.json"
    json_dump_atomic(receipt, {"changed_entries": 3})
    json_dump_atomic(receipt.with_suffix(".verification.json"), {})
    digest = "a"*64
    monkeypatch.setattr(verifier, "load_checkpoint", lambda _: SimpleNamespace(checkpoint_sha256=lambda: digest))
    identity = {"representation_sha256": digest}
    assert verifier._verify_reconstruction(tmp_path, {"changed_entries": 3}, receipt, identity)["passed"]
    assert not verifier._verify_reconstruction(tmp_path, {"changed_entries": 4}, receipt, identity)["passed"]
    assert not verifier._verify_reconstruction(tmp_path, {"changed_entries": 3}, receipt,
                                               {"representation_sha256": "b"*64})["passed"]


def worker_resources():
    return {"wall_seconds": 5., "cpu_seconds": 21., "peak_rss_bytes": 200,
        "observed_tree_cpu_seconds": 20., "cpu_accounting_complete_for_observed_processes": True}


def test_worker_launcher_refuses_budget_failure_and_keeps_receipt(tmp_path, monkeypatch):
    @contextmanager
    def sleep():
        state = {"requested": True, "succeeded": True, "released": None}
        yield state
        state["released"] = True

    def launch(argv, log, **limits):
        assert limits == {"wall_limit": 1200, "cpu_limit": 30000, "memory_limit": 24*1024**3}
        log.write_text("owned fake process", encoding="utf-8")
        return {"exit_code": 0, "budget_failure": "whole_tree_wall_budget", "resources": worker_resources()}

    monkeypatch.setattr(verifier, "_temporary_sleep_request", sleep)
    monkeypatch.setattr(verifier, "launch_owned_tree", launch)
    attempts = []
    with pytest.raises(RuntimeError, match="execution or resource gate"):
        verifier._worker("evaluate", [], "replay", tmp_path, {"wall_limit": 1200,
            "cpu_limit": 30000, "memory_limit": 24*1024**3}, attempts)
    assert attempts[0]["temporary_sleep_request"]["released"] is True
    assert attempts[0]["process_receipt_sha256"] == file_hash(tmp_path / "replay-process.json")
    assert verifier.read(tmp_path / "replay-process.json")["execution"]["budget_failure"] == "whole_tree_wall_budget"


def test_launcher_counters_cannot_hide_real_worker_cpu_or_memory():
    process = {"execution": {"resources": worker_resources()}}
    verifier._check_worker_body(process, {"wall_seconds": 4., "cpu_seconds": 19., "peak_rss_bytes": 190})
    with pytest.raises(ValueError, match="does not cover"):
        verifier._check_worker_body(process, {"wall_seconds": 4., "cpu_seconds": 100., "peak_rss_bytes": 500})


def test_host_replays_one_reconstruction_and_all_six_protected_workers(tmp_path, monkeypatch):
    base, merged, assessment, material, out = (tmp_path / name for name in ("parent", "merged", "assessment", "material", "out"))
    protocol, merge, auth, adapter = (tmp_path / name for name in ("protocol.json", "merge.json", "auth.json", "adapter.safetensors"))
    representation = "a"*64
    json_dump_atomic(protocol, {"checkpoint_sha256": verifier.PARENT, "source_sha256": {}, "resources":
        {"evaluation_worker_wall_seconds": 1200, "max_cpu_seconds_per_run": 30000, "max_memory_bytes": 24*1024**3}})
    json_dump_atomic(material / "manifest.json", {})
    json_dump_atomic(auth, {})
    adapter.write_bytes(b"fixture factors")
    adapter_sha = file_hash(adapter)
    original_merge = {"schema": "weightbridge.supervised_lora_merge_receipt.v1", "status": "ISOLATED_CANDIDATE",
        "base_sha256": verifier.PARENT, "candidate_sha256": representation, "base_unchanged": True,
        "authorization_sha256": file_hash(auth), "factor_sha256": "f"*64,
        "lineage_accepted": False, "production_deployment": False}

    def save_merge(path):
        json_dump_atomic(path, original_merge)
        json_dump_atomic(path.with_suffix(".verification.json"), {
            "schema": "weightbridge.supervised_baseline.merge_verification.v1", "merge_receipt_sha256": file_hash(path),
            "adapter_sha256": adapter_sha, "protocol_sha256": file_hash(protocol), "parent_checkpoint_sha256": verifier.PARENT,
            "candidate_sha256": representation, "lineage_authority": False, "production_authority": False,
            "resources": {"wall_seconds": 4., "cpu_seconds": 19., "peak_rss_bytes": 190}})

    save_merge(merge)
    json_dump_atomic(assessment / "independent.json", independent(
        protocol_sha256=file_hash(protocol), manifest_sha256=file_hash(material / "manifest.json"),
        representation_sha256=representation, merge_receipt_sha256=file_hash(merge)))
    observation = generation()
    observation["resources"] = {"generate_batches": 1, "batched_decode_steps": 3, "generated_examples": 1,
                                "padded_prompt_token_positions": 5}
    recorded = {"measurement": observation, "resources": {"wall_seconds": 4., "cpu_seconds": 19., "peak_rss_bytes": 190}}

    def prior(_assessment, _material, _protocol_path, _protocol, _independent, _identity, artifacts):
        stages = {}
        for stage in verifier.STAGES:
            roster, stage_auth = tmp_path / f"{stage}-roster.json", tmp_path / f"{stage}-auth.json"
            json_dump_atomic(roster, [])
            json_dump_atomic(stage_auth, {})
            artifacts[f"{stage}_roster"] = roster
            stages[stage] = (roster, stage_auth, recorded, recorded)
        return adapter, {"adapter_file_sha256": adapter_sha, "factor_sha256": "f"*64}, stages

    launched = []

    def launch(module, args, name, _out, _limits, attempts):
        launched.append((name, module, args))
        if module == "merge":
            save_merge(Path(args[args.index("--receipt")+1]))
        else:
            json_dump_atomic(Path(args[args.index("--out")+1]), recorded)
        record = {"execution": {"resources": worker_resources()}}
        attempts.append(record)
        return record

    monkeypatch.setattr(verifier, "_prior_evidence", prior)
    monkeypatch.setattr(verifier, "_worker", launch)
    monkeypatch.setattr(verifier, "_validated_worker", lambda path, *_: verifier.read(path))
    monkeypatch.setattr(verifier, "_gate", lambda *_: {"passed": True})
    monkeypatch.setattr(verifier, "load_checkpoint", lambda path: SimpleNamespace(
        checkpoint_sha256=lambda: verifier.PARENT if path == base else representation))
    result = verifier.run(base, merged, merge, auth, protocol, material, assessment, out)
    assert result["disposition"] == "VERIFIED_BASELINE_PASS"
    assert result["exact_reconstruction"]["passed"] and result["rollback"]["checkpoint_sha256"] == verifier.PARENT
    assert [name for name, _, _ in launched] == ["reconstruct"] + [f"{stage}-{model}" for stage in verifier.STAGES
        for model in ("reconstructed", "rollback")]
    assert all("--authorization" in args for _, _, args in launched)
    assert all(result["replays"][stage]["passed"] for stage in verifier.STAGES)
    assert not result["lineage_accepted"] and not result["production_authority"]
