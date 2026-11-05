"""Thin Phase III-R host gates over the unchanged Q training/verifier path.

No method in this module trains, changes a behavioral threshold, or promotes a
checkpoint. Q-compatible arm views preserve the independently checked worker
and reconstruction contracts. Protected per-row results stay host-private.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from experiments.phase_iiiq_baseline import study as qstudy
from experiments.phase_iiiq_baseline import verify_final as qverify
from experiments.phase_iiiq_baseline.data import read_manifest, load_split
from experiments.phase_iiiq_baseline.integrity import (
    PARENT, read, verify_sources, directory_hash, authorize_stage,
)
from experiments.phase_iiiq_baseline.measurement import compare

STAGES = qverify.STAGES


def retention(reference, candidate):
    """Paired correctness/validity retention; tied totals cannot hide losses."""
    result = compare(reference, candidate, min_strict_gain=0, min_family_gain=0)
    before, after = reference["rows"], candidate["rows"]
    result["lost_required_controls"] = [a["id"] for a, b in zip(before, after, strict=True)
        if a["group"] != "target" and ((a["correct"] and not b["correct"])
                                       or (a["valid"] and not b["valid"]))]
    result["correctness_disagreements"] = [a["id"] for a, b in zip(before, after, strict=True)
                                           if a["correct"] != b["correct"]]
    result["validity_disagreements"] = [a["id"] for a, b in zip(before, after, strict=True)
                                       if a["valid"] != b["valid"]]
    result["complete_continuation_disagreements"] = [a["id"] for a, b in zip(before, after, strict=True)
        if any(a.get(key) != b.get(key) for key in ("generated_token_ids", "text", "stop_reason"))]
    result["scope"] = "Measured paired suites only; no population-wide equivalence or full-continuation claim."
    return result


def _verified_q(receipt):
    if (receipt.get("schema") != "weightbridge.supervised_baseline.final_verification.v1"
            or receipt.get("passed") is not True or receipt.get("disposition") != "VERIFIED_BASELINE_PASS"
            or receipt.get("parent_sha256") != PARENT
            or (receipt.get("exact_reconstruction") or {}).get("passed") is not True
            or (receipt.get("rollback") or {}).get("passed") is not True
            or receipt.get("lineage_accepted") is not False or receipt.get("production_authority") is not False
            or receipt.get("error") is not None):
        raise ValueError("Replication requires exact independently verified reconstruction and original-parent rollback")
    for stage in STAGES:
        replay = receipt.get("replays", {}).get(stage) or {}
        if (replay.get("passed") is not True or any(
                (replay.get(name) or {}).get("exact_native_generation") is not True
                for name in ("reconstructed", "rollback"))):
            raise ValueError("Every protected stage must exactly replay before replication can pass")
    return receipt


def validate_replication_gate(protocol, path):
    """Rank-four training admission reads commitments, never protected rows."""
    value = read(path)
    if (value.get("schema") != "weightbridge.phase_iiir.replication_verification.v1"
            or value.get("passed") is not True or value.get("disposition") != "VERIFIED_REPLICATION_PASS"
            or value.get("study_sha256") != protocol.get("master_protocol_sha256")
            or value.get("global_data_manifest_sha256") != protocol.get("global_data_manifest_sha256")
            or value.get("parent_sha256") != PARENT or value.get("lineage_accepted") is not False
            or value.get("production_authority") is not False):
        raise ValueError("Rank-four training requires this study's fully verified rank-eight replication")
    source = Path(value["verification_path"])
    arm_protocol = Path(value["rank8_protocol_path"])
    if file_hash(source) != value["verification_sha256"] or file_hash(arm_protocol) != value["rank8_protocol_sha256"]:
        raise ValueError("Replication receipt or rank-eight frozen protocol identity changed")
    config = read(arm_protocol)
    verified = _verified_q(read(source))
    if (config["adapter"]["rank"] != 8 or config["adapter"]["alpha"] != 16
            or config["checkpoint_sha256"] != PARENT
            or config["master_protocol_sha256"] != value["study_sha256"]
            or config["global_data_manifest_sha256"] != value["global_data_manifest_sha256"]
            or verified["protocol_sha256"] != value["rank8_protocol_sha256"]
            or verified["representation_sha256"] != value["representation_sha256"]):
        raise ValueError("The replication belongs to another configuration or model")
    verify_sources(config["source_sha256"])
    # The inner verifier can finish just before the enclosing process exceeds
    # its budget. Only the completed outer receipt may admit the next arm.
    from .execute import validate_stage_execution
    stage = validate_stage_execution(arm_protocol.parent.parent, "rank8", "verify", arm_protocol)
    if stage["outcome"].get("disposition") != "VERIFIED_REPLICATION_PASS":
        raise ValueError("The enclosing rank-eight verification did not produce a replication pass")
    return value


def seal_replication(protocol_path, verification_path, out):
    """Name the R result only after the existing exact verifier has passed."""
    config = read(protocol_path)
    verified = _verified_q(read(verification_path))
    result = {"schema": "weightbridge.phase_iiir.replication_verification.v1", "passed": True,
        "disposition": "VERIFIED_REPLICATION_PASS", "parent_sha256": PARENT,
        "study_sha256": config["master_protocol_sha256"],
        "global_data_manifest_sha256": config["global_data_manifest_sha256"],
        "rank8_protocol_path": str(Path(protocol_path).resolve()), "rank8_protocol_sha256": file_hash(protocol_path),
        "verification_path": str(Path(verification_path).resolve()), "verification_sha256": file_hash(verification_path),
        "representation_sha256": verified["representation_sha256"],
        "lineage_accepted": False, "production_authority": False}
    if Path(out).exists():
        raise FileExistsError("Replication disposition is write-once")
    # Validate before creating an admission record; no transient passing record.
    if (config["adapter"]["rank"] != 8 or config["adapter"]["alpha"] != 16
            or verified["protocol_sha256"] != file_hash(protocol_path)):
        raise ValueError("Replication must use this exact rank-eight arm")
    verify_sources(config["source_sha256"])
    json_dump_atomic(out, result)
    return result


def _matched_arms(rank8_protocol, rank4_protocol):
    a, b = read(rank8_protocol), read(rank4_protocol)
    if (a["checkpoint_sha256"] != PARENT or b["checkpoint_sha256"] != PARENT
            or (a["adapter"]["rank"], a["adapter"]["alpha"]) != (8, 16)
            or (b["adapter"]["rank"], b["adapter"]["alpha"]) != (4, 8)
            or any(a[key] != b[key] for key in ("master_protocol_sha256", "global_data_manifest_sha256",
                                               "training", "inference", "endpoint", "gates"))):
        raise ValueError("Rank comparison requires both prospectively matched configurations")
    verify_sources(a["source_sha256"])
    verify_sources(b["source_sha256"])
    return a, b


def _bound_reference(protocol_path, assessment, replication):
    """The paired comparator must be the exact independently verified finalist."""
    prior = _verified_q(read(replication["verification_path"]))
    if (file_hash(protocol_path) != replication["rank8_protocol_sha256"]
            or file_hash(assessment / "independent.json") != prior["independent_sha256"]
            or any(file_hash(assessment / filename) != prior["input_sha256"][key]
                   for key, filename in (("assessment", "assessment.json"),
                       ("finalist_reference", "finalist-reference.json"), ("finalist_lock", "finalist-lock.json")))
            or read(assessment / "merged-development-gate.json")["representation_sha256"]
                != replication["representation_sha256"]):
        raise ValueError("Paired reference assessment differs from the verified rank-eight finalist")


def development_retention(rank8_protocol, rank4_protocol, rank8_assessment, rank4_assessment, *, merged=False):
    """Replay locked rankings and compare identical development cases."""
    _matched_arms(rank8_protocol, rank4_protocol)
    paths = [Path(rank8_assessment), Path(rank4_assessment)]
    protocols = [Path(rank8_protocol), Path(rank4_protocol)]
    assessments = [qstudy._replay_screening(path, protocol) for path, protocol in zip(paths, protocols, strict=True)]
    if file_hash(paths[0] / "rosters/development.json") != file_hash(paths[1] / "rosters/development.json"):
        raise ValueError("The rank comparison must use the same development presentations")
    workers = []
    for path, protocol, assessment in zip(paths, protocols, assessments, strict=True):
        selected = assessment["candidates"][assessment["winner"]]
        filename = "merged-development" if merged else f"development-{assessment['winner']}"
        if merged:
            gate = read(path / "merged-development-gate.json")
            if gate.get("passed") is not True:
                raise ValueError("Merged checkpoint must pass its own development gates first")
            checkpoint, adapter = gate["representation_sha256"], None
        else:
            checkpoint, adapter = PARENT, selected["adapter_sha256"]
        worker = qstudy._validated_worker(path / "evaluation" / f"{filename}.json",
            path / "rosters/development.json", protocol, checkpoint, adapter,
            "merged-development" if merged else "development")
        workers.append(worker)
    return retention(workers[0]["measurement"], workers[1]["measurement"])


def prepare_merge(base, protocol_path, assessment, authorization, merged, receipt, out):
    """Fresh-process adapter replay and existing separately authorized merge."""
    assessment, out = Path(assessment), Path(out)
    qstudy._host_output(base, out)
    out.mkdir(parents=True, exist_ok=False)
    screening = qstudy._replay_screening(assessment, protocol_path)
    reference = read(assessment / "finalist-reference.json")
    selected = screening["candidates"][screening["winner"]]
    protocol = read(protocol_path)
    adapter = Path(reference["adapter_directory"]) / "adapter.safetensors"
    replay_path = out / "adapter-practice-replay.json"
    replay = qstudy._launch(base, protocol_path, assessment / "rosters/practice.json", replay_path,
        adapter=adapter, adapter_sha=selected["adapter_sha256"], stage="practice")
    prior = read(assessment / "evaluation" / f"practice-{screening['winner']}.json")
    if not qverify._generation_equal(prior["measurement"], replay["measurement"]):
        raise ValueError("The selected saved adapter does not reproduce exact native practice generations")
    limits = {"wall_limit": protocol["resources"]["evaluation_worker_wall_seconds"],
        "cpu_limit": protocol["resources"]["max_cpu_seconds_per_run"],
        "memory_limit": protocol["resources"]["max_memory_bytes"]}
    attempts = []
    qverify._worker("merge", ["--base", base, "--protocol", protocol_path, "--adapter", adapter,
        "--expected-adapter", selected["adapter_sha256"], "--authorization", authorization,
        "--out", merged, "--receipt", receipt], "merge", out, limits, attempts)
    result = {"schema": "weightbridge.phase_iiir.prepared_merge.v1", "passed": True,
        "adapter_sha256": selected["adapter_sha256"], "replay_sha256": file_hash(replay_path),
        "merge_receipt_sha256": file_hash(receipt), "worker_attempts": attempts,
        "lineage_accepted": False, "production_authority": False}
    json_dump_atomic(out / "prepared.json", result)
    return result


def _checked_merged(base, merged, receipt_path, authorization, protocol_path, material, assessment):
    """The same exact merge/lock identity checks used by the Q host verifier."""
    screening = qstudy._replay_screening(assessment, protocol_path)
    ref, lock, receipt = (read(assessment / "finalist-reference.json"),
                         read(assessment / "finalist-lock.json"), read(receipt_path))
    policy = read(authorization)
    digest, files = directory_hash(ref["adapter_directory"])
    representation = load_checkpoint(merged).checkpoint_sha256()
    selected = screening["candidates"][screening["winner"]]
    if (screening["manifest_sha256"] != file_hash(material / "manifest.json")
            or ref["candidate_id"] != screening["winner"] or digest != lock["adapter_sha256"] or files != lock["adapter_files"]
            or any(lock[f"{stage}_gate"] != selected[f"{stage}_gate"] for stage in ("practice", "development"))
            or file_hash(Path(ref["adapter_directory"]) / "adapter.safetensors") != ref["adapter_file_sha256"]
            or policy.get("schema") != "weightbridge.supervised_lora_merge.v1"
            or policy.get("base_checkpoint_sha256") != PARENT or not policy.get("authority")
            or policy.get("stored_dtype") != "BF16"
            or policy.get("merge_math") != "float64_factor_product_then_round_once_into_BF16"
            or policy.get("production_authority", False) is not False or policy.get("lineage_authority", False) is not False
            or receipt.get("schema") != "weightbridge.supervised_lora_merge_receipt.v1"
            or receipt.get("status") != "ISOLATED_CANDIDATE" or receipt.get("base_sha256") != PARENT
            or receipt.get("candidate_sha256") != representation or receipt.get("factor_sha256") != ref["factor_sha256"]
            or receipt.get("base_unchanged") is not True or receipt.get("authorization_sha256") != file_hash(authorization)
            or load_checkpoint(base).checkpoint_sha256() != PARENT):
        raise ValueError("Merged checkpoint differs from its locked adapter and separate authorization")
    return screening, representation


def _merged_development(merged, protocol_path, assessment, identity):
    gates = {}
    for stage in ("practice", "development"):
        candidate = qstudy._launch(merged, protocol_path, assessment / "rosters" / f"{stage}.json",
            assessment / "evaluation" / f"merged-{stage}.json", checkpoint=identity["representation_sha256"],
            stage=f"merged-{stage}")
        parent = read(assessment / "evaluation" / f"{stage}-parent.json")
        gates[stage] = qstudy._gate(parent["measurement"], candidate["measurement"], read(protocol_path), stage)
    result = {"passed": all(g["passed"] for g in gates.values()), "gates": gates, **identity}
    json_dump_atomic(assessment / "merged-development-gate.json", result)
    return result


def qualify_bank_b(base, rank8_merged, rank8_protocol, rank8_assessment, replication_gate,
                   rank4_merged, rank4_receipt, rank4_authorization, rank4_protocol,
                   material, assessment, hidden_path):
    """Three models, identical bank B cases, ordered gates before every opening."""
    base, rank8_merged, rank8_protocol, rank8_assessment, replication_gate, rank4_merged, rank4_receipt, rank4_authorization, rank4_protocol, material, assessment, hidden_path = map(Path,
        (base, rank8_merged, rank8_protocol, rank8_assessment, replication_gate, rank4_merged, rank4_receipt,
         rank4_authorization, rank4_protocol, material, assessment, hidden_path))
    _, protocol = _matched_arms(rank8_protocol, rank4_protocol)
    replicated = validate_replication_gate(protocol, replication_gate)
    _bound_reference(rank8_protocol, rank8_assessment, replicated)
    qstudy._host_output(base, assessment)
    if (assessment / "bank-b.json").exists() or (assessment / "independent.json").exists():
        raise FileExistsError("Bank B evaluation is write-once")
    rank8_sha = replicated["representation_sha256"]
    if load_checkpoint(rank8_merged).checkpoint_sha256() != rank8_sha:
        raise ValueError("Bank B rank-eight reference is not the verified replication")
    pre = development_retention(rank8_protocol, rank4_protocol, rank8_assessment, assessment)
    bank = {"schema": "weightbridge.phase_iiir.bank_b.v1", "passed": False, "reason": "development_retention_failed",
        "study_sha256": protocol["master_protocol_sha256"], "rank8_sha256": rank8_sha,
        "replication_sha256": file_hash(replication_gate), "adapter_development_retention": pre,
        "merged_development_retention": None, "stages": {}, "lineage_accepted": False, "production_authority": False}
    if not pre["passed"]:
        json_dump_atomic(assessment / "bank-b.json", bank)
        return bank
    _, rank4_sha = _checked_merged(base, rank4_merged, rank4_receipt, rank4_authorization,
                                  rank4_protocol, material, assessment)
    identity = {"representation_sha256": rank4_sha, "protocol_sha256": file_hash(rank4_protocol),
                "manifest_sha256": file_hash(material / "manifest.json")}
    independent = {"schema": "weightbridge.supervised_baseline.independent.v1", "disposition": "HOLD", **identity,
        "merge_receipt_sha256": file_hash(rank4_receipt), "merged": None, "preservation_validation": None,
        "selection": None, "hidden": None, "exact_reconstruction": None, "lineage_accepted": False,
        "production_authority": False}
    try:
        merged_gate = _merged_development(rank4_merged, rank4_protocol, assessment, identity)
        independent["merged"] = merged_gate
        if not merged_gate["passed"]:
            bank["reason"] = "rank4_merged_development_failed"
        else:
            post = development_retention(rank8_protocol, rank4_protocol, rank8_assessment, assessment, merged=True)
            bank["merged_development_retention"] = post
            if not post["passed"]:
                bank["reason"] = "merged_development_retention_failed"
            else:
                manifest = read_manifest(material, protocol=rank4_protocol)
                for stage in STAGES:
                    # Only the previous combined pass admits this next bank stage.
                    auth = authorize_stage(assessment, stage, representation_sha256=rank4_sha,
                                           roster_sha256=manifest["splits"][stage]["sha256"])
                    auth_path = assessment / "authorizations" / f"{stage}.json"
                    json_dump_atomic(auth_path, auth)
                    ref_auth_path = assessment / "authorizations" / f"{stage}-rank8.json"
                    json_dump_atomic(ref_auth_path, {**auth, "representation_sha256": rank8_sha,
                        "replication_sha256": file_hash(replication_gate), "paired_reference_only": True})
                    rows = load_split(material, stage, manifest, protocol=rank4_protocol, host_authorized=True,
                                      hidden_path=hidden_path if stage == "hidden" else None)
                    roster = assessment / "private-independent-rosters" / f"{stage}.json"
                    json_dump_atomic(roster, rows)
                    observations = {}
                    for name, model, digest, grant in (("parent", base, PARENT, auth_path),
                            ("rank8", rank8_merged, rank8_sha, ref_auth_path), ("merged", rank4_merged, rank4_sha, auth_path)):
                        observations[name] = qstudy._launch(model, rank4_protocol, roster,
                            assessment / "evaluation" / f"{stage}-{name}.json", checkpoint=digest,
                            stage=stage, authorization=grant)
                    gate = {**qstudy._gate(observations["parent"]["measurement"], observations["merged"]["measurement"], protocol, stage),
                            **identity, "authorization_sha256": file_hash(auth_path)}
                    reference_gate = qstudy._gate(observations["parent"]["measurement"], observations["rank8"]["measurement"], protocol, stage)
                    paired = retention(observations["rank8"]["measurement"], observations["merged"]["measurement"])
                    json_dump_atomic(assessment / f"{stage}-gate.json", gate)
                    independent[stage.replace("-", "_")] = gate
                    stage_result = {"rank4_ordinary": gate, "rank8_reference": reference_gate, "rank4_retention": paired,
                        "passed": all(x["passed"] for x in (gate, reference_gate, paired))}
                    bank["stages"][stage] = stage_result
                    json_dump_atomic(assessment / f"bank-b-{stage}-gate.json", stage_result)
                    if not stage_result["passed"]:
                        bank["reason"] = f"bank_b_{stage}_failed"
                        break
                else:
                    bank.update(passed=True, reason="bank_b_behavior_qualified_exact_replay_required")
        independent["reason"] = qverify.PENDING_REASON if bank["passed"] else bank["reason"]
        verify_sources(protocol["source_sha256"])
        if load_checkpoint(base).checkpoint_sha256() != PARENT:
            raise ValueError("Original parent changed during Bank B evaluation")
    except BaseException as error:
        bank.update(passed=False, reason="incomplete_bank_b_evaluation", error={"type": type(error).__name__, "message": str(error)})
        independent.update(reason=bank["reason"], error=bank["error"])
        if not isinstance(error, Exception):
            json_dump_atomic(assessment / "independent.json", independent)
            json_dump_atomic(assessment / "bank-b.json", bank)
            raise
    json_dump_atomic(assessment / "independent.json", independent)
    json_dump_atomic(assessment / "bank-b.json", bank)
    return bank


def verify_comparison(rank8_merged, rank8_protocol, replication_gate, rank4_protocol,
                      assessment, rank4_verification, out):
    """Replay the paired reference after Q independently reconstructs rank four.

Q's final verifier already replays the parent and reconstructed rank-four
checkpoint. This bounded addition replays rank eight on those *same* Bank B
cases and recomputes the paired decisions, not a new independent sample.
"""
    rank8_merged, rank8_protocol, replication_gate, rank4_protocol, assessment, rank4_verification, out = map(Path,
        (rank8_merged, rank8_protocol, replication_gate, rank4_protocol, assessment, rank4_verification, out))
    qstudy._host_output(rank8_merged, out)
    out.mkdir(parents=True, exist_ok=False)
    wall, cpu = time.perf_counter(), time.process_time()
    attempts = []
    result = {"schema": "weightbridge.phase_iiir.comparison_verification.v1", "passed": False,
        "disposition": "HOLD", "reason": "paired_prior_evidence_refused", "rank4_adaptation_pass": False,
        "rank4_retention_pass": False, "replays": {}, "lineage_accepted": False, "production_authority": False}
    interrupted = None
    try:
        _, protocol = _matched_arms(rank8_protocol, rank4_protocol)
        replication = validate_replication_gate(protocol, replication_gate)
        _bound_reference(rank8_protocol, rank8_protocol.parent / "assessment", replication)
        final = _verified_q(read(rank4_verification))
        bank = read(assessment / "bank-b.json")
        if (bank.get("passed") is not True or bank.get("reason") != "bank_b_behavior_qualified_exact_replay_required"
                or bank.get("study_sha256") != protocol["master_protocol_sha256"]
                or bank.get("replication_sha256") != file_hash(replication_gate)
                or bank.get("rank8_sha256") != replication["representation_sha256"]
                or final["protocol_sha256"] != file_hash(rank4_protocol)
                or final["independent_sha256"] != file_hash(assessment / "independent.json")):
            raise ValueError("Bank B and exact rank-four verification identities disagree")
        reference_sha = replication["representation_sha256"]
        if load_checkpoint(rank8_merged).checkpoint_sha256() != reference_sha:
            raise ValueError("Rank-eight replay checkpoint is not the verified reference")
        result.update(study_sha256=protocol["master_protocol_sha256"], parent_sha256=PARENT,
            rank8_sha256=reference_sha, rank4_sha256=final["representation_sha256"],
            rank4_verification_sha256=file_hash(rank4_verification), bank_b_sha256=file_hash(assessment / "bank-b.json"),
            replication_sha256=file_hash(replication_gate))
        limits = {"wall_limit": protocol["resources"]["evaluation_worker_wall_seconds"],
            "cpu_limit": protocol["resources"]["max_cpu_seconds_per_run"],
            "memory_limit": protocol["resources"]["max_memory_bytes"]}
        result["worker_limits"] = limits
        artifacts = {"bank_b": assessment / "bank-b.json", "rank4_verification": rank4_verification,
                     "replication": replication_gate, "protocol": rank4_protocol}
        for stage in STAGES:
            roster = assessment / "private-independent-rosters" / f"{stage}.json"
            auth = assessment / "authorizations" / f"{stage}-rank8.json"
            before = qstudy._validated_worker(assessment / "evaluation" / f"{stage}-parent.json",
                roster, rank4_protocol, PARENT, None, stage)
            reference = qstudy._validated_worker(assessment / "evaluation" / f"{stage}-rank8.json",
                roster, rank4_protocol, reference_sha, None, stage)
            candidate = qstudy._validated_worker(assessment / "evaluation" / f"{stage}-merged.json",
                roster, rank4_protocol, final["representation_sha256"], None, stage)
            original = bank["stages"][stage]
            ordinary = qstudy._gate(before["measurement"], candidate["measurement"], protocol, stage)
            reference_gate = qstudy._gate(before["measurement"], reference["measurement"], protocol, stage)
            paired = retention(reference["measurement"], candidate["measurement"])
            if (not all(x["passed"] for x in (ordinary, reference_gate, paired))
                    or any(original["rank4_ordinary"][k] != v for k, v in ordinary.items())
                    or original["rank8_reference"] != reference_gate or original["rank4_retention"] != paired):
                raise ValueError(f"Recorded Bank B {stage} outcomes do not independently replay")
            grant = read(auth)
            if (grant.get("representation_sha256") != reference_sha or grant.get("roster_sha256") != file_hash(roster)
                    or grant.get("replication_sha256") != file_hash(replication_gate)):
                raise ValueError("Rank-eight replay authority differs from the locked comparison")
            for label, path in (("roster", roster), ("authorization", auth)):
                artifacts[f"{stage}_{label}"] = path
        initial_hashes = {k: file_hash(p) for k, p in artifacts.items()}
        result["input_sha256"] = initial_hashes
        result["reason"] = "paired_generation_replay_failed"
        for stage in STAGES:
            roster, auth = artifacts[f"{stage}_roster"], artifacts[f"{stage}_authorization"]
            evaluation = out / "private-evaluation" / f"{stage}-rank8.json"
            process = qverify._worker("evaluate", ["--base", rank8_merged, "--protocol", rank4_protocol,
                "--rows", roster, "--out", evaluation, "--expected-checkpoint", reference_sha,
                "--expected-rows", file_hash(roster), "--stage", stage, "--authorization", auth],
                f"{stage}-rank8", out, limits, attempts)
            observed = qstudy._validated_worker(evaluation, roster, rank4_protocol, reference_sha, None, stage)
            qverify._check_worker_body(process, observed["resources"])
            previous = read(assessment / "evaluation" / f"{stage}-rank8.json")
            reconstructed = qstudy._validated_worker(rank4_verification.parent / "private-evaluation" / f"{stage}-reconstructed.json",
                roster, rank4_protocol, final["representation_sha256"], None, stage)
            paired = retention(observed["measurement"], reconstructed["measurement"])
            exact = qverify._generation_equal(previous["measurement"], observed["measurement"])
            result["replays"][stage] = {"exact_rank8_native_generation": exact,
                "observations": len(observed["measurement"]["rows"]), "rank4_retention": paired,
                "evaluation_sha256": file_hash(evaluation), "passed": exact and paired["passed"]}
            process["work"] = {key: observed["measurement"]["resources"][key] for key in
                               ("generate_batches", "batched_decode_steps", "generated_examples", "padded_prompt_token_positions")}
            if not result["replays"][stage]["passed"]:
                raise ValueError(f"Bank B paired {stage} replay failed")
        if {k: file_hash(p) for k, p in artifacts.items()} != initial_hashes:
            raise ValueError("Comparison evidence changed during replay")
        verify_sources(protocol["source_sha256"])
        result.update(passed=True, disposition="VERIFIED_RANK4_ADAPTATION_AND_RETENTION_PASS",
            reason="independent_rank4_reconstruction_and_three_model_paired_replay_passed",
            rank4_adaptation_pass=True, rank4_retention_pass=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        if not isinstance(error, Exception):
            interrupted = error
    observed = [row["execution"]["resources"] for row in attempts if row["execution"] is not None]
    host_cpu = time.process_time()-cpu
    result.update(worker_attempts=attempts, resources={"wall_seconds": time.perf_counter()-wall,
        "host_cpu_seconds": host_cpu, "known_cpu_lower_bound_seconds": host_cpu+sum(r["observed_tree_cpu_seconds"] for r in observed),
        "exact_total_cpu_seconds": None,
        "observed_conservative_peak_rss_bytes": max((r["peak_rss_bytes"] for r in observed), default=None)})
    json_dump_atomic(out / "comparison-verification.json", result)
    if interrupted is not None:
        raise interrupted
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal-replication",))
    for name in ("protocol", "verification", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(seal_replication(args.protocol, args.verification, args.out)["disposition"])
