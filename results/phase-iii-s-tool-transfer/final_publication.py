"""Publish closed Phase S metadata after both explicit finalization transactions.

Report-only: no inference, stage authorization, raw rosters or raw worker reads.
The host supplies a private plan naming each stage's observed lifecycle state.
This checks metadata bindings; the host verifier remains scientifically authoritative.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

ARMS = ("sequential", "original-joint")
STAGES = ("historical-retention", "qualification", "confirmation")
STATES = {"COMPLETED", "FAILED", "UNOPENED", "INTERRUPTED"}
SCHEMA = "weightbridge.phase_iiis.final_publication_plan.v1"
MANIFEST = "ccc217d9139d214ecb241acb3099006a9cc35a0b3f85a6f49784adc3de9a813a"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def validate_plan(plan):
    if set(plan) != {"schema", "arms"} or plan["schema"] != SCHEMA or set(plan["arms"]) != set(ARMS):
        raise ValueError("Plan must explicitly declare both arms without private payloads")
    for arm in ARMS:
        if set(plan["arms"][arm]) != set(STAGES) or any(s not in STATES for s in plan["arms"][arm].values()):
            raise ValueError("Every stage needs an explicit observed lifecycle state")


def collect(study, plan, *, expected_manifest=MANIFEST):
    """Read only fixed metadata filenames selected by the declared states."""
    validate_plan(plan)
    study, gates = Path(study), []
    for arm in ARMS:
        prefix = f"{arm}/protected-v3"
        final = study/prefix/"finalize"
        tx, closure, scored = read(final/"completed.json"), read(final/"formal-stage-closure.json"), read(final/"final-verification.json")
        if (tx.get("schema") != "weightbridge.phase_iiis.private_protected_execution.v3"
                or tx.get("completed") is not True or tx.get("arm") != arm or tx.get("action") != "finalize"
                or tx.get("task_manifest_sha256") != expected_manifest
                or closure.get("schema") != "weightbridge.phase_iiis.formal_stage_closure.v3"
                or scored.get("schema") != "weightbridge.phase_iiis.final_verification.v1"
                or scored.get("arm") != arm or scored.get("task_manifest_sha256") != expected_manifest
                or tx.get("gate_sha256") != digest(final/"final-verification.json")
                or closure.get("frozen_verifier_sha256") != digest(final/"final-verification.json")
                or tx.get("input_sha256", {}).get("formal_stage_closure", {}).get("sha256") != digest(final/"formal-stage-closure.json")
                or set(closure.get("stages", {})) != set(STAGES)):
            raise ValueError("Both arms require closed, hash-bound finalization before final publication")
        ready = all(v.get("completed_valid") is True and v.get("passed") is True for v in closure["stages"].values())
        if (any(type(x.get("passed")) is not bool for x in (tx, closure, scored))
                or closure.get("ready_for_final_pass") is not ready
                or tx["passed"] != scored["passed"] or closure["passed"] != scored["passed"]
                or scored["passed"] and not ready):
            raise ValueError("Final success flags disagree or credit an incomplete lifecycle")
        for label, name in (("final", "final-verification.json"), ("closure", "formal-stage-closure.json"), ("final-transaction", "completed.json")):
            gates.append((f"{arm}-{label}", f"{prefix}/finalize/{name}"))
        for stage in STAGES:
            status, bound = plan["arms"][arm][stage], closure["stages"][stage]
            directory = study/prefix/stage
            if status == "UNOPENED":
                if directory.exists() or bound.get("completed_valid") or bound.get("disposition") != "UNOPENED":
                    raise ValueError("An opened or completed stage cannot be labeled unopened")
                continue
            if status in ("FAILED", "INTERRUPTED"):
                if bound.get("completed_valid") or bound.get("disposition") != "INTERRUPTED_INCOMPLETE":
                    raise ValueError("Declared interrupted stage differs from the closed final audit")
                if status == "FAILED":
                    failed = read(directory/"failed.json")
                    if (failed.get("completed") is not False or failed.get("passed") is not False
                            or bound.get("failed_sha256") != digest(directory/"failed.json")):
                        raise ValueError("Failure record is not bound to the finalized interruption")
                    gates.append((f"{arm}-{stage}-failed", f"{prefix}/{stage}/failed.json"))
                elif bound.get("failed_sha256") is not None:
                    raise ValueError("Use FAILED when the closed audit binds a failure receipt")
                # No partial gate or replay is promoted to a completed result.
                continue
            completed = read(directory/"completed.json")
            if (not bound.get("completed_valid") or completed.get("completed") is not True
                    or completed.get("arm") != arm or completed.get("action") != stage
                    or bound.get("completed_sha256") != digest(directory/"completed.json")
                    or bound.get("gate_sha256") != digest(directory/"gate.json")
                    or completed.get("gate_sha256") != digest(directory/"gate.json")
                    or completed.get("passed") != bound.get("passed")):
                raise ValueError("Declared completed stage differs from its finalized bindings")
            for label, name in ((stage, "gate.json"), (stage+"-transaction", "completed.json")):
                gates.append((f"{arm}-{label}", f"{prefix}/{stage}/{name}"))
            if stage != "historical-retention":
                if (bound.get("replay_receipt_sha256") != digest(directory/"replays.json")
                        or completed.get("replay_receipt_sha256") != digest(directory/"replays.json")):
                    raise ValueError("Completed formal stage lacks its bound replay receipt")
                gates.append((f"{arm}-{stage}-replays", f"{prefix}/{stage}/replays.json"))
    return gates


def publish(study, ledger, plan_path, out, *, expected_manifest=MANIFEST):
    if Path(out).exists():
        raise FileExistsError("Final public snapshots are write-once")
    gates = collect(study, read(plan_path), expected_manifest=expected_manifest)
    spec = importlib.util.spec_from_file_location("public_s_exporter", Path(__file__).with_name("export_outcome.py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cost = module.ledger_summary(module.Reader(study), ledger)
    if cost["unfinished_scope_names"] or not cost["complete_accounting"]:
        raise ValueError("Final publication cannot hide an active or unaccounted model scope")
    # No deferred postdevelopment state in a final snapshot.
    return module.export(study, ledger, out, gates, expected_manifest=expected_manifest)


def self_test():
    plan = {"schema": SCHEMA, "arms": {a: {s: "UNOPENED" for s in STAGES} for a in ARMS}}
    validate_plan(plan)
    for bad in ({**plan, "rows": []}, {**plan, "arms": {"sequential": plan["arms"]["sequential"]}}):
        try:
            validate_plan(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Implicit or data-bearing plan accepted")
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        write = lambda p, x: p.write_text(json.dumps(x), encoding="utf-8")
        manifest = {"source_sha256": {"fixture.py": "a"*64}}
        write(root/"manifest.json", manifest)
        identity = digest(root/"manifest.json")
        for arm in ARMS:
            directory = root/arm/"protected-v3/finalize"; directory.mkdir(parents=True)
            write(root/arm/"protocol.json", {"schema": "weightbridge.phase_iiis.protocol.v1", "arm": arm,
                "task_manifest_sha256": identity, "source_sha256": manifest["source_sha256"]})
            scored = {"schema": "weightbridge.phase_iiis.final_verification.v1", "arm": arm,
                "task_manifest_sha256": identity, "passed": False}
            write(directory/"final-verification.json", scored)
            closure = {"schema": "weightbridge.phase_iiis.formal_stage_closure.v3",
                "passed": False, "ready_for_final_pass": False,
                "frozen_verifier_sha256": digest(directory/"final-verification.json"),
                "stages": {s: {"completed_valid": False, "disposition": "UNOPENED"} for s in STAGES}}
            write(directory/"formal-stage-closure.json", closure)
            write(directory/"completed.json", {"schema": "weightbridge.phase_iiis.private_protected_execution.v3",
                "arm": arm, "action": "finalize", "task_manifest_sha256": identity, "completed": True,
                "passed": False, "gate_sha256": digest(directory/"final-verification.json"),
                "input_sha256": {"formal_stage_closure": {"sha256": digest(directory/"formal-stage-closure.json")}}})
        assert len(collect(root, plan, expected_manifest=identity)) == 6
        final = root/"sequential/protected-v3/finalize"
        closure = read(final/"formal-stage-closure.json")
        for stage in ("historical-retention", "qualification"):
            directory = root/"sequential/protected-v3"/stage; directory.mkdir()
            gate = {"schema": "weightbridge.phase_iiis.behavior_gate.v1", "stage": stage, "arm": "sequential",
                "task_manifest_sha256": identity, "passed": stage == "historical-retention",
                "disposition": "INSUFFICIENT_IMPROVEMENT_HEADROOM", "rows": [{"prompt": "PRIVATE_ROW"}],
                "private_seed": "PRIVATE_SEED", "lost_correct": ["PRIVATE_ID"],
                "input_sha256": {"worker": {"path": "C:/PRIVATE/output.json", "sha256": "b"*64}}}
            write(directory/"gate.json", gate)
            tx = {"schema": "weightbridge.phase_iiis.private_protected_execution.v3", "completed": True,
                "arm": "sequential", "action": stage, "passed": gate["passed"], "gate_sha256": digest(directory/"gate.json")}
            if stage == "qualification":
                write(directory/"replays.json", {"schema": "weightbridge.phase_iiis.private_stage_replay.v1", "passed": True})
                tx["replay_receipt_sha256"] = digest(directory/"replays.json")
            write(directory/"completed.json", tx)
            closure["stages"][stage] = {"completed_valid": True, "passed": tx["passed"],
                "completed_sha256": digest(directory/"completed.json"), "gate_sha256": tx["gate_sha256"],
                "replay_receipt_sha256": tx.get("replay_receipt_sha256"), "disposition": gate["disposition"]}
            plan["arms"]["sequential"][stage] = "COMPLETED"
        write(final/"formal-stage-closure.json", closure)
        tx = read(final/"completed.json")
        tx["input_sha256"]["formal_stage_closure"]["sha256"] = digest(final/"formal-stage-closure.json")
        write(final/"completed.json", tx)
        assert len(collect(root, plan, expected_manifest=identity)) == 11
        ledger, public = root/"ledger", root/"public"
        ledger.mkdir()
        private_plan = root/"PRIVATE_PLAN.json"; write(private_plan, plan)
        assert publish(root, ledger, private_plan, public, expected_manifest=identity)["passed"]
        availability = read(public/"stage-availability.json")
        assert availability["sealed_confirmation"] is None
        assert availability["protected_transactions"]["qualification"]["sequential"]["completed"] is True
        assert availability["protected_transactions"]["qualification"]["sequential"]["passed"] is False
        assert all("PRIVATE" not in p.read_text() for p in public.glob("*.json"))
        assert not (public/private_plan.name).exists()
        # Even recomputed hashes cannot turn missing confirmation into final PASS.
        final_paths = [final/n for n in ("final-verification.json", "formal-stage-closure.json", "completed.json")]
        saved = [p.read_bytes() for p in final_paths]
        scored = read(final_paths[0]); scored["passed"] = True; write(final_paths[0], scored)
        closure = read(final_paths[1]); closure["passed"] = True
        closure["frozen_verifier_sha256"] = digest(final_paths[0]); write(final_paths[1], closure)
        tx = read(final_paths[2]); tx["passed"] = True; tx["gate_sha256"] = digest(final_paths[0])
        tx["input_sha256"]["formal_stage_closure"]["sha256"] = digest(final_paths[1]); write(final_paths[2], tx)
        try:
            collect(root, plan, expected_manifest=identity)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete lifecycle reported final PASS with self-consistent hashes")
        for p, payload in zip(final_paths, saved): p.write_bytes(payload)
        qgate = root/"sequential/protected-v3/qualification/gate.json"
        original_gate = qgate.read_bytes(); qgate.write_bytes(b"{}")
        try:
            collect(root, plan, expected_manifest=identity)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed scored gate accepted")
        qgate.write_bytes(original_gate)
        (root/"original-joint/protected-v3/qualification").mkdir()
        try:
            collect(root, plan, expected_manifest=identity)
        except ValueError:
            pass
        else:
            raise AssertionError("Opened stage relabeled unopened")
    return {"explicit_both_arm_states": "PASS", "no_data_payload": "PASS", "closed_failed_finalization": "PASS",
        "completed_headroom_failure_with_unopened_confirmation": "PASS", "metadata_hash_tamper": "PASS",
        "opened_is_not_unopened": "PASS", "end_to_end_redaction_and_private_plan_exclusion": "PASS",
        "final_success_requires_complete_lifecycle": "PASS",
        "model_execution": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("study", "ledger", "plan", "out"):
        parser.add_argument("--"+arg, type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    elif all((args.study, args.ledger, args.plan, args.out)):
        print(json.dumps(publish(args.study, args.ledger, args.plan, args.out)))
    else:
        parser.error("Choose --self-test or --study/--ledger/--plan/--out")
