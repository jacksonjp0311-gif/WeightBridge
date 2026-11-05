"""Bounded, fixed study stages using the existing host evaluators."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

from experiments.phase_iiiq_baseline import study as qstudy, verify_final as qverify
from experiments.phase_iiiq_baseline.run_training import launch_owned_tree, _temporary_sleep_request
from weightbridge.core import file_hash, json_dump_atomic
from .integrity import ROOT, read, verify_sources
from . import study


def validate_stage_execution(root, arm, action, protocol=None):
    """Recheck measured process evidence rather than trusting a pass flag."""
    protocol = Path(protocol) if protocol is not None else Path(root)/arm/"protocol.json"
    config = read(protocol)
    prefix = Path(root)/"execution"/f"{arm}-{action}"
    recorded = read(prefix.with_suffix(".json"))
    worker_path = prefix.with_suffix(".worker.json")
    observed, sleep = recorded.get("execution") or {}, recorded.get("temporary_sleep_request") or {}
    usage, bounds = observed.get("resources") or {}, config["stage_resources"][action]
    resource_pass = all(math.isfinite(usage.get(key, float("inf")))
        and 0 <= usage[key] <= bounds[limit] for key, limit in (
            ("wall_seconds", "wall_seconds"), ("cpu_seconds", "cpu_seconds"), ("peak_rss_bytes", "memory_bytes")))
    if (recorded.get("schema") != "weightbridge.phase_iiir.stage_execution.v1"
            or recorded.get("passed") is not True or recorded.get("within_resource_limits") is not True
            or recorded.get("arm") != arm or recorded.get("action") != action
            or recorded.get("protocol_sha256") != file_hash(protocol)
            or recorded.get("outcome_sha256") != file_hash(worker_path)
            or recorded.get("outcome") != read(worker_path)
            or recorded["outcome"].get("passed") is not True or recorded.get("error") is not None
            or observed.get("exit_code") != 0 or observed.get("budget_failure") is not None
            or not resource_pass or usage.get("cpu_accounting_complete_for_observed_processes") is not True
            or (sleep.get("requested") and (sleep.get("succeeded") is not True or sleep.get("released") is not True))):
        raise ValueError("Enclosing study stage did not qualify")
    verify_sources(config["source_sha256"])
    return recorded


def previous_stage(root, arm, action, protocol):
    """An inner success cannot bypass an enclosing process-budget refusal."""
    previous = {"qualify": "assess", "verify": "qualify"}.get(action)
    if previous:
        return validate_stage_execution(root, arm, previous, protocol)


def worker(base, root, arm, action):
    base, root = Path(base), Path(root)
    local, other = root/arm, root/"rank8"
    protocol, material = local/"protocol.json", local/"material"
    assessment, merged = local/"assessment", local/"merged-checkpoint"
    merge_receipt, merge_auth = local/"merge-receipt.json", local/"merge-authorization.json"
    replication = root/"replication-verification.json"
    verify_sources(read(protocol)["source_sha256"])
    from .training import validate_main_execution
    validate_main_execution(local)
    previous_stage(root, arm, action, protocol)
    if arm == "rank4":
        study.validate_replication_gate(read(protocol), replication)
    if action == "assess":
        result = qstudy.assess(base, protocol, material, assessment, [local/"training"])
        passed = bool(result["winner"]) and not result["errors"]
    elif action == "qualify":
        if arm == "rank4":
            paired = study.development_retention(other/"protocol.json", protocol, other/"assessment", assessment)
            json_dump_atomic(local/"adapter-development-retention.json", paired)
            if not paired["passed"]:
                return {"passed": False, "reason": "rank4_development_retention_failed", "bank_b_opened": False}
        study.prepare_merge(base, protocol, assessment, merge_auth, merged, merge_receipt, local/"prepare-merge")
        if arm == "rank8":
            result = qstudy.finish(base, merged, merge_receipt, merge_auth, protocol, material, assessment,
                hidden_path=read(root/"freeze.json")["arms"][arm]["hidden_path"])
            passed = result.get("reason") == qverify.PENDING_REASON
        else:
            result = study.qualify_bank_b(base, other/"merged-checkpoint", other/"protocol.json", other/"assessment",
                replication, merged, merge_receipt, merge_auth, protocol, material, assessment,
                read(root/"freeze.json")["arms"][arm]["hidden_path"])
            passed = result.get("passed") is True
    elif action == "verify":
        result = qverify.run(base, merged, merge_receipt, merge_auth, protocol, material, assessment, local/"final-verification")
        passed = result.get("passed") is True
        if passed:
            if arm == "rank8":
                result = study.seal_replication(protocol, local/"final-verification/final-verification.json", replication)
            else:
                result = study.verify_comparison(other/"merged-checkpoint", other/"protocol.json", replication,
                    protocol, assessment, local/"final-verification/final-verification.json", local/"comparison-verification")
            passed = result.get("passed") is True
    else:
        raise ValueError("Unknown frozen stage")
    return {"passed": passed, "reason": result.get("reason"), "disposition": result.get("disposition"),
        "winner": result.get("winner"), "production_authority": False, "lineage_authority": False}


def run(base, root, arm, action):
    root = Path(root).resolve()
    if root.is_relative_to(ROOT) or root.is_relative_to(Path(base).resolve()):
        raise ValueError("Study stages require private storage outside the checkout and parent")
    protocol = root/arm/"protocol.json"
    config = read(protocol)
    verify_sources(config["source_sha256"])
    destination = root/"execution"
    destination.mkdir(exist_ok=True)
    prefix = destination/f"{arm}-{action}"
    if any(prefix.with_suffix(ext).exists() for ext in (".json", ".log", ".worker.json", ".claim.json")):
        raise FileExistsError("Study action already attempted; evidence cannot be overwritten")
    bounds = config["stage_resources"][action]
    json_dump_atomic(prefix.with_suffix(".claim.json"), {"protocol_sha256": file_hash(protocol), "action": action,
        "arm": arm, "source_sha256": file_hash(__file__), "limits": bounds})
    argv = [sys.executable, "-m", "experiments.phase_iiir_replication.execute", action,
        "--base", str(base), "--study", str(root), "--arm", arm, "--worker"]
    observed, failure, sleep = None, None, {}
    try:
        with _temporary_sleep_request() as sleep:
            if sleep["requested"] and not sleep["succeeded"]:
                raise RuntimeError("Temporary sleep prevention unavailable")
            observed = launch_owned_tree(argv, prefix.with_suffix(".log"), wall_limit=bounds["wall_seconds"],
                cpu_limit=bounds["cpu_seconds"], memory_limit=bounds["memory_bytes"])
        verify_sources(config["source_sha256"])
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    outcome = read(prefix.with_suffix(".worker.json")) if prefix.with_suffix(".worker.json").exists() else None
    usage = observed["resources"] if observed else {}
    within_limits = bool(observed) and all(math.isfinite(usage.get(key, float("inf")))
        and 0 <= usage[key] <= bounds[limit] for key, limit in (
            ("wall_seconds", "wall_seconds"), ("cpu_seconds", "cpu_seconds"), ("peak_rss_bytes", "memory_bytes")))
    result = {"schema": "weightbridge.phase_iiir.stage_execution.v1", "arm": arm, "action": action,
        "protocol_sha256": file_hash(protocol), "execution": observed, "temporary_sleep_request": sleep,
        "error": failure, "within_resource_limits": within_limits,
        "outcome": outcome, "passed": failure is None and within_limits and observed["exit_code"] == 0
            and observed["budget_failure"] is None and usage["cpu_accounting_complete_for_observed_processes"]
            and bool(outcome and outcome.get("passed")) and (not sleep["requested"] or sleep["released"] is True),
        "outcome_sha256": file_hash(prefix.with_suffix(".worker.json")) if outcome else None,
        "no_promotion_authority": True}
    json_dump_atomic(prefix.with_suffix(".json"), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("assess", "qualify", "verify"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--arm", choices=("rank8", "rank4"), required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        try:
            result = worker(args.base, args.study, args.arm, args.action)
        except Exception as error:
            result = {"passed": False, "error": {"type": type(error).__name__, "message": str(error)}}
        json_dump_atomic(args.study/"execution"/f"{args.arm}-{args.action}.worker.json", result)
    else:
        result = run(args.base, args.study, args.arm, args.action)
    print({"passed": result["passed"], "action": args.action, "arm": args.arm}, flush=True)
    raise SystemExit(0 if result["passed"] else 1)
