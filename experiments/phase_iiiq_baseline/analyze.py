"""Read-only post-hoc receipt summary; never loads a study roster or a model.

This module is outside the frozen execution closure. It cannot fit, rank,
authorize, or promote a candidate. Missing observations remain null.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from weightbridge.core import file_hash, json_dump_atomic
from .integrity import PARENT, read


def summarize(protocol_path, *, smoke_root=None, smoke_admission=None, run_roots=(), assessment_root=None):
    protocol_path = Path(protocol_path)
    protocol = read(protocol_path)
    if protocol.get("checkpoint_sha256") != PARENT:
        raise ValueError("Report requires the frozen Qwen parent identity")
    protocol_sha = file_hash(protocol_path)
    inputs = {"protocol": protocol_sha}

    def receipt(root, name, label):
        path = Path(root) / name
        if not path.is_file():
            return None
        value = read(path)
        recorded = value.get("protocol_sha256", value.get("identity", {}).get("protocol_sha256",
            value.get("host_metadata", {}).get("protocol_sha256")))
        if recorded is not None and recorded != protocol_sha:
            raise ValueError(f"Receipt protocol identity differs: {label}/{name}")
        inputs[f"{label}/{name}"] = file_hash(path)
        return value

    def training(root, label, *, process_tree=False):
        completed = receipt(root, "completed.json", label)
        failed = receipt(root, "failed.json", label)
        final = completed or failed
        inventory = receipt(root, "inventory.json", label)
        steps = [receipt(root, f"steps/{path.name}", label) for path in sorted((Path(root)/"steps").glob("*.json"))]
        checkpoints = []
        for path in sorted((Path(root)/"checkpoints").glob("step-*/receipt.json")):
            saved = receipt(root, path.relative_to(root).as_posix(), label)
            artifact = path.parent / "adapter.safetensors"
            digest = file_hash(artifact) if artifact.is_file() else None
            if digest is not None and digest != saved.get("sha256"):
                raise ValueError("Saved adapter hash differs from its checkpoint receipt")
            effective = saved.get("effective_update", {})
            checkpoints.append({"step": saved.get("step"), "adapter_sha256": digest,
                "factor_sha256": saved.get("factor_sha256"), "artifact_verified": digest is not None,
                **{key: effective.get(key) for key in ("proposed_update_norm", "stored_update_norm", "changed_entries",
                    "parent_selected_weight_norm", "relative_stored_update_norm", "rounding_residual_norm")}})
        process = receipt(root.parent, f"{root.name}-process.json", label) if process_tree else None
        cost = None
        if process:
            if (process.get("schema") != "weightbridge.supervised_baseline.training_process_tree.v1"
                    or process.get("attempt_directory", root.name) != root.name
                    or process.get("training_completed_receipt_sha256") != (file_hash(root/"completed.json") if completed else None)
                    or process.get("log_sha256") != file_hash(root.parent/f"{root.name}.log")):
                raise ValueError("Training process resource receipt differs from its owned attempt")
            observed = process["execution"]["resources"]
            cost = {"wall_seconds": observed.get("wall_seconds"), "cpu_seconds": None,
                "known_cpu_lower_bound_seconds": observed.get("cpu_seconds"),
                "conservative_cpu_upper_bound_seconds": None,
                "peak_rss_bytes": max(observed.get("peak_rss_bytes") or 0, observed.get("known_peak_rss_floor_bytes") or 0),
                "cpu_accounting_complete_for_observed_processes": observed.get("cpu_accounting_complete_for_observed_processes"),
                "source": "owned_process_tree", "coverage_scope": process["execution"].get("coverage_scope"),
                "memory_scope": process["execution"].get("memory_scope")}
        elif process_tree and final and final.get("resources"):
            body = final["resources"]
            cost = {"wall_seconds": body.get("wall_seconds"), "cpu_seconds": None,
                "known_cpu_lower_bound_seconds": body.get("cpu_seconds"), "conservative_cpu_upper_bound_seconds": None,
                "peak_rss_bytes": body.get("peak_rss_bytes"), "source": "training_worker_body_only"}
        state = process.get("status") if process else final.get("status") if final else "UNMEASURED_OR_INCOMPLETE"
        return {"status": state, "attempt_directory": root.name, "process_receipt": process, "process_cost": cost,
            "smoke": final.get("smoke") if final else None,
            "run_index": final.get("run_index") if final else (process or {}).get("run_index"),
            "counts": final.get("counts") if final else (steps[-1].get("counts") if steps else None),
            "resources": final.get("resources") if final else None, "base_unchanged": final.get("base_unchanged") if final else None,
            "all_factors_changed": final.get("all_factors_changed") if final else None,
            "all_factors_received_nonzero_gradient": final.get("all_factors_received_nonzero_gradient") if final else None,
            "inventory": {key: (inventory or {}).get(key) for key in ("trainable_parameters", "total_model_parameters",
                "trainable_fraction", "training_examples", "unique_answer_tokens")},
            "steps": [{key: row.get(key) for key in ("step", "loss", "learning_rate", "step_wall_seconds",
                "step_cpu_seconds", "unrounded_effective_update_norm")} for row in steps], "checkpoints": checkpoints}

    smoke, smoke_cost = None, None
    if smoke_root is not None:
        smoke_root = Path(smoke_root)
        gate = receipt(smoke_root, "smoke-gate.json", "smoke")
        zero = receipt(smoke_root, "zero/zero-init-check.json", "smoke")
        fresh = [receipt(smoke_root, f"fresh-{i}/fresh-adapter-check.json", "smoke") for i in (1, 2)]
        smoke = {"gate": gate, "training": training(Path(smoke_root)/"training", "smoke/training"),
            "matched_loss_before": (zero or {}).get("matched_loss_before", {}).get("mean_token_nll"),
            "matched_loss_after": [(row or {}).get("matched_loss_after", {}).get("mean_token_nll") for row in fresh],
            "scope": "Disposable training-set pipeline test; not independent neural improvement."}
        admission_path = Path(smoke_admission) if smoke_admission is not None else smoke_root.parent/"smoke-admission.json"
        admission = receipt(admission_path.parent, admission_path.name, "smoke-admission")
        if admission:
            if admission.get("source_gate_sha256") != file_hash(smoke_root/"smoke-gate.json"):
                raise ValueError("Smoke admission resource audit binds another raw gate")
            audit = admission["cost_audit"]
            for name, digest in audit.get("worker_execution_sha256", {}).items():
                execution = receipt(smoke_root, f"{name}/execution.json", "smoke")
                if execution is None or file_hash(smoke_root/name/"execution.json") != digest:
                    raise ValueError("Audited smoke worker resource receipt changed")
            smoke_cost = {"wall_seconds": audit.get("wall_seconds"), "cpu_seconds": audit.get("total_exact_cpu_seconds"),
                "known_cpu_lower_bound_seconds": audit.get("known_cpu_lower_bound_seconds"),
                "conservative_cpu_upper_bound_seconds": audit.get("conservative_cpu_upper_bound_seconds"),
                "peak_rss_bytes": audit.get("peak_known_rss_bytes"), "peak_rss_is_lower_bound": True,
                "cpu_accounting_complete": False, "source": "independently_replayed_smoke_admission"}
        elif gate:
            actual = [smoke["training"]["resources"]]
            for name in ("zero", "fresh-1", "fresh-2"):
                execution = receipt(smoke_root, f"{name}/execution.json", "smoke")
                actual.append((execution or {}).get("resources"))
            measured = [row for row in actual if row is not None]
            raw = gate.get("resources", {})
            lower = sum(row["cpu_seconds"] for row in measured)+raw.get("host_cpu_seconds", 0)+sum(
                row.get("cpu_seconds", 0) for row in gate.get("worker_processes", [])) if measured else None
            smoke_cost = {"wall_seconds": raw.get("wall_seconds"), "cpu_seconds": None,
                "known_cpu_lower_bound_seconds": lower, "conservative_cpu_upper_bound_seconds": None,
                "peak_rss_bytes": max((row["peak_rss_bytes"] for row in measured), default=None),
                "peak_rss_is_lower_bound": True, "cpu_accounting_complete": False,
                "source": "distinct_worker_body_receipts_plus_host_and_launcher_lower_bound"}
        smoke.update(admission=admission, corrected_resources=smoke_cost,
            raw_resource_warning="The original venv-launcher CPU/RSS aggregation excludes model descendants. Its complete-accounting flag is invalid and is not used in report totals. Original evidence is preserved.")
        smoke["work"] = {key: sum(row.get(key, 0) for row in [zero, *fresh] if row is not None)
            if zero is not None and all(fresh) else None for key in
            ("explicit_forward_calls", "backward_calls", "batched_decode_steps", "generated_response_records")}
    runs = [training(Path(root), f"training-attempt-{i}", process_tree=True) for i, root in enumerate(run_roots)]
    assessment = independent = None
    evaluation_costs, evaluation_work = [], []
    if assessment_root is not None:
        assessment = receipt(assessment_root, "assessment.json", "assessment")
        independent = receipt(assessment_root, "independent.json", "assessment")
        # Only already-scored practice/development workers; no roster or raw
        # protected-stage observations are opened by the reporting subsystem.
        if assessment:
            if any(not name or any(part in name for part in ("/", "\\", ".."))
                   for name in assessment.get("candidates", {})):
                raise ValueError("Reported candidate identities cannot select additional filesystem paths")
            names = [f"{stage}-parent.json" for stage in ("practice", "development")]
            names += [f"{stage}-{name}.json" for name, row in assessment.get("candidates", {}).items()
                      if not row.get("duplicate_of") for stage in ("practice", "development")]
            for name in names:
                worker = receipt(assessment_root, f"evaluation/{name}", "assessment")
                if worker:
                    evaluation_costs.append(worker.get("resources", {}))
                    measurement = worker.get("measurement", {})
                    evaluation_work.append({"batched_decode_steps": measurement.get("resources", {}).get("batched_decode_steps"),
                        "generated_response_records": worker.get("summary", {}).get("presentations")})
    costs = []
    if smoke_cost:
        costs.append({"phase": "disposable_smoke", **smoke_cost})
    costs += [{"phase": row["attempt_directory"], **row["process_cost"]} for row in runs if row["process_cost"]]
    if assessment:
        host = assessment.get("resources", {})
        costs.append({"phase": "practice_development_assessment", "wall_seconds": host.get("host_wall_seconds_including_workers"),
            "cpu_seconds": None, "known_cpu_lower_bound_seconds": host.get("host_cpu_seconds", 0)+sum(
                row.get("cpu_seconds", 0) for row in evaluation_costs), "conservative_cpu_upper_bound_seconds": None,
            "peak_rss_bytes": max((row.get("peak_rss_bytes", 0) for row in evaluation_costs), default=None),
            "cpu_accounting_complete": False})
    work = [{"phase": f"main_training_{i}", **row["counts"]} for i, row in enumerate(runs) if row["counts"]]
    if smoke:
        if smoke["training"]["counts"]:
            work.append({"phase": "smoke_training", **smoke["training"]["counts"]})
        work.append({"phase": "smoke_measurement_workers", **smoke["work"]})
    work += [{"phase": f"practice_development_worker_{i}", **row} for i, row in enumerate(evaluation_work)]
    report_code = {path.name: file_hash(path) for path in (Path(__file__), Path(__file__).with_name("figures.py")) if path.is_file()}
    return {"schema": "weightbridge.supervised_baseline.report.v1", "reporting_only": True,
        "checkpoint_sha256": protocol.get("checkpoint_sha256"), "protocol_sha256": protocol_sha,
        "source_manifest_sha256": hashlib.sha256(json.dumps(protocol.get("source_sha256", {}), sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(), "report_code_sha256": report_code, "input_sha256": inputs,
        "smoke": smoke, "main_runs": runs, "assessment": assessment, "independent": independent,
        "stages": {name: (independent or {}).get(name) for name in
            ("merged", "preservation_validation", "selection", "hidden", "exact_reconstruction")},
        "disposition": "HOLD", "qualification": "No PASS is inferred from training, smoke, or adapter execution.",
        "lineage_accepted": False, "production_authority": False, "cost_components": costs, "work_components": work,
        "measured_cost_totals": {"wall_seconds": sum(row["wall_seconds"] for row in costs) if costs and all(
            row.get("wall_seconds") is not None for row in costs) else None,
            "cpu_seconds": sum(row["cpu_seconds"] for row in costs) if costs and all(
            row.get("cpu_seconds") is not None for row in costs) else None,
            "known_cpu_lower_bound_seconds": sum(row.get("known_cpu_lower_bound_seconds", row.get("cpu_seconds"))
                for row in costs) if costs and all(row.get("known_cpu_lower_bound_seconds", row.get("cpu_seconds")) is not None for row in costs) else None,
            "conservative_cpu_upper_bound_seconds": sum(row.get("conservative_cpu_upper_bound_seconds", row.get("cpu_seconds"))
                for row in costs) if costs and all(row.get("conservative_cpu_upper_bound_seconds", row.get("cpu_seconds")) is not None for row in costs) else None,
            "peak_rss_bytes": max((row.get("peak_rss_bytes") or 0 for row in costs), default=None)},
        "cost_scope": "Recorded research components only. Smoke uses audited bounds or worker-body lower bounds. Main attempts use their owned process trees where available, including measured failed attempts, with no added inner training cost. Assessment adds host CPU to distinct practice/development workers. CPU exact totals remain null when process coverage is incomplete. Protected independent workers, tests and setup are excluded. No total concurrent-RSS claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--smoke-admission", type=Path)
    parser.add_argument("--runs", nargs="*", type=Path, default=[])
    parser.add_argument("--assessment", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Post-hoc summaries are write-once")
    json_dump_atomic(args.out, summarize(args.protocol, smoke_root=args.smoke, smoke_admission=args.smoke_admission, run_roots=args.runs,
                                        assessment_root=args.assessment))
