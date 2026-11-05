"""Rank-specific disposable Q smoke with prospective process-tree accounting.

The zero-init, causal loss, gradients, fresh reload and disabled-parent checks
are the existing Q checks. This host corrects launcher-only cost accounting
prospectively rather than reusing Q's post-hoc admission exception.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from weightbridge.core import file_hash, json_dump_atomic
from experiments.phase_iiiq_baseline.integrity import PARENT, ROOT, read, verify_sources
from experiments.phase_iiiq_baseline.data import PopulationSpec, validate
from experiments.phase_iiiq_baseline.smoke_checks import _IDENTITY, exact_predictions, validate_smoke
from experiments.phase_iiiq_baseline.run_training import launch_owned_tree, _temporary_sleep_request
from .training import claim_launch, validate_recipe


def validate_gate(protocol, authorization, bundle, rows, gate_path):
    """Independently replay the prospective smoke admission without inference."""
    from weightbridge.adaptation.training_policy import TrainingAuthorization
    from .training import verify_authorization
    protocol, authorization, bundle, rows, gate_path = map(Path, (protocol, authorization, bundle, rows, gate_path))
    config, gate, roster, material = read(protocol), read(gate_path), read(rows), read(bundle)
    auth = TrainingAuthorization.from_dict(read(authorization))
    rank = verify_authorization(config, auth, smoke=True)
    verify_sources(config["source_sha256"])
    validate(roster, "smoke", PopulationSpec(target_families=4))
    expected = {"protocol_sha256": file_hash(protocol), "authorization_sha256": file_hash(authorization),
        "training_bundle_sha256": file_hash(bundle), "roster_sha256": file_hash(rows), "parent_sha256": PARENT,
        "rank": rank, "alpha": rank*2}
    root = gate_path.parent
    if (gate.get("schema") != "weightbridge.phase_iiir.smoke_gate.v1" or gate.get("passed") is not True
            or gate.get("error") is not None or any(gate.get(key) != value for key, value in expected.items())
            or material["rows"] != roster or material["split_ids"] != ["smoke"]
            or material["protocol_sha256"] != file_hash(protocol) or auth.training_bundle_sha256 != file_hash(bundle)):
        raise ValueError("Smoke admission identity or raw execution failed")
    if read(root/"started.json") != {**expected, "launch_claim_sha256": gate["launch_claim_sha256"]}:
        raise ValueError("Smoke start identity changed")
    if file_hash(protocol.parent/"smoke-launch-claim.json") != gate["launch_claim_sha256"]:
        raise ValueError("Smoke launch claim changed")
    paths = {"zero": "zero/zero-init-check.json", "training": "training/completed.json",
             "fresh-1": "fresh-1/fresh-adapter-check.json", "fresh-2": "fresh-2/fresh-adapter-check.json"}
    if {name: file_hash(root/path) for name, path in paths.items()} != gate["worker_receipt_sha256"]:
        raise ValueError("Smoke worker receipt changed")
    zero, completed, first, second = (read(root/path) for path in paths.values())
    for key in ("protocol_sha256", "authorization_sha256", "parent_sha256"):
        if completed.get(key) != expected[key]:
            raise ValueError("Smoke completed worker identity differs")
    if (completed.get("smoke") is not True or completed.get("bundle_sha256") != file_hash(bundle)
            or completed.get("run_index") != 0):
        raise ValueError("Only this disposable training run may supply smoke evidence")
    adapter = Path(completed["checkpoints"][-1])/"adapter.safetensors"
    if not adapter.resolve().is_relative_to((root/"training").resolve()) or file_hash(adapter) != gate["adapter_sha256"]:
        raise ValueError("Smoke adapter identity differs")
    for receipt in (zero, first, second):
        if any(receipt.get(key) != value for key, value in (("checkpoint_sha256", PARENT),
                ("protocol_sha256", file_hash(protocol)), ("roster_sha256", file_hash(rows)), ("base_unchanged", True))):
            raise ValueError("Smoke inference worker identity or source restoration differs")
    for receipt in (first, second):
        if receipt.get("adapter_sha256") != gate["adapter_sha256"]:
            raise ValueError("Smoke fresh process used another adapter")
    for measurement in (zero["parent_measurement"], zero["zero_init_measurement"], first["trained_measurement"],
                        first["disabled_measurement"], second["trained_measurement"], second["disabled_measurement"]):
        for observed, frozen in zip(measurement["rows"], roster, strict=True):
            if any(observed.get(key) != frozen.get(key) for key in _IDENTITY):
                raise ValueError("Smoke semantic measurements changed the frozen roster")
    recomputed = validate_smoke(zero["parent_measurement"], first["trained_measurement"],
        second["trained_measurement"], first["disabled_measurement"], {**completed,
            "zero_init_measurement": zero["zero_init_measurement"], "matched_loss_before": zero["matched_loss_before"],
            "matched_loss_after": first["matched_loss_after"]}, gates=config["gates"]["smoke"])
    if not recomputed["passed"] or any(gate["checks"].get(key) != value for key, value in recomputed["checks"].items()):
        raise ValueError("Smoke quality does not reproduce its recorded gate")
    if not exact_predictions(zero["parent_measurement"], second["disabled_measurement"]):
        raise ValueError("Second fresh process did not restore the parent")
    processes = []
    for name, path in paths.items():
        process = read(root/f"{name}-process.json")
        observed, body = process["execution"], read(root/path)["resources"]
        if (process["worker"] != name or process["log_sha256"] != file_hash(root/f"{name}.log")
                or observed["exit_code"] != 0 or observed["budget_failure"] is not None
                or observed["resources"]["cpu_accounting_complete_for_observed_processes"] is not True
                or observed["resources"]["observed_tree_cpu_seconds"]+.01 < body["cpu_seconds"]
                or observed["resources"]["peak_rss_bytes"] < body["peak_rss_bytes"]):
            raise ValueError("Smoke process-tree evidence does not cover completed workers")
        processes.append(process)
    usage, limits = gate["resources"], config["resources"]
    children = sum(p["execution"]["resources"]["observed_tree_cpu_seconds"] for p in processes)
    peak = max(p["execution"]["resources"]["peak_rss_bytes"] for p in processes)
    sleep = gate["temporary_sleep_request"]
    if (processes != gate["worker_processes"] or usage["observed_tree_cpu_seconds"] != children
            or usage["cpu_seconds"] != children+usage["host_cpu_seconds"] or usage["peak_rss_bytes"] != peak
            or usage["wall_seconds"] > limits["smoke_max_wall_seconds"]
            or usage["cpu_seconds"] > limits["smoke_max_cpu_seconds"] or peak > limits["max_memory_bytes"]
            or (sleep["requested"] and (sleep["succeeded"] is not True or sleep["released"] is not True))
            or not all(gate["checks"].values())):
        raise ValueError("Smoke aggregate cost or sleep release did not satisfy its frozen admission")
    return gate


def run(base, protocol, authorization, bundle, rows, out):
    base, protocol, authorization, bundle, rows, out = map(Path, (base, protocol, authorization, bundle, rows, out))
    config, training, roster = read(protocol), read(bundle), read(rows)
    rank = validate_recipe(config)
    validate(roster, "smoke", PopulationSpec(target_families=4))
    if (training.get("rows") != roster or training.get("split_ids") != ["smoke"]
            or out.resolve().is_relative_to(base.resolve())):
        raise ValueError("Disposable smoke must use its own exact training roster outside the parent")
    claim, _ = claim_launch(protocol, authorization, bundle, out, smoke=True)
    out.mkdir(parents=True, exist_ok=False)
    wall, host_cpu = time.perf_counter(), time.process_time()
    limits, workers = config["resources"], []
    gate, error, worker_hashes = None, None, {}
    identity = {"protocol_sha256": file_hash(protocol), "authorization_sha256": file_hash(authorization),
        "training_bundle_sha256": file_hash(bundle), "roster_sha256": file_hash(rows),
        "parent_sha256": PARENT, "rank": rank, "alpha": rank*2, "launch_claim_sha256": file_hash(claim)}
    json_dump_atomic(out/"started.json", identity)

    def launch(name, module, *args):
        consumed_cpu = sum(row["execution"]["resources"]["observed_tree_cpu_seconds"] for row in workers)
        remaining_wall = limits["smoke_max_wall_seconds"]-(time.perf_counter()-wall)
        remaining_cpu = limits["smoke_max_cpu_seconds"]-consumed_cpu-(time.process_time()-host_cpu)
        if remaining_wall <= 0 or remaining_cpu <= 0:
            raise RuntimeError("Disposable smoke aggregate budget exhausted before worker launch")
        command = [sys.executable, "-m", f"experiments.phase_iiiq_baseline.{module}", *map(str, args)]
        observed = launch_owned_tree(command, out/f"{name}.log", wall_limit=remaining_wall,
            cpu_limit=remaining_cpu, memory_limit=limits["max_memory_bytes"], cwd=ROOT)
        record = {"worker": name, "execution": observed, "log_sha256": file_hash(out/f"{name}.log")}
        json_dump_atomic(out/f"{name}-process.json", record)
        workers.append(record)
        if observed["exit_code"] != 0 or observed["budget_failure"] is not None:
            raise RuntimeError(f"Disposable smoke worker {name} failed or exceeded its aggregate budget")
        if not observed["resources"]["cpu_accounting_complete_for_observed_processes"]:
            raise RuntimeError("Disposable smoke observed process-tree CPU counters are incomplete")

    with _temporary_sleep_request() as sleep_request:
        try:
            if sleep_request["requested"] and not sleep_request["succeeded"]:
                raise RuntimeError("Temporary Windows sleep prevention refused")
            launch("zero", "smoke_checks", "zero", "--base", base, "--protocol", protocol,
                "--rows", rows, "--out", out/"zero")
            zero = read(out/"zero/zero-init-check.json")
            if zero.get("zero_init_parent_identical") is not True:
                raise ValueError("Zero-initialized adapter changed parent generation")
            launch("training", "train", "--base", base, "--protocol", protocol,
                "--authorization", authorization, "--bundle", bundle, "--out", out/"training",
                "--smoke", "--run-index", 0)
            completed = read(out/"training/completed.json")
            if completed.get("status") != "COMPLETED" or not completed.get("checkpoints"):
                raise ValueError("Disposable training did not finish")
            checkpoint = Path(completed["checkpoints"][-1])
            if not checkpoint.resolve().is_relative_to((out/"training").resolve()):
                raise ValueError("Disposable adapter escaped its own training directory")
            adapter, fresh = checkpoint/"adapter.safetensors", []
            digest = file_hash(adapter)
            for name in ("fresh-1", "fresh-2"):
                launch(name, "smoke_checks", "fresh", "--base", base, "--protocol", protocol,
                    "--rows", rows, "--adapter", adapter, "--expected-sha", digest,
                    "--parent-measurement", out/"zero/parent-measurement.json", "--out", out/name)
                fresh.append(read(out/name/"fresh-adapter-check.json"))
            enriched = {**completed, "zero_init_measurement": zero["zero_init_measurement"],
                "matched_loss_before": zero["matched_loss_before"], "matched_loss_after": fresh[0]["matched_loss_after"]}
            gate = validate_smoke(zero["parent_measurement"], fresh[0]["trained_measurement"],
                fresh[1]["trained_measurement"], fresh[0]["disabled_measurement"], enriched, gates=config["gates"]["smoke"])
            gate["checks"]["second_disabled_parent_identical"] = exact_predictions(
                zero["parent_measurement"], fresh[1]["disabled_measurement"])
            gate["checks"]["second_matched_loss_identical"] = all(
                fresh[0]["matched_loss_after"].get(key) == fresh[1]["matched_loss_after"].get(key)
                for key in ("mean_token_nll", "row_ids", "tokenization_sha256", "supervised_tokens", "prompt_representation"))
            gate["checks"]["adapter_artifact_unchanged"] = file_hash(adapter) == digest
            gate["adapter_sha256"] = digest
            paths = {"zero": "zero/zero-init-check.json", "training": "training/completed.json",
                "fresh-1": "fresh-1/fresh-adapter-check.json", "fresh-2": "fresh-2/fresh-adapter-check.json"}
            worker_hashes = {name: file_hash(out/path) for name, path in paths.items()}
            for worker in workers:
                name = worker["worker"]
                body = read(out/paths[name])["resources"]
                observed = worker["execution"]["resources"]
                if (observed["observed_tree_cpu_seconds"]+.01 < body["cpu_seconds"]
                        or observed["peak_rss_bytes"] < body["peak_rss_bytes"]):
                    raise RuntimeError("Process-tree monitor did not cover a worker's own measured resource floor")
            verify_sources(config["source_sha256"])
        except BaseException as caught:
            error = {"type": type(caught).__name__, "message": str(caught)}
    host = time.process_time()-host_cpu
    child_cpu = sum(row["execution"]["resources"]["observed_tree_cpu_seconds"] for row in workers)
    peak = max((row["execution"]["resources"]["peak_rss_bytes"] for row in workers), default=0)
    elapsed = time.perf_counter()-wall
    gate = gate or {"checks": {}, "passed": False}
    gate["checks"].update(all_workers_completed=error is None and len(workers) == 4,
        aggregate_wall_budget=elapsed <= limits["smoke_max_wall_seconds"],
        aggregate_cpu_budget=host+child_cpu <= limits["smoke_max_cpu_seconds"],
        observed_peak_memory_budget=peak <= limits["max_memory_bytes"],
        temporary_sleep_request_acquired_and_released=not sleep_request["requested"] or (
            sleep_request["succeeded"] is True and sleep_request["released"] is True),
        process_tree_cpu_complete_for_observed_processes=bool(workers) and all(
            row["execution"]["resources"]["cpu_accounting_complete_for_observed_processes"] for row in workers))
    gate["passed"] = all(gate["checks"].values())
    gate.update(schema="weightbridge.phase_iiir.smoke_gate.v1", **identity,
        disposition="PASS" if gate["passed"] else "TRAINING_PIPELINE_FAILURE", error=error,
        failed_requirements=[key for key, value in gate["checks"].items() if not value],
        worker_receipt_sha256=worker_hashes, worker_processes=workers, temporary_sleep_request=sleep_request,
        resources={"wall_seconds": elapsed, "cpu_seconds": host+child_cpu, "host_cpu_seconds": host,
            "observed_tree_cpu_seconds": child_cpu, "peak_rss_bytes": peak},
        accounting_scope="Outer host plus each sequential owned process tree once; unobserved short-lived descendants cannot be excluded by polling. Peak is maximum of each tree's conservative summed process peaks.",
        no_main_training_authority=True, lineage_authority=False, production_authority=False)
    json_dump_atomic(out/"smoke-gate.json", gate)
    return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "protocol", "authorization", "bundle", "rows", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base, args.protocol, args.authorization, args.bundle, args.rows, args.out)
    print(result["disposition"], flush=True)
    raise SystemExit(0 if result["passed"] else 1)
