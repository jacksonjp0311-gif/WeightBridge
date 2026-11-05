"""Post-freeze host interface bridge; independently replays raw smoke evidence.

The original executed-smoke schema keeps protocol identity inside ``identity``.
This adds the flat field required by the already-frozen training admission API.
It changes no model code, source manifest, raw receipt, or quality threshold.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.adaptation.training_policy import TrainingAuthorization
from .data import PopulationSpec, validate
from .integrity import PARENT, ROOT, read, verify_sources
from .smoke_checks import _IDENTITY, exact_predictions, validate_smoke


def _same(actual, expected, message):
    if actual != expected:
        raise ValueError(message)


def _number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError("Smoke resource measurements must be finite nonnegative numbers")
    return value


def _cost_audit(smoke, completed, process_rows, raw_resources, limits):
    """Launcher CPU is not child CPU; retain the known lower bound honestly."""
    actual = {"training": completed["resources"]}
    identities = {}
    for name in ("zero", "fresh-1", "fresh-2"):
        path = Path(smoke) / name / "execution.json"
        receipt = read(path)
        if receipt.get("status") != "COMPLETED" or receipt.get("error") is not None:
            raise ValueError("Worker cleanup did not finish successfully")
        actual[name] = receipt["resources"]
        identities[name] = file_hash(path)
    for resource in actual.values():
        for field in ("cpu_seconds", "wall_seconds", "peak_rss_bytes"):
            _number(resource[field])
    worker_cpu = sum(resource["cpu_seconds"] for resource in actual.values())
    launchers = sum(_number(row["cpu_seconds"]) for row in process_rows)
    host = _number(raw_resources["host_cpu_seconds"])
    known_cpu = worker_cpu + launchers + host
    peak = max(resource["peak_rss_bytes"] for resource in actual.values())
    wall = _number(raw_resources["wall_seconds"])
    inner_wall = sum(resource["wall_seconds"] for resource in actual.values())
    if inner_wall > wall + .001:
        raise ValueError("Sequential worker intervals exceed the complete host wall interval")
    gap = max(0., wall - inner_wall)
    # Frozen preflight recorded thirty-two logical CPUs. Every worker was
    # sequential; even all cores busy throughout the unmeasured wall gap gives
    # this conservative upper bound, including launcher/host CPU once again.
    upper_cpu = known_cpu + gap*32
    return {"schema": "weightbridge.supervised_baseline.smoke_cost_audit.v1",
        "known_cpu_lower_bound_seconds": known_cpu, "worker_reported_cpu_seconds": worker_cpu,
        "launcher_cpu_seconds": launchers, "host_cpu_seconds": host,
        "conservative_cpu_upper_bound_seconds": upper_cpu,
        "total_exact_cpu_seconds": None,
        "summed_inner_wall_seconds": inner_wall, "unmeasured_wall_gap_seconds": gap,
        "recorded_logical_cores": 32,
        "upper_bound_assumptions": "Four sequential owned workers; inner wall intervals are disjoint subsets of complete host wall; at most the recorded 32 logical CPUs can execute during the residual gap. Host/launcher measured CPU retained, making the bound conservative.",
        "wall_seconds": wall, "peak_known_rss_bytes": peak,
        "raw_reported_cpu_seconds": raw_resources["cpu_seconds"],
        "raw_reported_peak_rss_bytes": raw_resources["peak_rss_bytes"],
        "raw_launcher_accounting_incomplete": True, "cpu_accounting_complete": False,
        "startup_and_post_receipt_child_cpu": "UNMEASURED",
        "startup_and_post_receipt_child_peak_rss": "UNMEASURED",
        "worker_execution_sha256": identities,
        "known_cpu_within_budget": known_cpu <= limits["smoke_max_cpu_seconds"],
        "conservative_cpu_upper_bound_within_budget": upper_cpu <= limits["smoke_max_cpu_seconds"],
        "known_peak_memory_within_budget": peak <= limits["max_memory_bytes"],
        "complete_wall_within_budget": wall <= limits["smoke_max_wall_seconds"],
        "scope": "Known compute lower bound includes body/cleanup receipts, launcher processes and host once. Launcher GetProcessTimes does not include its Python child. Full child startup/teardown CPU and peak memory remain unmeasured; no complete-accounting claim."}


def admit(protocol, authorization, bundle, rows, smoke, out):
    protocol, authorization, bundle, rows, smoke, out = map(Path, (protocol, authorization, bundle, rows, smoke, out))
    if out.resolve().is_relative_to(ROOT) or out.exists():
        raise ValueError("Host admission requires new private evidence outside the checkout")
    config, training, roster, gate = read(protocol), read(bundle), read(rows), read(smoke / "smoke-gate.json")
    verify_sources(config["source_sha256"])
    validate(roster, "smoke", PopulationSpec(target_families=4))
    _same(training["rows"], roster, "Smoke bundle is not the exact scored roster")
    _same(training["split_ids"], ["smoke"], "Only disposable smoke data may be admitted")
    identity = {"protocol_sha256": file_hash(protocol), "authorization_sha256": file_hash(authorization),
        "bundle_sha256": file_hash(bundle), "roster_sha256": file_hash(rows), "parent_sha256": PARENT,
        "source_sha256": config["source_sha256"]}
    _same(config["checkpoint_sha256"], PARENT, "Smoke parent differs")
    _same(training["protocol_sha256"], identity["protocol_sha256"], "Smoke bundle protocol differs")
    _same(gate.get("identity"), identity, "Raw smoke gate identity differs from authorized artifacts")
    _same(read(smoke / "started.json"), identity, "Raw smoke start identity differs")
    auth = TrainingAuthorization.from_dict(read(authorization))
    _same(auth.training_bundle_sha256, identity["bundle_sha256"], "Smoke authorization binds another bundle")
    _same(auth.parent_sha256, PARENT, "Smoke authorization binds another parent")
    paths = {"zero": smoke / "zero/zero-init-check.json", "training": smoke / "training/completed.json",
             "fresh-1": smoke / "fresh-1/fresh-adapter-check.json", "fresh-2": smoke / "fresh-2/fresh-adapter-check.json"}
    hashes = {name: file_hash(path) for name, path in paths.items()}
    _same(gate.get("worker_receipt_sha256"), hashes, "Raw worker receipt hash changed")
    zero, completed, first, second = (read(paths[name]) for name in paths)
    for worker in (zero, first, second):
        for field in ("checkpoint_sha256", "protocol_sha256", "roster_sha256"):
            _same(worker.get(field), PARENT if field == "checkpoint_sha256" else identity[field],
                  "Raw measurement worker identity differs")
        _same(worker.get("base_unchanged"), True, "Raw worker did not restore the immutable base")
    for field in ("protocol_sha256", "authorization_sha256", "bundle_sha256", "parent_sha256"):
        _same(completed.get(field), identity[field], "Raw training identity differs")
    _same(completed.get("smoke"), True, "Main training cannot be substituted for the smoke")
    checkpoint = Path(completed["checkpoints"][-1])
    if not checkpoint.resolve().is_relative_to((smoke / "training").resolve()):
        raise ValueError("Smoke adapter escapes its owned training directory")
    adapter = checkpoint / "adapter.safetensors"
    adapter_sha = file_hash(adapter)
    for worker in (first, second, gate):
        _same(worker.get("adapter_sha256"), adapter_sha, "Fresh worker adapter identity differs")
    measurements = (zero["parent_measurement"], zero["zero_init_measurement"],
        first["trained_measurement"], first["disabled_measurement"],
        second["trained_measurement"], second["disabled_measurement"])
    for measurement in measurements:
        for observed, frozen in zip(measurement["rows"], roster, strict=True):
            _same([observed.get(k) for k in _IDENTITY], [frozen.get(k) for k in _IDENTITY],
                  "Raw semantic observation differs from the frozen smoke roster")
    enriched = {**completed, "zero_init_measurement": zero["zero_init_measurement"],
        "matched_loss_before": zero["matched_loss_before"], "matched_loss_after": first["matched_loss_after"]}
    recomputed = validate_smoke(zero["parent_measurement"], first["trained_measurement"],
        second["trained_measurement"], first["disabled_measurement"], enriched, gates=config["gates"]["smoke"])
    checks = dict(recomputed["checks"])
    checks["second_disabled_parent_identical"] = exact_predictions(zero["parent_measurement"], second["disabled_measurement"])
    checks["second_matched_loss_identical"] = first["matched_loss_after"] == second["matched_loss_after"] or all(
        first["matched_loss_after"].get(key) == second["matched_loss_after"].get(key)
        for key in ("mean_token_nll", "row_ids", "tokenization_sha256", "supervised_tokens", "prompt_representation", "forward_calls"))
    process_rows = []
    for name in paths:
        process = read(smoke / f"{name}-process.json")
        _same(process.get("worker"), name, "Worker process identity differs")
        _same(process.get("log_sha256"), file_hash(smoke / f"{name}.log"), "Worker log changed")
        _number(process["wall_seconds"]); _number(process["cpu_seconds"]); _number(process["peak_rss_bytes"])
        process_rows.append(process)
    _same(gate.get("worker_processes"), process_rows, "Worker process resource receipts changed")
    limits, resources = config["resources"], gate["resources"]
    worker_cpu = sum(row["cpu_seconds"] for row in process_rows)
    peak = max(row["peak_rss_bytes"] for row in process_rows)
    _same(resources["worker_cpu_seconds"], worker_cpu, "Worker CPU aggregation differs")
    _same(resources["peak_rss_bytes"], peak, "Worker peak-memory aggregation differs")
    if not math.isclose(_number(resources["cpu_seconds"]), _number(resources["host_cpu_seconds"]) + worker_cpu, abs_tol=1e-7):
        raise ValueError("Aggregate smoke CPU accounting differs")
    cost = _cost_audit(smoke, completed, process_rows, resources, limits)
    checks.update(raw_gate_passed=gate.get("passed") is True and gate.get("error") is None,
        all_workers_completed=all(row.get("exit_code") == 0 and row.get("budget_failure") is None for row in process_rows),
        aggregate_wall_budget=cost["complete_wall_within_budget"],
        known_aggregate_cpu_budget=cost["known_cpu_within_budget"],
        conservative_aggregate_cpu_budget=cost["conservative_cpu_upper_bound_within_budget"],
        known_peak_memory_budget=cost["known_peak_memory_within_budget"],
        source_manifest_unchanged=gate.get("source_manifest_unchanged") is True)
    for key in recomputed["checks"]:
        _same(gate["checks"].get(key), checks[key], "Raw and independently recomputed quality checks disagree")
    usage = completed["resources"]; counts = completed["counts"]
    auth.check_usage(steps=counts["optimizer_steps"], answer_tokens=counts["answer_tokens"],
        input_tokens=counts["input_tokens"], wall_seconds=usage["wall_seconds"],
        cpu_seconds=usage["cpu_seconds"], rss_bytes=usage["peak_rss_bytes"])
    verify_sources(config["source_sha256"])
    _same({name: file_hash(path) for name, path in paths.items()}, hashes, "Raw worker receipt changed during replay")
    result = {"schema": "weightbridge.supervised_baseline.smoke_admission_bridge.v1",
        "passed": all(checks.values()), "protocol_sha256": identity["protocol_sha256"],
        "checks": checks, "source_gate_sha256": file_hash(smoke / "smoke-gate.json"),
        "worker_receipt_sha256": hashes, "adapter_sha256": adapter_sha,
        "cost_audit": cost, "cpu_accounting_complete": False,
        "host_bridge_code_sha256": file_hash(__file__), "frozen_identity": identity,
        "recomputed_quality": recomputed, "thresholds_modified": False,
        "production_authority": False, "lineage_authority": False,
        "scope": "Post-freeze host schema bridge only; independently replays unchanged raw evidence. No model execution or new learning claim."}
    json_dump_atomic(out, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "authorization", "bundle", "rows", "smoke", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(admit(args.protocol, args.authorization, args.bundle, args.rows, args.smoke, args.out)["passed"])
