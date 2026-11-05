"""Sequential, source-bound disposable smoke with aggregate host budgets."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from weightbridge.core import file_hash, json_dump_atomic
from .integrity import PARENT, ROOT, read, verify_sources
from .data import PopulationSpec, validate
from .smoke_checks import exact_predictions, validate_smoke


def _finished_cpu(process):
    """Read the complete exited Windows child CPU, including startup/cleanup."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    query = ctypes.WinDLL("kernel32", use_last_error=True).GetProcessTimes
    query.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)]*4)]
    query.restype = wintypes.BOOL
    creation, end, kernel, user = (wintypes.FILETIME() for _ in range(4))
    if not query(int(process._handle), creation, end, kernel, user):
        return None
    ticks = lambda item: (int(item.dwHighDateTime)<<32)+int(item.dwLowDateTime)
    return (ticks(kernel)+ticks(user))/10_000_000


def _worker(argv, name, root, budget):
    """Execute only this owned child, refusing aggregate resource excess."""
    log = root / f"{name}.log"
    wall = time.perf_counter()
    peak = sampled_cpu = 0.
    exceeded = None
    with log.open("xb") as stream:
        process = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        observed = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                used = observed.cpu_times()
                sampled_cpu = max(sampled_cpu, used.user+used.system)
                memory = observed.memory_info()
                peak = max(peak, getattr(memory, "peak_wset", memory.rss))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            if time.perf_counter()-budget["wall_start"] > budget["max_wall"]:
                exceeded = "aggregate_wall_budget"
            elif budget["completed_cpu"]+sampled_cpu+time.process_time()-budget["host_cpu_start"] > budget["max_cpu"]:
                exceeded = "aggregate_cpu_budget"
            elif peak > budget["max_memory"]:
                exceeded = "worker_memory_budget"
            if exceeded:
                # The host owns this new bounded worker. Other processes and
                # pre-existing experiments are never signaled or interrupted.
                process.kill()
                break
            time.sleep(.1)
        code = process.wait()
        complete_cpu = _finished_cpu(process)
    record = {"worker": name, "exit_code": code, "budget_failure": exceeded,
        "wall_seconds": time.perf_counter()-wall, "cpu_seconds": complete_cpu if complete_cpu is not None else sampled_cpu,
        "cpu_measurement": "complete_windows_process_times" if complete_cpu is not None else "sampled_process_lower_bound",
        "peak_rss_bytes": int(peak), "log_sha256": file_hash(log)}
    json_dump_atomic(root / f"{name}-process.json", record)
    budget["completed_cpu"] += record["cpu_seconds"]
    budget["workers"].append(record)
    if exceeded or code:
        raise RuntimeError(f"Smoke worker {name} failed: {exceeded or 'nonzero_exit'}")
    return record


def run(base, protocol, authorization, bundle, rows, out):
    base, protocol, authorization, bundle, rows, out = map(Path, (base, protocol, authorization, bundle, rows, out))
    if out.resolve().is_relative_to(base.resolve()) or out.resolve().is_relative_to(ROOT):
        raise ValueError("Smoke execution output must be isolated from original checkpoint and checkout")
    config, training, roster = read(protocol), read(bundle), read(rows)
    if not config.get("source_sha256"):
        raise ValueError("Smoke execution requires a frozen source manifest")
    verify_sources(config["source_sha256"])
    validate(roster, "smoke", PopulationSpec(target_families=4))
    if (training.get("rows") != roster or training.get("split_ids") != ["smoke"]
            or training.get("protocol_sha256") != file_hash(protocol) or config["checkpoint_sha256"] != PARENT):
        raise ValueError("Smoke training and scoring must use the same source-bound disposable roster")
    out.mkdir(parents=True, exist_ok=False)
    wall, cpu = time.perf_counter(), time.process_time()
    limits = config["resources"]
    budget = {"wall_start": wall, "host_cpu_start": cpu, "completed_cpu": 0., "workers": [],
        "max_wall": limits["smoke_max_wall_seconds"], "max_cpu": limits["smoke_max_cpu_seconds"],
        "max_memory": limits["max_memory_bytes"]}
    gate, error = None, None
    identity = {"protocol_sha256": file_hash(protocol), "authorization_sha256": file_hash(authorization),
        "bundle_sha256": file_hash(bundle), "roster_sha256": file_hash(rows), "parent_sha256": PARENT,
        "source_sha256": config["source_sha256"]}
    json_dump_atomic(out / "started.json", identity)
    def command(module, *args):
        return [sys.executable, "-m", f"experiments.phase_iiiq_baseline.{module}", *map(str, args)]
    try:
        _worker(command("smoke_checks", "zero", "--base", base, "--protocol", protocol,
            "--rows", rows, "--out", out/"zero"), "zero", out, budget)
        zero = read(out/"zero/zero-init-check.json")
        if zero.get("zero_init_parent_identical") is not True:
            raise ValueError("A zero-initialized adapter changed the parent predictions")
        _worker(command("train", "--base", base, "--protocol", protocol,
            "--authorization", authorization, "--bundle", bundle, "--out", out/"training", "--smoke", "--run-index", "0"),
            "training", out, budget)
        completed = read(out/"training/completed.json")
        if completed.get("status") != "COMPLETED" or not completed.get("checkpoints"):
            raise ValueError("Smoke training did not finish a saved adapter")
        checkpoint = Path(completed["checkpoints"][-1])
        if not checkpoint.resolve().is_relative_to((out/"training").resolve()):
            raise ValueError("Smoke checkpoint path escapes its owned training output")
        adapter = checkpoint/"adapter.safetensors"
        expected = file_hash(adapter)
        fresh = []
        for name in ("fresh-1", "fresh-2"):
            _worker(command("smoke_checks", "fresh", "--base", base, "--protocol", protocol,
                "--rows", rows, "--adapter", adapter, "--expected-sha", expected,
                "--parent-measurement", out/"zero/parent-measurement.json", "--out", out/name), name, out, budget)
            fresh.append(read(out/name/"fresh-adapter-check.json"))
        enriched = {**completed, "zero_init_measurement": zero["zero_init_measurement"],
            "matched_loss_before": zero["matched_loss_before"], "matched_loss_after": fresh[0]["matched_loss_after"]}
        gate = validate_smoke(zero["parent_measurement"], fresh[0]["trained_measurement"],
            fresh[1]["trained_measurement"], fresh[0]["disabled_measurement"], enriched, gates=config["gates"]["smoke"])
        gate["checks"]["second_disabled_parent_identical"] = exact_predictions(zero["parent_measurement"], fresh[1]["disabled_measurement"])
        gate["checks"]["adapter_artifact_unchanged"] = file_hash(adapter) == expected
        verify_sources(config["source_sha256"])
        gate.update(adapter_sha256=expected, worker_receipt_sha256={name: file_hash(out/path)
            for name, path in (("zero", "zero/zero-init-check.json"), ("training", "training/completed.json"),
                              ("fresh-1", "fresh-1/fresh-adapter-check.json"), ("fresh-2", "fresh-2/fresh-adapter-check.json"))})
    except BaseException as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
    host_cpu = time.process_time()-cpu
    measured_wall = time.perf_counter()-wall
    measured_cpu = host_cpu+budget["completed_cpu"]
    peak = max((entry["peak_rss_bytes"] for entry in budget["workers"]), default=0)
    gate = gate or {"checks": {}, "failed_requirements": [], "passed": False, "disposition": "FAIL"}
    gate["checks"].update(all_workers_completed=error is None and len(budget["workers"]) == 4,
        aggregate_wall_budget=measured_wall <= budget["max_wall"], aggregate_cpu_budget=measured_cpu <= budget["max_cpu"],
        peak_memory_budget=peak <= budget["max_memory"])
    gate["passed"] = all(gate["checks"].values())
    gate.update(schema="weightbridge.supervised_baseline.executed_smoke.v1", disposition="PASS" if gate["passed"] else "FAIL",
        failed_requirements=[key for key, value in gate["checks"].items() if not value], error=error,
        identity=identity, worker_processes=budget["workers"],
        resources={"wall_seconds": measured_wall, "cpu_seconds": measured_cpu, "host_cpu_seconds": host_cpu,
            "worker_cpu_seconds": budget["completed_cpu"], "peak_rss_bytes": peak,
            "cpu_accounting_complete": all(row["cpu_measurement"] == "complete_windows_process_times" for row in budget["workers"])},
        source_manifest_unchanged=error is None,
        no_main_training_authority=True, lineage_authority=False, production_authority=False,
        scope="Disposable smoke; every worker is a fresh process. A pass is required before the host may separately authorize main training. No development, selection or hidden roster is read.")
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
