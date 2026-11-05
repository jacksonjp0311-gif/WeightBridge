"""Post-freeze host launcher with observed process-tree resource accounting.

Windows virtual-environment python.exe can be a redirector whose counters do
not include the actual Python worker. Each observed descendant gets its own
retained query handle, so exited child CPU is measured before those handles
are closed. No frozen training or evaluation source is modified here.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from weightbridge.adaptation.training_policy import TrainingAuthorization
from weightbridge.core import file_hash, json_dump_atomic
from .integrity import PARENT, ROOT, read, verify_sources


class ObservedTree:
    """Only descendants of a newly owned child; never process-name selection."""

    def __init__(self, process):
        self.root_pid = process.pid
        self.entries = {}
        self._api = None
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)]*4)]
            kernel.GetProcessTimes.restype = wintypes.BOOL
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle.restype = wintypes.BOOL
            self._api = ctypes, wintypes, kernel
        self._remember(psutil.Process(process.pid))

    def _remember(self, process):
        try:
            key = (process.pid, process.create_time())
            if key in self.entries:
                return
            handle = None
            if self._api:
                handle = self._api[2].OpenProcess(0x0400 | 0x0010 | 0x00100000, False, process.pid)
            self.entries[key] = {"process": process, "handle": handle, "pid": process.pid,
                "create_time": key[1], "parent_pid_at_discovery": process.ppid(),
                "cpu_seconds": 0., "peak_rss_bytes": 0, "maximum_sampled_rss_bytes": 0,
                "cpu_counter_complete": False, "is_root_launcher": process.pid == self.root_pid}
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return

    def discover(self):
        # Continue following already observed children if a redirector exits.
        for entry in list(self.entries.values()):
            try:
                if entry["process"].create_time() != entry["create_time"]:
                    continue
                for child in entry["process"].children(recursive=True):
                    self._remember(child)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def _handle_cpu(self, entry):
        if not self._api or not entry["handle"]:
            return None
        ctypes, wintypes, kernel = self._api
        creation, end, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(entry["handle"], creation, end, kernel_time, user_time):
            return None
        ticks = lambda value: (int(value.dwHighDateTime)<<32)+int(value.dwLowDateTime)
        return (ticks(kernel_time)+ticks(user_time))/10_000_000, ticks(end) > 0

    def sample(self, *, final=False):
        self.discover()
        for entry in self.entries.values():
            exact_cpu = self._handle_cpu(entry)
            if exact_cpu is not None:
                entry["cpu_seconds"] = max(entry["cpu_seconds"], exact_cpu[0])
                entry["cpu_counter_complete"] = bool(final and exact_cpu[1])
            try:
                process = entry["process"]
                if process.create_time() != entry["create_time"]:
                    continue
                cpu = process.cpu_times()
                entry["cpu_seconds"] = max(entry["cpu_seconds"], cpu.user+cpu.system)
                memory = process.memory_info()
                entry["peak_rss_bytes"] = max(entry["peak_rss_bytes"], getattr(memory, "peak_wset", memory.rss))
                entry["maximum_sampled_rss_bytes"] = max(entry["maximum_sampled_rss_bytes"], memory.rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def kill_owned(self):
        self.discover()
        for entry in reversed(list(self.entries.values())):
            try:
                process = entry["process"]
                if process.create_time() == entry["create_time"] and process.is_running():
                    process.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def any_alive(self):
        for entry in self.entries.values():
            exact = self._handle_cpu(entry)
            if exact is not None:
                if not exact[1]:
                    return True
            else:
                try:
                    process = entry["process"]
                    if process.create_time() == entry["create_time"] and process.is_running():
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        return False

    def totals(self):
        return {"cpu_seconds": sum(row["cpu_seconds"] for row in self.entries.values()),
            "conservative_peak_rss_bytes": sum(row["peak_rss_bytes"] for row in self.entries.values()),
            "process_count": len(self.entries),
            "cpu_accounting_complete_for_observed_processes": bool(self.entries) and all(
                row["cpu_counter_complete"] for row in self.entries.values())}

    def records(self):
        return [{key: value for key, value in row.items() if key not in {"process", "handle"}}
                for row in self.entries.values()]

    def close(self):
        if self._api:
            for entry in self.entries.values():
                if entry["handle"]:
                    self._api[2].CloseHandle(entry["handle"])
                    entry["handle"] = None


def launch_owned_tree(argv, log_path, *, wall_limit, cpu_limit, memory_limit, cwd=ROOT):
    """Bound one new owned process tree and retain failures as measured evidence."""
    started, host_cpu = time.perf_counter(), time.process_time()
    process = tree = None
    failure = None
    try:
        with Path(log_path).open("xb") as stream:
            process = subprocess.Popen(list(map(str, argv)), cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=0x08000000 if os.name == "nt" else 0)
            tree = ObservedTree(process)
            while process.poll() is None or tree.any_alive():
                tree.sample()
                totals = tree.totals()
                if time.perf_counter()-started > wall_limit:
                    failure = "whole_tree_wall_budget"
                elif totals["cpu_seconds"]+time.process_time()-host_cpu > cpu_limit:
                    failure = "whole_tree_cpu_budget"
                elif totals["conservative_peak_rss_bytes"] > memory_limit:
                    failure = "conservative_tree_memory_budget"
                if failure:
                    tree.kill_owned()
                    break
                time.sleep(.1)
            code = process.wait()
            tree.sample(final=True)
            totals = tree.totals()
            consumed_host = time.process_time()-host_cpu
            return {"exit_code": code, "budget_failure": failure, "processes": tree.records(),
                "observed_process_count": totals["process_count"],
                "resources": {"wall_seconds": time.perf_counter()-started,
                    "cpu_seconds": totals["cpu_seconds"]+consumed_host, "host_cpu_seconds": consumed_host,
                    "observed_tree_cpu_seconds": totals["cpu_seconds"],
                    "peak_rss_bytes": totals["conservative_peak_rss_bytes"],
                    "cpu_accounting_complete_for_observed_processes": totals["cpu_accounting_complete_for_observed_processes"]},
                "memory_scope": "Conservative sum of lifetime peak working sets sampled for each observed owned process; not a simultaneous-RSS measurement.",
                "coverage_scope": "Root plus recursively observed descendants only. Retained Windows handles include observed child CPU after exit. Unobserved short-lived descendants cannot be excluded by polling alone."}
    finally:
        if process is not None and process.poll() is None:
            if tree:
                tree.kill_owned()
            else:
                process.kill()
            process.wait()
        if tree:
            tree.close()


def run(base, protocol, authorization, bundle, out, *, run_index, smoke_gate):
    base, protocol, authorization, bundle, out, smoke_gate = map(Path, (base, protocol, authorization, bundle, out, smoke_gate))
    if out.resolve().is_relative_to(base.resolve()) or out.resolve().is_relative_to(ROOT):
        raise ValueError("Training output must be outside original checkpoint and checkout")
    log, receipt = out.parent/f"run-{run_index}.log", out.parent/f"run-{run_index}-process.json"
    if any(path.exists() for path in (out, log, receipt)):
        raise FileExistsError("Training, log and process receipt paths must all be fresh")
    config, binding, training, gate = read(protocol), read(authorization), read(bundle), read(smoke_gate)
    verify_sources(config["source_sha256"])
    auth = TrainingAuthorization.from_dict(binding)
    if (config["checkpoint_sha256"] != PARENT or auth.parent_sha256 != PARENT
            or auth.training_bundle_sha256 != file_hash(bundle)
            or training.get("protocol_sha256") != file_hash(protocol)
            or set(training.get("split_ids", ())) != {"train", "replay"}
            or auth.smoke_gate_sha256 != file_hash(smoke_gate) or gate.get("passed") is not True
            or gate.get("protocol_sha256") != file_hash(protocol)
            or not 0 <= run_index < auth.max_runs):
        raise ValueError("Main training requires the exact admitted smoke, parent, protocol and training-only bundle")
    limits = config["resources"]
    wall_limit = min(limits["max_wall_seconds_per_run"], auth.max_wall_seconds_per_run)
    cpu_limit = min(limits["max_cpu_seconds_per_run"], auth.max_cpu_seconds_per_run)
    memory_limit = min(limits["max_memory_bytes"], auth.max_memory_bytes)
    out.parent.mkdir(parents=True, exist_ok=True)
    identity = {"protocol_sha256": file_hash(protocol), "authorization_sha256": file_hash(authorization),
        "bundle_sha256": file_hash(bundle), "smoke_gate_sha256": file_hash(smoke_gate), "parent_sha256": PARENT,
        "run_index": run_index, "host_wrapper_sha256": file_hash(__file__)}
    command = [sys.executable, "-m", "experiments.phase_iiiq_baseline.train", "--base", str(base.resolve()),
        "--protocol", str(protocol.resolve()), "--authorization", str(authorization.resolve()),
        "--bundle", str(bundle.resolve()), "--out", str(out.resolve()), "--run-index", str(run_index)]
    observed = launch_owned_tree(command, log, wall_limit=wall_limit, cpu_limit=cpu_limit, memory_limit=memory_limit)
    checks = {"worker_exit_success": observed["exit_code"] == 0, "no_tree_budget_refusal": observed["budget_failure"] is None,
        "complete_observed_process_cpu": observed["resources"]["cpu_accounting_complete_for_observed_processes"]}
    completed_path = out/"completed.json"
    completed = read(completed_path) if completed_path.is_file() else None
    if completed:
        body = completed["resources"]
        checks.update(completed_training=completed.get("status") == "COMPLETED" and completed.get("smoke") is False,
            completed_identity=all(completed.get(key) == identity[key] for key in ("protocol_sha256", "authorization_sha256", "bundle_sha256", "parent_sha256", "run_index")),
            worker_cpu_covered=observed["resources"]["observed_tree_cpu_seconds"]+.01 >= body["cpu_seconds"],
            worker_peak_covered=observed["resources"]["peak_rss_bytes"] >= body["peak_rss_bytes"])
        if not checks["worker_peak_covered"]:
            # Keep the independently reported known peak, while refusing the
            # incomplete tree-memory coverage rather than silently calling it complete.
            observed["resources"]["known_peak_rss_floor_bytes"] = body["peak_rss_bytes"]
    else:
        checks["completed_training"] = False
    try:
        verify_sources(config["source_sha256"])
        checks["frozen_sources_unchanged"] = True
    except ValueError:
        checks["frozen_sources_unchanged"] = False
    usage = observed["resources"]
    checks.update(whole_tree_wall_within_limit=usage["wall_seconds"] <= wall_limit,
        whole_tree_cpu_within_limit=usage["cpu_seconds"] <= cpu_limit,
        conservative_tree_memory_within_limit=usage["peak_rss_bytes"] <= memory_limit)
    result = {"schema": "weightbridge.supervised_baseline.training_process_tree.v1", **identity,
        "passed": all(checks.values()), "status": "COMPLETED" if all(checks.values()) else "HOLD",
        "checks": checks, "limits": {"wall_seconds": wall_limit, "cpu_seconds": cpu_limit, "memory_bytes": memory_limit},
        "execution": observed, "training_completed_receipt_sha256": file_hash(completed_path) if completed else None,
        "log_sha256": file_hash(log), "frozen_training_sources_modified": False,
        "production_authority": False, "lineage_authority": False}
    json_dump_atomic(receipt, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "protocol", "authorization", "bundle", "out", "smoke-gate"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-index", type=int, required=True)
    args = parser.parse_args()
    result = run(args.base, args.protocol, args.authorization, args.bundle, args.out,
        run_index=args.run_index, smoke_gate=args.smoke_gate)
    print(result["status"], flush=True)
    raise SystemExit(0 if result["passed"] else 1)
