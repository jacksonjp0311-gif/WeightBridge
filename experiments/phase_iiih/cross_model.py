"""Replay the shared endpoint on SmolLM2 with the frozen Phase III-F patch.

This is an engineering control. The old patch was not fitted to this task.
No result from this script qualifies a candidate or accesses a sealed roster.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from experiments.phase_iiif.run_matrix import _open
from weightbridge.adaptation.behavior_endpoint import compare_endpoint, decimal_endpoint, measure_endpoint
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.patches.huggingface import apply_safetensors_patch
from weightbridge.provenance import peak_rss_bytes


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("decimal_protocol.json")
MATERIAL = ROOT / "results/phase-iii-h/decimal-endpoint-v1/material"
ARCHIVE = ROOT / "results/phase-iii-f/matrix/llama"


def _read(path):
    return json.loads(path.read_bytes())


def worker(checkpoint, expected, out):
    started, cpu = time.perf_counter(), time.process_time()
    config = _read(PROTOCOL)
    roster = MATERIAL / "practice.json"
    manifest = _read(MATERIAL / "manifest.json")
    if (manifest["splits"]["practice"]["sha256"] != file_hash(roster)
            or manifest["protocol_sha256"] != file_hash(PROTOCOL)):
        raise ValueError("Practice roster commitment differs")
    contract = decimal_endpoint(config["inference"], required_wordings=tuple(config["endpoint"]["wordings"]),
                                max_candidate_evaluations=config["endpoint"]["max_candidate_evaluations"])
    with _open(checkpoint, config, expected) as (_, runner, runtime):
        measured = measure_endpoint(runner, _read(roster)["rows"], contract)
    json_dump_atomic(out, {"checkpoint_sha256": expected, "roster_sha256": file_hash(roster),
                          "model_contract": runtime.to_dict(), "measurement": measured,
                          "resources": {"cpu_seconds": time.process_time() - cpu,
                                        "wall_seconds": time.perf_counter() - started,
                                        "peak_rss_bytes": peak_rss_bytes()}})


def run(base, out):
    started = time.perf_counter()
    config = _read(PROTOCOL)
    historical = _read(ARCHIVE / "report.json")
    base_hash = historical["base_sha256"]
    patches = ARCHIVE / "patches"
    patch = _read(patches / "direct_low_rank.json")
    policy = _read(patches / "direct_low_rank-policy.json")
    stored = _read(patches / "direct_low_rank-receipt.json")
    contract = decimal_endpoint(config["inference"], required_wordings=tuple(config["endpoint"]["wordings"]),
                                max_candidate_evaluations=config["endpoint"]["max_candidate_evaluations"])

    def launch(path, expected, target):
        subprocess.run([sys.executable, "-m", "experiments.phase_iiih.cross_model", "worker", "--base", str(path),
                        "--sha256", expected, "--out", str(target)], cwd=ROOT, check=True,
                       timeout=config["inference"]["worker_timeout_seconds"])
        return _read(target)

    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="weightbridge-iiih-cross-") as temp:
        candidate = Path(temp) / "checkpoint"
        # Host-authorized historical patch; candidate is never the source path.
        receipt = apply_safetensors_patch(base, patch, policy, candidate)
        for field in ("base_sha256", "candidate_sha256", "patch_sha256", "policy_sha256", "tensor_receipts"):
            if stored[field] != receipt[field]:
                raise ValueError(f"Archived patch failed exact replay: {field}")
        before = launch(base, base_hash, out / "base.json")
        after = launch(candidate, receipt["candidate_sha256"], out / "candidate.json")
        rollback = launch(base, base_hash, out / "rollback.json")
    restored = rollback["measurement"]["rows"] == before["measurement"]["rows"]
    if not restored:
        raise ValueError("Rollback endpoint replay differs")
    gate = compare_endpoint(before["measurement"], after["measurement"], contract)
    json_dump_atomic(out / "report.json", {
        "schema": "weightbridge.phase_iiih.cross_model.v1", "source_phase": "III-F",
        "engineering_disposition": "PASS", "behavioral_adaptation_disposition": "NOT_TESTED",
        "reason": "Historical engineering patch; no task fitting, hidden access, or promotion",
        "protocol_sha256": file_hash(PROTOCOL), "rollback_exact": restored,
        "archived_receipt_sha256": file_hash(patches / "direct_low_rank-receipt.json"),
        "patch_receipt": receipt, "diagnostic_endpoint_comparison": gate,
        "artifacts": {name: file_hash(out / name) for name in ("base.json", "candidate.json", "rollback.json")},
        "resources": {"wall_seconds": time.perf_counter() - started,
                      "worker_cpu_seconds": sum(r["resources"]["cpu_seconds"] for r in (before, after, rollback)),
                      "max_worker_peak_rss_bytes": max(r["resources"]["peak_rss_bytes"] for r in (before, after, rollback))},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.command == "worker":
        if not args.sha256:
            parser.error("worker requires a checkpoint SHA-256")
        worker(args.base, args.sha256, args.out)
    else:
        run(args.base, args.out)
