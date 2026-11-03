"""Run the frozen natural-Qwen protocol in separate practice and evaluator processes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.core import file_hash, json_dump_atomic


def subprocess_stage(stage, state, output, arguments, timeout):
    resources_path = output.with_suffix(".resources.json")
    command = [sys.executable, str(Path(__file__).resolve()), stage, "--state", str(state),
               "--resources", str(resources_path)]
    for name, value in arguments.items():
        if isinstance(value, list):
            for path in value:
                command.extend((f"--{name}", str(path)))
        else:
            command.extend((f"--{name}", str(value)))
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace",
                                   timeout=timeout, check=False, cwd=ROOT)
    except subprocess.TimeoutExpired as exc:
        for stream, contents in (("stdout", exc.stdout), ("stderr", exc.stderr)):
            output.with_suffix(f".{stream}.txt").write_bytes(contents or b"")
        json_dump_atomic(output, {"stage": stage, "status": "TIMEOUT", "timeout_seconds": timeout,
                                  "wall_seconds": time.perf_counter() - started})
        raise
    output.with_suffix(".stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        output.with_suffix(".stdout.txt").write_text(completed.stdout, encoding="utf-8")
        raise RuntimeError(f"{stage} exited with {completed.returncode}. See {output.with_suffix('.stderr.txt')}")
    report = json.loads(completed.stdout)
    resources = json.loads(resources_path.read_bytes())
    json_dump_atomic(output, {"stage": stage, "wall_seconds": time.perf_counter() - started,
                              "process": resources, "report": report})
    return report


def run(base, materialized, out, timeout):
    protocol_path = materialized / "protocol.json"
    protocol = json.loads(protocol_path.read_bytes())
    out.mkdir(parents=True)
    state = out / "state"
    logs = out / "stages"
    logs.mkdir()
    status = subprocess_stage("init", state, logs / "init.json", {"base": base, "protocol": protocol_path}, timeout)
    accepted_holdouts = []
    decisions = []
    for entry in protocol["cycles"]:
        cycle = entry["id"]
        practice = materialized / "practice" / f"{cycle}.json"
        intent = materialized / "practice" / f"{cycle}-intent.json"
        holdout = materialized / "sealed" / f"{cycle}.json"
        search = subprocess_stage("search", state, logs / f"{cycle}-search.json",
                                  {"cycle": cycle, "practice": practice, "intent": intent}, timeout)
        verification = subprocess_stage("verify", state, logs / f"{cycle}-verify.json",
                                        {"cycle": cycle, "holdout": holdout, "retention": accepted_holdouts}, timeout)
        new_status = subprocess_stage("status", state, logs / f"{cycle}-status.json", {}, timeout)
        accepted = status["head_sha256"] != new_status["head_sha256"]
        if accepted:
            accepted_holdouts.append(holdout)
        decisions.append({
            "cycle_id": cycle, "family": entry["family"], "accepted": accepted,
            "parent_sha256": status["head_sha256"], "head_sha256": new_status["head_sha256"],
            "search": search, "verification": verification,
        })
        status = new_status
    final = subprocess_stage("final", state, logs / "final.json",
                             {"suite": materialized / "sealed" / "final.json"}, timeout)
    summary = {
        "schema": "weightbridge.phase_iii.study_run.v1", "protocol_sha256": file_hash(protocol_path),
        "original_sha256": protocol["base_sha256"], "final_head_sha256": status["head_sha256"],
        "accepted_cycles": sum(decision["accepted"] for decision in decisions),
        "cycles": decisions, "final": final,
    }
    json_dump_atomic(out / "summary.json", summary)
    return summary


def stage(args):
    from weightbridge.adaptation import controller

    if args.stage == "init":
        return controller.init_research(args.state, args.base, args.protocol)
    if args.stage == "search":
        return controller.search_cycle(args.state, args.cycle, args.intent, args.practice)
    if args.stage == "verify":
        return controller.verify_cycle(args.state, args.cycle, args.holdout, args.retention)
    if args.stage == "status":
        return controller.research_status(args.state)
    return controller.score_final(args.state, args.suite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    whole = commands.add_parser("run")
    for name in ("base", "materialized", "out"):
        whole.add_argument(f"--{name}", type=Path, required=True)
    whole.add_argument("--timeout-seconds", type=int, default=600)
    for action, paths in (
        ("init", ("base", "protocol")),
        ("search", ("intent", "practice")),
        ("verify", ("holdout",)),
        ("status", ()),
        ("final", ("suite",)),
    ):
        child = commands.add_parser(action)
        child.add_argument("--state", type=Path, required=True)
        child.add_argument("--resources", type=Path)
        for name in paths:
            child.add_argument(f"--{name}", type=Path, required=True)
        if action in ("search", "verify"):
            child.add_argument("--cycle", required=True)
        if action == "verify":
            child.add_argument("--retention", action="append", type=Path, default=[])
    args = parser.parse_args()
    if args.stage == "run":
        if args.timeout_seconds <= 0:
            parser.error("--timeout-seconds must be positive")
        result = run(args.base.resolve(), args.materialized.resolve(), args.out.resolve(), args.timeout_seconds)
    else:
        from weightbridge.provenance import peak_rss_bytes

        cpu_started = time.process_time()
        try:
            result = stage(args)
        finally:
            if args.resources:
                json_dump_atomic(args.resources, {"cpu_seconds": time.process_time() - cpu_started,
                                                   "peak_rss_bytes": peak_rss_bytes(),
                                                   "peak_vram_bytes": None})
    print(json.dumps(result, indent=2, sort_keys=True))
