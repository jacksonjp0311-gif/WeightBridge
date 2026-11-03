"""Run an interactive LLM-to-WeightBridge MCP toy discovery experiment.

The host keeps hidden examples in memory and only writes them after the agent
has finalized a practice-evaluated candidate. While running, send one JSON line
per MCP tool call on stdin, for example:

    {"tool":"get_diagnostic_report","arguments":{}}

The host returns the MCP result as JSON on stdout. This JSONL bridge is a local
experiment harness; an ordinary MCP host can connect to the same configured
server directly.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weightbridge.core import apply_patch, file_hash
from weightbridge.diagnostics.toy_mlp import ToyMLP
from weightbridge.evaluation.metrics import paired_bootstrap_interval
from weightbridge.provenance import append_experiment, peak_rss_bytes


MECHANISMS = (
    {"tensor": "input_weight", "index": [0, 0], "damaged_value": -2.0, "mechanism": "input_direction_reversal"},
    {"tensor": "hidden_bias", "index": [0], "damaged_value": -0.3, "mechanism": "hidden_threshold_shift"},
    {"tensor": "output_weight", "index": [0, 0], "damaged_value": 3.0, "mechanism": "output_class_collapse"},
    {"tensor": "output_bias", "index": [1], "damaged_value": 4.5, "mechanism": "output_intercept_shift"},
)


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _append_tool_exchange(path: Path, sequence: int, request: Any, response: dict) -> None:
    event = {"schema": "weightbridge.mcp_exchange.v1", "sequence": sequence,
             "timestamp_utc": datetime.now(timezone.utc).isoformat(),
             "request": request, "response": response}
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _correct_model() -> ToyMLP:
    return ToyMLP({
        "input_weight": np.array([[2.0, 0.0], [0.0, 2.0]]),
        "hidden_bias": np.array([-1.0, -1.0]),
        "output_weight": np.array([[-3.0, 3.0], [0.0, 0.0]]),
        "output_bias": np.array([0.0, 0.0]),
    })


def _make_split(gold_model: ToyMLP, seed: int, per_class: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    low = np.column_stack([rng.uniform(0.05, 0.45, per_class), rng.uniform(0.0, 1.0, per_class)])
    high = np.column_stack([rng.uniform(0.55, 0.95, per_class), rng.uniform(0.0, 1.0, per_class)])
    features = np.concatenate([low, high])
    labels = gold_model.forward(features).argmax(axis=1).astype(np.int64)
    order = rng.permutation(len(labels))
    return {"features": features[order], "labels": labels[order]}


def make_blinded_task(seed: int, per_class: int) -> dict:
    if type(seed) is not int or seed < 0 or type(per_class) is not int or per_class < 8:
        raise ValueError("seed must be nonnegative and per_class at least 8")
    gold = _correct_model()
    choice_rng = np.random.default_rng(seed ^ 0x5A17)
    oracle = dict(MECHANISMS[int(choice_rng.integers(0, len(MECHANISMS)))])
    damaged = {name: np.array(value, copy=True) for name, value in gold.tensors.items()}
    tensor_name = oracle["tensor"]
    index = tuple(oracle["index"])
    oracle["correct_value"] = float(damaged[tensor_name][index])
    damaged[tensor_name][index] = float(oracle["damaged_value"])
    oracle["corrective_delta"] = oracle["correct_value"] - oracle["damaged_value"]
    model = ToyMLP(damaged)
    return {
        "model": model,
        "practice": _make_split(gold, seed, per_class),
        "holdout": _make_split(gold, seed + 1, per_class),
        "oracle": oracle,
        "dataset": "synthetic-blinded-single-weight-defect-v1",
    }


def _score(checkpoint: Path, split: dict[str, np.ndarray]) -> dict:
    model = ToyMLP.load(checkpoint)
    logits = model.forward(split["features"])
    predictions = logits.argmax(axis=1)
    correct = predictions == split["labels"]
    return {"n": int(len(correct)), "correct": int(correct.sum()),
            "accuracy": float(correct.mean()), "per_example_correct": correct.astype(bool).tolist(),
            "class_errors": {str(label): int(np.count_nonzero((split["labels"] == label) & ~correct))
                             for label in (0, 1)},
            "completion_rate": 1.0, "token_limit_rate": None,
            "reasoning_loop_rate": None}


def _artifact_paths(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()) + [
        "experiment_record.json"]


def _finalize_experiment(output: Path, task: dict, *, seed: int, per_class: int,
                         finalized: dict | None, tool_calls: int,
                         started_wall: float, started_cpu: float) -> dict:
    base_path = output / "base.npz"
    base_hash = file_hash(base_path)
    holdout = task["holdout"]
    hidden_path = output / "hidden_holdout.npz"
    np.savez_compressed(hidden_path, features=holdout["features"], labels=holdout["labels"])
    _write_json(output / "host_oracle.json", {
        "schema": "weightbridge.blinded_task_oracle.v1",
        "note": "Written only after the agent finalized or stopped; never available in the MCP host config.",
        "seed": seed, **task["oracle"],
    })
    baseline = _score(base_path, holdout)
    candidate_result = None
    control_result = None
    candidate_id = None
    patch = None
    patch_receipt = None
    control_receipt = None
    paired_ci = None
    random_ci = None
    if finalized is not None and finalized.get("candidate_id"):
        candidate_id = finalized["candidate_id"]
        candidate_path = output / "candidates" / "sessions" / finalized["session_id"] / f"candidate-{candidate_id}.npz"
        patch_path = candidate_path.with_name(f"patch-{candidate_id}.json")
        receipt_path = candidate_path.with_suffix(candidate_path.suffix + ".receipt.json")
        patch = json.loads(patch_path.read_text(encoding="utf-8"))
        patch_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if file_hash(candidate_path) != finalized["candidate_sha256"]:
            raise RuntimeError("Finalized MCP candidate hash does not match the frozen receipt")
        candidate_result = _score(candidate_path, holdout)
        # One seeded random scalar control matched to the AI edit's realized magnitude.
        rng = np.random.default_rng(seed + 100003)
        all_coordinates = [(name, tuple(index)) for name, arr in task["model"].tensors.items()
                           for index in np.ndindex(arr.shape)]
        coordinate_number = int(rng.integers(0, len(all_coordinates)))
        control_tensor, control_index = all_coordinates[coordinate_number]
        magnitude = float(patch_receipt["max_abs_delta"])
        sign = int(rng.choice(np.array([-1, 1])))
        control_delta = magnitude * sign
        policy = {"schema": "weightbridge.policy.v1",
                  "allow_tensors": list(ToyMLP.tensor_names),
                  "max_entries": 1, "max_fraction": 1 / 12, "max_abs_delta": 8.0}
        control_patch = {
            "schema": "weightbridge.patch.v1", "base_sha256": base_hash,
            "reason": "Seeded random one-entry control matched by edit magnitude and parameter count.",
            "evidence_ids": ["mcp-pilot-random-control"],
            "changes": [{"tensor": control_tensor, "index": list(control_index),
                         "op": "add", "value": control_delta}],
            "projections": [],
        }
        control_path = output / "random_control.npz"
        control_receipt = apply_patch(base_path, control_patch, policy, control_path)
        _write_json(output / "random_control_patch.json", control_patch)
        control_result = _score(control_path, holdout)
        base_correct = np.asarray(baseline["per_example_correct"], dtype=float)
        candidate_correct = np.asarray(candidate_result["per_example_correct"], dtype=float)
        control_correct = np.asarray(control_result["per_example_correct"], dtype=float)
        paired_ci = paired_bootstrap_interval(base_correct, candidate_correct, seed=seed)
        random_ci = paired_bootstrap_interval(base_correct, control_correct, seed=seed + 1)

    _write_json(output / "evaluation.json", {
        "schema": "weightbridge.mcp_toy_evaluation.v1",
        "recorded_after_agent_finalization": True,
        "baseline": baseline, "ai_candidate": candidate_result,
        "random_control": control_result,
        "ai_vs_baseline_paired_bootstrap_95": paired_ci,
        "random_vs_baseline_paired_bootstrap_95": random_ci,
    })
    source_tensors = task["model"].tensors
    measurement_material = {
        "suite": task["dataset"], "seed": seed, "per_class": per_class,
        "base_sha256": base_hash, "practice_sha256": file_hash(output / "practice.npz"),
        "holdout_sha256": file_hash(hidden_path),
    }
    measurement_key = hashlib.sha256(json.dumps(measurement_material, sort_keys=True,
                                                separators=(",", ":")).encode()).hexdigest()
    experiment_id = f"codex-mcp-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    if candidate_result is None:
        outcome = "no_candidate_finalized"
        disposition = "INCONCLUSIVE"
        affected = {"tensor": None, "count": 0, "fraction": 0.0}
    else:
        delta = candidate_result["accuracy"] - baseline["accuracy"]
        outcome = "improved" if delta > 0 else "unchanged" if delta == 0 else "degraded"
        affected = {"tensor": patch_receipt["changes"][0]["tensor"],
                    "count": patch_receipt["changed_entries"],
                    "fraction": patch_receipt["changed_fraction"],
                    "max_abs_delta": patch_receipt["max_abs_delta"],
                    "candidate_tensor_index": patch_receipt["changes"][0]["index"]}
        # A single task is a discovery pilot; it cannot pass the planned multi-seed transfer gate.
        disposition = "INCONCLUSIVE"
    wall = time.perf_counter() - started_wall
    cpu = time.process_time() - started_cpu
    record = {
        "schema": "weightbridge.experiment.v1", "experiment_id": experiment_id,
        "measurement_key": measurement_key, "replicate_role": "single_instance_mcp_discovery_pilot",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "An LLM using standard MCP tools can infer and propose a bounded repair from weight summaries and practice-only failures without receiving the correction.",
        "source_model": {"id": "synthetic-toy-mlp-blinded-single-weight-defect-v1",
                         "sha256": base_hash,
                         "tensor_sha256": {name: hashlib.sha256(value.tobytes()).hexdigest()
                                           for name, value in source_tensors.items()}},
        "proposed_modification": patch,
        "affected_parameters": affected,
        "control_modification": None if control_receipt is None else {
            "tensor": control_receipt["changes"][0]["tensor"],
            "index": control_receipt["changes"][0]["index"],
            "delta": control_receipt["changes"][0]["after"] - control_receipt["changes"][0]["before"],
            "candidate_sha256": control_receipt["candidate_sha256"],
        },
        "evaluation": {
            "dataset": task["dataset"], "version": "1", "seed": seed,
            "practice_n": int(len(task["practice"]["labels"])),
            "holdout_n": int(len(holdout["labels"])), "practice_test_separated": True,
            "holdout_accessed_by_agent": False,
            "metrics": {"baseline": baseline, "ai_candidate": candidate_result,
                        "random_control": control_result},
            "ai_vs_baseline_paired_accuracy_ci": paired_ci,
            "random_vs_baseline_paired_accuracy_ci": random_ci,
            "pilot_outcome": outcome,
        },
        "resources": {"wall_seconds": wall, "cpu_seconds": cpu,
                      "peak_rss_bytes": peak_rss_bytes(), "peak_vram_bytes": None,
                      "inference_tokens": 0, "ai_tokens": None,
                      "python": platform.python_version(), "platform": platform.platform(),
                      "numpy_version": np.__version__, "mcp_sdk_version": version("mcp"),
                      "mcp_tool_calls": tool_calls,
                      "note": "AI token and MCP child-process resource use were not available from the host."},
        "agent": {"type": "LLM-directed-MCP-tool-caller",
                  "model": "Codex session; provider token meter unavailable",
                  "tool_calls": tool_calls,
                  "candidate_finalized": finalized is not None},
        "task_oracle": task["oracle"],
        "artifact_root": output.relative_to(ROOT).as_posix() if output.is_relative_to(ROOT) else str(output),
        "artifacts": _artifact_paths(output),
        "disposition": disposition,
        "scientific_scope": "one AI-directed synthetic discovery pilot; not evidence of transfer, Qwen improvement, or general learning",
    }
    _write_json(output / "experiment_record.json", record)
    append_experiment(ROOT / "results" / "experiment_ledger.jsonl", record)
    return record


async def run_session(output: Path, *, seed: int, per_class: int) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    task = make_blinded_task(seed, per_class)
    base_path = output / "base.npz"
    task["model"].save(base_path)
    np.savez_compressed(output / "practice.npz", **task["practice"])
    session_id = f"codex-{uuid.uuid4().hex}"
    config = {
        "schema": "weightbridge.agent_session.v1", "session_id": session_id,
        "base_checkpoint": str(base_path.resolve()),
        "practice_dataset": str((output / "practice.npz").resolve()),
        "candidate_directory": str((output / "candidates").resolve()),
        "policy": {"schema": "weightbridge.policy.v1", "allow_tensors": list(ToyMLP.tensor_names),
                   "max_entries": 1, "max_fraction": 1 / 12, "max_abs_delta": 8.0},
        "max_candidates": 8, "max_failure_examples": 48,
    }
    config_path = output / "host_session.json"
    _write_json(config_path, config)
    transcript_path = output / "agent_mcp_transcript.jsonl"
    with transcript_path.open("x", encoding="utf-8", newline="\n"):
        pass
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "weightbridge.agents.mcp_server", "--config", str(config_path)],
        cwd=str(ROOT), env=None,
    )
    tool_calls = 0
    finalized: dict | None = None
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as client:
            await client.initialize()
            listing = await client.list_tools()
            tools = [{"name": item.name, "description": item.description,
                      "input_schema": getattr(item, "inputSchema", None)} for item in listing.tools]
            sys.stdout.write(json.dumps({"ready": True, "experiment": "blinded-toy-MCP-discovery",
                                         "session_id": session_id, "tools": tools}, allow_nan=False) + "\n")
            sys.stdout.flush()
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                tool_calls += 1
                call_number = tool_calls
                if len(line.encode("utf-8")) > 65536:
                    response = {"ok": False, "error": "tool request exceeds 64 KiB"}
                    _append_tool_exchange(transcript_path, call_number,
                                          {"raw_request_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest()}, response)
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
                    continue
                request = None
                try:
                    request = json.loads(line)
                    if not isinstance(request, dict) or set(request) != {"tool", "arguments"}:
                        raise ValueError("request must have exactly tool and arguments fields")
                    if not isinstance(request["tool"], str) or not isinstance(request["arguments"], dict):
                        raise ValueError("tool must be text and arguments must be an object")
                    if tool_calls > 40:
                        raise ValueError("host MCP call budget (40 total) is exhausted")
                    result = await client.call_tool(request["tool"], request["arguments"])
                    if result.isError:
                        content = [getattr(item, "text", "") for item in result.content]
                        response = {"ok": False, "tool": request["tool"], "error": "\n".join(content)}
                        _append_tool_exchange(transcript_path, call_number, request, response)
                        sys.stdout.write(json.dumps(response, allow_nan=False) + "\n")
                        sys.stdout.flush()
                        continue
                    payload = getattr(result, "structuredContent", None)
                    if payload is None:
                        payload = {"content": [getattr(item, "text", "") for item in result.content]}
                    response = {"ok": True, "tool": request["tool"], "result": payload}
                    _append_tool_exchange(transcript_path, call_number, request, response)
                    if request["tool"] == "finalize_candidate":
                        finalized = payload
                        break
                    if request["tool"] == "stop_without_candidate":
                        finalized = {"session_id": session_id, "candidate_id": None, "stopped": True}
                        break
                    sys.stdout.write(json.dumps(response, allow_nan=False) + "\n")
                    sys.stdout.flush()
                except Exception as exc:
                    response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    if request is None:
                        request = {"raw_request": line.rstrip("\r\n")[:65536]}
                    _append_tool_exchange(transcript_path, call_number, request, response)
                    sys.stdout.write(json.dumps(response, allow_nan=False) + "\n")
                    sys.stdout.flush()

    if finalized is None:
        raise RuntimeError("Agent session ended without finalization or an explicit stop")
    record = _finalize_experiment(output, task, seed=seed, per_class=per_class,
                                  finalized=finalized, tool_calls=tool_calls,
                                  started_wall=_STARTED_WALL, started_cpu=_STARTED_CPU)
    sys.stdout.write(json.dumps({"experiment_complete": True,
                                 "experiment_id": record["experiment_id"],
                                 "disposition": record["disposition"],
                                 "pilot_outcome": record["evaluation"]["pilot_outcome"],
                                 "artifacts": record["artifacts"]}, allow_nan=False) + "\n")
    sys.stdout.flush()
    return record


_STARTED_WALL = time.perf_counter()
_STARTED_CPU = time.process_time()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a blinded toy discovery session through WeightBridge MCP")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=28431)
    parser.add_argument("--per-class", type=int, default=32)
    args = parser.parse_args()
    output = Path(args.out) if args.out else ROOT / "results" / "original_weight_edits" / f"mcp-pilot-{uuid.uuid4().hex[:10]}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    asyncio.run(run_session(output, seed=args.seed, per_class=args.per_class))


if __name__ == "__main__":
    main()
