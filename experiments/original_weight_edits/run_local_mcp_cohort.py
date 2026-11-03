"""Run a local Transformers diagnostic model against the real WeightBridge MCP broker."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
from threading import Thread
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.evaluation.freeze_cohort import freeze_protocol
from weightbridge.evaluation.mcp_cohort import _canonical_hash, _file_sha256, _read_json, summarize_cohort
from weightbridge.diagnostics.toy_mlp import ToyMLP
from weightbridge.provenance import peak_rss_bytes


SOURCE = Path(__file__).resolve()
TEMPLATE = SOURCE.with_name("mcp_llm_transfer_v2.template.json")
BROKER = SOURCE.with_name("run_mcp_agent_experiment.py")
MODEL_ID = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
TURN_LIMIT = 8
TOKEN_LIMIT = 160
THREADS = 6
REPEAT_LIMIT = 3

GUIDE = """Return one JSON object with keys tool (the tool name) and arguments (an object).
First call get_diagnostic_report with empty arguments.
The model computes h=tanh(x @ input_weight + hidden_bias), then logits=h @ output_weight + output_bias. Highest logit wins.
For submit_patch, arguments are tensor (exact name from diagnostics), index (array of zero-based integers, one per tensor axis), delta (signed number to add), and reason (short causal hypothesis).
For evaluate_practice or finalize_candidate, arguments contain candidate_id (the alias returned by submit_patch).
For stop_without_candidate, arguments contain reason (a short explanation).
Work from observed values. Evaluate each accepted candidate before deciding whether to finalize it."""

FORM_GUIDE = """You use WeightBridge MCP through a patch form. The host calls get_diagnostic_report and evaluate_practice for you.
The network computes h=tanh(x @ input_weight + hidden_bias), then logits=h @ output_weight + output_bias. Highest logit wins.
Use the observed activation and logit differences to locate a single faulty scalar.
Reply with only a JSON object with four keys: tensor (exact name), index (array of zero-based integers, one per tensor axis), delta (signed number to add), reason (short causal hypothesis).
After seeing practice feedback you can propose another patch, return an object with just select (an evaluated candidate alias to finalize), or return an object with just stop (a reason)."""


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_lines(stream, lines):
    for line in stream:
        lines.put(line)
    lines.put(None)


def decode_request(text):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    request = json.loads(text)
    if (not isinstance(request, dict) or set(request) != {"tool", "arguments"}
            or not isinstance(request["tool"], str) or not isinstance(request["arguments"], dict)):
        raise ValueError("expected a single tool request with an arguments object")
    return request


def form_request(text):
    form = json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
    if isinstance(form, dict):
        if set(form) == {"tensor", "index", "delta", "reason"}:
            return {"tool": "submit_patch", "arguments": form}
        if set(form) == {"select"}:
            return {"tool": "finalize_candidate", "arguments": {"candidate_id": form["select"]}}
        if set(form) == {"stop"}:
            return {"tool": "stop_without_candidate", "arguments": {"reason": form["stop"]}}
    raise ValueError("expected a patch form, a selection, or a stop reason")


def diagnostic_view(result):
    failures = result["practice_failures"]
    examples = []
    for label in (0, 1):
        group = [row for row in failures["examples"] if row["expected_class"] == label]
        for row in group[:2]:
            examples.append({
                "x": [round(value, 3) for value in row["input"]],
                "expected": label, "predicted": row["predicted_class"],
                "logits": [round(value, 3) for value in row["logits"]],
                "h": [round(value, 3) for value in row["hidden_activation"]],
            })
    return {
        "practice": result["practice_baseline"], "failure_examples": examples,
        "tensors": {name: {key: round(value, 3) if isinstance(value, float) else value
                           for key, value in stats.items() if key != "dtype"}
                    for name, stats in result["tensor_summary"].items()},
        "policy": result["host_policy"],
    }


def generate(tokenizer, model, messages):
    import torch

    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False,
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    prompt_tokens = encoded["input_ids"].shape[1]
    if prompt_tokens > 16_384:
        raise ValueError("host input token budget exceeded")
    with torch.inference_mode():
        tokens = model.generate(**encoded, do_sample=False, max_new_tokens=TOKEN_LIMIT)
    new_tokens = tokens[0, prompt_tokens:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), prompt_tokens, len(new_tokens)


def check_output_path(path):
    if os.name == "nt":
        receipt = (path / "trial-00" / "candidates" / "sessions" / ("codex-" + "0" * 32)
                   / ("candidate-" + "0" * 32 + ".npz.receipt.json"))
        if len(str(receipt).encode("utf-16-le")) // 2 >= 260:
            raise ValueError("candidate receipt exceeds the Windows path budget; use a short --out path")


def run_task(task_id, root, protocol_path, broker_python, tokenizer, model, prompt, mode,
             connectivity=None):
    output = root / task_id
    lines = Queue()
    started = time.perf_counter()
    started_cpu = time.process_time()
    aliases = {}
    base_sha = None
    diagnostic_shown = False
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": GUIDE}]
    events_path = root / f"{task_id}.model_turns.jsonl"
    with (root / f"{task_id}.broker.stderr.txt").open("x", encoding="utf-8") as stderr, \
            events_path.open("x", encoding="utf-8", newline="\n") as events:
        proc = subprocess.Popen(
            [str(broker_python), str(BROKER), "--cohort-manifest", str(protocol_path),
             "--task-id", task_id, "--out", str(output)],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
            text=True, encoding="utf-8",
        )
        Thread(target=read_lines, args=(proc.stdout, lines), daemon=True).start()

        def receive():
            line = lines.get(timeout=60)
            if line is None:
                raise RuntimeError(f"MCP broker closed early for {task_id}")
            return json.loads(line)

        def send(request):
            proc.stdin.write(json.dumps(request, allow_nan=False) + "\n")
            proc.stdin.flush()
            return receive()

        def log(event):
            events.write(json.dumps(event, ensure_ascii=True, allow_nan=False) + "\n")
            events.flush()

        try:
            ready = receive()
            if ready.get("ready") is not True:
                raise RuntimeError(f"MCP broker did not initialize for {task_id}")
            allowed = {tool["name"] for tool in ready["tools"]}
            if mode == "patch-form":
                diagnostic = send({"tool": "get_diagnostic_report", "arguments": {}})
                if not diagnostic.get("ok"):
                    raise RuntimeError("host diagnostic call failed")
                log({"host_call": "get_diagnostic_report"})
                base_sha = diagnostic["result"]["base_sha256"]
                view = diagnostic_view(diagnostic["result"])
                if connectivity:
                    checkpoint = output / "base.npz"
                    if _file_sha256(checkpoint) != base_sha:
                        raise RuntimeError("base checkpoint differs from the broker diagnostic")
                    view["logit_connectivity"] = ToyMLP.load(checkpoint).logit_connectivity()
                if connectivity is not None:
                    write_json(output / "diagnostics_sent.json", view)
                messages[-1]["content"] = FORM_GUIDE + "\nPractice diagnostics:\n" + json.dumps(
                    view, separators=(",", ":"))
            ended = False
            last_output = None
            repeats = 0
            stop_reason = "model turn budget exhausted"
            for turn in range(1, TURN_LIMIT + 1):
                text, input_tokens, output_tokens = generate(tokenizer, model, messages)
                messages.append({"role": "assistant", "content": text})
                log({"turn": turn, "raw_output": text, "input_tokens": input_tokens,
                     "output_tokens": output_tokens, "at_token_cap": output_tokens == TOKEN_LIMIT})
                repeats = repeats + 1 if text == last_output else 1
                last_output = text
                if repeats == REPEAT_LIMIT:
                    stop_reason = "model repeated the same response three times"
                    break
                try:
                    request = form_request(text) if mode == "patch-form" else decode_request(text)
                    tool, arguments = request["tool"], request["arguments"]
                    if tool not in allowed:
                        raise ValueError("tool is not available")
                    if tool == "submit_patch":
                        if set(arguments) != {"tensor", "index", "delta", "reason"} or not base_sha:
                            raise ValueError("provide tensor, index, delta and reason after diagnostics")
                        request["arguments"] = {"patch": {
                            "schema": "weightbridge.patch.v1", "base_sha256": base_sha,
                            "reason": arguments["reason"], "evidence_ids": ["practice-diagnostics"],
                            "changes": [{"tensor": arguments["tensor"], "index": arguments["index"],
                                         "op": "add", "value": arguments["delta"]}],
                        }}
                    candidate_id = arguments.get("candidate_id")
                    if isinstance(candidate_id, str) and candidate_id in aliases:
                        arguments["candidate_id"] = aliases[candidate_id]
                    json.dumps(request, allow_nan=False)
                except ValueError as exc:
                    log({"turn": turn, "transport_error": str(exc)})
                    messages.append({"role": "user", "content": "Invalid request. Use one JSON object in the specified format."})
                    continue
                response = send(request)
                if response.get("experiment_complete") is True:
                    ended = True
                    break
                result = response.get("result")
                if response.get("ok") and isinstance(result, dict):
                    if tool == "get_diagnostic_report":
                        base_sha = result.get("base_sha256")
                        response = ({"practice_diagnostics": diagnostic_view(result),
                                     "next": "Infer which single coordinate caused the errors. Propose a numerical patch, or stop if no diagnosis is supported."}
                                    if not diagnostic_shown else {"status": "See previous diagnostics"})
                        diagnostic_shown = True
                    if tool == "submit_patch" and result.get("candidate_id"):
                        alias = f"C{len(aliases) + 1}"
                        aliases[alias] = result["candidate_id"]
                        response = {"candidate_alias": alias, "next": "evaluate_practice"}
                        if mode == "patch-form":
                            evaluation = send({"tool": "evaluate_practice",
                                               "arguments": {"candidate_id": aliases[alias]}})
                            if not evaluation.get("ok"):
                                raise RuntimeError("host practice evaluation failed")
                            log({"host_call": "evaluate_practice", "candidate_alias": alias})
                            response = {"candidate_alias": alias,
                                        "practice_result": evaluation["result"]["result"],
                                        "next": "Select an evaluated alias, propose a revision, or stop."}
                    if tool == "evaluate_practice":
                        response = {"practice_result": result["result"],
                                    "next": "Choose whether to finalize the evaluated alias or revise the diagnosis."}
                if response.get("ok") is False:
                    response = {"error": str(response.get("error", ""))[:350],
                                "next": "Correct the request using the diagnostics. Do not repeat a rejected patch."}
                messages.append({"role": "user", "content": json.dumps(response, separators=(",", ":"))})
            if not ended:
                log({"host_stop": stop_reason})
                response = send({"tool": "stop_without_candidate",
                                 "arguments": {"reason": "Host stop: " + stop_reason}})
                if response.get("experiment_complete") is not True:
                    raise RuntimeError("budget stop did not finalize the experiment")
            if proc.wait(timeout=30) != 0:
                raise RuntimeError("MCP broker exited with an error")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=15)
    write_json(root / f"{task_id}.host_resources.json", {
        "wall_seconds": time.perf_counter() - started,
        "host_cpu_seconds": time.process_time() - started_cpu,
        "host_peak_rss_bytes": peak_rss_bytes(),
    })
    return _read_json(output / "experiment_record.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--broker-python", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, required=True,
                        help="fresh split seed offset, at least 20000, unique per study")
    parser.add_argument("--tasks", type=int, choices=range(1, 21), default=20,
                        metavar="1-20", help="execute a prefix of the frozen order")
    parser.add_argument("--mode", choices=("tool-json", "patch-form"), default="tool-json")
    parser.add_argument("--paired-connectivity", action="store_true",
                        help="compare control and connectivity views on identical fresh tasks")
    args = parser.parse_args()
    if args.seed_offset < 20_000:
        parser.error("--seed-offset must be at least 20000")
    if args.paired_connectivity and args.mode != "patch-form":
        parser.error("--paired-connectivity requires --mode patch-form")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    out = args.out.resolve()
    if not out.is_relative_to(ROOT):
        parser.error("--out must be within the checkout for portable evidence paths")
    arm_paths = {"control": out / "control", "probe": out / "probe"} if args.paired_connectivity else {"single": out}
    for path in arm_paths.values():
        check_output_path(path)
    broker_python = args.broker_python.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    checkpoint_files = {path.name: _file_sha256(path) for path in sorted(model_path.iterdir())
                        if path.is_file()}
    reference = _read_json(ROOT / "results/published_reproduction/p50c_rebuild_summary.json")
    checkpoint_sha = reference["base_checkpoint_sha256"]
    if checkpoint_files.get("model.safetensors-00001-of-00001.safetensors") != checkpoint_sha:
        raise ValueError("local checkpoint differs from the pinned Qwen base")
    template = _read_json(TEMPLATE)
    # Fresh study data and a balanced four-mechanism prefix for a bounded smoke.
    by_id = {task["task_id"]: task for task in template["tasks"]}
    for task in template["tasks"]:
        task["seed"] += args.seed_offset
    prefix = list({}.fromkeys(by_id[item]["mechanism"] for item in template["execution_order"]))
    first_ids = [next(item for item in template["execution_order"] if by_id[item]["mechanism"] == mechanism)
                 for mechanism in prefix]
    template["execution_order"] = first_ids + [item for item in template["execution_order"] if item not in first_ids]
    template["protocol_sha256"] = _canonical_hash(template)
    split_seeds = [seed for task in template["tasks"] for seed in (task["seed"], task["seed"] + 1)]
    if len(split_seeds) != len(set(split_seeds)):
        raise ValueError("cohort practice and holdout seeds overlap")
    prompt = (ROOT / template["prompt_path"]).read_text(encoding="utf-8")
    started = time.perf_counter()
    started_cpu = time.process_time()
    torch.set_num_threads(THREADS)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True,
    ).eval()
    out.mkdir(parents=True, exist_ok=False)
    identity = {
        "schema": "weightbridge.llm_identity_receipt.v1",
        "provider": "local-transformers", "model_id": MODEL_ID, "version": REVISION,
        "checkpoint_files_sha256": checkpoint_files,
        "host_source_sha256": _file_sha256(SOURCE), "planned_task_prefix": args.tasks,
        "interaction_mode": args.mode,
        "diagnostic_view": "first two failures per class rounded to three decimals; practice metrics and tensor summaries",
        "sampling": {"do_sample": False, "enable_thinking": False, "max_new_tokens": TOKEN_LIMIT,
                     "max_model_turns": TURN_LIMIT, "max_input_tokens": 16_384, "device": "cpu",
                     "identical_response_stop": REPEAT_LIMIT,
                     "dtype": "bfloat16", "torch_threads": THREADS, "torch": version("torch"),
                     "transformers": version("transformers")},
    }
    manifests = {}
    for arm, path in arm_paths.items():
        path.mkdir(exist_ok=True)
        draft_path = path / "study_template.json"
        write_json(draft_path, template)
        receipt_path = path / "local_model_identity.json"
        receipt = dict(identity, host_call_id=uuid.uuid4().hex)
        if args.paired_connectivity:
            receipt["connectivity_probe"] = arm == "probe"
            receipt["toy_model_source_sha256"] = _file_sha256(ROOT / "weightbridge/diagnostics/toy_mlp.py")
        write_json(receipt_path, receipt)
        manifest = path / "frozen_protocol.json"
        freeze_protocol(draft_path, receipt_path, manifest, cohort_id=(
            f"{out.name}-{arm}" if args.paired_connectivity else out.name),
            identity_provenance="Local checkpoint matched the pinned SHA-256; local files hashed, offline inference with a fresh chat per task.")
        manifests[arm] = manifest
    schedule = []
    for index, task_id in enumerate(template["execution_order"][:args.tasks]):
        arms = list(arm_paths)
        if index % 2:
            arms.reverse()
        schedule.extend({"task": task_id, "arm": arm} for arm in arms)
    if args.paired_connectivity:
        write_json(out / "paired_plan.json", {
            "schema": "weightbridge.connectivity_ablation.v1",
            "seed_offset": args.seed_offset, "schedule": schedule,
            "hypothesis": "The probe increases tasks with at least one accepted edit changing practice logits.",
            "primary": "fraction of paired tasks with any accepted edit changing practice logits by more than 1e-9",
            "secondary": ["practice class changes", "finalized holdout repairs", "accepted edits",
                          "model token counts", "CPU seconds"],
            "protocol_sha256": {arm: _read_json(path)["protocol_sha256"]
                                for arm, path in manifests.items()},
        })
    records = {arm: [] for arm in arm_paths}
    failure = None
    try:
        for step in schedule:
            task_id, arm = step["task"], step["arm"]
            print(f"Starting {arm} {task_id}", flush=True)
            record = run_task(task_id, arm_paths[arm], manifests[arm], broker_python,
                              tokenizer, model, prompt, args.mode,
                              arm == "probe" if args.paired_connectivity else None)
            records[arm].append(record)
            print(f"Completed {arm} {task_id}", flush=True)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for arm, path in arm_paths.items():
            write_json(path / "cohort_summary.json",
                       summarize_cohort(_read_json(manifests[arm]), records[arm]))
            resources = {"wall_seconds": time.perf_counter() - started,
                         "host_cpu_seconds": time.process_time() - started_cpu,
                         "host_peak_rss_bytes": peak_rss_bytes(), "completed_tasks": len(records[arm]),
                         "failure": failure}
            if args.paired_connectivity:
                measured = [_read_json(path / f"{row['protocol_binding']['task_id']}.host_resources.json")
                            for row in records[arm]]
                for field in ("wall_seconds", "host_cpu_seconds"):
                    resources[field] = sum(item[field] for item in measured)
            write_json(path / "host_resources.json", resources)
        if args.paired_connectivity:
            write_json(out / "host_resources.json", {
                "wall_seconds": time.perf_counter() - started,
                "host_cpu_seconds": time.process_time() - started_cpu,
                "host_peak_rss_bytes": peak_rss_bytes(),
                "completed_tasks": sum(len(rows) for rows in records.values()), "failure": failure,
            })


if __name__ == "__main__":
    main()
