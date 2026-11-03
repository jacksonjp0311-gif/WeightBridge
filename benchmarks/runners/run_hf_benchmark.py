"""Run a bounded, frozen JSON/JSONL text benchmark on one local HF model.

Expected math input rows: {"id": str, "prompt": str, "answer": str}.
IFEval rows also carry kind="ifeval", instruction_id_list and kwargs.
The file root may be {"split": ..., "items": [...]} or a JSONL stream.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weightbridge.inference import HFTextRunner, InferenceConfig
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.provenance import peak_rss_bytes
from weightbridge.evaluation.loops import loop_signals


def read_dataset(path: Path, split: str) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        doc = json.loads(raw)
        if isinstance(doc, dict):
            declared = doc.get("split")
            if declared is not None and declared != split:
                raise ValueError(f"dataset split is {declared!r}, requested {split!r}")
            rows = doc.get("items", doc.get("prompts"))
            suffix = doc.get("instruction_suffix", "")
        elif isinstance(doc, list):
            rows, suffix = doc, ""
        else:
            raise ValueError("JSON dataset must be a list or object")
    except json.JSONDecodeError:
        rows, suffix = [], ""
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            rows.append(row)
    if not isinstance(rows, list) or not rows:
        raise ValueError("dataset must contain at least one row")
    seen = set()
    normalized = []
    for row in rows:
        if not isinstance(row, dict) or not all(isinstance(row.get(key), str) and row[key]
                                                for key in ("id", "prompt")):
            raise ValueError("each dataset row requires nonempty string id and prompt")
        kind = row.get("kind", "math")
        if kind not in {"math", "ifeval"}:
            raise ValueError(f"unsupported dataset row kind: {kind!r}")
        if kind == "math" and not isinstance(row.get("answer"), str):
            raise ValueError("math rows require an answer string")
        if kind == "ifeval" and (not isinstance(row.get("instruction_id_list"), list)
                                  or not isinstance(row.get("kwargs"), list)
                                  or len(row["instruction_id_list"]) != len(row["kwargs"])
                                  or not row["instruction_id_list"]):
            raise ValueError("IFEval rows require matched instruction_id_list and kwargs lists")
        if row["id"] in seen:
            raise ValueError(f"duplicate problem id: {row['id']}")
        seen.add(row["id"])
        prompt = row["prompt"] + (f"\n{suffix}" if suffix else "")
        normalized.append({"id": row["id"], "prompt": prompt, "kind": kind,
                           **({"answer": row["answer"]} if kind == "math" else {
                               "instruction_id_list": row["instruction_id_list"], "kwargs": row["kwargs"]})})
    return normalized, digest


def normalize_answer(text: str) -> str:
    value = text.strip().replace(",", "")
    value = re.sub(r"\s+", "", value)
    value = value.strip("$ ")
    return value.lower()


def extract_answer(text: str) -> str | None:
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    index = text.rfind("\\boxed")
    marker = "\\boxed"
    if index < 0:
        index = text.rfind("\\fbox")
        marker = "\\fbox"
        if index < 0:
            return None
    rest = text[index + len(marker):]
    if rest.lstrip().startswith("{"):
        value = rest.lstrip()
        depth = 0
        for position, char in enumerate(value):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    answer = value[1:position].strip()
                    return normalize_answer(answer) if answer else None
        return None
    if marker == "\\boxed" and rest.startswith(" "):
        answer = rest.strip().split("$")[0].split("\n")[0].strip()
        return normalize_answer(answer) if answer else None
    return None


def math_verify_correct(prediction: str | None, answer: str) -> bool:
    """Use the authors' math-verify parsing and equivalence check when requested."""
    if prediction is None:
        return False
    try:
        import math_verify
    except ImportError as exc:
        raise RuntimeError("Install the benchmark extra to use math-verify grading") from exc
    gold = math_verify.parse("\\boxed{" + answer + "}", parsing_timeout=5)
    predicted = math_verify.parse("\\boxed{" + prediction + "}", parsing_timeout=5)
    if not gold or not predicted:
        return False
    try:
        return bool(math_verify.verify(gold, predicted, timeout_seconds=5))
    except Exception:
        return False


def ifeval_correct(row: dict, text: str, *, thinking_enabled: bool,
                   source_path: Path | None = None) -> tuple[bool, list[bool]]:
    """Apply Google's official IFEval instruction registry to one response."""
    if source_path is not None:
        path = str(source_path.resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        from instructions_registry import INSTRUCTION_DICT
    except ImportError as exc:
        raise RuntimeError("IFEval grading needs the official instruction_following_eval source path") from exc
    response = text.split("</think>")[-1].strip() if "</think>" in text else ("" if thinking_enabled else text.strip())
    follow = []
    for instruction_id, kwargs in zip(row["instruction_id_list"], row["kwargs"]):
        instruction = INSTRUCTION_DICT[instruction_id](instruction_id)
        clean_kwargs = {key: value for key, value in (kwargs or {}).items() if value is not None}
        instruction.build_description(**clean_kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=row["prompt"])
        follow.append(bool(response) and bool(instruction.check_following(response)))
    return all(follow), follow


def repetition_proxy(tokenizer, text: str) -> float:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < 16:
        return 0.0
    split = max(0, len(tokens) - 128)
    previous = tokens[max(0, split - 128):split]
    tail = tokens[split:]
    previous_grams = {tuple(previous[i:i + 8]) for i in range(max(0, len(previous) - 7))}
    tail_grams = [tuple(tail[i:i + 8]) for i in range(max(0, len(tail) - 7))]
    return sum(gram in previous_grams for gram in tail_grams) / len(tail_grams) if tail_grams else 0.0


def run(model_path: Path, dataset_path: Path, output: Path, *, variant: str, split: str,
        n: int, seed: int, max_new_tokens: int, max_input_tokens: int, temperature: float,
        top_p: float, top_k: int, presence_penalty: float, timeout_seconds: float,
        device: str, dtype: str, cpu_threads: int, thinking_enabled: bool,
        grader: str = "exact", ifeval_source: Path | None = None,
        max_rss_bytes: int = 8 * 1024**3) -> dict:
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    if type(n) is not int or not 1 <= n <= 16:
        raise ValueError("n must be in [1, 16]")
    output.mkdir(parents=True, exist_ok=False)
    setup_start = time.perf_counter()
    dataset_sha = None
    model_sha = None
    try:
        rows, dataset_sha = read_dataset(dataset_path, split)
        checkpoint = load_checkpoint(model_path)
        model_sha = checkpoint.checkpoint_sha256()
        runner = HFTextRunner.from_pretrained(str(model_path.resolve()), device=device, dtype=dtype,
                                              local_files_only=True, max_rss_bytes=max_rss_bytes)
        record_config = InferenceConfig(max_new_tokens=max_new_tokens,
                                        max_input_tokens=max_input_tokens,
                                        temperature=temperature, top_p=top_p, top_k=top_k,
                                        presence_penalty=presence_penalty, seed=seed,
                                        thinking_enabled=thinking_enabled,
                                        timeout_seconds=timeout_seconds, cpu_threads=cpu_threads,
                                        max_rss_bytes=max_rss_bytes)
    except Exception as exc:
        failure = {"schema": "weightbridge.hf_benchmark_failure.v1",
                   "timestamp_utc": datetime.now(timezone.utc).isoformat(), "variant": variant,
                   "model_path": str(model_path), "model_sha256": model_sha,
                   "dataset_path": str(dataset_path), "dataset_sha256": dataset_sha,
                   "split": split, "exception_type": type(exc).__name__, "error": str(exc)[:4000],
                   "setup_wall_seconds": time.perf_counter() - setup_start,
                   "peak_rss_bytes": peak_rss_bytes()}
        with (output / "run_failure.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(failure, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        raise
    generations = []
    start = time.perf_counter()
    cpu_start = time.process_time()
    for problem_index, row in enumerate(rows):
        for sample_index in range(n):
            sample_seed = seed + problem_index * n + sample_index
            config = InferenceConfig(**{**record_config.__dict__, "seed": sample_seed})
            try:
                result = runner.generate(row["prompt"], config)
                signals = loop_signals(result.text)
                if row["kind"] == "ifeval":
                    correct, instruction_results = ifeval_correct(
                        row, result.text, thinking_enabled=thinking_enabled, source_path=ifeval_source)
                    extracted = None
                    answer_completed = bool(result.text.strip())
                    expected = None
                else:
                    extracted = extract_answer(result.text)
                    correct = (math_verify_correct(extracted, row["answer"])
                               if grader == "math-verify" else extracted == normalize_answer(row["answer"]))
                    instruction_results = None
                    answer_completed = extracted is not None
                    expected = normalize_answer(row["answer"])
                generations.append({
                    "problem_id": row["id"], "sample_index": sample_index, "seed": sample_seed,
                    "kind": row["kind"], "status": "completed", "expected_answer": expected,
                    "extracted_answer": extracted, "correct": correct,
                    "answer_completed": answer_completed, "boxed_answer_completed": extracted is not None,
                    "ifeval_instruction_results": instruction_results,
                    "reasoning_loop_signals": signals,
                    "reasoning_loop": any(signals.values()),
                    "token_limit": result.finish_reason == "max_new_tokens",
                    "termination_reason": result.finish_reason,
                    "prompt_tokens": result.prompt_tokens, "generated_tokens": result.generated_tokens,
                    "elapsed_seconds": result.elapsed_seconds, "tokens_per_second": result.tokens_per_second,
                    "peak_rss_bytes": result.peak_rss_bytes, "peak_vram_bytes": result.peak_vram_bytes,
                    "tail_repetition_8gram_rate": repetition_proxy(runner.tokenizer, result.text),
                    "prompt_sha256": result.prompt_sha256, "generated_text": result.text,
                })
            except Exception as exc:
                generations.append({"problem_id": row["id"], "sample_index": sample_index,
                                    "seed": sample_seed, "status": "failed",
                                    "error_type": type(exc).__name__, "error": str(exc)[:2000]})
    elapsed = time.perf_counter() - start
    completed = [item for item in generations if item["status"] == "completed"]
    total = len(generations)
    summary = {
        "schema": "weightbridge.hf_benchmark_run.v1",
        "run_id": f"{variant}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{hashlib.sha256((model_sha+dataset_sha).encode()).hexdigest()[:8]}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "variant": variant,
        "model_path": str(model_path.resolve()), "model_sha256": model_sha,
        "dataset_path": str(dataset_path.resolve()), "dataset_sha256": dataset_sha,
        "split": split, "problem_count": len(rows), "samples_per_problem": n,
        "sampling": {"seed_base": seed, "temperature": temperature, "top_p": top_p,
                     "top_k": top_k, "presence_penalty": presence_penalty,
                     "thinking_enabled": thinking_enabled, "max_new_tokens": max_new_tokens,
                     "max_input_tokens": max_input_tokens, "timeout_seconds": timeout_seconds,
                     "max_rss_bytes": max_rss_bytes,
                     "device": device, "dtype": dtype, "cpu_threads": cpu_threads,
                     "grader": "official_ifeval" if any(row["kind"] == "ifeval" for row in rows) else grader},
        "summary": {
            "attempted_generations": total, "completed_generations": len(completed),
            "failures": total - len(completed),
            "accuracy": sum(bool(item["correct"]) for item in completed) / len(completed) if completed else None,
            "answer_completion_rate": sum(bool(item["answer_completed"]) for item in completed) / len(completed) if completed else None,
            "token_limit_rate": sum(bool(item["token_limit"]) for item in completed) / len(completed) if completed else None,
            "reasoning_loop_rate": sum(bool(item["reasoning_loop"]) for item in completed) / len(completed) if completed else None,
            "mean_tail_repetition_8gram_rate": sum(item["tail_repetition_8gram_rate"] for item in completed) / len(completed) if completed else None,
            "mean_generated_tokens": sum(item["generated_tokens"] for item in completed) / len(completed) if completed else None,
            "total_generated_tokens": sum(item["generated_tokens"] for item in completed),
            "mean_generation_seconds": sum(item["elapsed_seconds"] for item in completed) / len(completed) if completed else None,
            "mean_tokens_per_second": sum(item["tokens_per_second"] for item in completed) / len(completed) if completed else None,
        },
        "resources": {"model_load_seconds": runner.model_load_seconds, "model_parameters": runner.parameter_count,
                      "run_wall_seconds": elapsed, "cpu_seconds": time.process_time() - cpu_start,
                      "peak_rss_bytes": max((value for value in [peak_rss_bytes(),
                                                                  *(item.get("peak_rss_bytes") for item in completed)]
                                              if value is not None), default=None),
                      "peak_vram_bytes": max((value for value in (item.get("peak_vram_bytes") for item in completed)
                                               if value is not None), default=None)},
        "generations": generations,
        "grader": "official_ifeval" if any(row["kind"] == "ifeval" for row in rows) else grader,
        "scope_note": "Use math-verify plus prepared frozen dataset artifacts for author-compatible math scoring; IFEval needs the official instruction_following_eval source checkout.",
    }
    with (output / "run.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"run_id": summary["run_id"], "summary": summary["summary"],
                      "artifact": str(output / "run.json")}, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variant", default="candidate")
    parser.add_argument("--split", choices=["practice", "holdout", "development"], required=True)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--max-rss-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--grader", choices=["exact", "math-verify"], default="math-verify")
    parser.add_argument("--ifeval-source", type=Path, default=None,
                        help="Path to google-research/instruction_following_eval for official IFEval grading")
    args = parser.parse_args()
    run(Path(args.model), Path(args.dataset), Path(args.out), variant=args.variant, split=args.split,
        n=args.n, seed=args.seed, max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens, temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, presence_penalty=args.presence_penalty,
        timeout_seconds=args.timeout_seconds, device=args.device, dtype=args.dtype,
        cpu_threads=args.cpu_threads, thinking_enabled=not args.disable_thinking,
        grader=args.grader, ifeval_source=args.ifeval_source,
        max_rss_bytes=args.max_rss_bytes)


if __name__ == "__main__":
    main()
