"""Practice-only screen of a programmatically scored one-digit endpoint."""
from __future__ import annotations

import argparse
from collections import defaultdict
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.adaptation.content_scoring import _left_pad, _pad_id
from weightbridge.adaptation.model_adapter import select_hf_adapter
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.provenance import peak_rss_bytes


PROTOCOL = Path(__file__).with_name("digit_protocol.json")
DIGITS = tuple("0123456789")


def render_rows(config):
    rows = []
    for pair in config["target"]:
        a, b = pair
        for wording, template in enumerate(config["target_templates"]):
            rows.append({"item": f"product-{a}-{b}", "wording": wording, "preserve": False,
                         "prompt": template.format(a=a, b=b), "correct": str((a * b) % 10)})
    for digit in config["preservation"]:
        for wording, template in enumerate(config["preservation_templates"]):
            rows.append({"item": f"echo-{digit}", "wording": wording, "preserve": True,
                         "prompt": template.format(digit=digit), "correct": str(digit)})
    return rows


def _prepare(tokenizer, row, limit):
    def encode(value):
        return tuple(tokenizer.encode(value, add_special_tokens=False))

    prefix = encode(row["prompt"])
    if not prefix or len(prefix) + 1 > limit:
        raise ValueError("Digit screen prompt exceeds its frozen token budget")
    answers = []
    for digit in DIGITS:
        joint = encode(row["prompt"] + digit)
        if len(joint) != len(prefix) + 1 or joint[:-1] != prefix:
            raise ValueError("Digit is not a single exact continuation of the frozen prompt")
        answers.append(joint[-1])
    if len(set(answers)) != len(DIGITS) or tokenizer.eos_token_id in answers:
        raise ValueError("Digit continuations must be distinct non-EOS tokens")
    return prefix, answers


def qualify(rows, thresholds):
    groups = defaultdict(list)
    for row in rows:
        groups[row["item"]].append(row)
    targets = {key: group for key, group in groups.items() if not group[0]["preserve"]}
    controls = {key: group for key, group in groups.items() if group[0]["preserve"]}

    def solved(group):
        return all(row["ranker"] == row["exact"] == row["correct"] for row in group)

    stable = sum(group[0]["ranker"] is not None and len({row["ranker"] for row in group}) == 1
                 for group in targets.values())
    solved_target = [key for key, group in targets.items() if solved(group)]
    solved_preservation = [key for key, group in controls.items() if solved(group)]
    persistent = [key for key, group in targets.items()
                  if len({row["exact"] for row in group}) == 1 and
                  all(row["ranker"] == row["exact"] and row["exact"] not in (None, row["correct"]) for row in group)]
    parsed = [row for row in rows if row["exact"] is not None]
    agree = sum(row["ranker"] == row["exact"] for row in parsed)
    gates = {
        "ranker_stability": stable / len(targets) >= thresholds["min_ranker_stability"],
        "exact_parse_rate": len(parsed) / len(rows) >= thresholds["min_exact_parse_rate"],
        "ranker_generation_agreement": bool(parsed) and agree / len(parsed) >= thresholds["min_ranker_generation_agreement"],
        "solved_target_items": len(solved_target) >= thresholds["min_solved_target_items"],
        "persistent_target_errors": len(persistent) >= thresholds["min_persistent_target_errors"],
        "solved_preservation_items": len(solved_preservation) >= thresholds["min_solved_preservation_items"],
    }
    return {"disposition": "PASS_SCREEN" if all(gates.values()) else "HOLD", "gates": gates,
            "ranker_stable_items": [stable, len(targets)], "exact_parsed": [len(parsed), len(rows)],
            "ranker_generation_agreed": [agree, len(parsed)], "solved_target_items": solved_target,
            "persistent_target_errors": persistent, "solved_preservation_items": solved_preservation}


def run(model_key, checkpoint, out):
    import torch
    import transformers

    wall, cpu = time.perf_counter(), time.process_time()
    config = json.loads(PROTOCOL.read_bytes())
    expected = config["checkpoints"][model_key]
    settings = config["inference"]
    adapter = select_hf_adapter(checkpoint)
    if adapter.checkpoint.checkpoint_sha256() != expected:
        raise ValueError("Digit screen requires the frozen base checkpoint")
    out.mkdir(parents=True, exist_ok=False)
    freeze = {"schema": config["schema"], "protocol_sha256": file_hash(PROTOCOL), "checkpoint_sha256": expected,
              "source_sha256": {str(path.relative_to(ROOT).as_posix()): file_hash(path) for path in (
                  Path(__file__), ROOT / "weightbridge/adaptation/model_adapter.py",
                  ROOT / "weightbridge/adaptation/content_scoring.py", ROOT / "weightbridge/inference/huggingface.py",
              )}, "torch": str(torch.__version__), "transformers": transformers.__version__}
    json_dump_atomic(out / "freeze.json", freeze)
    torch.set_num_threads(settings["cpu_threads"])
    runner = adapter.load_runner(device=settings["device"], dtype=settings["dtype"],
                                 max_rss_bytes=settings["max_memory_bytes"])
    rows = render_rows(config)
    prepared = [_prepare(runner.tokenizer, row, settings["max_input_tokens"]) for row in rows]
    pad = _pad_id(runner.tokenizer)
    signature = inspect.signature(runner.model.forward).parameters
    score_calls = generate_calls = prompt_tokens = 0
    with torch.inference_mode():
        for start in range(0, len(rows), settings["batch_size"]):
            stop = start + settings["batch_size"]
            batch = prepared[start:stop]
            input_ids, attention = _left_pad(torch, [prefix for prefix, _ in batch], pad, runner.device)
            prompt_tokens += input_ids.numel()
            args = {"input_ids": input_ids, "attention_mask": attention}
            forward = {"use_cache": False}
            if "position_ids" in signature:
                forward["position_ids"] = (attention.cumsum(-1) - 1).masked_fill(attention == 0, 0)
            if "logits_to_keep" in signature:
                forward["logits_to_keep"] = 1
            scores = runner.model(**args, **forward).logits[:, -1].float().log_softmax(dim=-1)
            score_calls += 1
            generated = runner.model.generate(**args, do_sample=False, num_beams=1, pad_token_id=pad,
                                              max_new_tokens=settings["max_new_tokens"])
            generate_calls += 1
            if generated.shape != (input_ids.shape[0], input_ids.shape[1] + 1):
                raise ValueError("Generation differs from the frozen one-token endpoint")
            for row, (_, answers), score, token_id in zip(rows[start:stop], batch, scores,
                                                          generated[:, -1].tolist(), strict=True):
                values = score[answers].tolist()
                if not torch.isfinite(score[answers]).all().item():
                    raise ValueError("Digit scores must be finite")
                text = runner.tokenizer.decode([token_id], skip_special_tokens=False)
                maximum = max(values)
                row.update(
                    ranker=DIGITS[values.index(maximum)] if values.count(maximum) == 1 else None,
                    exact=text.strip() if text.strip() in DIGITS else None,
                    conditional_log_probabilities=dict(zip(DIGITS, values, strict=True)), digit_token_ids=answers,
                    generated_text=text, generated_token_id=token_id,
                )
    if (file_hash(PROTOCOL) != freeze["protocol_sha256"]
            or any(file_hash(ROOT / path) != digest for path, digest in freeze["source_sha256"].items())
            or adapter.checkpoint.checkpoint_sha256() != expected):
        raise ValueError("Frozen sources or checkpoint changed during measurement")
    raw = out / "predictions.json"
    json_dump_atomic(raw, {"rows": rows})
    report = {
        "schema": config["schema"], "model": model_key, "checkpoint_sha256": expected, "candidate_attempts": 0,
        "freeze_sha256": file_hash(out / "freeze.json"), "predictions_sha256": file_hash(raw),
        "qualification": qualify(rows, config["qualification"]),
        "resources": {"forward_calls": score_calls, "generate_calls": generate_calls, "prompt_tokens": prompt_tokens,
                      "wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
                      "peak_rss_bytes": peak_rss_bytes()},
    }
    json_dump_atomic(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen", "smollm2"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.model, args.checkpoint, args.out), indent=2))
