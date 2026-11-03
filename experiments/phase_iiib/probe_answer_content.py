"""Probe teacher-forced answer strings on the frozen common practice screen."""
from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiib.calibrate_choice_tokens import _metrics
from weightbridge.adaptation.bic import _runner
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.evaluation.hf_edit import _load_suite
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.provenance import peak_rss_bytes


RULE = {"primary_score": "mean log probability per answer token", "minimum_accuracy": 0.60,
        "minimum_order_consistency": 0.60, "minimum_predicted_position_share": 0.25,
        "requires_strict_accuracy_and_consistency_gain_over_ab": True}


def _content_row(row):
    question, left, right, instruction = row["text"].rsplit("\n", 3)
    first, second = row["presentation"]
    if (sorted((first, second)) != ["A", "B"] or not left.startswith(first + ". ")
            or not right.startswith(second + ". ") or instruction != "Reply with A or B only."):
        raise ValueError("Frozen practice prompt uses an unexpected answer format")
    options = [left[3:], right[3:]]
    if len(set(options)) != 2 or [first, second].index(row["label"]) != row["answer_position"]:
        raise ValueError("Frozen options or answer position are inconsistent")
    prompt = f"{question}\n- {options[0]}\n- {options[1]}\nReply with only the exact text of the correct option."
    return prompt, options


def _completions(runner, rows, max_input_tokens):
    tokenizer = runner.tokenizer
    template = bool(getattr(tokenizer, "chat_template", None))
    completions = []
    for row in rows:
        prompt, options = _content_row(row)
        if template:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        prefix = tokenizer.encode(prompt, add_special_tokens=not template)
        for option in options:
            suffix = tokenizer.encode(option, add_special_tokens=False)
            if not prefix or not suffix or len(prefix) + len(suffix) > max_input_tokens:
                raise ValueError("Answer content exceeds the frozen token budget or encodes to no tokens")
            joint = tokenizer.encode(prompt + option, add_special_tokens=not template)
            completions.append((prefix + suffix, suffix, joint == prefix + suffix))
    return completions


def _score(runner, torch, completions, batch_size):
    model, tokenizer = runner.model, runner.tokenizer
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Answer content scoring needs a pad or end token")
    forward = inspect.signature(model.forward).parameters
    scores = []
    total_tokens = 0
    with torch.inference_mode():
        for start in range(0, len(completions), batch_size):
            batch = completions[start:start + batch_size]
            width = max(len(sequence) for sequence, _, _ in batch)
            keep = 1 + max(len(answer) for _, answer, _ in batch)
            input_ids = torch.full((len(batch), width), pad, dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for index, (sequence, _, _) in enumerate(batch):
                input_ids[index, -len(sequence):] = torch.tensor(sequence)
                attention_mask[index, -len(sequence):] = 1
            kwargs = {"use_cache": False}
            if "logits_to_keep" in forward:
                kwargs["logits_to_keep"] = keep
            elif "num_logits_to_keep" in forward:
                kwargs["num_logits_to_keep"] = keep
            logits = model(input_ids=input_ids.to(runner.device), attention_mask=attention_mask.to(runner.device),
                           **kwargs).logits
            total_tokens += input_ids.numel()
            for index, (_, answer, joint_matches) in enumerate(batch):
                # The final logit is after the full answer. The preceding
                # logits predict each answer token, starting at the prefix.
                prediction = logits[index, -len(answer)-1:-1].float()
                wanted = torch.tensor(answer, device=prediction.device).unsqueeze(1)
                log_prob = prediction.gather(1, wanted).squeeze(1) - torch.logsumexp(prediction, dim=1)
                if not torch.isfinite(log_prob).all().item():
                    raise ValueError("Answer content scoring produced nonfinite log probabilities")
                scores.append({"answer_tokens": len(answer), "sum_log_probability": float(log_prob.sum().item()),
                               "mean_log_probability": float(log_prob.mean().item()),
                               "joint_tokenization_matches": joint_matches})
    return scores, {"forward_batches": math.ceil(len(completions) / batch_size),
                    "forward_examples": len(completions), "padded_input_token_positions": total_tokens}


def run(base, material, calibration, output):
    wall, cpu = time.perf_counter(), time.process_time()
    if output.exists():
        raise ValueError("Answer content probe report already exists")
    protocol_path = material / "protocol.json"
    protocol = json.loads(protocol_path.read_bytes())
    if protocol.get("schema") != "weightbridge.phase_iiib.qwen_protocol.v1" or protocol.get("scope") != "bounded-pilot":
        raise ValueError("Answer content probe requires the frozen bounded-pilot protocol")
    relative = protocol["cycles"][0]["practice"]["screen"]
    screen = (material / relative).resolve()
    if (not screen.is_relative_to(material / "practice")
            or file_hash(screen) != protocol["artifacts"][relative]):
        raise ValueError("Common practice screen differs from its frozen hash")
    suite = _load_suite(screen.read_bytes())
    if suite["choices"] != ["A", "B"]:
        raise ValueError("Common practice screen must use A/B choices")
    rows = suite["target"] + suite["sentinel"]
    checkpoint = load_checkpoint(base)
    baseline = json.loads(calibration.read_bytes())
    base_hash = protocol["model_sha256"]
    protocol_hash, screen_hash = file_hash(protocol_path), file_hash(screen)
    if (checkpoint.checkpoint_sha256() != base_hash or baseline.get("base_sha256") != base_hash
            or baseline.get("protocol_sha256") != protocol_hash or baseline.get("practice_screen_sha256") != screen_hash
            or baseline.get("schema") != "weightbridge.phase_iiib.choice_calibration.v1"):
        raise ValueError("Answer content probe and A/B calibration must bind the same base and practice data")
    inference = protocol["design"]["inference"]
    with _runner(checkpoint.root, inference) as (runner, torch):
        completions = _completions(runner, rows, inference["max_input_tokens"])
        scores, resources = _score(runner, torch, completions, inference["batch_size"])

    labels = ["first", "second"]
    paired = []
    for index, row in enumerate(rows):
        left, right = scores[2*index:2*index + 2]
        primary = int(right["mean_log_probability"] > left["mean_log_probability"])
        raw = int(right["sum_log_probability"] > left["sum_log_probability"])
        paired.append({**{key: row[key] for key in ("id", "item_key", "answer_position", "family")},
                       "label": labels[row["answer_position"]], "prediction": labels[primary],
                       "sum_score_prediction": labels[raw], "options": [left, right]})
    primary_metrics = _metrics(paired, labels)
    raw_metrics = _metrics([{**row, "prediction": row["sum_score_prediction"]} for row in paired], labels)
    ab_metrics = baseline["conditions"]["AB"]["metrics"]
    checks = {
        "accuracy": primary_metrics["accuracy"] >= RULE["minimum_accuracy"],
        "order_consistency": primary_metrics["order_consistency"] >= RULE["minimum_order_consistency"],
        "position_balance": primary_metrics["minority_prediction_count"] >=
                            math.ceil(len(rows) * RULE["minimum_predicted_position_share"]),
        "accuracy_gain_over_ab": primary_metrics["accuracy"] > ab_metrics["accuracy"],
        "consistency_gain_over_ab": primary_metrics["order_consistency"] > ab_metrics["order_consistency"],
    }
    if load_checkpoint(checkpoint.root).checkpoint_sha256() != base_hash:
        raise ValueError("Answer content source checkpoint changed")
    report = {
        "schema": "weightbridge.phase_iiib.answer_content_probe.v1", "base_sha256": base_hash,
        "protocol_sha256": protocol_hash, "practice_screen_sha256": screen_hash,
        "ab_calibration_sha256": file_hash(calibration), "source_sha256": file_hash(Path(__file__)),
        "scope": "Unedited pinned Qwen and frozen practice only. Changing the requested response also changes the prompt.",
        "length_caveat": "Mean log probability can favor a longer option with predictable suffixes. Sum scores favor shorter options. No stop token is scored.",
        "tokenization": "Every answer is appended as separately encoded tokens after the unchanged chat prefix.",
        "joint_tokenization_mismatches": sum(not value["joint_tokenization_matches"] for value in scores),
        "configuration": inference, "rule": RULE, "checks": checks, "practice_gate_passed": all(checks.values()),
        "ab_baseline": ab_metrics, "mean_log_probability": primary_metrics, "sum_log_probability": raw_metrics,
        "length_normalization_changes": sum(row["prediction"] != row["sum_score_prediction"] for row in paired),
        "rows_with_unequal_answer_lengths": sum(row["options"][0]["answer_tokens"] != row["options"][1]["answer_tokens"] for row in paired),
        "by_family": {family: _metrics([row for row in paired if row["family"] == family], labels)
                      for family in sorted({row["family"] for row in paired})},
        "paired_predictions": paired,
        "resources": {**resources, "wall_seconds": time.perf_counter() - wall,
                      "cpu_seconds": time.process_time() - cpu, "peak_rss_bytes": peak_rss_bytes(), "model_loads": 1},
    }
    json_dump_atomic(output, report)
    print(json.dumps({key: report[key] for key in ("practice_gate_passed", "checks", "ab_baseline",
                                                  "mean_log_probability", "sum_log_probability", "resources")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "material", "calibration", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.material.resolve(), args.calibration, args.out)
