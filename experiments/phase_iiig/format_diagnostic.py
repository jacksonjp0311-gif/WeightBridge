"""Measure prompt whitespace, tokenization, and generation cap effects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.adaptation.content_scoring import _left_pad, _pad_id
from weightbridge.adaptation.model_adapter import select_hf_adapter
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.provenance import peak_rss_bytes


PROTOCOL = Path(__file__).with_name("format_protocol.json")
DIGIT_LINE = re.compile(r"[ \t]*([0-9])[ \t]*")
HORIZONTAL_WHITESPACE = re.compile(r"[ \t]+")


def classify_digit(text, *, ended):
    """Accept a single ASCII digit on the first line after a line or EOS boundary."""
    line, boundary, _ = text.replace("\r\n", "\n").partition("\n")
    match = DIGIT_LINE.fullmatch(line)
    return {
        "digit_payload": match[1] if match else None,
        "complete_digit": match[1] if match and (boundary or ended) else None,
        "line_boundary": bool(boundary),
    }


def render(config):
    groups = {
        "copy": ("Copy the input digit. Write one digit on the answer line.",
                 [(f"Input {digit}", str(digit)) for digit in config["copy_examples"]],
                 [(f"copy-{digit}", f"Input {digit}", str(digit)) for digit in config["copy"]]),
        "last_digit_product": ("Write the last digit of the product on the answer line.",
                               [(f"{a} * {b}", str(a * b % 10)) for a, b in config["product_examples"]],
                               [(f"product-{a}-{b}", f"{a} * {b}", str(a * b % 10))
                                for a, b in config["last_digit_product"]]),
    }
    for family, (instruction, examples, items) in groups.items():
        for item, question, answer in items:
            for prefix_name, prefix in config["prefixes"].items():
                demonstration = [prefix.format(question=example) + " " + value
                                 for example, value in examples]
                for tail_name, tail in config["prompt_tails"].items():
                    prompt = "\n".join([instruction, *demonstration, prefix.format(question=question) + tail])
                    yield {"family": family, "item": item, "prefix": prefix_name,
                           "tail": tail_name, "prompt": prompt, "expected": answer}


def _continuation(ids, eos_id, cap):
    if eos_id in ids:
        end = ids.index(eos_id)
        return ids[:end], ids[:end + 1], True
    if len(ids) != cap:
        raise ValueError("Non-EOS generation stopped before the frozen cap")
    return ids, ids, False


def summarize(rows, caps):
    cells, transitions = {}, []
    by_prompt = {}
    for row in rows:
        key = (row["family"], row["prefix"], row["tail"], row["cap"])
        count = cells.setdefault(key, dict(presentations=0, whitespace_first=0, payload_parsed=0,
                                          payload_correct=0, complete_parsed=0, complete_correct=0))
        count["presentations"] += 1
        count["whitespace_first"] += row["whitespace_first"]
        count["payload_parsed"] += row["digit_payload"] is not None
        count["payload_correct"] += row["digit_payload"] == row["expected"]
        count["complete_parsed"] += row["complete_digit"] is not None
        count["complete_correct"] += row["complete_digit"] == row["expected"]
        by_prompt.setdefault((row["item"], row["prefix"], row["tail"]), {})[row["cap"]] = row
    for key, observed in by_prompt.items():
        shortest, longest = (observed[cap] for cap in (caps[0], caps[-1]))
        transitions.append({
            "item": key[0], "family": shortest["family"], "prefix": key[1], "tail": key[2],
            "whitespace_first": shortest["whitespace_first"],
            "whitespace_then_digit": shortest["whitespace_first"] and longest["digit_payload"] is not None,
            "whitespace_then_correct": shortest["whitespace_first"] and longest["digit_payload"] == shortest["expected"],
            "whitespace_then_wrong": shortest["whitespace_first"] and longest["digit_payload"] not in (None, shortest["expected"]),
            "prefix_consistent": all(
                row["generated_token_ids"] == longest["generated_token_ids"][:len(row["generated_token_ids"])]
                for row in observed.values()
            ),
        })
    return {"cells": [dict(family=k[0], prefix=k[1], tail=k[2], cap=k[3], **v)
                      for k, v in cells.items()], "cap_transitions": transitions}


def run(model_key, checkpoint, out):
    import torch
    import transformers

    wall, cpu = time.perf_counter(), time.process_time()
    config = json.loads(PROTOCOL.read_bytes())
    settings = config["inference"]
    adapter = select_hf_adapter(checkpoint)
    expected = config["checkpoints"][model_key]
    if adapter.checkpoint.checkpoint_sha256() != expected:
        raise ValueError("Unexpected checkpoint for frozen diagnostic")
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / "weightbridge/adaptation/content_scoring.py",
               ROOT / "weightbridge/adaptation/model_adapter.py",
               ROOT / "weightbridge/inference/huggingface.py"]
    freeze = {
        "schema": config["schema"], "checkpoint_sha256": expected,
        "protocol_sha256": file_hash(PROTOCOL),
        "source_sha256": {path.relative_to(ROOT).as_posix(): file_hash(path) for path in sources},
        "torch": str(torch.__version__), "transformers": transformers.__version__,
    }
    json_dump_atomic(out / "freeze.json", freeze)
    torch.set_num_threads(settings["cpu_threads"])
    runner = adapter.load_runner(device=settings["device"], dtype=settings["dtype"],
                                 max_rss_bytes=settings["max_memory_bytes"])
    tokenizer = runner.tokenizer
    rows = list(render(config))
    prefixes = [tokenizer.encode(row["prompt"], add_special_tokens=False) for row in rows]
    if not all(0 < len(prefix) <= settings["max_input_tokens"] for prefix in prefixes):
        raise ValueError("Prompt exceeds the frozen token budget")
    eos = tokenizer.eos_token_id
    if type(eos) is not int:
        raise ValueError("Diagnostic requires a single tokenizer EOS token")

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    results = []
    generate_calls = padded_positions = generated_tokens = 0
    pad = _pad_id(tokenizer)
    with torch.inference_mode():
        for cap in config["caps"]:
            for start in range(0, len(rows), settings["batch_size"]):
                stop = start + settings["batch_size"]
                batch = prefixes[start:stop]
                input_ids, attention = _left_pad(torch, batch, pad, runner.device)
                generated = runner.model.generate(
                    input_ids=input_ids, attention_mask=attention, do_sample=False, num_beams=1,
                    max_new_tokens=cap, pad_token_id=pad, eos_token_id=eos,
                    use_cache=True, return_dict_in_generate=False,
                )
                if (generated.ndim != 2 or generated.shape[0] != len(batch)
                        or not torch.equal(generated[:, :input_ids.shape[1]], input_ids)):
                    raise ValueError("Generation returned unexpected prompt sequences")
                generate_calls += 1
                padded_positions += input_ids.numel()
                sequences = generated[:, input_ids.shape[1]:].tolist()
                for row, prompt_ids, generated_ids in zip(rows[start:stop], batch, sequences, strict=True):
                    if not 1 <= len(generated_ids) <= cap:
                        raise ValueError("Generation returned an invalid number of tokens")
                    text_ids, retained_ids, ended = _continuation(generated_ids, eos, cap)
                    pieces = [decode([token]) for token in retained_ids]
                    text = decode(text_ids)
                    generated_tokens += len(retained_ids)
                    results.append({
                        **row, "cap": cap, "prompt_token_ids": prompt_ids,
                        "prompt_tail_token_text": [decode([token]) for token in prompt_ids[-4:]],
                        "generated_token_ids": retained_ids, "generated_token_text": pieces,
                        "text": text, "stop": "eos" if ended else "cap",
                        "whitespace_first": bool(text_ids and HORIZONTAL_WHITESPACE.fullmatch(pieces[0])),
                        **classify_digit(text, ended=ended),
                    })
    if (adapter.checkpoint.checkpoint_sha256() != expected or file_hash(PROTOCOL) != freeze["protocol_sha256"]
            or any(file_hash(ROOT / name) != digest for name, digest in freeze["source_sha256"].items())):
        raise ValueError("Frozen diagnostic inputs changed during execution")
    raw = out / "predictions.json"
    json_dump_atomic(raw, {"rows": results})
    report = {
        "schema": config["schema"], "model": model_key, "checkpoint_sha256": expected,
        "freeze_sha256": file_hash(out / "freeze.json"), "predictions_sha256": file_hash(raw),
        "resources": {"generate_calls": generate_calls, "padded_prompt_tokens": padded_positions,
                      "generated_tokens": generated_tokens, "wall_seconds": time.perf_counter() - wall,
                      "cpu_seconds": time.process_time() - cpu, "peak_rss_bytes": peak_rss_bytes()},
        **summarize(results, config["caps"]),
    }
    json_dump_atomic(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen", "smollm2"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.model, args.checkpoint, args.out)
    print(json.dumps({"model": args.model, "resources": result["resources"]}, indent=2))
