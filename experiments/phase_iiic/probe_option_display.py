"""Pair labeled and bulleted options on frozen practice questions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiic import run_measurement
from experiments.phase_iiic.measurement_roster import FAMILIES, PROTOCOL
from weightbridge.adaptation import content_scoring
from weightbridge.adaptation.bic import _runner
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.provenance import peak_rss_bytes


DISPLAYS = ("labels", "bullets")
RULES = ("sum", "mean", "generation")


def _prompt(row, display):
    labeled = row["content_prompt"]
    question, first, second, instruction = labeled.rsplit("\n", 3)
    expected = [f"{label}. {answer}" for label, answer in zip(row["labels"], row["answers"], strict=True)]
    if [first, second] != expected:
        raise ValueError("Frozen practice display differs from its two options")
    if display == "labels":
        return labeled
    return "\n".join((question, f"- {row['answers'][0]}", f"- {row['answers'][1]}", instruction))


def _choice(row, display, rule):
    condition = row["displays"][display]
    if rule == "generation":
        return condition["generation"]["completed_parsed_position"]
    return condition["likelihood"][rule]["position"]


def _measure(runner, rows, config):
    resources = {}
    for display in DISPLAYS:
        prompts = [_prompt(row, display) for row in rows]
        inference, generation = config["inference"], config["generation"]
        options = content_scoring.prepare_continuations(
            runner.tokenizer, prompts, [row["answers"] for row in rows],
            max_input_tokens=inference["max_input_tokens"], include_eos=config["scoring"]["include_eos"],
        )
        scores, score_resources = content_scoring.score_continuations(runner, options, batch_size=inference["batch_size"])
        pairs = run_measurement._paired_scores(scores, [row["correct_position"] for row in rows],
                                              run_measurement.CONTENT_RULES)
        generated, generate_resources = content_scoring.generate_responses(
            runner, prompts, max_input_tokens=inference["max_input_tokens"],
            max_new_tokens=generation["max_new_tokens"], batch_size=generation["batch_size"],
        )
        for row, prompt, (likelihood, tokens), response in zip(rows, prompts, pairs, generated, strict=True):
            parsed = run_measurement.parse_response(response["text"], row["answers"], row["labels"], row["aliases"], "content")
            normalized = run_measurement._normalized(response["text"])
            row["displays"][display] = {"prompt": prompt, "likelihood": likelihood, "continuations": tokens,
                                        "generation": {**response, "parsed_position": parsed,
                                                       "completed_parsed_position": parsed if response["stop_reason"] == "eos" else None,
                                                       "exact_label_shaped_output": normalized in {"a", "b", "x", "y"}}}
        resources[display] = {"scoring": score_resources, "generation": generate_resources}
    return resources


def _summaries(rows, settings):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["item_id"]].append(row)
    if any(len(pair) != 2 or {row["correct_position"] for row in pair} != {0, 1} for pair in grouped.values()):
        raise ValueError("Display comparison requires a reversed presentation pair for each item")
    summaries, item_records = {}, {}
    for display in DISPLAYS:
        generated = [row["displays"][display]["generation"] for row in rows]
        summaries[display] = {
            "eos_completed": sum(result["stop_reason"] == "eos" for result in generated),
            "exact_label_shaped_outputs": sum(result["exact_label_shaped_output"] for result in generated),
            "label_output_counts": dict(Counter(run_measurement._normalized(result["text"]) for result in generated
                                               if result["exact_label_shaped_output"])),
            "parsed": sum(result["parsed_position"] is not None for result in generated),
            "completed_and_parsed": sum(result["completed_parsed_position"] is not None for result in generated),
        }
        for rule in RULES:
            per_item = {}
            for item_id, pair in grouped.items():
                positions = [_choice(row, display, rule) for row in pair]
                identities = [row["order"][position] if position is not None else None
                              for row, position in zip(pair, positions, strict=True)]
                per_item[item_id] = {"family": pair[0]["family"], "chosen_option_ids": identities,
                                     "correct_presentations": identities.count(0),
                                     "same_option_in_both_orders": identities[0] is not None and identities[0] == identities[1]}
            summaries[display][rule] = {
                "correct_presentations": sum(item["correct_presentations"] for item in per_item.values()),
                "both_orders_correct": sum(item["correct_presentations"] == 2 for item in per_item.values()),
                "same_option_in_both_orders": sum(item["same_option_in_both_orders"] for item in per_item.values()),
            }
            item_records.setdefault(display, {})[rule] = per_item
    differences = {}
    for rule in RULES:
        rng = np.random.default_rng(settings["seed"])
        strata, pooled = [], []
        for family in FAMILIES:
            changes = np.array([(item_records["bullets"][rule][item_id]["correct_presentations"] -
                                 item_records["labels"][rule][item_id]["correct_presentations"]) / 2
                                for item_id, pair in grouped.items() if pair[0]["family"] == family])
            sampled = rng.choice(changes, size=(settings["replicates"], len(changes)), replace=True)
            strata.append(sampled.mean(axis=1))
            pooled.extend(changes)
        draws = np.stack(strata).mean(axis=0)
        tail = (1 - settings["interval"]) / 2
        differences[rule] = {"accuracy_difference": float(np.mean(pooled)),
                             "question_cluster_interval": np.quantile(draws, [tail, 1 - tail]).tolist()}
    return summaries, item_records, differences


def run(base, material, out):
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    config = json.loads(PROTOCOL.read_bytes())
    manifest_path = material / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    checkpoint = load_checkpoint(base)
    model_hash = checkpoint.checkpoint_sha256()
    if (manifest["protocol_sha256"] != file_hash(PROTOCOL)
            or not model_hash == manifest["model_sha256"] == config["model_sha256"]):
        raise ValueError("Display probe requires the frozen practice and pinned source checkpoint")
    practice = run_measurement._load_split(material, manifest, "practice")
    selected = [item for family in FAMILIES for item in [row for row in practice if row["family"] == family][:2]]
    if len(selected) != 6:
        raise ValueError("Display probe requires two practice items per family")
    rows = [{**row, "displays": {}} for row in run_measurement._rows(selected) if row["paraphrase"] == 0]
    with _runner(checkpoint.root, config["inference"]) as (runner, _):
        resources = _measure(runner, rows, config)
    summaries, per_item, differences = _summaries(rows, config["bootstrap"])
    if load_checkpoint(checkpoint.root).checkpoint_sha256() != model_hash:
        raise ValueError("Display probe source checkpoint changed")
    report = {"schema": "weightbridge.phase_iiic.option_display_probe.v1", "scope": "exploratory frozen practice only",
              "hypothesis": "Removing visible labels may improve completed content responses under the same instruction.",
              "model_sha256": model_hash, "protocol_sha256": file_hash(PROTOCOL),
              "practice_sha256": manifest["artifacts"]["practice"]["sha256"],
              "material_manifest_sha256": file_hash(manifest_path),
              "source_sha256": {path.relative_to(ROOT).as_posix(): file_hash(path) for path in
                                 (Path(__file__), Path(run_measurement.__file__), Path(content_scoring.__file__))},
              "selection": "First two frozen practice items per family, both orders of the first paraphrase",
              "underlying_items": len(selected), "presentations_per_display": len(rows),
              "inference": config["inference"], "generation": config["generation"], "summaries": summaries,
              "per_item": per_item, "bullets_minus_labels": differences, "bootstrap": config["bootstrap"],
              "rows": rows, "resources": {**resources, "wall_seconds": time.perf_counter() - start_wall,
                                           "cpu_seconds": time.process_time() - start_cpu,
                                           "peak_rss_bytes": peak_rss_bytes(), "model_loads": 1}}
    json_dump_atomic(out, report)
    print(json.dumps({key: report[key] for key in ("summaries", "bullets_minus_labels", "resources")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "material", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.material.resolve(), args.out)
