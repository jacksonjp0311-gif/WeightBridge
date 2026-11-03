"""Measure frozen Phase III-D grounding on the original pinned checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiid.analysis import analyze_split, select_practice
from weightbridge.adaptation.bic import _runner
from weightbridge.adaptation.content_scoring import (
    generate_responses,
    prepare_continuations,
    score_continuations,
)
from weightbridge.adaptation.grounding import parse_generation
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.provenance import peak_rss_bytes


PROTOCOL = Path(__file__).with_name("protocol.json")
ROSTER_SCHEMA = "weightbridge.phase_iiid.grounding_roster.v1"
MANIFEST_SCHEMA = "weightbridge.phase_iiid.grounding_manifest.v1"
SPLITS = ("practice", "selection", "qualification")
DISPLAYS = ("labeled", "bullets")
CONTEXTS = ("task", "counterfactual", "neutral_absent", "neutral_generic", "irrelevant")
GENERATION_CONTEXTS = ("task", "counterfactual")


def _load_split(material, manifest, split, seen):
    path = material / f"{split}.json"
    expected = manifest["artifacts"][split]
    if file_hash(path) != expected["sha256"]:
        raise ValueError(f"Frozen {split} roster hash differs from the manifest")
    doc = json.loads(path.read_bytes())
    items = doc["items"]
    semantic_keys = {item["semantic_key"] for item in items}
    if (doc.get("schema") != ROSTER_SCHEMA or doc.get("split") != split
            or len(items) != expected["underlying_items"] or len(semantic_keys) != len(items)
            or seen & semantic_keys):
        raise ValueError(f"Frozen {split} roster is malformed or overlaps a previous split")
    seen.update(semantic_keys)
    return items


def _rows(items):
    rows = []
    for item in items:
        presentations = item["presentations"]
        renders = {(tuple(row["order"]), row["paraphrase"]) for row in presentations}
        if len(presentations) != 4 or renders != {(order, variant) for order in ((0, 1), (1, 0))
                                                 for variant in (0, 1)}:
            raise ValueError("Each item requires two paraphrases crossed with two option orders")
        for presentation in presentations:
            order = presentation["order"]
            rows.append({**presentation, "item_id": item["id"], "family": item["family"],
                         "answers": [item["answers"][choice] for choice in order],
                         "aliases": [item["aliases"][choice] for choice in order],
                         "correct_position": order.index(0), "counterfactual_position": order.index(1)})
    return rows


def _resources(wall, cpu, resources):
    return {**resources, "wall_seconds": time.perf_counter() - wall,
            "cpu_seconds": time.process_time() - cpu, "peak_rss_bytes": peak_rss_bytes()}


def _score(runner, prompts, answers, inference, *, letter=False):
    wall, cpu = time.perf_counter(), time.process_time()
    continuations = prepare_continuations(runner.tokenizer, prompts, answers,
                                          max_input_tokens=inference["max_input_tokens"], include_eos=not letter)
    if letter and any(len(option.answer_ids) != 1 for option in continuations):
        raise ValueError("Frozen letter baseline requires one token for each label")
    records, resources = score_continuations(runner, continuations, batch_size=inference["batch_size"])
    for option, record in zip(continuations, records, strict=True):
        record["answer_token_ids"] = list(option.answer_ids)
    pairs = list(zip(records[::2], records[1::2], strict=True))
    return pairs, _resources(wall, cpu, resources)


def _generate(runner, rows, prompts, display, config):
    wall, cpu = time.perf_counter(), time.process_time()
    generation = config["generation"]
    results, resources = generate_responses(
        runner, prompts, max_input_tokens=config["inference"]["max_input_tokens"],
        max_new_tokens=generation["max_new_tokens"], batch_size=generation["batch_size"],
    )
    parsed = []
    for row, result in zip(rows, results, strict=True):
        decision = parse_generation(result["text"], display=display, labels=row["labels"],
                                    answers=row["answers"], aliases=row["aliases"],
                                    completed=result["stop_reason"] == "eos")
        parsed.append({**result, **decision})
    return parsed, _resources(wall, cpu, resources)


def _evaluate_split(runner, items, config, split):
    rows = _rows(items)
    results = [{key: value for key, value in row.items() if key != "displays"} for row in rows]
    for row in results:
        row["displays"] = {}
    inference = config["inference"]
    answers = [row["answers"] for row in rows]
    resources = {}
    for display in DISPLAYS:
        for result in results:
            result["displays"][display] = {"contexts": {}, "generation": {}}
        measured = {"contexts": {}, "generation": {}}
        for context in CONTEXTS:
            prompts = [row["displays"][display][context] for row in rows]
            pairs, measured["contexts"][context] = _score(runner, prompts, answers, inference)
            for result, pair in zip(results, pairs, strict=True):
                result["displays"][display]["contexts"][context] = pair
        if display == "labeled":
            prompts = [row["displays"][display]["letter_prompt"] for row in rows]
            pairs, measured["letter"] = _score(runner, prompts, [row["labels"] for row in rows],
                                                inference, letter=True)
            for result, pair in zip(results, pairs, strict=True):
                result["displays"][display]["letter"] = pair
        print(f"{split} {display} likelihoods recorded", file=sys.stderr, flush=True)
        for context in GENERATION_CONTEXTS:
            prompts = [row["displays"][display][context] for row in rows]
            generated, measured["generation"][context] = _generate(runner, rows, prompts, display, config)
            for result, response in zip(results, generated, strict=True):
                result["displays"][display]["generation"][context] = response
            print(f"{split} {display} {context} generation recorded", file=sys.stderr, flush=True)
        resources[display] = measured
    return results, resources


def _source_hashes():
    paths = [Path(__file__).with_name(name) for name in ("analysis.py", "roster.py", "run_grounding.py")] + [
        ROOT / "weightbridge/adaptation/content_scoring.py",
        ROOT / "weightbridge/adaptation/grounding.py",
        ROOT / "weightbridge/adaptation/bic.py",
    ]
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths}


def run(base, material, output):
    wall, cpu = time.perf_counter(), time.process_time()
    manifest_path = material / "manifest.json"
    manifest, config = json.loads(manifest_path.read_bytes()), json.loads(PROTOCOL.read_bytes())
    manifest_hash = file_hash(manifest_path)
    checkpoint = load_checkpoint(base)
    base_hash, protocol_hash = checkpoint.checkpoint_sha256(), file_hash(PROTOCOL)
    if (base_hash != config["model_sha256"] or manifest.get("schema") != MANIFEST_SCHEMA
            or manifest["model_sha256"] != base_hash or manifest["protocol_sha256"] != protocol_hash
            or manifest["generator_sha256"] != file_hash(Path(__file__).with_name("roster.py"))):
        raise ValueError("Grounding requires the frozen manifest and original pinned checkpoint")
    output.mkdir(parents=True)
    source_hashes = _source_hashes()
    json_dump_atomic(output / "execution-sources.json", {
        "schema": "weightbridge.phase_iiid.grounding_execution.v1", "base_sha256": base_hash,
        "protocol_sha256": protocol_hash, "material_manifest_sha256": manifest_hash,
        "source_sha256": source_hashes, "inference": config["inference"], "generation": config["generation"],
        "generation_contexts": GENERATION_CONTEXTS, "scored_contexts": CONTEXTS,
    })
    seen = set()
    summaries = {}
    decision_path = output / "practice-decision.json"
    with _runner(checkpoint.root, config["inference"]) as (runner, _):
        for split in SPLITS:
            if (_source_hashes() != source_hashes or file_hash(PROTOCOL) != protocol_hash
                    or file_hash(manifest_path) != manifest_hash):
                raise ValueError("Frozen implementation or protocol changed during measurement")
            split_wall, split_cpu = time.perf_counter(), time.process_time()
            rows, phases = _evaluate_split(runner, _load_split(material, manifest, split, seen), config, split)
            predictions = output / f"{split}-predictions.json"
            resources = _resources(split_wall, split_cpu, {"phases": phases})
            json_dump_atomic(predictions, {"split": split, "rows": rows, "resources": resources})
            if split == "practice":
                decision = select_practice(rows, config)
                json_dump_atomic(decision_path, {**decision, "protocol_sha256": protocol_hash,
                                                 "practice_predictions_sha256": file_hash(predictions)})
            summaries[split] = {"predictions_sha256": file_hash(predictions),
                                "resources": resources, "analysis": analyze_split(rows, config, decision)}
            print(f"{split} closed with {len(rows)} presentations", file=sys.stderr, flush=True)
    if (_source_hashes() != source_hashes or file_hash(PROTOCOL) != protocol_hash
            or file_hash(manifest_path) != manifest_hash
            or load_checkpoint(checkpoint.root).checkpoint_sha256() != base_hash):
        raise ValueError("Frozen implementation, protocol, or checkpoint changed during measurement")
    report = {
        "schema": "weightbridge.phase_iiid.grounding_report.v1", "base_sha256": base_hash,
        "protocol_sha256": protocol_hash, "material_manifest_sha256": manifest_hash,
        "execution_sources_sha256": file_hash(output / "execution-sources.json"),
        "practice_decision_sha256": file_hash(decision_path), "splits": summaries,
        "resources": _resources(wall, cpu, {"model_loads": 1}),
    }
    json_dump_atomic(output / "report.json", report)
    print(json.dumps(report["resources"], indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "material", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.material, args.out)
