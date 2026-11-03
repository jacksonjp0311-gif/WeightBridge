"""Run the prospectively frozen Phase III-E Qwen measurement and archive evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiid.run_grounding import _generate, _resources, _score
from experiments.phase_iiie.qwen_analysis import DIRECTIONS, analyze, classify_headroom, lock
from experiments.phase_iiie.qwen_roster import CONTEXTS, FAMILIES, LABELS, ORDERS, PROTOCOL, RESERVE, SPLITS
from weightbridge.adaptation.bic import _runner
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint


def _read(path):
    return json.loads(path.read_bytes())


def _sources():
    paths = [Path(__file__).with_name(name) for name in (
        "qwen_measure.py", "qwen_roster.py", "qwen_analysis.py", "qwen_protocol.json",
    )] + [ROOT / path for path in (
        "experiments/phase_iiid/analysis.py", "experiments/phase_iiid/roster.py",
        "experiments/phase_iiid/run_grounding.py", "weightbridge/adaptation/bic.py",
        "weightbridge/adaptation/content_scoring.py", "weightbridge/adaptation/counterfactual.py",
        "weightbridge/adaptation/grounding.py", "weightbridge/inference/huggingface.py",
    )]
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths}


def _load_rows(path, split, config, manifest, seen):
    binding = manifest["artifacts"][split]
    if file_hash(path) != binding["sha256"]:
        raise ValueError(f"Roster hash changed for {split}")
    record = _read(path)
    items = record["items"]
    keys = {item["semantic_key"] for item in items}
    if (record["split"] != split or len(keys) != len(items) or keys & seen
            or len(items) != binding["underlying_items"]
            or Counter(item["family"] for item in items) != dict.fromkeys(FAMILIES, config["items_per_family"][split])):
        raise ValueError(f"Invalid, imbalanced or overlapping {split} roster")
    seen.update(keys)
    rows = []
    expected = {(order, labels) for order in ORDERS for labels in LABELS}
    for item in items:
        rendered = {(tuple(row["order"]), tuple(row["labels"])) for row in item["rows"]}
        if len(item["rows"]) != len(expected) or rendered != expected:
            raise ValueError(f"Roster is missing crossed presentations for {item['id']}")
        rows.extend({**row, "item_id": item["id"], "family": item["family"]} for row in item["rows"])
    return rows


def _measure(runner, rows, config, split):
    raw = [{key: row[key] for key in ("order", "labels", "answers", "item_id", "family")} for row in rows]
    resources = {}
    for group, prompt_key, answers_key, contexts in (
        ("content", "prompts", "answers", CONTEXTS),
        ("letter", "letter_prompts", "labels", DIRECTIONS),
    ):
        resources[group] = {}
        for target in raw:
            target[group] = {}
        options = [row[answers_key] for row in rows]
        for context in contexts:
            prompts = [row[prompt_key][context] for row in rows]
            scored, usage = _score(runner, prompts, options, config["inference"], letter=group == "letter")
            resources[group][context] = usage
            for target, scores in zip(raw, scored, strict=True):
                target[group][context] = scores
            print(f"{split} {group}/{context} closed with {len(scored)} presentations", file=sys.stderr, flush=True)
    indexes = [i for i, row in enumerate(rows) if row["labels"] == list(LABELS[0])]
    selected = [rows[i] for i in indexes]
    resources["generation"] = {}
    for context in DIRECTIONS:
        prompts = [row["generation_prompts"][context] for row in selected]
        responses, usage = _generate(runner, selected, prompts, "bullets", config)
        resources["generation"][context] = usage
        for index, response in zip(indexes, responses, strict=True):
            raw[index].setdefault("generation", {})[context] = response
        print(f"{split} generation/{context} closed with {len(responses)} responses", file=sys.stderr, flush=True)
    return raw, resources


def _admission(summaries, selected, config):
    ranker = {name: report["rankers"][selected]["passed"] for name, report in summaries.items()}
    generation = {name: report["generation"]["pass"] for name, report in summaries.items()}
    target_min = config["headroom_gate"]["persistent_errors_per_split"]
    preserve_min = config["headroom_gate"]["preserved_items_per_split"]
    headroom = {}
    for family in FAMILIES:
        headroom[family] = {}
        for split in ("practice", "selection"):
            bank = summaries[split]["headroom"]
            targets = [item for item, state in bank[family].items() if state == "persistently_wrong"]
            controls = [item for other in bank if other != family for item, state in bank[other].items()
                        if state == "solved"]
            headroom[family][split] = {"targets": targets, "controls": controls,
                                       "passed": len(targets) >= target_min and len(controls) >= preserve_min}
    eligible = [family for family in FAMILIES if all(split["passed"] for split in headroom[family].values())]
    admission = all(ranker.values()) and bool(eligible)
    return {"ranker_gates": ranker, "generation_gates": generation, "headroom": headroom,
            "eligible_families": eligible if admission else [], "candidate_attempts": 0, "reserve_opened": False,
            "disposition": "PASS_MEASUREMENT_AND_HEADROOM" if admission else "HOLD",
            "hold_reason": None if admission else ("MEASUREMENT" if not all(ranker.values()) else "HEADROOM")}


def run(base, material, output):
    wall, cpu = time.perf_counter(), time.process_time()
    config = _read(PROTOCOL)
    manifest_path = material / "manifest.json"
    manifest = _read(manifest_path)
    manifest_hash, protocol_hash, sources = file_hash(manifest_path), file_hash(PROTOCOL), _sources()
    checkpoint = load_checkpoint(base)
    checkpoint_hash = checkpoint.checkpoint_sha256()
    if (checkpoint_hash != config["model_sha256"] or manifest["protocol_sha256"] != protocol_hash
            or manifest["roster_source_sha256"] != sources["experiments/phase_iiie/qwen_roster.py"]):
        raise ValueError("Run requires the pinned checkpoint and original frozen protocol")

    def verify_sources():
        if _sources() != sources or file_hash(manifest_path) != manifest_hash:
            raise ValueError("Frozen sources or material manifest changed during scoring")

    output.mkdir(parents=True, exist_ok=False)
    execution_path = output / "execution.json"
    json_dump_atomic(execution_path, {
        "schema": "weightbridge.phase_iiie.qwen_execution.v1",
        "checkpoint_sha256": checkpoint_hash, "protocol_sha256": protocol_hash,
        "manifest_sha256": manifest_hash, "source_sha256": sources,
        "platform": platform.platform(), "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "tokenizers", "numpy", "safetensors", "psutil",
        )},
    })
    seen = set()
    summaries = {}
    decision_path = output / "practice-lock.json"
    with _runner(checkpoint.root, config["inference"]) as (runner, _):
        for split in SPLITS:
            verify_sources()
            if split != "practice":
                selected = _read(decision_path)["selected"]
            rows = _load_rows(material / f"{split}.json", split, config, manifest, seen)
            start_wall, start_cpu = time.perf_counter(), time.process_time()
            raw, resources = _measure(runner, rows, config, split)
            split_usage = _resources(start_wall, start_cpu, resources)
            raw_path = output / f"{split}-predictions.json"
            json_dump_atomic(raw_path, {"split": split, "rows": raw, "resources": split_usage})
            analysis, projections = analyze(raw, config)
            raw_hash = file_hash(raw_path)
            if split == "practice":
                selected = lock(analysis, config)
                json_dump_atomic(decision_path, {"selected": selected, "practice_sha256": raw_hash,
                                                 "protocol_sha256": protocol_hash})
                decision_hash = file_hash(decision_path)
            if file_hash(decision_path) != decision_hash:
                raise ValueError("Practice lock changed")
            summaries[split] = {**analysis, "predictions_sha256": raw_hash, "resources": split_usage,
                                "headroom": classify_headroom(raw, projections, selected)}
            print(f"{split} analysis closed with selected method {selected}", file=sys.stderr, flush=True)
    verify_sources()
    if load_checkpoint(checkpoint.root).checkpoint_sha256() != checkpoint_hash:
        raise ValueError("Checkpoint identity changed during measurement")
    report = {
        "schema": "weightbridge.phase_iiie.qwen_report.v1", "checkpoint_sha256": checkpoint_hash,
        "execution_sha256": file_hash(execution_path), "manifest_sha256": manifest_hash,
        "protocol_sha256": protocol_hash, "practice_lock_sha256": decision_hash, "selected": selected,
        "splits": summaries, "admission": _admission(summaries, selected, config),
        "resources": _resources(wall, cpu, {"model_loads": 1}),
    }
    json_dump_atomic(output / "report.json", report)
    return report


def archive(material, run_path, output):
    """Replay analysis and export only completed measurement evidence."""
    report = _read(run_path / "report.json")
    manifest = _read(material / "manifest.json")
    execution = _read(run_path / "execution.json")
    decision = _read(run_path / "practice-lock.json")
    config = _read(PROTOCOL)
    if (file_hash(run_path / "execution.json") != report["execution_sha256"]
            or file_hash(run_path / "practice-lock.json") != report["practice_lock_sha256"]
            or file_hash(material / "manifest.json") != report["manifest_sha256"]
            or report["manifest_sha256"] != execution["manifest_sha256"]
            or execution["source_sha256"] != _sources()
            or not report["checkpoint_sha256"] == execution["checkpoint_sha256"] == config["model_sha256"]
            or not report["protocol_sha256"] == decision["protocol_sha256"] == manifest["protocol_sha256"]
            == execution["protocol_sha256"] == file_hash(PROTOCOL)):
        raise ValueError("Measurement source bindings do not verify")
    files = {output / "protocol.json": PROTOCOL, output / "material/manifest.json": material / "manifest.json"}
    seen = set()
    summaries = {}
    for split in SPLITS:
        roster = material / f"{split}.json"
        rows = _load_rows(roster, split, config, manifest, seen)
        predictions = run_path / f"{split}-predictions.json"
        payload = _read(predictions)
        raw = payload["rows"]
        raw_hash = file_hash(predictions)
        if (payload["split"] != split or any(
                any(original[key] != scored[key] for key in ("order", "labels", "answers", "item_id", "family"))
                for original, scored in zip(rows, raw, strict=True))):
            raise ValueError(f"Measured {split} rows differ from the roster")
        analysis, projections = analyze(raw, config)
        if split == "practice":
            selected = lock(analysis, config)
            if selected != report["selected"] or decision != {
                    "selected": selected, "practice_sha256": raw_hash, "protocol_sha256": report["protocol_sha256"]}:
                raise ValueError("Practice lock does not reproduce")
        summary = {**analysis, "predictions_sha256": raw_hash, "resources": payload["resources"],
                   "headroom": classify_headroom(raw, projections, selected)}
        if summary != report["splits"][split]:
            raise ValueError(f"Measurement summary differs on {split}")
        summaries[split] = summary
        files[output / "material" / roster.name] = roster
        files[output / predictions.name] = predictions
    if _admission(summaries, selected, config) != report["admission"]:
        raise ValueError("Admission does not reproduce")
    for name in ("execution.json", "practice-lock.json", "report.json"):
        files[output / name] = run_path / name
    output.mkdir(parents=True, exist_ok=False)
    for destination, source in files.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    json_dump_atomic(output / "artifact_manifest.json", {
        "schema": "weightbridge.phase_iiie.qwen_archive.v1",
        "artifacts": {path.relative_to(output).as_posix(): file_hash(path) for path in files},
        "excluded_reserve_commitment": manifest["artifacts"][RESERVE]["sha256"],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "archive"))
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.action == "run" and args.base:
        print(json.dumps(run(args.base, args.material, args.run)["admission"], indent=2))
    elif args.action == "archive" and args.archive:
        archive(args.material, args.run, args.archive)
    else:
        parser.error("Run requires --base and archive requires --archive")
