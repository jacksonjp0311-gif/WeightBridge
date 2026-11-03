"""Replay and archive the completed cross-model experiment without weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiif.matrix_data import PROTOCOL, load_split, read_json, render_rows
from experiments.phase_iiif.matrix_scoring import compare, qualify, summarize
from experiments.phase_iiif.run_matrix import _sources
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.patches.huggingface import _canonical_sha256


def _replay(measured, expected):
    for got, original in zip(measured["rows"], expected, strict=True):
        left, right = [choice["sum_log_probability"] for choice in got["score"]]
        labels = got["choices"]
        ranker = labels[0] if left > right else labels[1] if right > left else None
        response = got["generation"]
        label = response["text"].strip().casefold()
        exact = label if response["stop_reason"] == "eos" and label in labels else None
        if (any(got[key] != original[key] for key in ("id", "item", "wording", "family", "preserve"))
                or got["correct"] != original["answers"][0].strip()
                or set(labels) != {answer.strip() for answer in original["answers"]}
                or got["ranker"] != ranker or got["exact"] != exact):
            raise ValueError("Recorded prediction cannot be replayed against the frozen item")
    if measured["summary"] != summarize(measured["rows"]):
        raise ValueError("Recorded summary does not match raw predictions")


def _verify(run, material, config, manifest):
    report = read_json(run / "report.json")
    execution = read_json(run / "execution.json")
    lock = read_json(run / "candidate-lock.json")
    if (report.get("schema") != "weightbridge.phase_iiif.matrix_report.v1"
            or not report["base_sha256"] == execution["base_sha256"] == report["contract"]["checkpoint_sha256"]
            or not report["manifest_sha256"] == execution["manifest_sha256"] == file_hash(material / "manifest.json")
            or execution["source_sha256"] != lock["source_sha256"]
            or execution["source_sha256"] != _sources()
            or report["candidates"] != lock["candidates"]
            or lock["practice_sha256"] != file_hash(run / "practice.json")
            or report["intent"] != config["intent"]):
        raise ValueError("Run does not bind the original checkpoint, plans, manifest and sources")
    for name, digest in report["artifacts"].items():
        path = Path(name)
        if (path.is_absolute() or ".." in path.parts or "checkpoints" in path.parts or path.suffix != ".json"
                or ":" in name or "\\" in name or not (run / path).resolve().is_relative_to(run.resolve())):
            raise ValueError("Archive inventory must contain only local JSON evidence")
        if file_hash(run / path) != digest:
            raise ValueError(f"Run artifact changed: {name}")
    practice = read_json(run / "practice.json")
    _replay(practice["measurement"], render_rows(load_split(material, "practice", manifest, config["roster_counts_per_split"])))
    if report["practice"] != qualify(practice["measurement"]["rows"], config["qualification"]):
        raise ValueError("Practice gate differs on replay")
    eval_items = load_split(material, "evaluation", manifest, config["roster_counts_per_split"])
    evaluation = render_rows(eval_items)

    def verifier(tag, expected_sha):
        data = read_json(run / "verification" / f"{tag}.json")
        if (data["checkpoint_sha256"] != expected_sha
                or data["evaluation_sha256"] != manifest["artifacts"]["evaluation"]["sha256"]
                or data["contract"]["family"] != report["contract"]["family"]):
            raise ValueError("Independent verifier checkpoint or roster hash differs")
        _replay(data["measurement"], evaluation)
        return data
    baseline = verifier("base", report["base_sha256"])
    restored = verifier("rollback", report["base_sha256"])
    if (report["rollback"]["exact_replay"] != (baseline["measurement"]["rows"] == restored["measurement"]["rows"])
            or report["baseline_summary"] != baseline["measurement"]["summary"]):
        raise ValueError("Independent baseline or rollback differs on replay")
    for method, record in report["candidates"].items():
        candidate = record["candidate"]
        if candidate is None:
            if method in report["comparison"]:
                raise ValueError("A missing candidate has a reported comparison")
            continue
        observed = verifier(method, candidate["checkpoint_sha256"])
        patch = read_json(run / "patches" / f"{method}.json")
        policy = read_json(run / "patches" / f"{method}-policy.json")
        receipt = read_json(run / "patches" / f"{method}-receipt.json")
        if (receipt != candidate["receipt"] or receipt["patch_sha256"] != _canonical_sha256(patch)
                or receipt["policy_sha256"] != _canonical_sha256(policy)
                or receipt["base_sha256"] != report["base_sha256"]
                or receipt["candidate_sha256"] != candidate["checkpoint_sha256"]):
            raise ValueError("Candidate receipt cannot be reproduced")
        metrics = compare(baseline["measurement"]["rows"], observed["measurement"]["rows"],
                          config["intent"]["objective"]["min_gain"], config["intent"]["risk"]["max_regressions"])
        if method == config["engineering_method"]:
            metrics.update(behavior_gate=metrics["disposition"], disposition="ENGINEERING_ONLY")
        metrics.update(summary=observed["measurement"]["summary"], worker_resources=observed["worker_resources"])
        if metrics != report["comparison"][method]:
            raise ValueError("Independent behavior gate differs on replay")
    return report


def archive(material, first_run, second_run, out):
    config = read_json(PROTOCOL)
    manifest = read_json(material / "manifest.json")
    if (manifest["protocol_sha256"] != file_hash(PROTOCOL)
            or manifest["generator_sha256"] != file_hash(Path(__file__).with_name("matrix_data.py"))):
        raise ValueError("Archived protocol or roster generator changed")
    runs = {}
    reports = {}
    for run in (first_run, second_run):
        report = _verify(run, material, config, manifest)
        family = report["contract"]["family"]
        if family not in {"qwen3_5", "llama"} or family in runs:
            raise ValueError("Archive requires one Qwen3.5 and one Llama report")
        runs[family], reports[family] = run, report
    files = {"protocol.json": PROTOCOL}
    for filename in ("manifest.json", "practice.json", "evaluation.json"):
        files[f"material/{filename}"] = material / filename
    for name, run in runs.items():
        files[f"{name}/report.json"] = run / "report.json"
        files.update({f"{name}/{path}": run / path for path in reports[name]["artifacts"]})
    out.mkdir(parents=True, exist_ok=False)
    for name, path in files.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if file_hash(target) != file_hash(path):
            raise ValueError(f"Archive copy differs: {name}")
    result = {
        "schema": "weightbridge.phase_iiif.matrix_archive.v1",
        "artifacts": {path: file_hash(out / path) for path in files},
        "models": {family: {key: report[key] for key in
                            ("base_sha256", "engineering_disposition", "scientific_disposition", "sequential_disposition")}
                   for family, report in reports.items()},
    }
    json_dump_atomic(out / "artifact_manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("material", "first-run", "second-run", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(archive(args.material, args.first_run, args.second_run, args.out)["models"], indent=2))
