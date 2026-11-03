"""Replay and archive Phase III-G evidence without exporting an unopened roster."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiig.endpoint import parse_integer
from experiments.phase_iiig.roster import PROTOCOL, load_split
from experiments.phase_iiig.run_trial import _scope, _sources, compare, qualify
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.patches.huggingface import _canonical_sha256


def _read(path):
    return json.loads(path.read_bytes())


def replay_rows(measured, expected):
    rows = measured["rows"]
    for prediction, row in zip(rows, expected, strict=True):
        if (any(prediction[key] != value for key, value in row.items())
                or prediction["exact"] != parse_integer(prediction["text"], prediction["stop_reason"])):
            raise ValueError("Behavioral prediction differs from its frozen row or raw output")
    return rows


def verify_trial(run, material, protocol, manifest, load, expected_sources):
    report, lock = (_read(run / name) for name in ("report.json", "candidate-lock.json"))
    manifest_sha = file_hash(material / "manifest.json")
    model = report["model"]
    if (report["schema"] != "weightbridge.phase_iiig.trial.v1" or lock["model"] != model
            or report["base_sha256"] != protocol["checkpoints"][model]
            or lock["base_sha256"] != report["base_sha256"]
            or report["manifest_sha256"] != manifest_sha or lock["manifest_sha256"] != manifest_sha
            or report["source_sha256"] != expected_sources or lock["source_sha256"] != expected_sources
            or lock["winner"] != report["winner"] or lock["family"] != report["selected_family"]):
        raise ValueError("Run does not bind the committed model, sources and roster")
    files = {"report.json": run / "report.json"}
    for name, digest in report["artifacts"].items():
        relative = PurePosixPath(name)
        path = run / relative
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name
                or relative.suffix != ".json" or "checkpoints" in relative.parts
                or not path.resolve().is_relative_to(run.resolve()) or file_hash(path) != digest):
            raise ValueError(f"Trial artifact is unsafe or changed at {name}")
        files[name] = path
    family_lock = _read(files["family-lock.json"])
    if (family_lock["practice_sha256"] != file_hash(files["practice.json"])
            or family_lock["manifest_sha256"] != manifest_sha
            or family_lock["selected_family"] != report["selected_family"]):
        raise ValueError("Practice lock is inconsistent")
    practice = replay_rows(_read(files["practice.json"]), load("practice"))
    qualification = {family: qualify(practice, family, protocol["qualification"])
                     for family in report["practice"]}
    passes = [family for family, result in qualification.items() if result["disposition"] == "PASS"]
    family = min(passes, key=lambda name: (-len(qualification[name]["repeatable_errors"]),
                                         -len(qualification[name]["solved"]["target"]), name), default=None)
    if qualification != report["practice"] or family != report["selected_family"]:
        raise ValueError("Practice gate or family choice differs on replay")
    if family is None:
        if report["behavioral_qualification"] != "HOLD" or report["candidate_attempts"] or report["hidden_opened"]:
            raise ValueError("Unqualified measurement advanced to fitting")
        return report, files
    selection = replay_rows(_read(files["selection.json"]), _scope(load("selection"), family))
    selection_gate = qualify(selection, family, protocol["qualification"])
    if selection_gate != report["selection"] or selection_gate["disposition"] != report["behavioral_qualification"]:
        raise ValueError("Selection qualification differs on replay")

    def independent(split, tag, expected_sha):
        payload = _read(files[f"verification/{tag}.json"])
        if (payload["split"] != split or payload["family"] != family or payload["checkpoint_sha256"] != expected_sha
                or payload["roster_sha256"] != manifest["splits"][split]["sha256"]):
            raise ValueError("Independent verifier used another roster or checkpoint")
        return replay_rows(payload["measurement"], _scope(load(split), family))

    if lock["candidates"]:
        before = independent("selection", "selection-base", report["base_sha256"])
        eligible = {}
        for method, record in lock["candidates"].items():
            if record["candidate"] is None:
                continue
            digest = record["candidate"]["checkpoint_sha256"]
            receipt = _read(files[f"patches/{method}-receipt.json"])
            patch = _read(files[f"patches/{method}.json"])
            policy = _read(files[f"patches/{method}-policy.json"])
            if (receipt != record["candidate"]["receipt"] or receipt["candidate_sha256"] != digest
                    or receipt["base_sha256"] != report["base_sha256"]
                    or receipt["patch_sha256"] != _canonical_sha256(patch)
                    or receipt["policy_sha256"] != _canonical_sha256(policy)):
                raise ValueError("Candidate receipt differs from the stored plan and patch")
            changed = independent("selection", f"selection-{method}", digest)
            result = compare(before, changed, family, protocol["selection_gate"],
                             protocol["qualification"]["min_parse_rate"])
            if result != record["selection"]:
                raise ValueError("Stored candidate selection score differs on replay")
            if result["disposition"] == "PASS":
                eligible[method] = digest
        methods = protocol["intent"]["methods"]
        winner = min(eligible, key=lambda method: (-lock["candidates"][method]["selection"]["target_item_gain"],
                                                   methods.index(method)), default=None)
        if lock["eligible"] != eligible or lock["winner"] != winner:
            raise ValueError("Selection lock differs from measured results")
    if report["hidden_opened"] != (lock["winner"] is not None):
        raise ValueError("Hidden access differs from the selection lock")
    if report["hidden_opened"]:
        before = independent("hidden", "hidden-base", report["base_sha256"])
        hidden = {}
        for method, digest in lock["eligible"].items():
            changed = independent("hidden", f"hidden-{method}", digest)
            hidden[method] = changed
            result = compare(before, changed, family, protocol["hidden_gate"],
                             protocol["qualification"]["min_parse_rate"])
            if result != report["candidates"][method]["hidden"]:
                raise ValueError("Independent candidate gate differs on replay")
        if report["single_cycle"] == "PASS":
            winner = lock["winner"]
            digest = lock["eligible"][winner]
            if (report["accepted_candidate_sha256"] != digest
                    or report["candidates"][winner]["hidden"]["disposition"] != "PASS"
                    or independent("hidden", "restart", digest) != hidden[winner]
                    or independent("hidden", "rollback", report["base_sha256"]) != before):
                raise ValueError("Accepted candidate failed restart or rollback replay")
    return report, files


def archive(material, runs, out, *, followup=False, scaling=False):
    if scaling:
        from experiments.phase_iiig.roster_scaling import PROTOCOL as protocol_path, load_split as read_split
        from experiments.phase_iiig.run_scaling import _sources as source_hashes
    elif followup:
        from experiments.phase_iiig.roster_followup import PROTOCOL as protocol_path, load_split as read_split
        from experiments.phase_iiig.run_followup import _sources as source_hashes
    else:
        protocol_path, read_split, source_hashes = PROTOCOL, load_split, _sources
    config = _read(protocol_path)
    manifest_path = material / "manifest.json"
    manifest = _read(manifest_path)

    def load(split):
        return read_split(material, split, manifest)

    paths = {"protocol.json": protocol_path, "material/manifest.json": manifest_path}
    reports = {}
    for run in runs:
        model = _read(run / "report.json")["model"]
        eligible = config.get("eligible_families", {}).get(model, config["target_families"])

        def model_split(split):
            return [row for row in load(split) if row["family"] in eligible or row["group"] == "preserve"]

        report, files = verify_trial(run, material, config, manifest, model_split, source_hashes())
        model = report["model"]
        if model in reports:
            raise ValueError("Duplicate model in the trial matrix")
        reports[model] = report
        paths.update({f"{model}/{name}": path for name, path in files.items()})
    if set(reports) != set(config["checkpoints"]):
        raise ValueError("Archive needs every predeclared model report")
    lock_hashes = {model: file_hash(paths[f"{model}/candidate-lock.json"]) for model in reports}
    for model, report in reports.items():
        if report["hidden_opened"]:
            independent = _read(paths[f"{model}/independent-lock.json"])
            if independent["both_candidate_locks"] != lock_hashes:
                raise ValueError("Hidden evaluation does not bind every candidate lock")
    splits = ["practice", "selection"]
    if any(report["hidden_opened"] for report in reports.values()):
        splits.append("hidden")
    for split in splits:
        load(split)
        paths[f"material/{split}.json"] = material / f"{split}.json"
    out.mkdir(parents=True, exist_ok=False)
    inventory = {}
    for name, source in paths.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        inventory[name] = file_hash(target)
        if inventory[name] != file_hash(source):
            raise ValueError(f"Trial copy differs at {name}")
    document = {"schema": "weightbridge.phase_iiig.archive.v1", "artifacts": inventory,
                "hidden_exported": "hidden" in splits,
                "models": {model: {key: report[key] for key in ("behavioral_qualification", "single_cycle",
                                                               "sequential_readiness", "selected_family")}
                           for model, report in reports.items()}}
    json_dump_atomic(out / "artifact_manifest.json", document)
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("material", "qwen", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--smollm2", type=Path)
    version = parser.add_mutually_exclusive_group()
    version.add_argument("--followup", action="store_true")
    version.add_argument("--scaling", action="store_true")
    args = parser.parse_args()
    if args.scaling == (args.smollm2 is not None):
        parser.error("Use --smollm2 for the two-model trials and omit it for --scaling")
    runs = (args.qwen,) if args.scaling else (args.qwen, args.smollm2)
    result = archive(args.material, runs, args.out, followup=args.followup, scaling=args.scaling)
    print(json.dumps(result["models"], indent=2))
