"""Run a frozen behavioral qualification and single-cycle weight trial."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiif.run_matrix import _fit, _open
from experiments.phase_iiig.endpoint import answer_pair, bare_runner, measure
from experiments.phase_iiig.roster import PROTOCOL, load_split
from weightbridge.adaptation.meta_controller import EvidenceState
from weightbridge.adaptation.model_adapter import Capability
from weightbridge.adaptation.universal_awl import (
    HostAuthorization, ModelEvidence, compile_awl_v3, parse_awl_v3, recommend_awl_v3,
)
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.provenance import peak_rss_bytes


def _json(path):
    return json.loads(path.read_bytes())


def _sources():
    names = [
        "experiments/phase_iiig/run_trial.py", "experiments/phase_iiig/endpoint.py",
        "experiments/phase_iiig/roster.py", "experiments/phase_iiig/protocol.json",
        "experiments/phase_iiif/run_matrix.py",
        "weightbridge/adaptation/model_adapter.py", "weightbridge/adaptation/content_scoring.py",
        "weightbridge/adaptation/universal_awl.py", "weightbridge/adaptation/universal_update.py",
        "weightbridge/adaptation/meta_controller.py", "weightbridge/inference/huggingface.py",
        "weightbridge/patches/huggingface.py",
    ]
    return {name: file_hash(ROOT / name) for name in names}


def _material(material):
    manifest = _json(material / "manifest.json")
    if (manifest["protocol_sha256"] != file_hash(PROTOCOL)
            or manifest["roster_source_sha256"] != file_hash(Path(__file__).with_name("roster.py"))):
        raise ValueError("Roster and protocol commitments differ")
    return manifest


def _items(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["item"]].append(row)
    for group in grouped.values():
        if (len(group) != 2 or len({r["wording"] for r in group}) != 2
                or len({(r["family"], r["group"], r["answer"]) for r in group}) != 1):
            raise ValueError("Each item needs two consistent and distinct prompt wordings")
    return grouped


def _scope(rows, family):
    return [row for row in rows if row["family"] == family or row["group"] == "preserve"]


def qualify(rows, family, thresholds):
    rows = _scope(rows, family)
    groups = _items(rows)
    solved = {"target": [], "positive": [], "preserve": []}
    errors = {}
    for item, (first, second) in groups.items():
        if first["exact"] == second["exact"] == first["answer"]:
            solved[first["group"]].append(item)
        elif first["group"] == "target" and first["exact"] is not None and first["exact"] == second["exact"]:
            errors[item] = first["exact"]
    parsed = sum(row["exact"] is not None for row in rows)
    checks = {
        "parse_rate": parsed >= thresholds["min_parse_rate"] * len(rows),
        "positive_controls": len(solved["positive"]) >= thresholds["min_positive_solved"],
        "preservation_controls": len(solved["preserve"]) >= thresholds["min_preserve_solved"],
        "solved_target": len(solved["target"]) >= thresholds["min_target_solved"],
        "repeatable_errors": len(errors) >= thresholds["min_repeatable_errors"],
    }
    return {"disposition": "PASS" if all(checks.values()) else "HOLD",
            "gates": checks, "parsed": [parsed, len(rows)], "solved": solved, "repeatable_errors": errors}


def compare(base, candidate, family, gate, parse_rate):
    base, candidate = _scope(base, family), _scope(candidate, family)
    before, after = _items(base), _items(candidate)
    regressions = defaultdict(list)
    corrected, lost = [], []
    for first, second in zip(base, candidate, strict=True):
        for field in ("id", "item", "family", "group", "wording", "prompt", "answer"):
            if first[field] != second[field]:
                raise ValueError("Paired behavioral measurement rows differ")
        if first["exact"] == first["answer"] and second["exact"] != second["answer"]:
            regressions[first["group"]].append(first["id"])
    for item, group in before.items():
        if group[0]["group"] != "target":
            continue
        was = all(row["exact"] == row["answer"] for row in group)
        now = all(row["exact"] == row["answer"] for row in after[item])
        if now and not was:
            corrected.append(item)
        if was and not now:
            lost.append(item)
    parsed = sum(row["exact"] is not None for row in candidate)
    gain = len(corrected) - len(lost)
    passes = (gain >= gate["min_target_gain"] and sum(map(len, regressions.values())) <= gate["max_regressions"]
              and parsed >= len(candidate) * parse_rate)
    return {"disposition": "PASS" if passes else "FAIL", "target_item_gain": gain,
            "corrected_items": corrected, "lost_items": lost, "regressions": dict(regressions),
            "candidate_parsed": [parsed, len(candidate)]}


def _fit_rows(runner, measured, family, qualification, settings):
    errors = qualification["repeatable_errors"]
    target_items = set(list(errors)[:8])
    preserved = {item for group in qualification["solved"].values() for item in group}
    fit = []
    for row in _scope(measured, family):
        item = row["item"]
        if item not in target_items and item not in preserved:
            continue
        answer = row["answer"]
        rival = errors[item] if item in target_items else str(int(answer) + 1)
        fit.append({"id": row["id"], "prompt": row["prompt"], "preserve": item in preserved,
                    "answers": answer_pair(runner, row["prompt"], answer, rival,
                                           max_input_tokens=settings["max_input_tokens"])})
    return fit


def _worker(base, expected, material, split, family, out, manifest_sha, seal, seal_sha):
    start, cpu = time.perf_counter(), time.process_time()
    config, manifest = _json(PROTOCOL), _material(material)
    if file_hash(material / "manifest.json") != manifest_sha:
        raise ValueError("Verifier roster commitment differs")
    if split == "hidden":
        if seal is None or file_hash(seal) != seal_sha:
            raise ValueError("Hidden evaluation requires an immutable candidate lock")
        lock = _json(seal)
        if (lock["family"] != family or lock["manifest_sha256"] != manifest_sha
                or lock["source_sha256"] != _sources()
                or set(lock["both_candidate_locks"]) != set(config["checkpoints"])
                or expected not in {lock["base_sha256"], *lock["eligible"].values()}):
            raise ValueError("Checkpoint is outside the frozen independent evaluation scope")
    rows = _scope(load_split(material, split, manifest), family)
    with _open(base, config, expected) as (_, runner, contract):
        measured = measure(runner, rows, config["inference"])
    json_dump_atomic(out, {
        "schema": "weightbridge.phase_iiig.verifier.v1", "checkpoint_sha256": expected,
        "family": family, "split": split, "roster_sha256": manifest["splits"][split]["sha256"],
        "contract": contract.to_dict(), "measurement": measured,
        "resources": {"wall_seconds": time.perf_counter() - start, "cpu_seconds": time.process_time() - cpu,
                      "peak_rss_bytes": peak_rss_bytes()},
    })


def _independent(base, expected, material, split, family, out, manifest_sha, timeout, seal=None):
    args = [sys.executable, str(Path(__file__).resolve()), "worker", "--base", str(base),
            "--checkpoint-sha256", expected, "--material", str(material), "--split", split,
            "--family", family, "--out", str(out), "--manifest-sha256", manifest_sha]
    if seal is not None:
        args.extend(["--seal", str(seal), "--seal-sha256", file_hash(seal)])
    subprocess.run(args, check=True, timeout=timeout)
    result = _json(out)
    if (result["checkpoint_sha256"] != expected or result["split"] != split or result["family"] != family
            or result["roster_sha256"] != _material(material)["splits"][split]["sha256"]):
        raise ValueError("Independent verification returned different identities")
    return result


def run(model, base, material, out):
    start, cpu = time.perf_counter(), time.process_time()
    config, manifest = _json(PROTOCOL), _material(material)
    manifest_sha, sources = file_hash(material / "manifest.json"), _sources()
    settings, thresholds = config["inference"], config["qualification"]
    timeout = settings["worker_timeout_seconds"]
    base_sha = config["checkpoints"][model]
    practice = load_split(material, "practice", manifest)
    selection = load_split(material, "selection", manifest)
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "weightbridge.phase_iiig.trial.v1", "model": model, "base_sha256": base_sha,
              "manifest_sha256": manifest_sha, "source_sha256": sources,
              "behavioral_qualification": "HOLD", "single_cycle": "HOLD", "sequential_readiness": "HOLD",
              "hidden_opened": False, "candidates": {}, "candidate_attempts": 0}
    candidates = report["candidates"]
    with _open(base, config, base_sha, gradient_probe=True) as (adapter, runner, contract):
        report["contract"] = contract.to_dict()
        pre = measure(runner, practice, settings)
        families = sorted({row["family"] for row in practice if row["group"] == "target"})
        evaluations = {family: qualify(pre["rows"], family, thresholds) for family in families}
        passed = [family for family in families if evaluations[family]["disposition"] == "PASS"]
        family = min(passed, key=lambda name: (-len(evaluations[name]["repeatable_errors"]),
                                               -len(evaluations[name]["solved"]["target"]), name), default=None)
        report.update(practice=evaluations, selected_family=family)
        json_dump_atomic(out / "practice.json", pre)
        json_dump_atomic(out / "family-lock.json", {"selected_family": family,
                         "practice_sha256": file_hash(out / "practice.json"), "manifest_sha256": manifest_sha})
        print(f"{model}: practice qualification {family or 'HOLD'}", flush=True)
        if family is not None:
            preselection = measure(runner, _scope(selection, family), settings)
            json_dump_atomic(out / "selection.json", preselection)
            selection_qualification = qualify(preselection["rows"], family, thresholds)
            report["selection"] = selection_qualification
            if selection_qualification["disposition"] == "PASS":
                report["behavioral_qualification"] = "PASS"
                if Capability.COMPUTE_GRADIENTS in contract.capabilities:
                    specification = config["intent"]
                    intent = parse_awl_v3({**specification,
                        "objective": {**specification["objective"], "family": family},
                        "preserve": [f"{family}-easy", "copy"]})
                    host = HostAuthorization(base_sha, intent.methods,
                                             tuple(r.tensor_name for r in contract.editable_regions),
                                             (family, *intent.preserve), intent.practice_data_id,
                                             intent.evaluation_data_id, intent.limits)
                    fit = _fit_rows(runner, pre["rows"], family, evaluations[family], settings)
                    json_dump_atomic(out / "fit-rows.json", {"rows": fit})
                    plan = compile_awl_v3(intent, contract, host, method="constrained")
                    evidence = EvidenceState(True, len(evaluations[family]["repeatable_errors"]),
                                             intent.limits.max_forward_calls, None)
                    report["meta_decision"] = recommend_awl_v3(
                        plan, ModelEvidence(contract.family, base_sha, evidence)).record()
                    fit_config = {**config, "inference": {
                        **settings, "batch_size": settings.get("fit_batch_size", settings["batch_size"])}}
                    for method in intent.methods:
                        report["candidate_attempts"] += 1
                        candidates[method] = _fit(base, adapter, bare_runner(runner), contract, method, intent,
                                                  host, fit, fit_config, out)
                        print(f"{model}: {method} {candidates[method]['disposition']}", flush=True)
                else:
                    report["reason"] = "checkpoint_gradient_capability_unavailable"
    del runner
    gc.collect()
    verification = out / "verification"
    verification.mkdir()
    eligible = {}
    if candidates:
        baseline = _independent(base, base_sha, material, "selection", family,
                                verification / "selection-base.json", manifest_sha, timeout)
        for method, record in candidates.items():
            if record["candidate"] is None:
                continue
            digest = record["candidate"]["checkpoint_sha256"]
            candidate = _independent(out / "checkpoints" / method, digest, material, "selection", family,
                                     verification / f"selection-{method}.json", manifest_sha, timeout)
            paired = compare(baseline["measurement"]["rows"], candidate["measurement"]["rows"], family,
                             config["selection_gate"], thresholds["min_parse_rate"])
            record["selection"] = paired
            if paired["disposition"] == "PASS":
                eligible[method] = digest
        report["single_cycle"] = "FAIL"
    methods = config["intent"]["methods"]
    winner = min(eligible, key=lambda method: (-candidates[method]["selection"]["target_item_gain"],
                                                methods.index(method)), default=None)
    seal = out / "candidate-lock.json"
    report["winner"] = winner
    json_dump_atomic(seal, {"schema": "weightbridge.phase_iiig.candidate_lock.v1",
        "model": model, "family": family, "base_sha256": base_sha, "winner": winner, "eligible": eligible,
        "manifest_sha256": manifest_sha, "source_sha256": sources, "candidates": candidates})
    if (file_hash(material / "manifest.json") != manifest_sha or _sources() != sources
            or load_checkpoint(base).checkpoint_sha256() != base_sha):
        raise ValueError("Pinned sources, roster, or original checkpoint changed")
    report["resources"] = {"wall_seconds": time.perf_counter() - start,
                           "host_cpu_seconds": time.process_time() - cpu, "peak_rss_bytes": peak_rss_bytes()}
    json_dump_atomic(out / "preparation.json", report)
    return report


def finish(base, material, out, peer):
    start, cpu = time.perf_counter(), time.process_time()
    config = _json(PROTOCOL)
    thresholds, timeout = config["qualification"], config["inference"]["worker_timeout_seconds"]
    report = _json(out / "preparation.json")
    lock_paths = (out / "candidate-lock.json", peer / "candidate-lock.json")
    lock, other = map(_json, lock_paths)
    manifest_sha, sources = file_hash(material / "manifest.json"), _sources()
    if ({lock["model"], other["model"]} != set(config["checkpoints"])
            or any(record["manifest_sha256"] != manifest_sha or record["source_sha256"] != sources
                   or record["base_sha256"] != config["checkpoints"][record["model"]]
                   for record in (lock, other))
            or lock["model"] != report["model"] or lock["candidates"] != report["candidates"]
            or lock["winner"] != report["winner"] or lock["family"] != report["selected_family"]
            or load_checkpoint(base).checkpoint_sha256() != lock["base_sha256"]):
        raise ValueError("Both model candidate locks must bind the same frozen experiment")
    winner, family, base_sha = lock["winner"], lock["family"], lock["base_sha256"]
    eligible, candidates = lock["eligible"], report["candidates"]
    verification = out / "verification"
    if winner is not None:
        seal = out / "independent-lock.json"
        bindings = {record["model"]: file_hash(path) for record, path in zip((lock, other), lock_paths, strict=True)}
        json_dump_atomic(seal, {**lock, "schema": "weightbridge.phase_iiig.independent_lock.v1",
                               "both_candidate_locks": bindings})
        report["hidden_opened"] = True
        baseline = _independent(base, base_sha, material, "hidden", family, verification / "hidden-base.json",
                                manifest_sha, timeout, seal)
        observed = {}
        for method, digest in eligible.items():
            observed[method] = _independent(out / "checkpoints" / method, digest, material, "hidden", family,
                verification / f"hidden-{method}.json", manifest_sha, timeout, seal)
            candidates[method]["hidden"] = compare(baseline["measurement"]["rows"],
                observed[method]["measurement"]["rows"], family, config["hidden_gate"], thresholds["min_parse_rate"])
        if candidates[winner]["hidden"]["disposition"] == "PASS":
            restart = _independent(out / "checkpoints" / winner, eligible[winner], material, "hidden", family,
                                   verification / "restart.json", manifest_sha, timeout, seal)
            rollback = _independent(base, base_sha, material, "hidden", family, verification / "rollback.json",
                                    manifest_sha, timeout, seal)
            report["restart_exact"] = restart["measurement"]["rows"] == observed[winner]["measurement"]["rows"]
            report["rollback_exact"] = rollback["measurement"]["rows"] == baseline["measurement"]["rows"]
            if report["restart_exact"] and report["rollback_exact"]:
                report.update(single_cycle="PASS", accepted_candidate_sha256=eligible[winner])
    if (file_hash(material / "manifest.json") != manifest_sha or _sources() != sources
            or load_checkpoint(base).checkpoint_sha256() != base_sha):
        raise ValueError("Pinned sources, roster, or original checkpoint changed")
    report["resources"]["wall_seconds"] += time.perf_counter() - start
    report["resources"]["host_cpu_seconds"] += time.process_time() - cpu
    peak = peak_rss_bytes()
    report["resources"]["peak_rss_bytes"] = max(report["resources"]["peak_rss_bytes"] or 0, peak or 0)
    report["artifacts"] = {path.relative_to(out).as_posix(): file_hash(path)
                           for path in sorted(out.rglob("*.json"))
                           if "checkpoints" not in path.relative_to(out).parts}
    json_dump_atomic(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "finish", "worker"))
    for field in ("base", "material", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--model", choices=("qwen", "smollm2"))
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--split", choices=("selection", "hidden"))
    parser.add_argument("--family")
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--seal-sha256")
    parser.add_argument("--peer", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if not all((args.checkpoint_sha256, args.manifest_sha256, args.split, args.family)):
            parser.error("worker requires checkpoint and manifest hashes, split, and family")
        _worker(args.base, args.checkpoint_sha256, args.material, args.split, args.family,
                args.out, args.manifest_sha256, args.seal, args.seal_sha256)
    elif args.command == "run":
        if not args.model:
            parser.error("run requires model")
        result = run(args.model, args.base, args.material, args.out)
        print(json.dumps({key: result[key] for key in ("behavioral_qualification", "single_cycle")}), flush=True)
    else:
        if not args.peer:
            parser.error("finish requires the other model's locked run directory")
        result = finish(args.base, args.material, args.out, args.peer)
        print(json.dumps({key: result[key] for key in ("behavioral_qualification", "single_cycle")}), flush=True)
