"""Run a frozen Phase III-B comparison using separate compiler and verifier processes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.adaptation.awl_v2 import compile_awl_v2
from weightbridge.adaptation.bic import score_binary_margins
from weightbridge.adaptation.controls import matched_random_patch
from weightbridge.adaptation.gradient_control import fit_gradient_control
from weightbridge.adaptation.lineage import init_lineage, record_lineage_decision, record_lineage_rejection
from weightbridge.core import EditRejected, file_hash, json_dump_atomic
from weightbridge.evaluation.hf_edit import evaluate_hf_edit
from weightbridge.inference import InferenceError
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches.huggingface import _to_float64, apply_safetensors_patch
from weightbridge.provenance.resources import peak_rss_bytes


SOURCE_FILES = (
    "experiments/phase_iiib/qwen_material.py", "experiments/phase_iiib/run_qwen_ablation.py",
    "weightbridge/adaptation/awl_v2.py", "weightbridge/adaptation/bic.py",
    "weightbridge/adaptation/bic_math.py", "weightbridge/adaptation/controls.py",
    "weightbridge/adaptation/gradient_control.py", "weightbridge/adaptation/lineage.py",
    "weightbridge/agents/hf_direction.py", "weightbridge/evaluation/hf_edit.py",
    "weightbridge/patches/huggingface.py",
)


def _read(path):
    return json.loads(Path(path).read_bytes())


def _resources(wall, cpu):
    return {"wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
            "process_peak_rss_bytes": peak_rss_bytes()}


def _requested_norm(patch):
    return math.sqrt(sum(float(np.square(np.asarray(edit["values"])).sum()) for edit in patch["slices"]))


def _clip_requested_norm(patch, cap):
    norm = _requested_norm(patch)
    if not math.isfinite(norm):
        raise ValueError("Requested update norm is nonfinite")
    if norm > cap:
        for edit in patch["slices"]:
            edit["values"] = (np.asarray(edit["values"]) * (cap / norm)).tolist()
    return norm


def _realized_scope(base, candidate, patch, policy):
    before, after = load_checkpoint(base), load_checkpoint(candidate)
    scopes = []
    for edit in patch["slices"]:
        tensor = edit["tensor"]
        ref = before.tensors[tensor]
        read = [checkpoint.read_raw(tensor, max_bytes=policy["max_tensor_bytes"]) for checkpoint in (before, after)]
        original, updated = [_to_float64(raw, ref.dtype).reshape(ref.shape)[:, edit["indices"]] for raw in read]
        delta = updated - original
        scopes.append({"tensor": tensor, "dtype": ref.dtype, "coordinates": int(delta.size),
                       "changed": int(np.count_nonzero(delta)), "norm": float(np.linalg.norm(delta))})
    return {"tensors": scopes, "coordinates": sum(row["coordinates"] for row in scopes),
            "changed": sum(row["changed"] for row in scopes),
            "norm": math.sqrt(sum(row["norm"] ** 2 for row in scopes))}


def _paired(base, candidate, suite, inference):
    rows = suite["target"] + suite["sentinel"]
    choices = suite["choices"]
    before = score_binary_margins(base, rows, inference, choices=choices)
    after = before if base == candidate else score_binary_margins(candidate, rows, inference, choices=choices)
    pairs = []
    for raw, initial, updated in zip(rows, before["rows"], after["rows"], strict=True):
        if raw["id"] != initial["id"] or raw["id"] != updated["id"]:
            raise ValueError("Scorer returned mismatched row identities")
        pairs.append({"id": raw["id"], "label": raw["label"], "family": raw["family"],
                      "item_key": raw["item_key"], "answer_position": raw["answer_position"],
                      "presentation": raw["presentation"],
                      "base_prediction": initial["prediction"], "candidate_prediction": updated["prediction"],
                      "base_margin": initial["margin"], "candidate_margin": updated["margin"],
                      "base_signed_margin": initial["signed_margin"],
                      "candidate_signed_margin": updated["signed_margin"]})
    return {"target": pairs[:len(suite["target"])], "sentinel": pairs[len(suite["target"]):],
            "base_resources": before["resources"], "candidate_resources": None if base == candidate else after["resources"]}


def _metrics(target, sentinel):
    base = sum(row["base_prediction"] == row["label"] for row in target)
    candidate = sum(row["candidate_prediction"] == row["label"] for row in target)
    return {"base_correct": base, "candidate_correct": candidate, "target_count": len(target),
            "target_gain": candidate - base,
            "target_regressions": sum(row["base_prediction"] == row["label"] != row["candidate_prediction"] for row in target),
            "sentinel_flips": sum(row["base_prediction"] != row["candidate_prediction"] for row in sentinel),
            "sentinel_count": len(sentinel)}


def _margin_groups(pair):
    groups = {}
    rows = pair["target"] + pair["sentinel"]
    for family in sorted({row["family"] for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        raw_delta = [row["candidate_margin"] - row["base_margin"] for row in family_rows]
        signed_delta = [row["candidate_signed_margin"] - row["base_signed_margin"] for row in family_rows]
        groups[family] = {"n": len(family_rows), "raw_delta_mean": float(np.mean(raw_delta)),
                          "signed_delta_mean": float(np.mean(signed_delta)),
                          "prediction_flips": sum(row["base_prediction"] != row["candidate_prediction"]
                                                  for row in family_rows)}
    return groups


def _compile(base, output, job):
    wall, cpu = time.perf_counter(), time.process_time()
    design, arm = job["design"], job["arm"]
    inference, policy = design["inference"], job["policy"]
    practice, intent = job["practice"], job["intent"]
    result = {"arm": arm, "candidate_attempts": 1, "practice_rows": len(practice["target"] + practice["sentinel"]),
              "eligible": False}
    try:
        if arm in ("random", "adam"):
            reference = job["reference"]
            reference_recompiled = reference is None
            if reference_recompiled:
                reference, _ = compile_awl_v2(
                    base, practice, intent, policy=policy, inference=inference,
                    authorized_preservation_families=tuple(intent["preserve"]),
                    max_radius=design["frob_cap"], max_probe_batches=design["probe_batch_cap"],
                )
                if reference is None:
                    raise ValueError("The control reference produced no patch")
                _clip_requested_norm(reference, design["frob_cap"])
            controls = design["controls"]
            if arm == "random":
                patch = matched_random_patch(reference, seed=controls["random_seed"] + job["cycle_index"])
            else:
                patch = fit_gradient_control(base, practice, reference, inference,
                                             seed=controls["adam_seed"] + job["cycle_index"],
                                             steps=controls["adam_steps"], learning_rate=controls["adam_learning_rate"])
            n_rows = len(practice["target"] + practice["sentinel"])
            batches = math.ceil(n_rows / inference["batch_size"])
            sentinel_batches = batches - len(practice["target"]) // inference["batch_size"]
            steps = controls["adam_steps"] if arm == "adam" else 0
            diagnostics = {"method": arm, "reference_recompiled": reference_recompiled,
                           "fit_forward_batches": steps * batches + (sentinel_batches if steps else 0)
                           + (batches if reference_recompiled else 0), "fit_backward_batches": steps * batches}
        else:
            patch, diagnostics = compile_awl_v2(
                base, practice, intent, policy=policy, inference=inference,
                authorized_preservation_families=tuple(intent["preserve"]),
                max_radius=design["frob_cap"], max_probe_batches=design["probe_batch_cap"],
            )
        result["synthesis"] = diagnostics
        if patch is None:
            result["disposition"] = "NO_PATCH"
            return result
        result["unclipped_requested_norm"] = _clip_requested_norm(patch, design["frob_cap"])
        result["requested_norm"] = _requested_norm(patch)
        json_dump_atomic(output / "patch.json", patch)
        candidate = output / "model"
        receipt = apply_safetensors_patch(base, patch, policy, candidate)
        scope = _realized_scope(base, candidate, patch, policy)
        if scope["changed"] != receipt["changed_entries"]:
            raise ValueError("Realized scope differs from the immutable patch receipt")
        result.update(scope=scope, candidate_sha256=receipt["candidate_sha256"])
        if not scope["changed"]:
            result["disposition"] = "PRECISION_NOOP"
            return result
        pair = _paired(base, candidate, job["screen"], inference)
        metrics = _metrics(pair["target"], pair["sentinel"])
        result["practice"] = {**metrics, "margin_groups": _margin_groups(pair), "pairs": pair}
        result["eligible"] = (metrics["target_gain"] >= 1 and not metrics["target_regressions"]
                              and not metrics["sentinel_flips"])
        result["disposition"] = "FINALIST" if result["eligible"] else "PRACTICE_REJECTED"
    except (EditRejected, InferenceError, ValueError, RuntimeError) as exc:
        result.update(disposition="ERROR", reason=f"{type(exc).__name__}: {exc}")
    finally:
        result["resources"] = _resources(wall, cpu)
    return result


def _paired_format_metrics(verification, prior_ids):
    target = verification["target"]
    sentinel = [row for row in verification["sentinel"] if row["id"] not in prior_ids]
    retention = [row for row in verification["sentinel"] if row["id"] in prior_ids]
    return {**_metrics(target, sentinel), "prior_count": len(retention),
            "prior_flips": sum(row["base_prediction"] != row["candidate_prediction"] for row in retention)}


def _verify(base, candidate, output, holdouts, retained, design):
    wall, cpu = time.perf_counter(), time.process_time()
    reports, metrics = {}, {}
    for name, source in holdouts.items():
        suite = _read(source)
        for row, _ in retained[name].values():
            suite["sentinel"].append(row)
        combined = output / f"verify-suite-{name}.json"
        report_path = output / f"verify-{name}.json"
        json_dump_atomic(combined, suite)
        verification = evaluate_hf_edit(base, candidate, combined, report_path, **design["inference"])
        reports[name] = {"suite": combined, "report": report_path, "result": verification}
        metrics[name] = _paired_format_metrics(verification, retained[name])
        metrics[name]["parent_prior_mismatches"] = sum(
            row["base_prediction"] != retained[name][row["id"]][1]
            for row in verification["sentinel"] if row["id"] in retained[name]
        )
    gates = design["acceptance"]
    passed = metrics["ab"]["target_gain"] >= gates["min_cycle_ab_gain"] and all(
        value["target_regressions"] <= gates["max_cycle_target_regressions"]
        and value["sentinel_flips"] <= gates["max_cycle_sentinel_flips"]
        and value["prior_flips"] <= gates["max_prior_prediction_flips"]
        and not value["parent_prior_mismatches"] for value in metrics.values()
    )
    return reports, {"accepted_by_verifier": passed, "formats": metrics, "resources": _resources(wall, cpu)}


def _final(base, candidate, suites, inference, cache):
    formats = {}
    clusters = {}
    for name, path in suites.items():
        key = (str(candidate), file_hash(path))
        if key not in cache:
            pair = _paired(base, candidate, _read(path), inference)
            cache[key] = {**_metrics(pair["target"], pair["sentinel"]), "pairs": pair,
                          "margin_groups": _margin_groups(pair)}
        formats[name] = cache[key]
        for row in formats[name]["pairs"]["target"]:
            change = int(row["candidate_prediction"] == row["label"]) - int(row["base_prediction"] == row["label"])
            clusters.setdefault(row["item_key"], []).append(change)
    cluster_means = np.array([np.mean(changes) for changes in clusters.values()])
    draws = np.random.default_rng(20261103).choice(cluster_means, size=(5000, len(cluster_means)), replace=True)
    return {"formats": formats, **{field: sum(row[field] for row in formats.values())
                                   for field in ("base_correct", "candidate_correct", "target_count", "target_gain",
                                                 "target_regressions", "sentinel_flips", "sentinel_count")},
            "underlying_item_count": len(clusters),
            "item_cluster_bootstrap_95": np.quantile(draws.mean(axis=1), [0.025, 0.975]).tolist()}


def _worker(base, output, job_path, timeout):
    command = [sys.executable, str(Path(__file__).resolve()), "compile", "--base", str(base),
               "--out", str(output), "--job", str(job_path)]
    try:
        completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        (output / "compile.stderr.txt").write_bytes(exc.stderr or b"")
        return {"disposition": "TIMEOUT", "eligible": False, "candidate_attempts": 1}
    if completed.returncode:
        # Keep the exception type and final error without machine-local traceback paths.
        lines = completed.stderr.decode("utf-8", errors="replace").strip().splitlines()
        return {"disposition": "PROCESS_ERROR", "eligible": False, "candidate_attempts": 1,
                "reason": lines[-1] if lines else f"Exited {completed.returncode}"}
    return _read(output / "compile.json")


def run(base, material, output, admission, *, timeout=600):
    total_wall, total_cpu = time.perf_counter(), time.process_time()
    protocol_path = material / "protocol.json"
    protocol = _read(protocol_path)
    gate = _read(admission)
    if (gate.get("schema") != "weightbridge.phase_iiib.synthetic_report.v1"
            or gate.get("exploratory_gate_passed") is not True
            or not gate.get("gates") or not all(check is True for check in gate["gates"].values())):
        raise ValueError("The frozen synthetic gate must pass before Qwen execution")
    if protocol.get("schema") != "weightbridge.phase_iiib.qwen_protocol.v1":
        raise ValueError("A frozen Phase III-B Qwen protocol is required")
    if load_checkpoint(base).checkpoint_sha256() != protocol["model_sha256"]:
        raise ValueError("Checkpoint differs from the frozen protocol")
    if type(timeout) is not int or not 1 <= timeout <= 3600:
        raise ValueError("Compiler timeout must be in [1, 3600] seconds")
    design = protocol["design"]
    if design["candidate_budget_per_cycle"] != 1:
        raise ValueError("The comparison runner implements exactly one candidate per cycle")
    # Verify only practice material here. The verifier checks sealed hashes
    # when opening each finalized cycle and the cumulative final suite.
    def artifact(relative):
        path = (material / relative).resolve()
        if not path.is_relative_to(material) or file_hash(path) != protocol["artifacts"][relative]:
            raise ValueError("Artifact differs from the frozen protocol")
        return path

    source_hashes = {name: file_hash(ROOT / name) for name in SOURCE_FILES if (ROOT / name).is_file()}
    output.mkdir(parents=True)
    json_dump_atomic(output / "execution-freeze.json", {
        "protocol_sha256": file_hash(protocol_path), "synthetic_admission_sha256": file_hash(admission),
        "source_sha256": source_hashes, "compiler_timeout_seconds": timeout,
    })
    arms = (*design["methods"], "random", "adam")
    states, kept, results = {}, {}, {}
    for arm in arms:
        work = output / arm
        work.mkdir()
        states[arm] = init_lineage(work / "lineage", base)
        kept[arm] = {name: {} for name in design["choice_formats"]}
        results[arm] = {"cycles": []}

    for cycle_index, cycle in enumerate(protocol["cycles"]):
        cycle_id = cycle["id"]
        jobs, reference = {}, None
        a_parent = states["A"]["head_sha256"]
        for arm in arms:
            work = output / arm / cycle_id
            work.mkdir()
            state = states[arm]
            base_for_arm = Path(state["head_path"]).resolve()
            control = arm not in design["methods"]
            job_path = work / "compile-input.json"
            job = {
                "arm": arm, "cycle_index": cycle_index, "policy": protocol["policy"], "design": design,
                "practice": _read(artifact(cycle["practice"]["generic" if arm == "A" or control else "aware"])),
                "screen": _read(artifact(cycle["practice"]["screen"])),
                "intent": _read(artifact(cycle["intents"]["A" if control else arm])),
                "reference": reference if control and state["head_sha256"] == a_parent else None,
            }
            json_dump_atomic(job_path, job)
            attempt = _worker(base_for_arm, work, job_path, timeout)
            jobs[arm] = (base_for_arm, work, attempt)
            if arm == "A" and (work / "patch.json").is_file():
                reference = _read(work / "patch.json")
            print(f"{cycle_id} {arm} practice {attempt['disposition']}", file=sys.stderr, flush=True)

        holdouts = {name: artifact(path) for name, path in cycle["holdouts"].items()}
        for arm, (parent, work, attempt) in jobs.items():
            state = states[arm]
            candidate = work / "model"
            report = {"cycle_id": cycle_id, "family": cycle["family"], "parent_sha256": state["head_sha256"],
                      "attempt": attempt, "accepted": False}
            if attempt.get("scope", {}).get("changed") and candidate.is_dir():
                # Include sealed diagnostics for practice-rejected stored
                # candidates. The practice decision is already frozen.
                verification, measured = _verify(parent, candidate, work, holdouts, kept[arm], design)
                accept = attempt["eligible"] and measured["accepted_by_verifier"]
                reason = ("Practice finalist passed both formats and prior retention" if accept else
                          "Practice gate rejected proposal" if not attempt["eligible"] else
                          "Independent format or retention gate rejected proposal")
                ab = verification["ab"]
                states[arm] = record_lineage_decision(
                    output / arm / "lineage", candidate, ab["report"], suite=ab["suite"], cycle_id=cycle_id,
                    accept=accept, reason=reason, expected_parent_sha256=state["head_sha256"],
                    source_holdout=holdouts["ab"],
                )
                if accept:
                    for name, source in holdouts.items():
                        source_doc = _read(source)
                        for group in ("target", "sentinel"):
                            rows = source_doc[group]
                            outcomes = verification[name]["result"][group][:len(rows)]
                            for row, outcome in zip(rows, outcomes, strict=True):
                                kept[arm][name][row["id"]] = (row, outcome["candidate_prediction"])
                measured["artifact_sha256"] = {
                    name: {kind: file_hash(files[kind]) for kind in ("suite", "report")}
                    for name, files in verification.items()
                }
                report.update(verification=measured, accepted=accept)
            else:
                states[arm] = record_lineage_rejection(
                    output / arm / "lineage", cycle_id=cycle_id, reason=attempt["disposition"],
                    expected_parent_sha256=state["head_sha256"],
                )
            report["head_sha256"] = states[arm]["head_sha256"]
            json_dump_atomic(work / "decision.json", report)
            results[arm]["cycles"].append(report)
            print(f"{cycle_id} {arm} accepted={report['accepted']}", file=sys.stderr, flush=True)

    # None of the final suites are read before every candidate decision closes.
    finals = {name: artifact(path) for name, path in protocol["final"].items()}
    cache = {}
    for arm in arms:
        results[arm]["final"] = _final(base, Path(states[arm]["head_path"]).resolve(), finals,
                                       design["inference"], cache)
        results[arm]["accepted_cycles"] = sum(cycle["accepted"] for cycle in results[arm]["cycles"])
    strongest = max(results[arm]["final"]["target_gain"] for arm in arms if arm != "D")
    observed = results["D"]["final"]
    acceptance = design["acceptance"]
    qualifies = (results["D"]["accepted_cycles"] >= acceptance["min_accepted_full_cycles"]
                 and observed["target_gain"] >= acceptance["min_final_gain"]
                 and observed["target_gain"] - strongest >= acceptance["min_final_gain_over_strongest_full_baseline"]
                 and observed["target_regressions"] <= acceptance["max_final_target_regressions"]
                 and observed["sentinel_flips"] <= acceptance["max_final_sentinel_flips"])
    if any(file_hash(ROOT / name) != digest for name, digest in source_hashes.items()):
        raise ValueError("Experiment source changed during the frozen run")
    report = {"schema": "weightbridge.phase_iiib.qwen_ablation.v1", "scope": protocol["scope"],
              "protocol_sha256": file_hash(protocol_path), "model_sha256": protocol["model_sha256"],
              "disposition": ("EXPLORATORY" if protocol["scope"] == "bounded-pilot" else
                              "PASS" if qualifies else "FAIL"),
              "strongest_baseline_final_gain": strongest, "arms": results,
              "host_resources": _resources(total_wall, total_cpu)}
    json_dump_atomic(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "compile"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--material", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.stage == "compile":
        if args.job is None:
            parser.error("--job is required to compile")
        result = _compile(args.base.resolve(), args.out.resolve(), _read(args.job))
        json_dump_atomic(args.out / "compile.json", result)
        print(result["disposition"])
    else:
        if args.material is None or args.admission is None:
            parser.error("--material and --admission are required for a run")
        result = run(args.base.resolve(), args.material.resolve(), args.out.resolve(), args.admission.resolve(),
                     timeout=args.timeout)
        print(json.dumps({"scope": result["scope"], "disposition": result["disposition"],
                          "accepted": {arm: value["accepted_cycles"] for arm, value in result["arms"].items()}}, indent=2))
