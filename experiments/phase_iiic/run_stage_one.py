"""Run a bounded Stage 1 only after measurement and headroom admission."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiic.measurement_roster import PROTOCOL
from experiments.phase_iiic.run_measurement import _evaluate_split, _load_split, _rows
from weightbridge.adaptation import content_scoring
from weightbridge.adaptation.behavior_stage import (
    StageOneAdmission, compare_behavior_rows, compile_stage_one_candidate, stage_one_admission,
)
from weightbridge.adaptation.bic import _runner
from weightbridge.adaptation.lineage import init_lineage, record_lineage_decision, record_lineage_rejection
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.provenance import peak_rss_bytes


STAGE_PROTOCOL = Path(__file__).with_name("stage_one_protocol.json")
ARMS = ("A", "B", "C", "D")


def _read(path):
    return json.loads(path.read_bytes())


def _practice(items):
    return [{"item_id": row["item_id"], **{
        view: {"id": row["id"], "prompt": row[f"{view}_prompt"], "answers": row[choices],
               "correct": row["correct_position"]}
        for view, choices in (("letter", "labels"), ("content", "answers"))}}
        for row in _rows(items)]


def _choose_practice(items, admission, settings):
    by_family = {entry[0]: entry[1:] for entry in admission.practice_scope}
    count = settings["preservation_items_per_arm"]
    for family in settings["family_priority"]:
        if family not in by_family:
            continue
        targets, controls = by_family[family]
        protected_family = settings["protected_family"][family]
        generic, task = [], []
        for item in items:
            if item["id"] in controls:
                (task if item["family"] == protected_family else generic).append(item)
        if len(generic) >= count and len(task) >= count:
            return family, _practice([item for item in items if item["id"] in targets]), {
                "generic": _practice(generic[:count]), "task_aware": _practice(task[:count]),
            }
    return None


def _measure(base, items, config, family, scoring):
    wall, cpu = time.perf_counter(), time.process_time()
    with _runner(base, config["inference"]) as (runner, _):
        raw, resources = _evaluate_split(runner, items, config)
    paired = []
    for row in raw:
        content = row["conditions"]["content"]
        paired.append({"id": row["id"], "item_id": row["item_id"], "correct": row["correct_position"],
                       "role": "target" if row["family"] == family else "preservation",
                       "scored": content["likelihood"][scoring]["position"],
                       "generated": content["generation"]["eligible_position"]})
    return {"paired": paired, "raw": raw, "resources": {"phases": resources,
            "wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
            "peak_rss_bytes": peak_rss_bytes()}}


def _lineage_report(work, before, after, receipt, roster_sha256):
    """Project completed generation to the lineage verifier's paired schema."""
    suite = {"schema": "weightbridge.hf_edit_suite.v1", "choices": ["0", "1", "unparsed"],
             "target": [], "sentinel": []}
    pairs = {"target": [], "sentinel": []}
    prediction = lambda row: "unparsed" if row["generated"] is None else str(row["generated"])
    for parent, candidate, raw in zip(before["paired"], after["paired"], before["raw"], strict=True):
        group = "target" if parent["role"] == "target" else "sentinel"
        label = str(parent["correct"])
        suite[group].append({"id": parent["id"], "text": raw["content_prompt"], "label": label})
        pairs[group].append({"id": parent["id"], "label": label, "base_prediction": prediction(parent),
                             "candidate_prediction": prediction(candidate)})
    suite_path = work / "verification-suite.json"
    json_dump_atomic(suite_path, suite)
    old = sum(row["base_prediction"] == row["label"] for row in pairs["target"])
    new = sum(row["candidate_prediction"] == row["label"] for row in pairs["target"])
    flips = sum(row["base_prediction"] != row["candidate_prediction"] for row in pairs["sentinel"])
    report_path = work / "lineage-verification.json"
    json_dump_atomic(report_path, {"schema": "weightbridge.hf_edit_evaluation.v1", **pairs,
        "measurement": "completed_parsed_generation", "source_roster_sha256": roster_sha256,
        "base_sha256": receipt["base_sha256"], "candidate_sha256": receipt["candidate_sha256"],
        "suite_sha256": file_hash(suite_path),
        "receipt_sha256": file_hash(work / "candidate/model/.weightbridge_receipt.json"),
        "base_target_correct": old, "candidate_target_correct": new, "target_gain": new - old,
        "sentinel_prediction_flips": flips, "min_target_gain": 1, "max_sentinel_flips": 0,
        "disposition": "PASS" if new > old and not flips else "FAIL"})
    return report_path, suite_path


def run(base, material, measurement, output):
    admission = stage_one_admission(PROTOCOL, measurement)
    output.mkdir(parents=True)
    decision = {"schema": "weightbridge.phase_iiic.stage_one_admission.v1", **asdict(admission),
                "stage_one_protocol_sha256": file_hash(STAGE_PROTOCOL)}
    if not admission.admitted:
        json_dump_atomic(output / "report.json", {**decision, "disposition": "HOLD", "candidate_attempts": 0})
        return "HOLD"
    manifest_path = material / "manifest.json"
    qualified = _read(measurement)
    if (file_hash(manifest_path) != qualified["material_manifest_sha256"]
            or file_hash(content_scoring.__file__) != qualified["scoring_source_sha256"]
            or load_checkpoint(base).checkpoint_sha256() != admission.base_sha256):
        raise ValueError("Stage 1 requires the qualified scorer, frozen material and checkpoint")
    manifest, config, settings = _read(manifest_path), _read(PROTOCOL), _read(STAGE_PROTOCOL)
    prepared = _choose_practice(_load_split(material, manifest, "practice"), admission, settings)
    if prepared is None:
        json_dump_atomic(output / "report.json", {**decision, "disposition": "HOLD", "candidate_attempts": 0,
                                                  "stage_reason": "INSUFFICIENT_MATCHED_PRESERVATION_CONTROLS"})
        return "HOLD"
    family, targets, preservation = prepared
    tensor = f"model.language_model.layers.{settings['layer']}.mlp.down_proj.weight"
    policy = {"schema": "weightbridge.policy.v1", "allow_tensors": [tensor],
              **{key: settings[key] for key in ("max_entries", "max_fraction", "max_abs_delta")}}
    outcomes = {}
    for arm in ARMS:
        work = output / arm
        init_lineage(work / "lineage", base)
        job = {"admission": asdict(admission), "arm": arm, "family": family, "targets": targets,
               "preservation": preservation, "tensor_names": [tensor], "inference": config["inference"],
               "policy": policy, "radius": settings["radius"]}
        path = work / "authorized-practice.json"
        json_dump_atomic(path, job)
        command = [sys.executable, str(Path(__file__).resolve()), "--job", str(path), "--base", str(base),
                   "--out", str(work / "candidate")]
        try:
            proc = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=settings["worker_timeout_seconds"])
            outcomes[arm] = _read(work / "candidate/candidate.json") if proc.returncode == 0 else {
                "disposition": "ERROR", "reason": proc.stderr.decode(errors="replace").strip().splitlines()[-1:]
            }
        except subprocess.TimeoutExpired:
            outcomes[arm] = {"disposition": "TIMEOUT"}
    json_dump_atomic(output / "compiled.json", outcomes)

    def score_stage(split, arms):
        if not arms:
            return []
        items = _load_split(material, manifest, split)
        parent = _measure(base, items, config, family, admission.scoring)
        json_dump_atomic(output / f"{split}-parent.json", parent)
        accepted = []
        for arm in arms:
            work = output / arm
            candidate_path = work / "candidate/model"
            receipt = outcomes[arm]["receipt"]
            candidate = _measure(candidate_path, items, config, family, admission.scoring)
            if load_checkpoint(candidate_path).checkpoint_sha256() != receipt["candidate_sha256"]:
                raise ValueError("Persisted candidate differs from the patch receipt")
            gate = compare_behavior_rows(parent["paired"], candidate["paired"])
            outcomes[arm][split] = gate
            json_dump_atomic(work / f"{split}-measurement.json", {"candidate": candidate, "gate": gate})
            if gate["passed"]:
                accepted.append(arm)
            if split == "sealed-adaptation":
                report, suite = _lineage_report(work, parent, candidate, receipt, manifest["artifacts"][split]["sha256"])
                record_lineage_decision(work / "lineage", candidate_path, report, suite=suite, cycle_id="cycle-01",
                                        accept=gate["passed"], reason="Independent content and generation gate",
                                        expected_parent_sha256=admission.base_sha256)
        if load_checkpoint(base).checkpoint_sha256() != admission.base_sha256:
            raise ValueError("Original checkpoint changed during verification")
        return accepted

    stored = [arm for arm, result in outcomes.items() if result["disposition"] == "PENDING_SELECTION"]
    selected = score_stage("selection", stored)
    json_dump_atomic(output / "selection.json", {"selected_arms": selected, "outcomes": outcomes})
    verified = score_stage("sealed-adaptation", selected)
    for arm in ARMS:
        if arm not in selected:
            record_lineage_rejection(output / arm / "lineage", cycle_id="cycle-01",
                                     reason="No independently eligible finalist", expected_parent_sha256=admission.base_sha256)
    status = "PASS" if verified else "FAIL"
    json_dump_atomic(output / "report.json", {**decision, "disposition": status, "candidate_attempts": len(ARMS),
                                              "selected_arms": selected, "verified_arms": verified, "outcomes": outcomes})
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("base", "out", "material", "measurement", "job"):
        parser.add_argument(f"--{arg}", type=Path, required=arg in ("base", "out"))
    args = parser.parse_args()
    if args.job:
        job = _read(args.job)
        admission = StageOneAdmission(**job.pop("admission"))
        print(compile_stage_one_candidate(admission, base=args.base, output=args.out, **job)["disposition"])
    else:
        if args.material is None or args.measurement is None:
            parser.error("--material and --measurement are required")
        print(run(args.base.resolve(), args.material.resolve(), args.measurement.resolve(), args.out.resolve()))
