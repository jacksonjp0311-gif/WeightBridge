"""Recompute Phase III-I endpoint gates from public evidence and summarize paired outcomes."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.phase_iiih.run_decimal_endpoint import _paired, endpoint, qualify
from experiments.phase_iiii.decimal_roster import PROTOCOL, read_manifest
from experiments.phase_iiii.verification import source_hashes
from weightbridge.adaptation.behavior_endpoint import compare_endpoint
from weightbridge.core import file_hash


def _read(path):
    return json.loads(path.read_bytes())


def summarize(trial, material):
    report, config = _read(trial / "report.json"), _read(PROTOCOL)
    read_manifest(material)
    if report["sources"] != source_hashes() or report["manifest_sha256"] != file_hash(material / "manifest.json"):
        raise ValueError("Phase III-I sources or roster commitment differ")
    for name, expected in report["artifacts"].items():
        path = (trial / name).resolve()
        if not path.is_relative_to(trial.resolve()) or file_hash(path) != expected:
            raise ValueError(f"Evidence hash differs for {name}")
    lock = _read(trial / "candidate-lock.json")
    if (file_hash(trial / "candidate-lock.json") != report["selection"]["candidate_lock_sha256"]
            or lock["sources"] != report["sources"] or lock["manifest_sha256"] != report["manifest_sha256"]):
        raise ValueError("Finalist lock differs from the report")
    practice = _read(trial / "practice.json")["measurement"]
    contract = endpoint(config)
    qualification = qualify(practice, config)
    if qualification != report["qualification"]:
        raise ValueError("Archived practice qualification differs")
    winner = lock["winner"]
    if winner != report["selection"]["winner"] or lock["selection"] != report["selection"]["selection"]:
        raise ValueError("Archived selection decision differs")
    selection_dir = trial / "verification"
    selection_base = _read(selection_dir / "selection-base.json")["measurement"] if "base" in lock["reports"] else None
    for name, digest in lock["reports"].items():
        path = selection_dir / f"selection-{name}.json"
        if file_hash(path) != digest:
            raise ValueError(f"Independent selection evidence differs for {name}")
        if name != "base" and _paired(selection_base, _read(path)["measurement"], config, config["selection_gate"]) != lock["selection"][name]:
            raise ValueError(f"Independent selection verdict differs for {name}")
    diagnostic_path = trial / "diagnosis.json"
    diagnostic = _read(diagnostic_path) if diagnostic_path.exists() else {}
    fitted_forks = [fork for fork in diagnostic.get("forks", []) if fork["question_id"] in diagnostic["target_items"]]
    arms = {}
    for arm in report["arms"]:
        raw = _read(trial / f"arm-{arm}.json")
        probe = raw["probe"]
        observed = probe.get("endpoint")
        row = {"reason": probe["reason"], "layer": raw["layer"], "realized_norm": probe.get("realized_update_norm"),
               "changed_entries": probe.get("changed_entries"), "duplicate_of": raw.get("duplicate_of"),
               "probe_resources": probe.get("resources"), "training_resources": raw.get("diagnostics", {}).get("resources")}
        if observed is not None:
            guard = compare_endpoint(practice, observed["measurement"], contract, min_repeatable_corrections=1)
            corrected = sorted(set(guard["corrected_repeatable_target_items"]) & qualification["repeatable_errors"].keys())
            guard["corrected_repeatable_negative_items"] = corrected
            guard["passed"] = bool(guard["passed"] and corrected)
            if guard != observed["guard"] or probe.get("restored_exactly") is not True:
                raise ValueError(f"Archived practice guard or restoration differs for arm {arm}")
            fork, trajectory = probe.get("fork") or {}, probe.get("trajectory") or {}
            candidate_rows = {record["id"]: record for record in observed["measurement"]["rows"]}
            fork_paths = []
            for observed_fork in fitted_forks:
                candidate_row = candidate_rows[observed_fork["prompt_id"]]
                ids, prefix = candidate_row["generated_token_ids"], observed_fork["shared_token_ids"]
                position = len(prefix)
                retained = ids[:position] == prefix
                accepted = retained and len(ids) > position and ids[position] in observed_fork["accepted_token_ids"]
                parsed = contract.parse(candidate_row["text"], candidate_row["stop_reason"])
                fork_paths.append({"prompt_id": observed_fork["prompt_id"], "retained": retained,
                                   "accepted": accepted, "completed_correct": parsed.completed and
                                   parsed.answer == contract.question(candidate_row).answer})
            row.update(endpoint_pass=guard["passed"], target_item_gain=guard["target_item_gain"],
                       repeated_items_corrected=len(corrected), regressions=len(guard["prohibited_regressions"]),
                       lost_validity=len(guard["lost_valid_presentations"]),
                       whole_answer_loss_gain=probe.get("margin", {}).get("surrogate_gain"),
                       fork_loss_gain=fork.get("cross_entropy_gain"), accepted_next_tokens=fork.get("accepted_next_token_count"),
                       fork_count=fork.get("probe_count"), new_nonpositive_margins=trajectory.get("new_nonpositive_margin_positions"),
                       sampled_positions=trajectory.get("positions"), mean_kl=trajectory.get("mean_kl"),
                       endpoint_decode_steps=observed["measurement"]["resources"]["batched_decode_steps"],
                       actual_fork_paths=fork_paths)
        arms[arm] = row
    scan = _read(trial / "scan.json") if (trial / "scan.json").exists() else None
    loss_improvements = [arm for arm in arms.values() if (arm.get("fork_loss_gain") or 0) > 0]
    return {"schema": "weightbridge.phase_iiii.analysis.v1", "single_cycle": report["single_cycle"],
            "source_checkpoint": report["base_sha256"], "report_sha256": file_hash(trial / "report.json"),
            "arms": arms, "winner": winner, "scan": scan, "fork_replay_status": dict(Counter(
                fork["replay_status"] for fork in diagnostic.get("forks", []))),
            "fork_offsets": dict(Counter(len(fork["shared_token_ids"]) for fork in diagnostic.get("forks", []))),
            "target_item_gain_given_fork_loss_gain": [sum(arm["target_item_gain"] > 0 for arm in loss_improvements),
                                                     len(loss_improvements)],
            "trial_resources": report["resources"], "selection_resources": report["selection"]["verifier_resources"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--material", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.trial, args.material), indent=2))
