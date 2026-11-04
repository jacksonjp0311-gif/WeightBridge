"""Extract the Phase III-K fitted fork evidence without revisiting any model data."""

from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path
from statistics import mean

from weightbridge.core import file_hash, json_dump_atomic


SOURCE_COMMIT = "4b10f9dfe415439eb7e260714ea75eaee20f80b3"
TRIAL = Path("results/phase-iii-k/jspace-v1/trial")


def audit(trial):
    report_path = trial / "report.json"
    report = json.loads(report_path.read_bytes())
    hashes = {"report.json": file_hash(report_path)}

    def read(name):
        path = trial / name
        digest = file_hash(path)
        if digest != report["artifacts"][name]:
            raise ValueError(f"Phase III-K receipt differs from its archive: {name}")
        hashes[name] = digest
        return json.loads(path.read_bytes())

    parent = read("parent-adaptation.json")
    diagnosis = read("diagnosis.json")
    arm = read("arm-C1.json")
    observed = read(f"arm-{arm['duplicate_of']}.json")
    if arm["probe"]["candidate_tensor_sha256"] != observed["probe"]["candidate_tensor_sha256"]:
        raise ValueError("Phase III-K rounded-arm alias does not match the observed state")
    rows = {row["id"]: row for row in parent["measurement"]["rows"]}
    forks = {fork["prompt_id"]: fork for fork in diagnosis["forks"]}
    comparisons = defaultdict(list)
    fitted = []
    predictions = arm["geometry"]["target_predictions"]
    measured = observed["probe"]["fork"]
    for projection, candidate_token, candidate_margin in zip(
            predictions, measured["candidate_predicted_ids"], measured["candidate_decision_margins"], strict=True):
        prompt = projection["prompt_id"]
        row, fork = rows[prompt], forks[prompt]
        truth, other = sorted(row["operands"], key=Decimal, reverse=True)
        if Decimal(truth) != Decimal(row["answer"]):
            raise ValueError("Archived correct answer disagrees with exact comparison")
        precision = "shorter" if len(truth.partition(".")[2]) < len(other.partition(".")[2]) else "longer"
        comparisons[precision].append(projection["slope"])
        fitted.append({
            "item": row["item"], "wording": row["wording"], "tier": row["tier"], "correct_answer_length": precision,
            "operands": row["operands"], "correct_answer": row["answer"], "parent_answer": row["exact"],
            "shared_token_ids": fork["shared_token_ids"], "accepted_tokens": fork["replay"]["accepted"],
            "actual_token": fork["replay"]["actual"], "predicted_margin_slope": projection["slope"],
            "candidate_token": candidate_token, "candidate_margin": candidate_margin,
            "measured_accepted_token": candidate_token in fork["accepted_token_ids"],
        })
    fitted_items = {row["item"] for row in fitted}
    omitted = [row for row in parent["measurement"]["rows"]
               if row["wording"] == 0 and row["item"] in parent["qualification"]["repeatable_errors"]
               and row["item"] not in fitted_items]
    return {
        "schema": "weightbridge.phase_iiil.phase_k_fork_audit.v1", "source_commit": SOURCE_COMMIT,
        "source_sha256": hashes, "fitted_presentations": fitted,
        "mean_slope_by_correct_answer_length": {key: mean(values) for key, values in comparisons.items()},
        "repeatable_unfitted_errors": [{"item": row["item"], "tier": row["tier"], "operands": row["operands"],
                                       "correct_answer": row["answer"], "parent_answer": row["exact"]} for row in omitted],
        "observed_arm": arm["duplicate_of"], "candidate_tensor_sha256": observed["probe"]["candidate_tensor_sha256"],
        "corrected_repeatable_items": observed["probe"]["endpoint"]["guard"]["corrected_repeatable_target_items"],
        "selection": report["reason"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, default=TRIAL)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    json_dump_atomic(args.out, audit(args.trial))


if __name__ == "__main__":
    main()
