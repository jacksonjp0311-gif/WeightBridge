"""Rebuild and score an archived Qwen repair with the product evaluator."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.core import json_dump_atomic
from weightbridge.evaluation.hf_edit import evaluate_hf_edit
from weightbridge.patches import apply_safetensors_patch

EXAMPLES = {
    "wording": ("qwen-wording-ablation-v1", "case-12", "multi", 10),
    "ai-pilot": ("qwen-ai-edit-pilot-v1", "case-03", "guided", 3),
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def apply_archived(source, out, folder, name, policy):
    with gzip.open(folder / f"{name}_patch.json.gz", "rt", encoding="utf-8") as stream:
        patch = json.load(stream)
    expected = read(folder / f"{name}_receipt.json")
    receipt = apply_safetensors_patch(source, patch, policy, out)
    for field in ("base_sha256", "candidate_sha256", "patch_sha256", "policy_sha256", "changed_entries"):
        if receipt[field] != expected[field]:
            raise ValueError(f"Rebuilt {name} receipt differs in {field}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pristine", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--example", choices=EXAMPLES, default="wording")
    args = parser.parse_args()
    args.out.mkdir()
    evidence_name, case_name, arm, min_gain = EXAMPLES[args.example]
    evidence = ROOT / "results" / evidence_name
    case = evidence / case_name
    policy = read(evidence / "manifest.json")["policy"]
    damaged, repaired = args.out / "damaged", args.out / "repaired"
    apply_archived(args.pristine, damaged, case / "evaluator", "fault", policy)
    apply_archived(damaged, repaired, case / "search", arm, policy)
    suite = args.out / "holdout.json"
    holdout = read(case / "evaluator/holdout.json")
    json_dump_atomic(suite, {"schema": "weightbridge.hf_edit_suite.v1", "choices": ["A", "B"], **holdout})
    report = evaluate_hf_edit(damaged, repaired, suite, args.out / "report.json",
                              min_target_gain=min_gain, dtype="bfloat16")
    expected = read(case / "evaluation/scores.json")
    for group in ("target", "sentinel"):
        observed = [(row["base_prediction"], row["candidate_prediction"]) for row in report[group]]
        archived = [(a["prediction"], b["prediction"])
                    for a, b in zip(expected["damaged"][group], expected[arm][group], strict=True)]
        if observed != archived:
            raise ValueError(f"Reloaded {group} predictions differ from archived scores")
    print(json.dumps({key: report[key] for key in (
        "disposition", "base_target_correct", "candidate_target_correct", "target_count",
        "sentinel_count", "sentinel_prediction_flips",
    )}, indent=2))
    if report["disposition"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
