"""Replay the published K/L archives and extract the Phase III-L causal limitations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.phase_iiik.analyze import summarize as analyze_k
from experiments.phase_iiil.analyze import summarize as analyze_l
from weightbridge.core import file_hash, json_dump_atomic


ROOT = Path(__file__).resolve().parents[2]
START_COMMIT = "07cb0bdcf099de444f798527d4ba32f15532f04b"


def audit():
    results, hashes = {}, {}
    for phase, folder, summarize in (("k", "jspace-v1", analyze_k), ("l", "semantic-v1", analyze_l)):
        root = ROOT / f"results/phase-iii-{phase}" / folder
        actual = summarize(root / "trial", root / "material")
        expected = json.loads((root / "analysis.json").read_bytes())
        if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError(f"Phase III-{phase.upper()} archive does not replay")
        results[phase] = actual
        paths = [root / "analysis.json", root / "material/manifest.json"]
        paths += [root / "trial" / name for name in ("fit-report.json", "candidate-lock.json", "independent-outcome.json")
                  if (root / "trial" / name).exists()]
        hashes.update({path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths})
    l = results["l"]
    arms = l["arms"]
    practice = arms["B2"]["practice_gate"]
    transfer, selection, hidden = l["transfer"]["B2"], l["selection"]["B2"], l["independent"]
    geometry = l["directions"]["2"]["directions"]
    same_fields = ("update_norm", "group_raw_mean_improvement", "item_predicted_margin_change",
                   "protected_predicted_margins")
    same_mean = all(geometry["weighted_mean"][key] == geometry["maximin"][key] for key in same_fields)
    public_pairs = {
        "practice": practice["candidate_truth_flip_matrix"]["both_correct"],
        "development_validation": transfer["candidate_truth_flip_matrix"]["both_correct"],
        "selection": selection["candidate_truth_flip_matrix"]["both_correct"],
    }
    checks = {
        "practice_1_to_4": (practice["parent_target_correct"], practice["candidate_target_correct"]) == (1, 4),
        "validation_25_preserved": transfer["passed"] and transfer["protected_presentations"] == 25,
        "selection_0_to_1_zero_regression": (selection["parent_target_correct"], selection["candidate_target_correct"]) == (0, 1)
                                           and not selection["regressions"] and selection["passed"],
        "sealed_gain3_regression1_rejected": hidden["target_item_gain"] == 3 and hidden["regressions"] == 1
                                             and hidden["disposition"] == "FAIL",
        "no_accepted_checkpoint": not hidden.get("accepted_candidate_sha256"),
        "no_complete_public_truthflip": all(value == 0 for value in public_pairs.values()),
        "larger_maximin_matches_normalized_group_mean": same_mean,
        "realized_norms_differ": arms["A2"]["realized_norm"] != arms["B2"]["realized_norm"],
    }
    if not all(checks.values()):
        raise ValueError("A reported Phase III-L finding could not be verified")
    code = ("weightbridge/adaptation/semantic_update.py", "weightbridge/adaptation/semantic_consensus.py",
            "experiments/phase_iiil/run_semantic.py", "experiments/phase_iiil/semantic_protocol.json",
            "experiments/phase_iiil/verification.py")
    hashes.update({path: file_hash(ROOT / path) for path in code})
    return {"schema": "weightbridge.phase_iiim.previous_audit.v1", "starting_commit": START_COMMIT,
            "replayed": ["phase-iii-k", "phase-iii-l"], "verified_findings": checks,
            "source_sha256": hashes, "public_complete_truthflip_families": public_pairs,
            "phase_l_realized_norms": {name: arms[name]["realized_norm"] for name in ("A2", "B2")},
            "limitations": ["Phase III-L changed CE to margin and item normalization together.",
                "Equal subgroup counts make equal-group and global unit means equivalent.",
                "The stored intervention norms differed, so causal direction effects were confounded.",
                "The sealed failure is preserved as aggregate evidence and supplies no new fitting examples."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    json_dump_atomic(args.out, audit())
