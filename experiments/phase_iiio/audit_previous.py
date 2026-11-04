"""Replay published M/N analyses and N's post-hoc local bound certificates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.phase_iiin.audit_previous import audit as prior_audit
from experiments.phase_iiin.analyze import summarize as analyze_n
from experiments.phase_iiin.local_pair_bounds import analyze as pair_bounds
from weightbridge.core import file_hash, json_dump_atomic

ROOT = Path(__file__).resolve().parents[2]


def audit():
    prior = prior_audit()
    root = ROOT / "results/phase-iii-n/paired-v1"
    actual = analyze_n(root / "trial", root / "material")
    if json.dumps(actual, sort_keys=True) != json.dumps(json.loads((root / "analysis.json").read_bytes()), sort_keys=True):
        raise ValueError("Phase III-N canonical archive replay differs")
    expected = json.loads((root / "local-pair-bounds.json").read_bytes())
    replay = pair_bounds(root / "trial")
    for key in ("families", "input_sha256", "scope", "used_for_selection_or_fitting"):
        if expected[key] != replay[key]:
            raise ValueError("Phase III-N local averaged-margin certificate does not replay")
    paths = (root / "analysis.json", root / "local-pair-bounds.json", root / "trial/fit-report.json",
             root / "trial/assessment.json", root / "material/manifest.json", root / "reconstruction-B2.json")
    return {"schema": "weightbridge.phase_iiio.previous_audit.v1",
        "starting_commit": "e2dbbf353501dddab927c850414ab5938381380d",
        "replayed": ["phase-iii-k", "phase-iii-l", "phase-iii-m", "phase-iii-n"], "prior_audit": prior,
        "phase_n_canonical_replay_exact": True, "phase_n_local_certificates_replay_exact": True,
        "phase_n_disposition": actual["disposition"], "phase_n_reason": actual["reason"],
        "phase_n_practice_B2": {key: actual["arms"]["B2"]["practice_gate"][key] for key in (
            "target_gain", "complete_family_gain", "regressions", "lost_valid")},
        "phase_n_development_B2": actual["development"]["B2"]["gate"],
        "source_sha256": {path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths},
        "scope": "Read-only published evidence; no old disclosed development or hidden result supplies a new fitting example."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    json_dump_atomic(args.out, audit())
