"""Reproduce published L/M analyses without modifying their archives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.phase_iiim.audit_previous import audit as audit_k_l
from experiments.phase_iiim.analyze import summarize as analyze_m
from weightbridge.core import file_hash, json_dump_atomic

ROOT = Path(__file__).resolve().parents[2]


def audit():
    previous = audit_k_l()
    folder = ROOT / "results/phase-iii-m/factorial-v1"
    actual = analyze_m(folder / "trial", folder / "material")
    if json.dumps(actual, sort_keys=True) != json.dumps(json.loads((folder / "analysis.json").read_bytes()), sort_keys=True):
        raise ValueError("Phase III-M analysis failed canonical exact replay")
    paths = (folder / "analysis.json", folder / "trial/fit-report.json", folder / "trial/assessment.json",
             folder / "material/manifest.json", folder / "trial/basis.json")
    return {"schema": "weightbridge.phase_iiin.previous_audit.v1",
            "starting_commit": "178db583894278b9fc52150f8789a1a5f2eadb49",
            "replayed": ["phase-iii-k", "phase-iii-l", "phase-iii-m"], "k_l_audit": previous,
            "phase_m_analysis_canonical_exact": True,
            "source_sha256": {path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths},
            "phase_m_disposition": actual.get("disposition"),
            "scope": "Published aggregate old hidden outcome is audit evidence only; no old hidden rows were opened."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    json_dump_atomic(args.out, audit())
