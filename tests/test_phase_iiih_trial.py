from pathlib import Path

import pytest

from experiments.phase_iiig import roster_decimal_targeted
from experiments.phase_iiih import decimal_roster, run_decimal_endpoint
from weightbridge.core import file_hash


MATERIAL = Path(__file__).resolve().parents[1] / "results/phase-iii-h/decimal-endpoint-v1/material"


def test_frozen_public_rosters_are_disjoint_and_exclude_disclosed_questions():
    manifest = decimal_roster.read_manifest(MATERIAL)
    blocked = decimal_roster._blocked_identities()
    for split in ("practice", "selection"):
        rows = decimal_roster.load_split(MATERIAL, split, manifest)
        identities = {roster_decimal_targeted._identity(row["family"], row["operands"]) for row in rows}
        assert len(identities) == 24
        assert not identities & blocked
        blocked |= identities
    assert not (MATERIAL / "hidden.json").exists()
    with pytest.raises(ValueError, match="explicit host path"):
        decimal_roster.load_split(MATERIAL, "hidden", manifest)


def test_independent_worker_requires_candidate_lock_before_loading_hidden(tmp_path):
    private = tmp_path / "hidden.json"
    private.write_text("not a valid roster")
    with pytest.raises(ValueError, match="requires a candidate seal"):
        run_decimal_endpoint._worker(
            tmp_path / "base", "unused", MATERIAL, "hidden", tmp_path / "report.json",
            file_hash(MATERIAL / "manifest.json"), hidden=private,
        )


def test_surrogate_fidelity_does_not_conflate_proxy_improvement_and_endpoint_pass():
    def probe(identifier, surrogate_gain, target_gain, *, regressions=(), passed=False):
        return {"id": identifier, "margin": {"surrogate_gain": surrogate_gain},
                "endpoint": {"guard": {"passed": passed, "target_item_gain": target_gain,
                                       "prohibited_regressions": regressions, "lost_valid_presentations": []}}}

    evidence = run_decimal_endpoint.surrogate_fidelity({"corrected": {"probes": [
        {"id": "no-op"},
        probe("score-only", .25, 0),
        probe("real-but-regressing", .5, 1, regressions=["correct-copy"]),
        probe("real-with-lower-proxy", -.05, 1, passed=True),
    ]}})
    assert evidence["behavior_improvements_given_surrogate_improvement"] == [1, 2]
    assert evidence["wilson_95_descriptive"][0] < .5 < evidence["wilson_95_descriptive"][1]
    assert [entry["practice_pass"] for entry in evidence["pairs"]] == [False, False, True]
