import json
from pathlib import Path

import pytest

from experiments.phase_iiig.roster_decimal_targeted import _identity
from experiments.phase_iiik import archive_trial
from experiments.phase_iiik.baseline import PUBLIC_SPLITS
from experiments.phase_iiik.decimal_roster import PROTOCOL, _blocked_identities, load_split, read_manifest
from experiments.phase_iiik.run_jspace import _read_fit, _selection_keys


MATERIAL = Path(__file__).resolve().parents[1] / "results/phase-iii-k/jspace-v1/material"


def test_frozen_roles_keep_underlying_questions_and_wordings_disjoint():
    manifest = read_manifest(MATERIAL)
    blocked = _blocked_identities()
    for split in PUBLIC_SPLITS:
        rows = load_split(MATERIAL, split, manifest)
        identities = {_identity(row["family"], row["operands"]) for row in rows}
        assert len(rows) == 2 * len(identities)
        assert identities.isdisjoint(blocked)
        assert all({row["wording"] for row in rows if row["item"] == item} == {0, 1}
                   for item in {row["item"] for row in rows})
        blocked.update(identities)
    assert not (MATERIAL / "hidden.json").exists()


def test_fit_refuses_preservation_validation_receipt_even_if_renamed(tmp_path):
    receipt = tmp_path / "adaptation.json"
    receipt.write_text(json.dumps({"split": "preservation-validation", "measurement": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind the frozen partition"):
        _read_fit(tmp_path, MATERIAL, read_manifest(MATERIAL), json.loads(PROTOCOL.read_bytes()))


def _record(margin, interval=True):
    safe = {"lower": .005, "upper": .0125} if interval else None
    return {"active_measurements": [{"measured_margin": margin}], "floor": .0625,
            "geometry": {"local_feasibility": {"per_target": [{"safe_interval": safe}]}}}


def test_interior_ranking_can_disagree_with_max_gain_and_rejects_zero_margin():
    rows = {"C1": _record(.25), "C2": _record(.125), "C3": _record(0), "C4": _record(.50, False)}
    gains = {"C1": 1, "C2": 2, "C3": 4, "C4": 3}
    def observed(name):
        return {"realized_update_norm": .01, "endpoint": {"guard": {"passed": True, "target_item_gain": gains[name]}}}
    result = _selection_keys(rows, observed)
    assert result["max_gain"]["arm"] == "C3"
    assert result["interior"]["arm"] == "C1"


def test_archive_refuses_hidden_artifacts_even_if_listed(tmp_path, monkeypatch):
    source = tmp_path / "trial"
    source.mkdir()
    monkeypatch.setattr(archive_trial, "source_hashes", lambda: {})
    (source / "report.json").write_text(json.dumps({
        "schema": "weightbridge.phase_iiik.jspace_report.v1", "sources": {},
        "artifacts": {"verification/hidden-base.json": "digest"},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="public evidence allowlist"):
        archive_trial.archive_trial(source, tmp_path / "archive")
