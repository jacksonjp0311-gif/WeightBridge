import json
from pathlib import Path

import pytest

from experiments.phase_iiii import archive_trial
from experiments.phase_iiii.decimal_roster import PROTOCOL, _blocked_identities, load_split, read_manifest
from experiments.phase_iiii.run_causal_fork import _read_practice
from experiments.phase_iiig.roster_decimal_targeted import _identity


ROOT = Path(__file__).resolve().parents[1]
MATERIAL = ROOT / "results/phase-iii-i/causal-fork-v1/material"


def test_fresh_populations_do_not_reuse_disclosed_items():
    manifest = read_manifest(MATERIAL)
    excluded = _blocked_identities()
    for split in ("practice", "selection"):
        rows = load_split(MATERIAL, split, manifest)
        items = {_identity(row["family"], row["operands"]) for row in rows}
        assert len(items) == 24
        assert items.isdisjoint(excluded)
        excluded.update(items)
    assert not (MATERIAL / "hidden.json").exists()


def test_selection_receipt_cannot_be_used_for_fitting(tmp_path):
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"split": "selection", "measurement": {}}))
    with pytest.raises(ValueError, match="frozen parent, endpoint and practice roster"):
        _read_practice(record, MATERIAL, read_manifest(MATERIAL), json.loads(PROTOCOL.read_bytes()))


def test_archive_refuses_hidden_artifacts_even_if_listed_in_report(tmp_path, monkeypatch):
    source = tmp_path / "trial"
    source.mkdir()
    monkeypatch.setattr(archive_trial, "source_hashes", lambda: {})
    (source / "report.json").write_text(json.dumps({
        "schema": "weightbridge.phase_iiii.causal_fork_report.v1", "sources": {},
        "artifacts": {"verification/hidden-base.json": "digest"},
    }))
    output = tmp_path / "archive"
    with pytest.raises(ValueError, match="public evidence allowlist"):
        archive_trial.archive_trial(source, output)
    assert not output.exists()
