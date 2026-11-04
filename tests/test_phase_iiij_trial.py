import json
from pathlib import Path

import pytest

from experiments.phase_iiig.roster_decimal_targeted import _identity
from experiments.phase_iiij.decimal_roster import PROTOCOL, _blocked_identities, load_split, read_manifest
from experiments.phase_iiij.run_preservation import _preservation_group, _rank_measurement, _read_practice


MATERIAL = Path(__file__).resolve().parents[1] / "results/phase-iii-j/preservation-v1/material"


def test_public_populations_exclude_all_disclosed_question_identities():
    manifest = read_manifest(MATERIAL)
    blocked = _blocked_identities()
    for split in ("practice", "selection"):
        rows = load_split(MATERIAL, split, manifest)
        identities = {_identity(row["family"], row["operands"]) for row in rows}
        assert len(rows) == 48 and len(identities) == 24
        assert identities.isdisjoint(blocked)
        blocked.update(identities)
    assert not (MATERIAL / "hidden.json").exists()


def test_fitter_refuses_selection_receipt(tmp_path):
    receipt = tmp_path / "selection.json"
    receipt.write_text(json.dumps({"split": "selection", "measurement": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen practice roster"):
        _read_practice(receipt, MATERIAL, read_manifest(MATERIAL), json.loads(PROTOCOL.read_bytes()))


def test_mixed_sign_is_distinct_and_qualified_result_wins_dose_formulation():
    assert _preservation_group({"group": "positive", "tier": "mixed-sign"}) == "mixed-sign"
    assert _preservation_group({"group": "target", "tier": "mixed-sign"}) == "mixed-sign"
    assert _preservation_group({"group": "positive", "tier": "positive-far"}) == "positive"
    assert _preservation_group({"group": "preserve", "tier": None}) == "copy"
    more_but_regresses = {"passed": False, "prohibited_regressions": ["protected"],
                         "corrected_repeatable_negative_items": ["a", "b"], "target_item_gain": 4}
    qualifies = {"passed": True, "prohibited_regressions": [],
                 "corrected_repeatable_negative_items": ["a"], "target_item_gain": 1}
    assert _rank_measurement(qualifies) > _rank_measurement(more_but_regresses)
