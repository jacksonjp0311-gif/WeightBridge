import json
import shutil

import pytest

from experiments.phase_iiie.qwen_handoff import decide
from experiments.phase_iiie.qwen_roster import FAMILIES, PROTOCOL, SPLITS
from weightbridge.core import file_hash


@pytest.mark.parametrize("qualified, action", [(False, "measure"), (True, "reachability")])
def test_qwen_handoff_never_recommends_an_edit_from_measurement_alone(tmp_path, qualified, action):
    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf8")

    shutil.copyfile(PROTOCOL, tmp_path / "protocol.json")
    config = json.loads(PROTOCOL.read_bytes())
    gates = dict.fromkeys(SPLITS, qualified)
    headroom = {family: {split: {"targets": [], "controls": []} for split in ("practice", "selection")}
                for family in FAMILIES}
    headroom["date-order"] = {
        split: {"targets": [f"{split}-target-{i}" for i in range(2)], "controls": ["word", "fraction"]}
        for split in ("practice", "selection")
    }
    report = {"schema": "weightbridge.phase_iiie.qwen_report.v1", "selected": "paired",
              "protocol_sha256": file_hash(PROTOCOL), "checkpoint_sha256": config["model_sha256"],
              "splits": {split: {"rankers": {"paired": {"passed": gate}}} for split, gate in gates.items()},
              "admission": {"ranker_gates": gates, "candidate_attempts": 0, "reserve_opened": False,
                            "headroom": headroom, "eligible_families": ["date-order"] if qualified else [],
                            "hold_reason": None if qualified else "MEASUREMENT"}}
    write("report.json", report)
    write("artifact_manifest.json", {"schema": "weightbridge.phase_iiie.qwen_archive.v1",
                                    "artifacts": {"protocol.json": file_hash(PROTOCOL),
                                                  "report.json": file_hash(tmp_path / "report.json")}})
    assert decide(tmp_path)["decision"]["action"] == "hold"
    result = decide(tmp_path, diagnostic_budget=1.)
    assert result["decision"]["action"] == action
    assert all(item.startswith("practice") for items in result["eligible_practice_targets"].values() for item in items)
    write("report.json", {**report, "selected": "raw"})
    with pytest.raises(ValueError, match="hashes differ"):
        decide(tmp_path)
