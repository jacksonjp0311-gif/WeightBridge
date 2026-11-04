from collections import defaultdict
from decimal import Decimal
import json

import pytest

from experiments.phase_iiip import semantic_roster as roster


def test_fresh_family_partitions_exclude_O_and_keep_all_eight_exact_outcomes(tmp_path):
    material, hidden = tmp_path / "private" / "material", tmp_path / "private-hidden.json"
    manifest = roster.freeze(material, hidden, hidden_seed="p-fixture-hidden")
    old_used, old_anchors = roster._blocked()
    anchors = set()
    for split in roster.SPLITS:
        rows = roster.load_split(material, split, manifest, hidden_path=hidden if split == "hidden" else None)
        families = defaultdict(list)
        for row in rows:
            assert roster.generator.renderer._identity(row["family"], row["operands"]) not in old_used
            if row["group"] == "target":
                families[row["pair_id"]].append(row)
                assert row["answer"] == max(row["operands"], key=Decimal)
        for members in families.values():
            assert len(members) == 8
            pair = [next(row for row in members if row["counterfactual"] == side and row["order"] == row["wording"] == 0)
                    for side in (0, 1)]
            assert {row["truth_precision"] for row in pair} == {"shorter", "longer"}
            assert pair[0]["answer"] != pair[1]["answer"]
            anchor = Decimal(pair[0]["operands"][0])
            assert anchor not in old_anchors and anchor not in anchors
            anchors.add(anchor)
    assert roster.read_manifest(material) == manifest
    with pytest.raises(FileExistsError):
        roster.freeze(material, hidden)


def test_hidden_requires_explicit_host_path_and_exact_identity(tmp_path):
    material, hidden = tmp_path / "material", tmp_path / "hidden.json"
    manifest = roster.freeze(material, hidden, hidden_seed="p-sealing-fixture")
    with pytest.raises(ValueError, match="explicit host"):
        roster.load_split(material, "hidden", manifest)
    payload = json.loads(hidden.read_bytes())
    payload["rows"][0]["answer"] = "0"
    hidden.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="partition differs"):
        roster.load_split(material, "hidden", manifest, hidden_path=hidden)


def test_initial_freeze_refuses_public_material_and_nested_hidden(tmp_path):
    with pytest.raises(ValueError, match="private host"):
        roster.freeze(roster.ROOT / "public-do-not-create", tmp_path / "hidden.json")
    with pytest.raises(ValueError, match="private host"):
        roster.freeze(tmp_path / "material", tmp_path / "material" / "hidden.json")


def test_protocol_has_one_changed_geometry_and_bounded_separate_track_A():
    config = json.loads(roster.PROTOCOL.read_bytes())
    assert config["track"] == "A" and len(config["roster_counts"]) == 7
    assert config["intent"]["limits"]["max_rank"] == 1
    assert config["intent"]["limits"]["max_update_entries"] == 16384
    assert config["intent"]["limits"]["max_update_norm"] == .025
    assert config["probe_design"]["fixed_layer"] == 23
    assert config["probe_design"]["target_margin"] == .0625
    assert config["endpoint"]["max_candidate_evaluations"] == 4
    assert set(config["probe_design"]["candidate_slots"]) == {"A1", "A2", "C1", "C2"}
    assert config["probe_design"]["outer_search"]["maximum_starts"] == 4
    assert config["probe_design"]["outer_search"]["maximum_iterations"] == 8
    assert config["probe_design"]["outer_search"]["maximum_line_search_steps"] == 4
    assert any("phase-iii-o" in path for path in roster.DISCLOSED)
    assert not any("track_b" in path or "reference_edit" in path for path in roster._sources())


def test_manifest_source_or_protocol_change_refuses_replay(tmp_path, monkeypatch):
    material, hidden = tmp_path / "material", tmp_path / "hidden.json"
    roster.freeze(material, hidden, hidden_seed="p-manifest-fixture")
    original = roster._sources()
    monkeypatch.setattr(roster, "_sources", lambda: {**original, "unexpected.py": "0" * 64})
    with pytest.raises(ValueError, match="frozen source or protocol"):
        roster.read_manifest(material)
