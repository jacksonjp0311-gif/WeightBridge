from collections import defaultdict
from decimal import Decimal
from pathlib import Path
import json

import pytest

from experiments.phase_iiin import semantic_roster as roster


def test_fresh_family_partitions_and_opposing_exact_outcomes(tmp_path):
    material, hidden = tmp_path / "private" / "material", tmp_path / "private-hidden.json"
    manifest = roster.freeze(material, hidden, hidden_seed="n-fixture-hidden")
    anchors = set()
    for split in roster.SPLITS:
        rows = roster.load_split(material, split, manifest, hidden_path=hidden if split == "hidden" else None)
        families = defaultdict(list)
        for row in rows:
            if row["group"] == "target":
                families[row["pair_id"]].append(row)
                assert row["answer"] == max(row["operands"], key=Decimal)
        for members in families.values():
            assert len(members) == 8
            pair = [next(r for r in members if r["counterfactual"] == side and r["order"] == r["wording"] == 0)
                    for side in (0, 1)]
            assert {r["truth_precision"] for r in pair} == {"shorter", "longer"}
            assert pair[0]["answer"] != pair[1]["answer"]
            anchor = Decimal(pair[0]["operands"][0])
            assert anchor not in anchors
            anchors.add(anchor)
    with pytest.raises(FileExistsError):
        roster.freeze(material, hidden)


def test_hidden_requires_distinct_explicit_host_path_and_hash(tmp_path):
    material, hidden = tmp_path / "material", tmp_path / "hidden.json"
    manifest = roster.freeze(material, hidden, hidden_seed="n-sealing-fixture")
    with pytest.raises(ValueError, match="explicit host"):
        roster.load_split(material, "hidden", manifest)
    payload = json.loads(hidden.read_bytes())
    payload["rows"][0]["answer"] = "0"
    hidden.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="partition differs"):
        roster.load_split(material, "hidden", manifest, hidden_path=hidden)


def test_initial_freeze_refuses_public_checkout_paths(tmp_path):
    with pytest.raises(ValueError, match="private host"):
        roster.freeze(roster.ROOT / "public-do-not-create", tmp_path / "hidden.json")
