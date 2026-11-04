from decimal import Decimal
import json

import pytest

from experiments.phase_iiim import semantic_roster as study


def test_all_semantic_relatives_stay_in_one_fresh_partition(tmp_path):
    material, hidden = tmp_path / "material", tmp_path / "host" / "hidden.json"
    hidden.parent.mkdir()
    manifest = study.freeze(material, hidden, hidden_seed="phase-m-unit-hidden")
    used_pairs, used_anchors = study._blocked()
    new_anchors = set()
    for split in study.SPLITS:
        rows = study.load_split(material, split, manifest, hidden_path=hidden if split == "hidden" else None)
        families = {}
        for row in rows:
            identity = study.previous.renderer._identity(row["family"], row["operands"])
            assert identity not in used_pairs
            if row["group"] == "target":
                families.setdefault(row["pair_id"], []).append(row)
                assert max(map(Decimal, row["operands"])) == Decimal(row["answer"])
        for family in families.values():
            assert len(family) == 8
            assert len({(row["counterfactual"], row["order"], row["wording"]) for row in family}) == 8
            shorter = min(family[0]["operands"], key=lambda value: len(value.partition(".")[2]))
            assert Decimal(shorter) not in used_anchors | new_anchors
            new_anchors.add(Decimal(shorter))
        # A comparison's repeated renderings belong to the same partition.
        used_pairs.update(study.previous.renderer._identity(row["family"], row["operands"]) for row in rows)
    with pytest.raises(FileExistsError):
        study.freeze(material, tmp_path / "other-hidden.json")


def test_sealed_paths_and_modified_oracle_rows_are_rejected(tmp_path):
    material, hidden = tmp_path / "material", tmp_path / "hidden.json"
    manifest = study.freeze(material, hidden, hidden_seed="phase-m-unit-hidden")
    with pytest.raises(ValueError, match="explicit host path"):
        study.load_split(material, "hidden", manifest)
    with pytest.raises(ValueError, match="explicit host path"):
        study.load_split(material, "selection", manifest, hidden_path=hidden)
    path = material / "adaptation.json"
    data = json.loads(path.read_bytes())
    data["rows"][0]["answer"] = "1.0"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        study.load_split(material, "adaptation", manifest)
