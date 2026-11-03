from collections import Counter
from fractions import Fraction
import json

import pytest

from experiments.phase_iiid import roster


def test_grounding_roster_counterbalances_and_holds_options_fixed():
    rows = [roster._question("test-seed", "practice", family, index, set())
            for family in roster.FAMILIES for index in range(2)]
    for row in rows:
        assert len(row["presentations"]) == 4
        if row["family"] == "word-order":
            assert row["answers"][0] < row["answers"][1]
        elif row["family"] == "date-order":
            assert row["aliases"][0][-1] < row["aliases"][1][-1]
        else:
            first, second = map(Fraction, row["answers"])
            assert first > second
            for position, expected in enumerate((first, second)):
                assert all(Fraction(alias) == expected for alias in row["aliases"][position])

        orders = Counter(tuple(view["order"]) for view in row["presentations"])
        assert orders == {(0, 1): 2, (1, 0): 2}
        for view in row["presentations"]:
            options = [row["answers"][index] for index in view["order"]]
            for display, prefixes in (("labeled", [f"{label}. " for label in view["labels"]]),
                                      ("bullets", ["- ", "- "])):
                prompts = view["displays"][display]
                expected = [prefix + option for prefix, option in zip(prefixes, options, strict=True)]
                keys = ("task", "counterfactual", "neutral_absent", "neutral_generic", "irrelevant")
                for key in keys:
                    assert prompts[key].splitlines()[-3:] == [*expected, roster.INSTRUCTION]
                assert len({prompts[key] for key in keys}) == len(keys)
                assert prompts["neutral_absent"].splitlines() == [*expected, roster.INSTRUCTION]
        assert row == roster._question("test-seed", "practice", row["family"],
                                      0 if set(row["presentations"][0]["labels"]) == {"A", "B"} else 1, set())


def test_frozen_material_is_disjoint_and_repeatable(tmp_path, monkeypatch):
    config = json.loads(roster.PROTOCOL.read_bytes())

    class Checkpoint:
        def checkpoint_sha256(self):
            return config["model_sha256"]

    monkeypatch.setattr(roster, "load_checkpoint", lambda path: Checkpoint())
    first = tmp_path / "first"
    assert roster.freeze(tmp_path, first) == roster.freeze(tmp_path, tmp_path / "second")
    with pytest.raises(FileExistsError):
        roster.freeze(tmp_path, first)

    manifest = json.loads((first / "manifest.json").read_bytes())
    material = [json.loads((first / f"{split}.json").read_bytes()) for split in roster.SPLITS]
    keys = [row["semantic_key"] for split in material for row in split["items"]]
    assert len(keys) == len(set(keys)) == manifest["underlying_items"] == 24
    for split in material:
        assert Counter(row["family"] for row in split["items"]) == dict.fromkeys(roster.FAMILIES, 2)
        assert roster.file_hash(first / f"{split['split']}.json") == manifest["artifacts"][split["split"]]["sha256"]


def test_roster_refuses_unpinned_model_before_creating_artifacts(tmp_path, monkeypatch):
    class Checkpoint:
        def checkpoint_sha256(self):
            return "unpinned-model"

    monkeypatch.setattr(roster, "load_checkpoint", lambda path: Checkpoint())
    out = tmp_path / "refused"
    with pytest.raises(ValueError, match="pinned Qwen checkpoint"):
        roster.freeze(tmp_path, out)
    assert not out.exists()
