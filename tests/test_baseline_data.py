"""Exact supervised-data tests; generated fixtures are not model measurements."""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from experiments.phase_iiiq_baseline import data


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    root = tmp_path_factory.mktemp("baseline-data")
    protocol = root / "protocol.json"
    protocol.write_text(json.dumps({"data": {"seed_prefix": "test-q-baseline-data-v1",
        "populations": {name: asdict(spec) for name, spec in data.DEFAULT_POPULATIONS.items()}}}), encoding="utf-8")
    material, hidden = root / "private" / "material", root / "sealed" / "hidden.json"
    manifest = data.freeze(material, hidden, protocol=protocol,
        host_seeds={"selection": "test-only-selection-secret", "hidden": "test-only-hidden-secret"})
    populations = {split: data.load_split(material, split, manifest, protocol=protocol,
        host_authorized=True, hidden_path=hidden if split == "hidden" else None) for split in data.SPLITS}
    return root, protocol, material, hidden, manifest, populations


def test_default_supervised_counts_and_equal_weight_without_error_selection(frozen):
    _, protocol, material, _, manifest, populations = frozen
    assert data.read_manifest(material, protocol=protocol) == manifest
    assert len(populations["train"]) == len(populations["replay"]) == 1024
    assert len(populations["smoke"]) == 32
    assert len(populations["preservation-validation"]) == 320
    assert Counter(row["family"] for row in populations["replay"]) == {
        "larger-decimal": 512, "copy": 256, "integer-add": 128, "integer-successor": 128}
    assert all(row["group"] == "target" for row in populations["train"])
    assert not manifest["selection_is_error_based"]
    for rows in populations.values():
        assert len({row["id"] for row in rows}) == len({row["prompt"] for row in rows}) == len(rows)
        assert all(row["training_weight"] == 1. and "correct" not in row and "parent_correct" not in row for row in rows)
        assert all(row["completion"] == " "+data.oracle(row["family"], row["operands"])+"\n" for row in rows)
    assert isinstance(json.loads((material / "train.json").read_bytes()), list)


def test_full_families_are_exact_balanced_and_never_split(frozen):
    _, _, _, _, manifest, populations = frozen
    used, blocked_anchors = data._blocked()
    seen_keys, seen_families, seen_anchors = set(), set(), set()
    for split, rows in populations.items():
        keys = {data.semantic_key(row["family"], row["operands"]) for row in rows}
        assert not keys & used and not keys & seen_keys
        seen_keys.update(keys)
        family_ids = {row["family_id"] for row in rows}
        assert not family_ids & seen_families
        seen_families.update(family_ids)
        target = defaultdict(list)
        for row in rows:
            if row["group"] == "target":
                target[row["pair_id"]].append(row)
                assert Decimal(row["answer"]) == max(map(Decimal, row["operands"]))
                assert all(10 <= abs(int(Decimal(value))) <= 9999 for value in row["operands"])
        assert len(target) == manifest["splits"][split]["population"]["target_families"]
        for forms in target.values():
            assert len(forms) == 8
            assert {(row["counterfactual"], row["order"], row["wording"]) for row in forms} == {
                (c, o, w) for c in (0, 1) for o in (0, 1) for w in (0, 1)}
            originals = [next(row for row in forms if row["counterfactual"] == c and row["order"] == row["wording"] == 0) for c in (0, 1)]
            assert {row["truth_precision"] for row in originals} == {"shorter", "longer"}
            assert originals[0]["answer"] != originals[1]["answer"]
            anchor = Decimal(originals[0]["operands"][0])
            assert anchor not in blocked_anchors and anchor not in seen_anchors
            seen_anchors.add(anchor)


def test_numeric_equivalent_forms_share_the_same_semantic_identity():
    assert data.semantic_key("larger-decimal", ["-1.20", "-2.000"]) == data.semantic_key("larger-decimal", ["-2", "-1.2000"])
    assert data.semantic_key("copy", ["1234.5000"]) == data.semantic_key("copy", ["1234.5"])
    assert data.semantic_key("integer-add", ["17", "2"]) == data.semantic_key("integer-add", ["2", "17"])


def test_fixed_demonstrations_and_disclosed_timing_examples_are_excluded():
    used, anchors = data._blocked()
    for family, examples in {**data.generator.renderer.DEMONSTRATIONS, **data.ARITHMETIC_DEMOS}.items():
        assert all(data.semantic_key(family, operands) in used for operands in examples)
    assert all(data.semantic_key("larger-decimal", operands) in used for operands in data.PREFLIGHT_COMPARISONS)
    assert {Decimal("-1000.1"), Decimal("-801.1"), Decimal("-4.2")} <= anchors


def test_numeric_equivalent_copy_is_not_reused_as_a_new_unique_task(monkeypatch):
    monkeypatch.setattr(data.generator.renderer, "_draw", lambda *args: ("1738.2400",))
    with pytest.raises(ValueError, match="fresh preservation"):
        data.generate("replay", data.PopulationSpec(copy_controls=1), "constant-copy", {data.semantic_key("copy", ["1738.24"])}, set())


def test_train_labels_cover_both_outcomes_and_not_repeated_failed_token_forks(frozen):
    rows = frozen[-1]["train"]
    comparisons = defaultdict(list)
    for row in rows:
        comparisons[row["comparison_id"]].append(row)
    assert len(comparisons) == 256
    assert all(len(forms) == 4 and len({row["completion"] for row in forms}) == 1 for forms in comparisons.values())
    assert all("fork" not in row and "accepted_token_ids" not in row for row in rows)


def test_arithmetic_oracle_and_exact_final_prompt_tail(frozen):
    for row in frozen[-1]["replay"]:
        if row["family"] == "integer-add":
            expected = Decimal(row["operands"][0])+Decimal(row["operands"][1])
            tail = f"Integers: {row['operands'][0]} and {row['operands'][1]}\nAnswer:"
        elif row["family"] == "integer-successor":
            expected = Decimal(row["operands"][0])+1
            tail = f"Integer: {row['operands'][0]}\nAnswer:"
        else:
            continue
        assert Decimal(row["answer"]) == expected
        assert "\n".join(row["prompt"].splitlines()[-2:]) == tail
    with pytest.raises(ValueError, match="integers"):
        data.oracle("integer-add", ["1.1", "2"])


@pytest.mark.parametrize("change", ["answer", "completion", "group", "wording", "order"])
def test_malformed_supervision_and_transformations_are_refused(frozen, change):
    rows = deepcopy(frozen[-1]["smoke"])
    if change == "answer":
        rows[0]["answer"] = "0"
    elif change == "completion":
        rows[0]["completion"] = rows[0]["answer"]
    elif change == "group":
        rows[0]["group"] = "preserve"
    elif change == "wording":
        rows[0]["wording"] = 7
    else:
        rows[0]["order"] = 1-rows[0]["order"]
    with pytest.raises(ValueError):
        data.validate(rows, "smoke", data.DEFAULT_POPULATIONS["smoke"])


def test_host_seeds_are_committed_without_exposing_sealed_seed_values(frozen):
    _, _, _, _, manifest, _ = frozen
    raw = json.dumps(manifest)
    assert "test-only-selection-secret" not in raw and "test-only-hidden-secret" not in raw
    assert manifest["splits"]["selection"]["seed_commitment"] == hashlib.sha256(b"test-only-selection-secret").hexdigest()
    assert manifest["artifacts"]["hidden.json"] == manifest["splits"]["hidden"]["sha256"]


def test_sealed_access_requires_explicit_host_and_distinct_hidden_path(frozen):
    _, protocol, material, hidden, manifest, _ = frozen
    with pytest.raises(ValueError, match="host authorization"):
        data.load_split(material, "selection", manifest, protocol=protocol)
    with pytest.raises(ValueError, match="host authorization"):
        data.load_split(material, "hidden", manifest, protocol=protocol, hidden_path=hidden)
    with pytest.raises(ValueError, match="host authorization"):
        data.load_split(material, "hidden", manifest, protocol=protocol, host_authorized=True)
    with pytest.raises(ValueError, match="outside"):
        data.load_split(material, "hidden", manifest, protocol=protocol, host_authorized=True, hidden_path=material / "hidden.json")


def test_freeze_and_load_are_write_once_and_hash_bound(frozen, tmp_path):
    _, protocol, material, hidden, manifest, _ = frozen
    with pytest.raises(FileExistsError):
        data.freeze(material, hidden, protocol=protocol)
    with pytest.raises(ValueError, match="private host"):
        data.freeze(data.ROOT / "never-create", tmp_path / "hidden.json", protocol=protocol)
    isolated = tmp_path / "material"
    isolated.mkdir()
    damaged = deepcopy(frozen[-1]["smoke"])
    damaged[0]["answer"] = "0"
    (isolated / "smoke.json").write_text(json.dumps(damaged), encoding="utf-8")
    with pytest.raises(ValueError, match="identity differs"):
        data.load_split(isolated, "smoke", manifest, protocol=protocol)


def test_generation_replays_exactly_and_obeys_bounded_counts():
    spec = data.PopulationSpec(target_families=4, comparison_controls=2, copy_controls=2, addition_controls=2, successor_controls=2)
    first = data.generate("smoke", spec, "replay-fixture", set(), set())
    second = data.generate("smoke", spec, "replay-fixture", set(), set())
    assert first == second
    with pytest.raises(ValueError, match="balanced"):
        data.PopulationSpec(target_families=3)
    with pytest.raises(ValueError, match="bounded"):
        data.PopulationSpec(copy_controls=True)
