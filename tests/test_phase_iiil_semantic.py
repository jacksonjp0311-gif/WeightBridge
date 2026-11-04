from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace

import pytest

from experiments.phase_iiil import semantic_diagnostics as diagnostics, semantic_roster as roster
from weightbridge.adaptation.behavior_endpoint import canonical_decimal


@pytest.fixture
def frozen(tmp_path):
    material, hidden = tmp_path / "material", tmp_path / "host-hidden.json"
    manifest = roster.freeze(material, hidden, hidden_seed="test-only-phase-iiil-hidden")
    assert roster.read_manifest(material) == manifest
    return material, hidden, manifest


def _observed(rows, *, wrong=frozenset(), invalid=frozenset()):
    observed = []
    for row in rows:
        valid = row["id"] not in invalid
        correct = valid and row["id"] not in wrong
        if correct:
            answer = row["answer"]
        elif valid:
            answer = next(value for value in row["operands"] if value != row["answer"])
        else:
            answer = None
        observed.append({**row, "exact": canonical_decimal(answer) if answer is not None else None,
                         "status": "correct" if correct else "incorrect" if valid else "incomplete",
                         "valid": valid, "completed": valid, "correct": correct})
    return {"endpoint": {"id": "test"}, "rows": observed}


def test_freeze_keeps_each_counterfactual_family_and_both_orders_together(frozen):
    material, hidden, manifest = frozen
    all_ids = roster._blocked_identities()
    all_anchors = set()
    for split in roster.SPLITS:
        rows = roster.load_split(material, split, manifest, hidden_path=hidden if split == "hidden" else None)
        target = [row for row in rows if row["group"] == "target"]
        pairs = {row["pair_id"] for row in target}
        assert len(pairs) == manifest["splits"][split]["pairs"]
        assert len(target) == 8 * len(pairs)
        assert len({row["id"] for row in rows}) == len(rows)
        for pair_id in pairs:
            forms = [row for row in target if row["pair_id"] == pair_id]
            base, flip = (next(row for row in forms if (row["counterfactual"], row["order"], row["wording"]) == (variant, 0, 0))
                          for variant in (0, 1))
            assert base["operands"][0] == flip["operands"][0]
            assert Decimal(base["operands"][0]) not in all_anchors
            all_anchors.add(Decimal(base["operands"][0]))
            assert {base["truth_precision"], flip["truth_precision"]} == {"shorter", "longer"}
            for representative in (base, flip):
                identity = roster.renderer._identity("larger-decimal", representative["operands"])
                assert identity not in all_ids
                all_ids.add(identity)
            for row in forms:
                original = base if row["counterfactual"] == 0 else flip
                assert row["operands"] == (original["operands"][::-1] if row["order"] else original["operands"])
                assert row["answer"] == max(row["operands"], key=Decimal)
        assert {(row["tier"], row["truth_precision"]) for row in target} == {
            tuple(group.split("/")) for group in roster.GROUPS
        }
    assert not (material / "hidden.json").exists()
    with pytest.raises(ValueError, match="explicit host path"):
        roster.load_split(material, "hidden", manifest)


def test_roster_rejects_mutation_and_mismatched_oracle(frozen):
    material, _, manifest = frozen
    rows = roster.load_split(material, "adaptation", manifest)
    changed = deepcopy(rows)
    changed[0]["answer"] = next(value for value in rows[0]["operands"] if value != rows[0]["answer"])
    with pytest.raises(ValueError, match="Decimal oracle"):
        roster._validate(changed, "adaptation", roster._read(roster.PROTOCOL)["roster_counts"]["adaptation"])
    path = material / "adaptation.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="roster hash differs"):
        roster.load_split(material, "adaptation", manifest)


def test_semantic_gate_requires_all_four_and_any_lost_presentation_fails(frozen):
    material, _, manifest = frozen
    rows = roster.load_split(material, "adaptation", manifest)
    focus = rows[0]["comparison_id"]
    first_order = {row["id"] for row in rows if row["comparison_id"] == focus and row["order"] == 0}
    second_order = {row["id"] for row in rows if row["comparison_id"] == focus and row["order"] == 1}
    parent = _observed(rows, wrong=first_order | second_order)
    summary = diagnostics.summarize(parent)
    assert summary["repeatable_errors"] == summary["all_four_repeatable_errors"] == [focus]
    assert len(summary["eligible_fit_items"][focus]) == 2
    assert summary["truth_flip_matrix"]["only_counterfactual_correct"] == 1
    partial = diagnostics.compare_semantic(parent, _observed(rows, wrong=second_order))
    assert not partial["passed"] and partial["target_gain"] == 0
    loss = next(row["id"] for row in rows if row["group"] == "positive")
    regressed = diagnostics.compare_semantic(parent, _observed(rows, invalid={loss}))
    assert not regressed["passed"] and regressed["target_gain"] == 1
    assert regressed["regressions"] == regressed["lost_valid"] == [loss]
    passed = diagnostics.compare_semantic(parent, _observed(rows), min_repeatable_corrections=1)
    assert passed["passed"] and passed["corrected_repeatable_target_comparisons"] == [focus]
    assert passed["family_gain_interval"]["clusters"] == summary["pair_families"]


def test_teacher_preference_and_position_are_measured_without_qualifying_incomplete_error(frozen):
    material, _, manifest = frozen
    rows = roster.load_split(material, "adaptation", manifest)
    focus = rows[0]["comparison_id"]
    one_order = {row["id"] for row in rows if row["comparison_id"] == focus and row["order"] == 0}
    teacher = {row["id"]: canonical_decimal(row["answer"]) for row in rows if row["comparison_id"] == focus}
    summary = diagnostics.summarize(_observed(rows, wrong=one_order), teacher_preferences=teacher)
    assert summary["repeatable_errors"] == [focus]
    assert summary["all_four_repeatable_errors"] == []
    assert sum(cell["teacher_greedy_disagreements"] for cell in summary["subgroups"].values()) == 2
    incomplete = diagnostics.summarize(_observed(rows, wrong=one_order, invalid={min(one_order)}))
    assert incomplete["repeatable_errors"] == [] and incomplete["status_counts"]["incomplete"] == 1


def test_fork_diagnostic_detects_distinguishing_digit_inside_native_token():
    class Tokenizer:
        def decode(self, tokens, *, skip_special_tokens=False):
            pieces = {1: " -", 2: "138", 3: "142", 4: "\n"}
            return "".join(pieces[token] for token in tokens)

        def encode(self, text, *, add_special_tokens=False):
            return [2 if "138" in text else 3]

    fork = SimpleNamespace(accepted_token_ids=(2,), actual_token_id=3, shared_token_ids=(1,),
                           decoding={"eos_token_id": 0}, status="semantic", replay_status="matched")
    result = diagnostics.describe_fork_tokens(fork, Tokenizer(), {"answer": "-138.1", "operands": ["-138.1", "-142.1"]})
    assert result["at_first_distinguishing_digit"] and not result["starts_at_text_distinction"]
    assert result["accepted_tokens"][0]["kind"] == result["actual_kind"] == "digit"
