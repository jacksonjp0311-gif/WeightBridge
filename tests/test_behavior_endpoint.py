from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from weightbridge.adaptation.behavior_endpoint import (
    CandidateEvaluationBudget,
    build_decimal_fit_rows,
    compare_endpoint,
    decimal_endpoint,
    measure_endpoint,
    validate_decimal_fit_row,
)


ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "results" / "phase-iii-g" / "negative-decimal-v6"


class _Tokenizer:
    eos_token_id = 999
    chat_template = "must not be used by bare prompt contracts"

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(character) for character in text]


def _endpoint(*, max_candidate_evaluations=2):
    return decimal_endpoint({"max_input_tokens": 350, "max_new_tokens": 12, "batch_size": 4},
                            max_candidate_evaluations=max_candidate_evaluations)


def _example_rows():
    rows = []
    for item, group, operands in (("A", "target", ("-9.40", "-9.450")), ("B", "target", ("-4.2", "-4.3")),
                                  ("C", "positive", ("5.3", "6.2")), ("D", "preserve", ("312.90",))):
        family = "copy" if group == "preserve" else "larger-decimal"
        value_line = (f"Decimal: {operands[0]}" if family == "copy"
                      else f"Values: {operands[0]} or {operands[1]}")
        for wording in (0, 1):
            rows.append({"id": f"{item}{wording}", "item": item, "wording": wording, "group": group,
                         "family": family, "operands": operands, "answer": operands[1 if item == "C" else 0],
                         "prompt": f"Wording {wording}\n{value_line}\nAnswer:"})
    return rows


def _measure(texts, rows=None, *, contract=None, budget=None):
    runner = SimpleNamespace(tokenizer=_Tokenizer())
    rows = _example_rows() if rows is None else rows
    contract = contract or _endpoint()

    def decoder(adapted, prompts, **settings):
        assert runner.tokenizer.chat_template and adapted.tokenizer.chat_template is None
        assert prompts == [row["prompt"] for row in rows]
        assert settings == {"max_input_tokens": 350, "max_new_tokens": 12, "batch_size": 4}
        return ([{"text": text, "stop_reason": reason, "generated_token_ids": [1]} for text, reason in texts],
                {"generate_batches": 2})

    return measure_endpoint(runner, rows, contract, decoder=decoder, candidate_budget=budget)


def _answers(*, correction=False):
    values = ("-9.4" if correction else "-9.450", "-9.4" if correction else "-9.450",
              "-4.2", "-4.2", "6.2", "6.2", "312.9", "312.9")
    return [(value, "eos") for value in values]


def test_generation_distinguishes_completion_grammar_and_out_of_option():
    texts = _answers()
    texts[:4] = [(" -9.400\r\nNext prompt", "token_budget"), ("-9.4", "token_budget"),
                 ("-4.4", "eos"), ("-4.2 or -4.3\n", "token_budget")]
    measurement = _measure(texts)
    observations = measurement["rows"][:4]
    assert [row["status"] for row in observations] == ["correct", "incomplete", "out_of_option", "invalid_grammar"]
    assert [row["grammar_valid"] for row in observations] == [True, False, True, False]
    assert [row["completed"] for row in observations] == [True, False, True, True]
    assert [row["valid"] for row in observations] == [True, False, False, False]


def test_guard_requires_both_wordings_and_protects_every_parent_correct_presentation():
    contract = _endpoint()
    parent = _measure(_answers(), contract=contract)
    repaired = _measure(_answers(correction=True), contract=contract)
    result = compare_endpoint(parent, repaired, contract, min_repeatable_corrections=1)
    assert result["passed"] and result["corrected_repeatable_target_items"] == ["A"]
    assert result["target_item_gain"] == 1 and result["target_presentation_gain"] == 2
    texts = _answers(correction=True)
    texts[3] = ("-4.2", "token_budget")
    texts[7] = ("312.0", "eos")
    result = compare_endpoint(parent, _measure(texts), contract)
    assert not result["passed"] and result["prohibited_regressions"] == ["B1", "D1"]
    assert result["lost_valid_presentations"] == ["B1"]
    texts = _answers()
    texts[0] = ("-9.4", "eos")
    result = compare_endpoint(parent, _measure(texts), contract)
    assert not result["passed"] and result["target_presentation_gain"] == 1 and result["target_item_gain"] == 0


def test_guard_rejects_comparison_of_changed_roster_or_missing_wording():
    rows = _example_rows()
    contract = _endpoint()
    parent = _measure(_answers(), contract=contract)
    rows[0]["prompt"] = rows[0]["prompt"].replace("Wording 0", "Unfrozen wording")
    with pytest.raises(ValueError, match="same frozen roster"):
        compare_endpoint(parent, _measure(_answers(correction=True), rows), contract)
    with pytest.raises(ValueError, match="missing required wordings"):
        _measure(_answers()[1:], rows[1:])


def test_candidate_budget_does_not_count_parent_and_refuses_extra_decode():
    contract = _endpoint(max_candidate_evaluations=1)
    budget = CandidateEvaluationBudget(contract)
    _measure(_answers(), contract=contract)
    _measure(_answers(correction=True), contract=contract, budget=budget)
    assert budget.used == 1
    with pytest.raises(ValueError, match="budget exhausted"):
        _measure(_answers(correction=True), contract=contract, budget=budget)


def test_fit_builder_keeps_real_displayed_rivals_and_separately_tags_copy_decoys():
    contract = _endpoint()
    parent = _measure(_answers(), contract=contract)
    runner = SimpleNamespace(tokenizer=_Tokenizer())
    fit = build_decimal_fit_rows(runner, parent, contract, target_items=["A"], preservation_items=["B", "C", "D"],
                                 max_rows=8)
    by_id = {row.id: row for row in fit}
    assert by_id["A0"].answers == (" -9.40", " -9.450")
    assert by_id["B0"].answers == (" -4.2", " -4.3") and by_id["B0"].kind == "task_choice"
    assert by_id["D0"].answers == (" 312.90", " 313.9") and by_id["D0"].kind == "auxiliary_copy_decoy"
    old = build_decimal_fit_rows(runner, parent, contract, target_items=["A"], preservation_items=["B"],
                                 max_rows=4, mode="legacy_decoy_control")
    assert old[2].kind == "legacy_comparison_decoy" and old[2].answers[1] == " -3.2"
    with pytest.raises(ValueError, match="other displayed operand"):
        validate_decimal_fit_row(parent["rows"][2], old[2].to_solver_row())
    wrong_generation = _answers()
    wrong_generation[0] = ("-9.35", "eos")
    with pytest.raises(ValueError, match="repeatable completed wrong-option"):
        build_decimal_fit_rows(runner, _measure(wrong_generation), contract,
                               target_items=["A"], preservation_items=["B"], max_rows=4)


def test_archived_v6_target_competitor_and_preservation_artificial_rival_are_not_equivalent():
    practice = {row["id"]: row for row in json.loads((V6 / "material" / "practice.json").read_text())["rows"]}
    fit = json.loads((V6 / "qwen" / "fit-rows.json").read_text())["rows"]
    comparison = next(row for row in fit if row["preserve"] and practice[row["id"]]["family"] == "larger-decimal")
    copy = next(row for row in fit if practice[row["id"]]["family"] == "copy")
    assert validate_decimal_fit_row(practice[fit[0]["id"]], fit[0]).kind == "task_choice"
    assert validate_decimal_fit_row(practice[copy["id"]], copy).kind == "auxiliary_copy_decoy"
    with pytest.raises(ValueError, match="other displayed operand"):
        validate_decimal_fit_row(practice[comparison["id"]], comparison)
    assert validate_decimal_fit_row(practice[comparison["id"]], comparison,
                                    mode="legacy_decoy_control").kind == "legacy_comparison_decoy"
    with pytest.raises(ValueError, match="displayed decimal question"):
        validate_decimal_fit_row({**practice[fit[0]["id"]], "operands": ["-1.1", "-1.2"]}, fit[0])
    with pytest.raises(ValueError, match="semantic decimal answer"):
        row = practice[fit[0]["id"]]
        validate_decimal_fit_row({**row, "answer": next(value for value in row["operands"] if value != row["answer"])},
                                 fit[0])
