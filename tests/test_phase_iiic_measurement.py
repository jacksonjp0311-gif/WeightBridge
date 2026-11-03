import hashlib
import json

import pytest

from experiments.phase_iiic.measurement_roster import FAMILIES, SCHEMA, _question
from experiments.phase_iiic.run_measurement import _bootstrap, _headroom, _load_split, _metrics, parse_response


def _item_rows(split, family, index, *, likelihood_correct, generated):
    rows = []
    for paraphrase in (0, 1):
        for order in ([0, 1], [1, 0]):
            correct = order.index(0)
            likelihood = correct if likelihood_correct else 1 - correct
            generation = correct if generated is True else 1 - correct if generated is False else None
            rows.append({"item_id": f"{split}/{family}/{index}", "family": family, "order": order,
                         "paraphrase": paraphrase, "correct_position": correct, "labels": ["A", "B"],
                         "conditions": {"content": {"generation": {
                             "eligible_position": generation, "parsed_position": generation,
                             "stop_reason": "eos" if generation is not None else "token_budget",
                         }, "likelihood": {"mean": {"position": likelihood}}}}})
    return rows


def test_response_parser_uses_complete_exact_answer_and_refuses_ambiguity():
    answers = ["15 July 2240", "1 August 2240"]
    aliases = [["2240-07-15"], ["2240-08-01"]]
    assert parse_response("  july 15, 2240 ", answers, ["B", "A"], [["July 15, 2240"], []], "content") == 0
    assert parse_response("2240-08-01", answers, ["B", "A"], aliases, "content") == 1
    assert parse_response(" B\n", answers, ["B", "A"], aliases, "letter") == 0
    assert parse_response("The answer is 2240-07-15", answers, ["B", "A"], aliases, "content") is None
    assert parse_response("shared", answers, ["B", "A"], [["shared"], ["shared"]], "content") is None


def test_free_generation_order_uses_answer_identity():
    rows = _item_rows("practice", "word-order", 1, likelihood_correct=True, generated=True)
    metrics = _metrics(rows, "content", "generation")
    assert (metrics["both_order_correct"], metrics["stable_order_pairs"], metrics["all_four_correct"]) == (2, 2, 1)
    for row in rows:
        row["conditions"]["content"]["generation"]["eligible_position"] = 0
    metrics = _metrics(rows, "content", "generation")
    assert (metrics["both_order_correct"], metrics["stable_order_pairs"], metrics["all_four_correct"]) == (0, 0, 0)


def test_headroom_refuses_incomplete_generation_and_requires_cross_family_controls():
    data = {}
    for split in ("practice", "selection"):
        data[split] = _item_rows(split, "fraction-order", 1, likelihood_correct=False, generated=False)
        data[split] += _item_rows(split, "fraction-order", 2, likelihood_correct=False, generated=None)
        data[split] += _item_rows(split, "word-order", 1, likelihood_correct=True, generated=True)
    thresholds = {"minimum_persistent_errors_per_split": 1, "minimum_cross_family_controls_per_split": 1}
    result = _headroom(data, "mean", True, thresholds)
    assert result["eligible_families"] == ["fraction-order"]
    assert result["by_family"]["fraction-order"]["practice"]["persistent_error_ids"] == ["practice/fraction-order/1"]
    assert _headroom(data, "mean", False, thresholds)["eligible_families"] == []


def test_paired_interval_resamples_underlying_question_clusters():
    rows = []
    for family in FAMILIES:
        rows += _item_rows("practice", family, 1, likelihood_correct=False, generated=False)
        rows += _item_rows("practice", family, 2, likelihood_correct=True, generated=True)
    predict = lambda row: row["conditions"]["content"]["likelihood"]["mean"]["position"]
    config = {"bootstrap": {"replicates": 4000, "seed": 2026110317, "interval": 0.95}}
    interval = _bootstrap(rows, predict, lambda row: 1 - row["correct_position"], config)
    assert interval == pytest.approx([1 / 6, 5 / 6])


def test_roster_fingerprint_refuses_edited_split(tmp_path):
    row = _question("seed", "practice", "date-order", 0, set())
    assert row == _question("seed", "practice", "date-order", 0, set())
    path = tmp_path / "practice.json"
    path.write_text(json.dumps({"schema": SCHEMA, "split": "practice", "items": [row]}), encoding="utf-8")
    manifest = {"artifacts": {"practice": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                           "underlying_items": 1}}}
    assert _load_split(tmp_path, manifest, "practice") == [row]
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        _load_split(tmp_path, manifest, "practice")
