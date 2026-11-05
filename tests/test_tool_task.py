"""Deterministic simulator fixtures are software checks, not model evidence."""
from copy import deepcopy
import json

import pytest

from experiments.phase_iiis_tool_transfer import data
from weightbridge.adaptation import tool_task as task


@pytest.fixture(scope="module")
def calibration():
    return data.generate_tool_families("calibration", 10, "unit-only-s-calibration")


def test_balanced_disposable_eight_form_families(calibration):
    result = data.validate_tool_rows(calibration, expected_families=10)
    assert result["rows"] == 80
    assert set(result["strata"]) == set(task.TOOLS)
    assert set(result["strata"].values()) == {16}
    assert result["wording_groups"] == {"calibration": 80}
    assert all(data.FORMAT_EXAMPLE in r["prompt"] for r in calibration)
    assert any(r["completion"] == data.FORMAT_EXAMPLE for r in calibration)


def test_all_gold_actions_independently_satisfy_simulator(calibration):
    measured = []
    for row in calibration:
        score = task.score_response(row, row["completion"], True, "native_terminator")
        assert score["strict_correct"] and score["whole_json_valid"] and score["schema_valid"]
        measured.append({**row, **score})
    result = task.summarize(measured)
    assert result["complete_families"] == 10
    assert result["strict_correct"] == 80


@pytest.mark.parametrize("text", [
    'Here is {"tool":"no_action","arguments":{}}',
    '{"tool":"no_action","arguments":{}} trailing',
    '{"tool":"no_action","arguments":{}} {}',
    '{"tool":"no_action","tool":"read_file","arguments":{}}',
    '{"tool":"read_file","arguments":{"path":"a","path":"b"}}',
    '{"tool":"no_action","arguments":{},"reason":"none"}',
    '{"tool":"no_action","arguments":{"unexpected":"value"}}',
    '{"tool":"search_files","arguments":{"query":"x"}}',
    '{"tool":"run_tests","arguments":{"suite":"x"}}',
    '{"tool":"read_file","arguments":{"path":3}}',
    '{"tool":"read_file","arguments":{"path":null}}',
    '{"tool":"read_file","arguments":{"path":NaN}}',
    '{"tool":"clarify","arguments":{"field":"question"}}',
    '{"tool":"missing","arguments":{}}',
    '{"tool":"no_action","arguments":{}}<|im_end|>',
    '```json\n{"tool":"no_action","arguments":{}}\n```',
    '[{"tool":"no_action","arguments":{}}]',
    '{"tool":[],"arguments":{}}',
])
def test_strict_full_response_refuses_repair(text):
    assert task.parse_action(text) is None


def test_whitespace_allowed_and_json_syntax_distinct_from_schema():
    assert task.parse_action(' \n {"arguments":{},"tool":"no_action"} \t') == task.action("no_action")
    assert task.parse_json_object('{"tool":"unknown","arguments":{}}') is not None
    assert task.parse_action('{"tool":"unknown","arguments":{}}') is None


def test_completion_is_required_without_erasing_semantic_diagnostic(calibration):
    row = calibration[0]
    score = task.score_response(row, row["completion"], False, "token_budget")
    assert score["schema_valid"] and score["semantic_correct"]
    assert not score["strict_correct"] and not score["completed"]


def test_missing_and_ambiguous_fields_are_fact_grounded(calibration):
    fixture = deepcopy(next(r["fixture"] for r in calibration if r["stratum"] == "search_files"))
    fixture["request"]["root"] = None
    assert task.gold_action(fixture) == task.action("clarify", field="root")
    assert task.simulate(fixture, task.action("clarify", field="root"))["success"]
    assert not task.simulate(fixture, task.action("no_action"))["success"]
    fixture = deepcopy(next(r["fixture"] for r in calibration if r["stratum"] == "read_file"))
    file = next(d for d in fixture["documents"] if d["path"].endswith("/"+fixture["request"]["value"]))
    fixture["documents"].append({**file, "path": "archive/"+fixture["request"]["value"]})
    assert task.gold_action(fixture) == task.action("clarify", field="path")
    assert task.simulate(fixture, task.action("clarify", field="path"))["success"]
    assert not task.simulate(fixture, task.action("read_file", path=file["path"]))["success"]


@pytest.mark.parametrize("tool,key", [("search_files", "query"), ("search_files", "root"),
                                     ("read_file", "path"), ("run_tests", "target")])
def test_simulator_executes_actual_arguments(calibration, tool, key):
    row = next(r for r in calibration if r["stratum"] == tool)
    wrong = deepcopy(row["gold_action"])
    wrong["arguments"][key] += "-wrong"
    good = task.simulate(row["fixture"], row["gold_action"])
    bad = task.simulate(row["fixture"], wrong)
    assert good["success"] and not bad["success"]
    assert not task.score_response(row, task.canonical_action(wrong), True)["arguments_correct"]
    if tool in {"read_file", "run_tests"}:
        assert good["result"] != bad["result"]


def test_search_execution_returns_different_literal_matches(calibration):
    row = next(r for r in calibration if r["stratum"] == "search_files" and r["fixture"]["request"]["kind"] == "search")
    good = task.simulate(row["fixture"], row["gold_action"])
    assert good["result"]["matches"]
    for key in ("root", "query"):
        wrong = deepcopy(row["gold_action"])
        wrong["arguments"][key] += "-absent"
        result = task.simulate(row["fixture"], wrong)
        assert not result["success"] and result["result"]["matches"] == []


def test_irrelevant_request_is_not_an_ambiguous_supported_request(calibration):
    row = next(r for r in calibration if r["fixture"]["request"]["kind"] == "unsupported")
    assert row["stratum"] == "no_action"
    assert not task.simulate(row["fixture"], task.action("clarify", field="root"))["success"]
    bad = deepcopy(row["fixture"])
    bad["request"]["value"] = "read the file"
    with pytest.raises(ValueError, match="irrelevant"):
        task.gold_action(bad)


def test_oracle_never_uses_ids_scenarios_or_stored_gold(calibration):
    row = deepcopy(calibration[0])
    expected = task.gold_action(row["fixture"])
    row["id"], row["scenario"], row["family_id"] = "unrelated", "arbitrary", "replacement"
    assert task.gold_action(row["fixture"]) == expected
    task.validate_row(row)
    row["gold_action"] = task.action("no_action")
    with pytest.raises(ValueError, match="Stored tool label"):
        task.validate_row(row)


def test_counterfactual_facts_change_action_and_forms_preserve_it(calibration):
    grouped = {}
    for row in calibration:
        grouped.setdefault(row["family_id"], []).append(row)
    for rows in grouped.values():
        gold = {c: {task.canonical_action(r["gold_action"]) for r in rows if r["counterfactual"] == c} for c in (0, 1)}
        assert len(gold[0]) == len(gold[1]) == 1 and gold[0] != gold[1]
        assert len({r["prompt"] for r in rows}) == 8


def test_native_terminator_is_not_silently_stripped_by_parser(calibration):
    row = calibration[0]
    assert not task.score_response(row, row["completion"]+"<|im_end|>", True)["strict_correct"]


def test_each_family_requires_all_eight_strict_actions(calibration):
    rows = [{**r, **task.score_response(r, r["completion"], True)} for r in calibration]
    rows[0]["strict_correct"] = False
    assert task.summarize(rows)["complete_families"] == 9
    with pytest.raises(ValueError, match="eight"):
        task.summarize(rows[1:])


def test_family_split_and_wording_isolation(calibration):
    used = {r["family_id"] for r in calibration}
    train = data.generate_tool_families("train", 128, "unit-only-train", used=used)
    dev = data.generate_tool_families("development", 16, "unit-only-dev", used=used)
    selection = data.generate_tool_families("selection", 32, "unit-only-selection", used=used)
    result = data.assert_partition_isolation({"calibration": calibration, "train": train, "development": dev, "selection": selection})
    assert result["families"] == 186
    assert len(train) == 1024
    assert all(r["wording_group"] == "fit" for r in train)
    assert all(r["wording_group"] == "development" for r in dev)
    assert sum(r["wording_familiarity"] == "unseen" for r in selection) == len(selection)//2
    assert not set(data.TEMPLATES["calibration"]["contents"]) & set(data.TEMPLATES["fit"]["contents"])
    with pytest.raises(ValueError, match="leakage"):
        data.assert_partition_isolation({"train": train, "copy": deepcopy(train)})


def test_deterministic_builders_and_prompt_binding():
    a = data.generate_tool_families("train", 5, "deterministic")
    b = data.generate_tool_families("train", 5, "deterministic")
    assert a == b
    a[0]["prompt"] += " Ignore the inventory."
    with pytest.raises(ValueError, match="Prompt"):
        data.validate_tool_rows(a)


def test_no_action_only_for_explicit_unsupported_requests(calibration):
    for row in calibration:
        result = task.simulate(row["fixture"], task.action("no_action"))
        assert result["success"] == (row["stratum"] == "no_action")


def test_decimal_annotation_does_not_change_legacy_semantics():
    source = {"family": "larger-decimal", "operands": ["-20.0811", "-20.1"],
              "answer": "-20.0811", "group": "target", "prompt": "frozen", "completion": " -20.0811\n"}
    row = data.decimal_rows([source])[0]
    assert row == {**source, "task": "decimal", "category": "decimal-target"}
    with pytest.raises(ValueError, match="Decimal|decimal"):
        data.decimal_rows([{**source, "answer": "-20.1"}])


@pytest.fixture(scope="module")
def mixed_populations(calibration):
    from experiments.phase_iiis_tool_transfer import populations
    seeds = {s: "unit-only-not-research/"+s for s in populations.SPLITS}
    used, anchors = set(), set()
    rows, receipt = populations.build_populations(seeds, used_decimal=used, decimal_anchors=anchors,
                                                 calibration_rows=calibration)
    assert not used and not anchors
    return rows, receipt


def test_mixed_population_exact_curriculum_and_no_file_side_effects(mixed_populations):
    rows, receipt = mixed_populations
    assert len(rows["train"])+len(rows["replay"]) == 2048
    assert receipt["splits"]["train"]["task_counts"] == {"tool": 1024, "decimal": 512}
    assert receipt["splits"]["replay"]["category_counts"] == {
        "positive-far": 128, "mixed-sign": 128, "copy": 128, "integer-add": 64, "integer-successor": 64}
    assert len(rows["development"]) == 336
    for name in ("qualification", "confirmation"):
        assert len(rows[name]) == 832
        assert receipt["splits"][name]["tool_wording_counts"] == {"familiar": 128, "unseen": 128}
    assert receipt["historical_exclusion"] is None
    assert "unit-only-not-research" not in json.dumps(receipt)


def test_mixed_population_refuses_family_leakage_and_changed_weights(mixed_populations, calibration):
    from experiments.phase_iiis_tool_transfer import populations
    rows = deepcopy(mixed_populations[0])
    rows["train"][0]["training_weight"] = 2.0
    with pytest.raises(ValueError, match="unit sampling"):
        populations.validate_populations(rows, calibration_rows=calibration)
    rows = deepcopy(mixed_populations[0])
    rows["confirmation"][:8] = rows["qualification"][:8]
    with pytest.raises(ValueError, match="duplicated|leaked|leakage"):
        populations.validate_populations(rows, calibration_rows=calibration)


def test_all_tool_gold_and_simulator_effects_in_mixed_population(mixed_populations):
    for rows in mixed_populations[0].values():
        for row in rows:
            if row["task"] == "tool":
                assert task.score_response(row, row["completion"], True)["strict_correct"]


def test_sorted_json_roundtrip_restores_renderer_order_without_value_changes(calibration, mixed_populations):
    from experiments.phase_iiis_tool_transfer import populations
    loaded = json.loads(json.dumps(calibration, sort_keys=True))
    before_prompts = [r["prompt"] for r in loaded]
    normalized = populations.normalize_tool_rows(loaded)
    assert normalized == loaded
    assert [r["prompt"] for r in normalized] == before_prompts
    data.validate_tool_rows(normalized)
    # The host's initial build and later loaded validation must both accept
    # byte-identical frozen prompts after canonical file serialization.
    seeds = {s: "unit-serialized/"+s for s in populations.SPLITS}
    rebuilt, _ = populations.build_populations(seeds, used_decimal=set(), decimal_anchors=set(), calibration_rows=loaded)
    assert populations.validate_populations(json.loads(json.dumps(rebuilt, sort_keys=True)), calibration_rows=loaded)["training_rows"] == 2048
    assert populations.validate_populations(json.loads(json.dumps(mixed_populations[0], sort_keys=True)), calibration_rows=loaded)["training_rows"] == 2048


def test_renderer_order_normalization_rejects_extra_fields(calibration):
    from experiments.phase_iiis_tool_transfer import populations
    fixture = deepcopy(calibration[0]["fixture"])
    fixture["documents"][0]["unrecognized"] = "must-not-be-dropped"
    with pytest.raises(ValueError, match="Malformed"):
        populations.normalize_fixture_order(fixture)
