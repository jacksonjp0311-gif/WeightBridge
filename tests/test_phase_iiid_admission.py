import json

import pytest

from experiments.phase_iiid import admission, roster, run_grounding
from experiments.phase_iiid.analysis import analyze_split, select_practice
from weightbridge.core import file_hash, json_dump_atomic


def _score(position, correct):
    value = -0.1 if position == correct else -6.1
    return {"sum_log_probability": value, "mean_log_probability": value,
            "terminated_sum_log_probability": value - 1, "answer_tokens": 1, "answer_token_ids": [position]}


def _rows_for(items, *, ranker_headroom, ranker_fails_counterfactual, strict_correct):
    rows = run_grounding._rows(items)
    for row in rows:
        target = row["correct_position"]
        changed = row["counterfactual_position"]
        wrong = ranker_headroom and row["family"] == "word-order"
        task_prediction = 1 - target if wrong else target
        changed_prediction = 1 - changed if wrong or ranker_fails_counterfactual else changed
        def response(correct, display):
            if strict_correct:
                return {"text": row["answers"][correct], "strict_position": correct, "semantic_position": correct,
                        "stop_reason": "eos", "semantic_kind": "content"}
            if display == "labeled":
                return {"text": row["labels"][correct], "strict_position": None, "semantic_position": correct,
                        "stop_reason": "eos", "semantic_kind": "label"}
            return {"text": "unknown", "strict_position": None, "semantic_position": None,
                    "stop_reason": "eos", "semantic_kind": "unrecognized"}
        for display in run_grounding.DISPLAYS:
            row["displays"][display] = {
                "contexts": {
                    context: [_score(position, prediction) for position in (0, 1)]
                    for context, prediction in (("task", task_prediction), ("counterfactual", changed_prediction),
                                                ("irrelevant", None), ("neutral_absent", None), ("neutral_generic", None))
                },
                "generation": {"task": response(target, display), "counterfactual": response(changed, display)},
            }
        row["displays"]["labeled"]["letter"] = [_score(position, task_prediction) for position in (0, 1)]
    return rows


def _evidence(tmp_path, monkeypatch, *, headroom, strict, ranker_failure=()):
    protocol_hash = file_hash(roster.PROTOCOL)
    config = json.loads(roster.PROTOCOL.read_bytes())
    class Checkpoint:
        def checkpoint_sha256(self):
            return config["model_sha256"]
    monkeypatch.setattr(roster, "load_checkpoint", lambda _: Checkpoint())
    material, measurement = tmp_path / "material", tmp_path / "measurement"
    roster.freeze(tmp_path, material)
    (material / "sealed-adaptation.json").unlink()
    measurement.mkdir()
    execution = measurement / "execution-sources.json"
    manifest_hash = file_hash(material / "manifest.json")
    json_dump_atomic(execution, {
        "schema": admission.EXECUTION_SCHEMA, "base_sha256": config["model_sha256"],
        "protocol_sha256": protocol_hash, "material_manifest_sha256": manifest_hash,
        "source_sha256": run_grounding._source_hashes(), "inference": config["inference"],
        "generation": config["generation"], "generation_contexts": run_grounding.GENERATION_CONTEXTS,
        "scored_contexts": run_grounding.CONTEXTS,
    })
    splits = {}
    decision_path = measurement / "practice-decision.json"
    for split in run_grounding.SPLITS:
        items = json.loads((material / f"{split}.json").read_bytes())["items"]
        rows = _rows_for(items, ranker_headroom=headroom, ranker_fails_counterfactual=split in ranker_failure,
                         strict_correct=strict)
        predictions = measurement / f"{split}-predictions.json"
        json_dump_atomic(predictions, {"split": split, "rows": rows, "resources": {}})
        digest = file_hash(predictions)
        if split == "practice":
            decision = select_practice(rows, config)
            json_dump_atomic(decision_path, {**decision, "protocol_sha256": protocol_hash,
                                            "practice_predictions_sha256": digest})
        splits[split] = {"predictions_sha256": digest, "resources": {}, "analysis": analyze_split(rows, config, decision)}
    json_dump_atomic(measurement / "report.json", {
        "schema": admission.REPORT_SCHEMA, "base_sha256": config["model_sha256"],
        "protocol_sha256": protocol_hash, "material_manifest_sha256": manifest_hash,
        "execution_sources_sha256": file_hash(execution), "practice_decision_sha256": file_hash(decision_path),
        "splits": splits,
    })
    return material, measurement


def test_qualifying_ranker_authorizes_only_practice_ids(tmp_path, monkeypatch):
    material, measurement = _evidence(tmp_path, monkeypatch, headroom=True, strict=False)
    result = admission.assess(material, measurement)
    assert result["disposition"] == "PASS"
    assert result["qualification"]["ranker"] and not result["qualification"]["strict_generation"]
    assert result["eligible_families"] == ["word-order"]
    scope = result["authorized_practice"]["word-order"]
    assert len(scope["target_item_ids"]) == 2 and len(scope["preservation_item_ids"]) == 4
    assert all(item.startswith("practice/") for ids in scope.values() for item in ids)
    target = scope["target_item_ids"][0]
    records = result["population"]["practice"]
    assert records["ranker"]["word-order"][target]["task"]["classification"] == "persistent_wrong"
    assert records["strict_generation"]["word-order"][target]["task"]["classification"] == "invalid"
    assert records["semantic_generation"]["word-order"][target]["task"]["classification"] == "solved"
    assert not result["sealed_adaptation_opened"]


def test_generation_qualification_cannot_authorize_failed_ranker(tmp_path, monkeypatch):
    material, measurement = _evidence(tmp_path, monkeypatch, headroom=False, strict=True,
                                      ranker_failure=("qualification",))
    result = admission.assess(material, measurement)
    assert result["qualification"]["strict_generation"]
    assert not result["qualification"]["ranker"]
    assert result["hold_reason"] == "QUALIFIED_GENERATION_WITHOUT_QUALIFIED_RANKER_SURROGATE"
    assert result["authorized_practice"] == {}


def test_no_headroom_holds_even_if_both_measurements_pass(tmp_path, monkeypatch):
    material, measurement = _evidence(tmp_path, monkeypatch, headroom=False, strict=True)
    result = admission.assess(material, measurement)
    assert result["qualification"]["ranker"] and result["qualification"]["strict_generation"]
    assert result["hold_reason"] == "NO_REPEATABLE_RANKER_HEADROOM"


def test_rehashed_modified_practice_decision_is_refused(tmp_path, monkeypatch):
    material, measurement = _evidence(tmp_path, monkeypatch, headroom=True, strict=False)
    decision_path = measurement / "practice-decision.json"
    decision = json.loads(decision_path.read_bytes())
    decision["selected_ranker"]["rule"]["display"] = "bullets"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    report_path = measurement / "report.json"
    report = json.loads(report_path.read_bytes())
    report["practice_decision_sha256"] = file_hash(decision_path)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="does not replay from practice"):
        admission.assess(material, measurement)


def test_tampered_qualification_predictions_are_refused(tmp_path, monkeypatch):
    material, measurement = _evidence(tmp_path, monkeypatch, headroom=True, strict=False)
    predictions = measurement / "qualification-predictions.json"
    predictions.write_text(predictions.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="qualification prediction evidence differs"):
        admission.assess(material, measurement)


def test_rehashed_generation_parse_must_match_the_recorded_answer(tmp_path, monkeypatch):
    material, measurement = _evidence(tmp_path, monkeypatch, headroom=False, strict=True)
    predictions = measurement / "qualification-predictions.json"
    recorded = json.loads(predictions.read_bytes())
    recorded["rows"][0]["displays"]["labeled"]["generation"]["task"]["text"] = "unknown"
    predictions.write_text(json.dumps(recorded), encoding="utf-8")
    report_path = measurement / "report.json"
    report = json.loads(report_path.read_bytes())
    report["splits"]["qualification"]["predictions_sha256"] = file_hash(predictions)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="generation decision does not replay"):
        admission.assess(material, measurement)
