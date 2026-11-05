from copy import deepcopy
import json

import pytest

from experiments.phase_iiis_tool_transfer import measurement as m
from experiments.phase_iiis_tool_transfer import verification as v
from weightbridge.adaptation.tool_task import canonical_action, gold_action


SETTINGS = {"cpu_threads": 6, "batch_size": 8, "max_input_tokens": 512, "max_new_tokens": 96}
END = {"schema": "weightbridge.native_tool_completion.v1", "end_token_id": 99, "end_token_surface": "<end>"}


def rows(count=8):
    result = []
    for index in range(count):
        for cf in (0, 1):
            path = f"docs/item-{index}-{cf}.txt"
            fixture = {"request": {"kind": "search" if index % 2 == 0 else "contents",
                "value": f"query-{index}-{cf}" if index % 2 == 0 else path, "root": "docs"},
                "roots": [{"path": "docs", "supported": True}],
                "documents": [{"path": path, "content": f"query-{index}-{cf}", "readable": True}], "tests": []}
            gold = gold_action(fixture)
            for wording in (0, 1):
                for ordering in (0, 1):
                    result.append({"task": "tool", "category": "tool", "id": f"{index}-{cf}-{wording}-{ordering}",
                        "family_id": f"family-{index}", "scenario": "test", "wording_group": "fit",
                        "counterfactual": cf, "wording": wording, "ordering": ordering, "fixture": fixture,
                        "gold_action": gold, "stratum": gold["tool"], "completion": canonical_action(gold),
                        "prompt": f"visible fixture {index} {cf} {wording} {ordering}"})
    return result


def measured(population, wrong=(), incomplete=()):
    recorded = []
    for index, row in enumerate(population):
        text = row["completion"] if index not in wrong else '{"tool":"no_action","arguments":{}}'
        output = {"text": text, "generated_token_ids": [5, 99] if index not in incomplete else [5]*96,
            "prefix_token_ids": [3], "prefix_tokens": 1, "native_end_token_id": 99,
            "unexpected_special_token_ids": [], "stop_reason": "eos" if index not in incomplete else "token_budget"}
        recorded.append({**deepcopy(row), **output, **m._score(row, output)})
    return {"endpoint": m.tool_endpoint(SETTINGS).manifest(), "completion": END, "rows": recorded}


def test_eight_forms_and_complete_family_gate():
    population = rows(32)
    original = measured(population, wrong=range(256))
    candidate = measured(population)
    gate = m.compare_tools(original, candidate, formal=True)
    assert gate["passed"]
    assert gate["complete_family_gain"] == 32
    assert gate["corrected_real_actions"] == ["read_file", "search_files"]
    assert m.summarize_tools(candidate)["complete_families"] == 32
    candidate["rows"].pop()
    with pytest.raises(ValueError, match="eight"):
        m.summarize_tools(candidate)


def test_development_does_not_inherit_formal_family_counts():
    population = rows(1)
    before, after = measured(population, wrong=range(8)), measured(population)
    assert m.compare_tools(before, after)["passed"]
    formal = m.compare_tools(before, after, formal=True)
    assert not formal["passed"]
    assert formal["disposition"] == "INSUFFICIENT_IMPROVEMENT_HEADROOM"


def test_schema_and_semantic_preservation_survive_incomplete_response():
    population = rows(1)
    before = measured(population, incomplete=[0])
    assert before["rows"][0]["semantic_correct"]
    assert before["rows"][0]["schema_valid"]
    assert not before["rows"][0]["strict_correct"]
    after = measured(population, wrong=[0])
    gate = m.tool_retention(before, after)
    assert not gate["passed"]
    assert gate["lost_semantic_correct"] == [population[0]["id"]]


def test_equal_totals_cannot_hide_response_or_family_losses():
    population = rows(2)
    before, after = measured(population, wrong=[0]), measured(population, wrong=[8])
    assert m.summarize_tools(before)["strict_correct"] == m.summarize_tools(after)["strict_correct"]
    loss = m.tool_retention(before, after)
    assert not loss["passed"]
    assert loss["lost_complete_families"] == ["family-1"]


def test_formal_semantic_gain_cannot_be_format_only():
    population = rows(8)
    before, after = measured(population, incomplete=range(64)), measured(population)
    result = m.compare_tools(before, after, formal=True)
    assert result["strict_response_gain"] == 64
    assert result["corrected_schema_valid_semantic_errors"] == []
    assert not result["passed"]
    assert result["disposition"] == "INSUFFICIENT_IMPROVEMENT_HEADROOM"


def test_native_completion_and_unexpected_special_tokens():
    population = rows(1)
    candidate = measured(population)
    m.checked_tools(candidate, population, SETTINGS, END)
    wrong = deepcopy(candidate)
    wrong["rows"][0]["generated_token_ids"] = [5]
    with pytest.raises(ValueError, match="termination"):
        m.checked_tools(wrong, population, SETTINGS, END)
    output = {key: candidate["rows"][0][key] for key in ("text", "stop_reason", "generated_token_ids", "native_end_token_id")}
    output["unexpected_special_token_ids"] = [5]
    assert not m._score(population[0], output)["schema_valid"]


def test_scores_are_recomputed_and_task_identity_is_immutable():
    population = rows(1)
    candidate = measured(population, wrong=[0])
    candidate["rows"][0]["strict_correct"] = True
    with pytest.raises(ValueError, match="replay"):
        m.checked_tools(candidate, population, SETTINGS, END)
    before, after = measured(population), measured(population)
    after["rows"][0]["prompt"] += " modified"
    with pytest.raises(ValueError, match="identities"):
        m.tool_retention(before, after)


def test_exact_native_replay_is_own_representation_not_parent():
    population = rows(1)
    parent, candidate = measured(population, wrong=[0]), measured(population)
    assert not v.exact_native_replay(parent, candidate)
    assert v.exact_native_replay(candidate, deepcopy(candidate))
    replay = deepcopy(candidate)
    replay["rows"][0]["generated_token_ids"] = [6, 99]
    assert not v.exact_native_replay(candidate, replay)
    replay = deepcopy(candidate)
    replay["rows"][0]["text"] += " "
    assert not v.exact_native_replay(candidate, replay)


def protocol(tmp_path):
    p = {"checkpoint_sha256": "1"*64, "task_manifest_sha256": "2"*64, "source_sha256": {},
         "endpoints": {"tool": SETTINGS, "decimal": {**SETTINGS, "max_input_tokens": 128, "max_new_tokens": 16}},
         "tool_completion": END}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(p))
    return path


def test_host_authorization_parent_and_task_binding(tmp_path):
    path = protocol(tmp_path)
    auth = tmp_path / "auth.json"
    v.authorize_evaluation(auth, path, stage="calibration", task="tool", roster_sha256="3"*64,
                           representation_sha256="4"*64)
    v.validate_authorization(path, auth, stage="calibration", task="tool", expected_rows="3"*64,
                             expected_checkpoint="4"*64)
    with pytest.raises(ValueError, match="another"):
        v.validate_authorization(path, auth, stage="calibration", task="decimal", expected_rows="3"*64,
                                 expected_checkpoint="4"*64)
    with pytest.raises(FileExistsError):
        v.authorize_evaluation(auth, path, stage="calibration", task="tool", roster_sha256="3"*64,
                              representation_sha256="4"*64)


def test_independent_authorization_refuses_unlocked_or_failed_stages(tmp_path):
    path = protocol(tmp_path)
    out = tmp_path / "independent.json"
    with pytest.raises(ValueError, match="locked"):
        v.authorize_evaluation(out, path, stage="qualification", task="tool", roster_sha256="3"*64,
                              representation_sha256="4"*64)
    assert not out.exists()
    with pytest.raises(ValueError, match="SHA-256"):
        v.sha("f"*62)
    with pytest.raises(ValueError, match="Unknown"):
        v.authorize_evaluation(out, path, stage="qualificaton", task="tool", roster_sha256="3"*64,
                              representation_sha256="4"*64)


def test_worker_refuses_before_reading_protected_roster(tmp_path):
    from experiments.phase_iiis_tool_transfer.evaluate import run
    path = protocol(tmp_path)
    with pytest.raises(ValueError, match="authorization"):
        run(tmp_path / "nonexistent-model", path, tmp_path / "sealed-nonexistent-roster", tmp_path / "out.json",
            expected_checkpoint="4"*64, expected_rows="3"*64, task="tool", stage="qualification", authorization=None)
    assert not (tmp_path / "out.json").exists()


def test_missing_replay_is_not_a_pass():
    result = v.final_replay_gate({"qualification": None}, reconstruction={"passed": True},
        original_restore={"passed": True}, required_keys=("qualification",))
    assert not result["passed"]
    sequential = v.final_replay_gate({"qualification": {"passed": True, "exact_native_generation": True}},
        reconstruction={"passed": True}, original_restore={"passed": True}, sequential=True,
        required_keys=("qualification",))
    assert not sequential["passed"]


def test_native_decoder_has_no_forced_format_or_truncation():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    class Tokenizer:
        chat_template = "native"
        all_special_ids = [99, 55]
        pad_token_id = 0
        def encode(self, text, add_special_tokens=False):
            return [99] if text == "<end>" else [3, 4]
        def decode(self, ids, skip_special_tokens=False):
            assert not skip_special_tokens
            if ids == [99]:
                return "<end>"
            return " ".join(str(i) for i in ids)
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
            return "native:" + messages[0]["content"]
    class Model:
        def generate(self, **kwargs):
            assert set(kwargs) == {"input_ids", "attention_mask", "do_sample", "num_beams", "max_new_tokens",
                "use_cache", "pad_token_id", "eos_token_id", "return_dict_in_generate"}
            assert kwargs["do_sample"] is False and kwargs["eos_token_id"] == 99
            tails = torch.tensor([[55, 99]+[0]*94, [5]*96])
            return torch.cat((kwargs["input_ids"], tails), dim=1)
    runner = SimpleNamespace(tokenizer=Tokenizer(), model=Model(), device="cpu")
    observations, costs = m.generate_tools(runner, ["one", "two"], SETTINGS, END)
    assert observations[0]["generated_token_ids"] == [55, 99]
    assert observations[0]["unexpected_special_token_ids"] == [55]
    assert observations[0]["text"] == "55"
    assert observations[1]["stop_reason"] == "token_budget"
    assert len(observations[1]["generated_token_ids"]) == 96
    assert costs["batched_decode_steps"] == 96


def test_protected_authorization_checks_actual_prerequisite_receipts(tmp_path):
    from weightbridge.core import file_hash
    path = protocol(tmp_path)
    lock = tmp_path / "lock.json"
    identity = {"passed": True, "protocol_sha256": file_hash(path), "task_manifest_sha256": "2"*64,
                "representation_sha256": "4"*64}
    lock.write_text(json.dumps(identity))
    prerequisites = {}
    for name in v._requirements("qualification"):
        receipt = tmp_path / f"{name}.json"
        receipt.write_text(json.dumps(identity))
        prerequisites[name] = receipt
    broken = prerequisites["historical-retention"]
    broken.write_text(json.dumps({**identity, "passed": False}))
    out = tmp_path / "authorization.json"
    with pytest.raises(ValueError, match="failed"):
        v.authorize_evaluation(out, path, stage="qualification", task="tool", roster_sha256="3"*64,
            representation_sha256="4"*64, finalist_lock_path=lock, prerequisite_paths=prerequisites)
    assert not out.exists()
    broken.write_text(json.dumps(identity))
    v.authorize_evaluation(out, path, stage="qualification", task="tool", roster_sha256="3"*64,
        representation_sha256="4"*64, finalist_lock_path=lock, prerequisite_paths=prerequisites)
    v.validate_authorization(path, out, stage="qualification", task="tool", expected_rows="3"*64,
                             expected_checkpoint="4"*64)


def test_secondary_reference_losses_block_even_if_own_parent_improves():
    population = rows(8)
    own = measured(population, wrong=range(64))
    reference = measured(population)
    candidate = measured(population, wrong=[0])
    assert m.compare_tools(own, candidate)["passed"]
    assert not m.compare_tools(own, candidate, required_references=[reference])["passed"]
