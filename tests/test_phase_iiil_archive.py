import hashlib
import json

import pytest

from experiments.phase_iiil.archive_trial import _FIT_FILES, _WORKER_FILES, _private_fields, _public_bytes
from experiments.phase_iiil.analyze import _checked_measurement, _subgroup_outcomes
from experiments.phase_iiil.semantic_diagnostics import compare_semantic
from experiments.phase_iiil.semantic_roster import PROTOCOL, freeze, load_split
from experiments.phase_iiih.run_decimal_endpoint import endpoint
from weightbridge.adaptation.behavior_endpoint import measure_endpoint


def test_fit_archive_refuses_hidden_path_and_unbound_receipt(tmp_path):
    evidence = tmp_path / "parent-adaptation.json"
    evidence.write_bytes(b"{}\n")
    expected = hashlib.sha256(evidence.read_bytes()).hexdigest()
    assert _public_bytes(tmp_path, evidence.name, _FIT_FILES, expected) == b"{}\n"
    for private_name in ("hidden.json", "../parent-adaptation.json", "checkpoints/A1/config.json",
                         "verification/hidden-A1.json", "worker-stdout.txt"):
        with pytest.raises(ValueError, match="allowlist"):
            _public_bytes(tmp_path, private_name, _FIT_FILES, expected)
    evidence.write_bytes(b'{"changed":true}\n')
    with pytest.raises(ValueError, match="recorded hash"):
        _public_bytes(tmp_path, evidence.name, _FIT_FILES, expected)


def test_worker_allowlist_and_hidden_aggregate_reject_raw_outputs():
    assert _WORKER_FILES.fullmatch("verification/transfer-validation-B1.json")
    assert _WORKER_FILES.fullmatch("verification/selection-base.json")
    assert not _WORKER_FILES.fullmatch("verification/hidden-A1.json")
    assert _private_fields({"disposition": "HOLD", "counts": {"regressions": 1}}) == set()
    assert _private_fields({"attempts": [{"rows": [{"prompt": "secret"}]}]}) == {"rows", "prompt"}


def test_archive_reparses_generation_before_replaying_gate(tmp_path):
    material = tmp_path / "public"
    manifest = freeze(material, tmp_path / "hidden.json", hidden_seed="archive-test")
    rows = load_split(material, "adaptation", manifest)
    contract = endpoint(json.loads(PROTOCOL.read_bytes()))

    def decoder(_, prompts, **limits):
        answers = {row["prompt"]: row["answer"] for row in rows}
        return [{"text": f" {answers[prompt]}\n", "stop_reason": "token_budget"} for prompt in prompts], {}

    class Runner:
        tokenizer = object()

    measured = measure_endpoint(Runner(), rows, contract, decoder=decoder)
    assert _checked_measurement(measured, rows, contract) == measured
    measured["rows"][0]["correct"] = False
    with pytest.raises(ValueError, match="inconsistent parsing"):
        _checked_measurement(measured, rows, contract)


def test_subgroup_report_keeps_one_semantic_comparison_tied_across_four_forms(tmp_path):
    material = tmp_path / "public"
    manifest = freeze(material, tmp_path / "hidden.json", hidden_seed="subgroup-report-test")
    rows = load_split(material, "adaptation", manifest)
    contract = endpoint(json.loads(PROTOCOL.read_bytes()))
    focus, fail = rows[0]["comparison_id"], False

    def decoder(_, prompts, **limits):
        lookup = {row["prompt"]: row for row in rows}
        results = []
        for prompt in prompts:
            row = lookup[prompt]
            answer = row["answer"]
            if fail and row["comparison_id"] == focus:
                answer = next(value for value in row["operands"] if value != row["answer"])
            results.append({"text": answer + "\n", "stop_reason": "token_budget"})
        return results, {}

    class Runner:
        tokenizer = object()

    fail = True
    parent = measure_endpoint(Runner(), rows, contract, decoder=decoder)
    fail = False
    candidate = measure_endpoint(Runner(), rows, contract, decoder=decoder)
    gate = compare_semantic(parent, candidate)
    result = _subgroup_outcomes(parent, candidate, gate)
    group = f"{rows[0]['tier']}/{rows[0]['truth_precision']}"
    assert result[group]["candidate_correct_presentations"] - result[group]["parent_correct_presentations"] == 4
    assert result[group]["candidate_strict_correct"] - result[group]["parent_strict_correct"] == 1
    assert result[group]["new_strict_corrections"] == [focus]
