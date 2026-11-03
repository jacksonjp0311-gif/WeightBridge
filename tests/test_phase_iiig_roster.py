import json

import pytest

from experiments.phase_iiig.roster import PROTOCOL, SPLITS, _identity, freeze, load_split


def test_arithmetic_truth_is_paired_and_disjoint(tmp_path):
    output = tmp_path / "material"
    manifest = freeze(output, hidden_seed="test-hidden")
    previous = set()
    for split in SPLITS:
        rows = load_split(output, split, manifest)
        population = set()
        for first, second in zip(rows[::2], rows[1::2], strict=True):
            assert first["item"] == second["item"]
            assert first["answer"] == second["answer"]
            assert first["wording"] == 0 and second["wording"] == 1
            assert first["prompt"] != second["prompt"]
            assert all(row["prompt"].endswith("Answer:") for row in (first, second))
            family, operands = first["family"], first["operands"]
            if family == "addition":
                truth = operands[0] + operands[1]
            elif family == "subtraction":
                truth = operands[0] - operands[1]
            else:
                truth, = operands
            assert first["answer"] == str(truth)
            identity = _identity(family, operands)
            assert identity not in previous and identity not in population
            population.add(identity)
        previous |= population
    assert "test-hidden" not in (output / "manifest.json").read_text()


def test_roster_freeze_keeps_development_fixed_and_refuses_tampering(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    one = freeze(first, hidden_seed="hidden-one")
    two = freeze(second, hidden_seed="hidden-two")
    assert one["splits"]["practice"] == two["splits"]["practice"]
    assert one["splits"]["selection"] == two["splits"]["selection"]
    assert one["splits"]["hidden"]["sha256"] != two["splits"]["hidden"]["sha256"]
    practice = first / "practice.json"
    changed = json.loads(practice.read_bytes())
    changed["rows"][0]["answer"] = "wrong"
    practice.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="roster hash differs"):
        load_split(first, "practice", one)
    with pytest.raises(FileExistsError):
        freeze(first)
    assert json.loads(PROTOCOL.read_bytes())["endpoint"]["answer_prefix"] == "Answer:"
