import json

from experiments.phase_iiig import roster
from experiments.phase_iiig.roster_followup import PROTOCOL, SPLITS, _identity, freeze, load_split


def test_followup_keeps_gates_and_disjoint_model_specific_tasks(tmp_path):
    previous_config = json.loads(roster.PROTOCOL.read_bytes())
    config = json.loads(PROTOCOL.read_bytes())
    for key in ("qualification", "selection_gate", "hidden_gate"):
        assert config[key] == previous_config[key]
    assert config["eligible_families"] == {
        "qwen": ["multiplication"], "smollm2": ["first-digit", "last-digit"],
    }
    output = tmp_path / "followup"
    manifest = freeze(output, hidden_seed="test-private")
    seen = set()
    for split in SPLITS:
        rows = load_split(output, split, manifest)
        for first, second in zip(rows[::2], rows[1::2], strict=True):
            assert first["item"] == second["item"]
            assert first["prompt"].splitlines()[1:] == second["prompt"].splitlines()[1:]
            assert first["prompt"].splitlines()[0] != second["prompt"].splitlines()[0]
            assert first["answer"] == second["answer"]
            family, operands = first["family"], first["operands"]
            identity = _identity(family, operands)
            assert identity not in seen
            seen.add(identity)
            if family == "multiplication":
                result = str(operands[0] * operands[1])
            elif family == "first-digit":
                result = str(operands[0])[0]
            elif family == "last-digit":
                result = str(operands[0])[-1]
            else:
                assert len(str(operands[0])) == 3
                result = str(operands[0])
            assert first["answer"] == result
