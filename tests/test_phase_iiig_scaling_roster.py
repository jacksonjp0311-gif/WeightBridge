import json
from collections import Counter

from experiments.phase_iiig import roster_followup, roster_scaling


def test_scaling_keeps_acceptance_gates_and_distinct_operand_regimes(tmp_path):
    config = json.loads(roster_scaling.PROTOCOL.read_bytes())
    previous = json.loads(roster_followup.PROTOCOL.read_bytes())
    for key in ("qualification", "selection_gate", "hidden_gate"):
        assert config[key] == previous[key]
    assert config["eligible_families"] == {"qwen": ["multiplication"]}
    output = tmp_path / "scaling"
    manifest = roster_scaling.freeze(output, hidden_seed="test-private")
    seen = set()
    for split in roster_scaling.SPLITS:
        rows = roster_scaling.load_split(output, split, manifest)
        widths = Counter()
        for first, second in zip(rows[::2], rows[1::2], strict=True):
            assert first["item"] == second["item"]
            assert first["prompt"].splitlines()[1:] == second["prompt"].splitlines()[1:]
            assert first["prompt"].splitlines()[0] != second["prompt"].splitlines()[0]
            assert first["answer"] == second["answer"]
            family, operands = first["family"], first["operands"]
            identity = roster_scaling._identity(family, operands)
            assert identity not in seen
            seen.add(identity)
            if first["group"] == "preserve":
                assert len(str(operands[0])) == 4
                truth = operands[0]
            else:
                if first["group"] == "positive":
                    assert operands[0] in (10, 11)
                else:
                    widths[tuple(map(lambda value: len(str(value)), operands))] += 1
                truth = operands[0] * operands[1]
            assert first["answer"] == str(truth)
        expected = config["roster_counts"][split]["target"] // 3
        assert widths == {(3, 1): expected, (3, 3): expected, (4, 2): expected}
