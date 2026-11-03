import json
from collections import Counter

from experiments.phase_iiig import roster_replication, roster_scaling


def test_replication_reserves_new_operands_and_bounds_fit_scope(tmp_path):
    config = json.loads(roster_replication.PROTOCOL.read_bytes())
    scaling = json.loads(roster_scaling.PROTOCOL.read_bytes())
    for key in ("qualification", "selection_gate", "hidden_gate"):
        assert config[key] == scaling[key]
    limits = config["fit_limits"]
    for counts in config["roster_counts"].values():
        assert 2 * (limits["max_repeatable_error_items"] + limits["max_solved_target_items"]
                    + counts["positive"] + counts["preserve"]) <= limits["max_rows"] == 64
    output = tmp_path / "replication"
    manifest = roster_replication.freeze(output, hidden_seed="test-private")
    seen = set()
    for split in roster_replication.SPLITS:
        rows = roster_replication.load_split(output, split, manifest)
        widths = Counter()
        for first, second in zip(rows[::2], rows[1::2], strict=True):
            assert first["item"] == second["item"]
            assert first["answer"] == second["answer"]
            assert first["prompt"].splitlines()[1:] == second["prompt"].splitlines()[1:]
            family, operands = first["family"], first["operands"]
            identity = roster_replication._identity(family, operands)
            assert identity not in seen
            seen.add(identity)
            if first["group"] == "preserve":
                assert len(str(operands[0])) == 5
                truth = operands[0]
            else:
                if first["group"] == "positive":
                    assert 12 <= operands[0] <= 27 and operands[1] == 2
                else:
                    widths[tuple(len(str(value)) for value in operands)] += 1
                truth = operands[0] * operands[1]
            assert first["answer"] == str(truth)
        expected = config["roster_counts"][split]["target"] // 3
        assert widths == {(4, 1): expected, (4, 3): expected, (5, 2): expected}
