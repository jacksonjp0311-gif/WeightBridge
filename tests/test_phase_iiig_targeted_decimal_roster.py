from collections import Counter
from decimal import Decimal
import json

from experiments.phase_iiig import roster_decimal, roster_decimal_targeted


def test_targeted_decimal_freeze_keeps_gates_and_separates_control_competence(tmp_path):
    config = json.loads(roster_decimal_targeted.PROTOCOL.read_bytes())
    previous = json.loads(roster_decimal.PROTOCOL.read_bytes())
    for key in ("qualification", "selection_gate", "hidden_gate", "fit_limits"):
        assert config[key] == previous[key]
    output = tmp_path / "targeted"
    manifest = roster_decimal_targeted.freeze(output, hidden_seed="test-private")
    expected = Counter(roster_decimal_targeted.TARGET_CELLS)
    seen = set()
    for split in roster_decimal_targeted.SPLITS:
        rows = roster_decimal_targeted.load_split(output, split, manifest)
        cells = Counter()
        for first, second in zip(rows[::2], rows[1::2], strict=True):
            assert first["item"] == second["item"]
            assert first["prompt"].splitlines()[1:] == second["prompt"].splitlines()[1:]
            operands = first["operands"]
            identity = roster_decimal_targeted._identity(first["family"], operands)
            assert identity not in seen
            seen.add(identity)
            assert first["answer"] == second["answer"] == max(operands, key=Decimal)
            if first["group"] == "preserve":
                assert len(operands[0].split(".")[1]) == 4
                continue
            assert max(len(value.split(".")[1]) for value in operands) == 4
            side = first["truth_side"]
            assert first["answer"] == operands[side]
            precision = first["truth_precision"]
            assert (len(operands[side].split(".")[1]) == 4) == (precision == "longer")
            left, right = map(Decimal, operands)
            tier = first["tier"]
            if first["group"] == "positive":
                assert max(left, right) > 0
            else:
                if tier.startswith("negative"):
                    assert left < 0 and right < 0
                    if tier == "negative-far":
                        assert Decimal("2") < abs(left - right) < Decimal("9")
                cells[tier, precision, side] += 1
        assert cells == expected
