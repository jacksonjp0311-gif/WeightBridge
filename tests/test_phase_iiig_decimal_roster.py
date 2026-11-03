from collections import Counter
from decimal import Decimal
from itertools import product
import json

from experiments.phase_iiig import roster_decimal, roster_replication


def test_decimal_roster_balances_sign_precision_and_answer_side(tmp_path):
    config = json.loads(roster_decimal.PROTOCOL.read_bytes())
    previous = json.loads(roster_replication.PROTOCOL.read_bytes())
    for key in ("qualification", "selection_gate", "hidden_gate", "fit_limits"):
        assert config[key] == previous[key]
    output = tmp_path / "decimal"
    manifest = roster_decimal.freeze(output, hidden_seed="test-private")
    expected_cells = dict.fromkeys(product(roster_decimal.STRATA, ("shorter", "longer"), (0, 1)), 2)
    seen = set()
    for split in roster_decimal.SPLITS:
        rows = roster_decimal.load_split(output, split, manifest)
        cells = Counter()
        for first, second in zip(rows[::2], rows[1::2], strict=True):
            assert first["item"] == second["item"]
            assert first["answer"] == second["answer"]
            assert first["prompt"].splitlines()[1:] == second["prompt"].splitlines()[1:]
            assert first["prompt"].splitlines()[0] != second["prompt"].splitlines()[0]
            operands = first["operands"]
            identity = roster_decimal._identity(first["family"], operands)
            assert identity not in seen
            seen.add(identity)
            assert first["answer"] == max(operands, key=Decimal)
            group = first["group"]
            if group == "preserve":
                assert len(operands) == 1 and "." in operands[0]
                continue
            side = first["truth_side"]
            assert first["answer"] == operands[side]
            if group == "positive":
                assert abs(Decimal(operands[0]) - Decimal(operands[1])) > Decimal("10")
                continue
            precision = first["truth_precision"]
            fractions = [len(value.split(".")[1]) for value in operands]
            assert (fractions[side] > fractions[1 - side]) == (precision == "longer")
            left, right = map(Decimal, operands)
            tier = first["tier"]
            if tier == "positive-close":
                assert left > 0 and right > 0 and abs(left - right) <= Decimal("0.09")
            elif tier == "negative-close":
                assert left < 0 and right < 0 and abs(left - right) <= Decimal("0.09")
            else:
                assert left * right < 0
            cells[tier, precision, side] += 1
        assert cells == expected_cells
