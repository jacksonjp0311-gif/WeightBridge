import pytest

from experiments.phase_iiig.endpoint import parse_integer
from experiments.phase_iiig.decimal_endpoint import parse_decimal


@pytest.mark.parametrize(("text", "stop", "expected"), [
    (" 247 ", "eos", "247"),
    ("\t-85\r\nNext", "token_budget", "-85"),
    ("20\n", "token_budget", "20"),
    ("20", "token_budget", None),
    ("\n20", "eos", None),
    ("0.5", "eos", None),
    ("24 extra", "eos", None),
    ("002", "eos", None),
    ("Answer: 8", "eos", None),
    (" ", "eos", None),
])
def test_strict_first_line_integer(text, stop, expected):
    assert parse_integer(text, stop) == expected


@pytest.mark.parametrize(("text", "stop", "expected"), [
    (" 9.110 ", "eos", "9.11"),
    ("-.900\nNext", "token_budget", "-0.9"),
    ("9.1", "token_budget", None),
    ("9.1 > 9.09\n", "token_budget", None),
    ("9.1e0", "eos", None),
])
def test_strict_completed_decimal(text, stop, expected):
    assert parse_decimal(text, stop) == expected
