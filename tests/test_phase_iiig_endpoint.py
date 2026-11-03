import pytest

from experiments.phase_iiig.endpoint import parse_integer


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
