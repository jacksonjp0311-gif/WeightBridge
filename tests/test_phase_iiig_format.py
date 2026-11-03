import json

import pytest

from experiments.phase_iiig.format_diagnostic import PROTOCOL, _continuation, classify_digit, render


@pytest.mark.parametrize("text,ended,payload,complete", [
    (" ", False, None, None),
    (" 7", False, "7", None),
    (" 7", True, "7", "7"),
    ("\t7 \r\nother", False, "7", "7"),
    ("\n7", True, None, None),
    ("７\n", False, None, None),
    ("77\n", False, None, None),
    ("7 apples\n", True, None, None),
])
def test_digit_format_separates_bounded_payload_from_completed_line(text, ended, payload, complete):
    result = classify_digit(text, ended=ended)
    assert result["digit_payload"] == payload
    assert result["complete_digit"] == complete


def test_frozen_format_prompt_crosses_separator_and_tail_for_each_item():
    rows = list(render(json.loads(PROTOCOL.read_bytes())))
    assert len(rows) == 40
    groups = {}
    for row in rows:
        groups.setdefault(row["item"], set()).add((row["prefix"], row["tail"], row["expected"]))
        assert row["prompt"].endswith(" ") == (row["tail"] == "space")
    assert all(len(cells) == 4 and len({value for _, _, value in cells}) == 1 for cells in groups.values())


def test_early_eos_excludes_batched_padding():
    assert _continuation([11, 22, 0], 22, 3) == ([11], [11, 22], True)
    assert _continuation([11, 12], 22, 2) == ([11, 12], [11, 12], False)
    with pytest.raises(ValueError):
        _continuation([11], 22, 2)
