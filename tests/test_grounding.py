import math

import pytest

from weightbridge.adaptation.grounding import parse_generation, prior_subtracted_scores


def _parse(text, *, display="labels", completed=True, labels=("A", "B"),
           answers=("red fox", "brown bear"), aliases=(("Fox",), ("Bear",))):
    return parse_generation(text, display=display, labels=labels, answers=answers, aliases=aliases,
                            completed=completed)


@pytest.mark.parametrize(("text", "strict", "semantic", "kind"), [
    ("  RED\n FOX ", 0, 0, "content"),
    ("bear", 1, 1, "content"),
    (" b\n", None, 1, "label"),
    (" A.   FOX ", None, 0, "label_content"),
    ("B) Brown bear", None, 1, "label_content"),
    ("B: Bear", None, 1, "label_content"),
    ("B - Bear", None, 1, "label_content"),
    ("A fox", None, 0, "label_content"),
    ("A: Bear", None, None, "conflict"),
    ("A: fox or bear", None, None, "unrecognized"),
    ("I choose fox", None, None, "unrecognized"),
    ("Bear.", None, None, "unrecognized"),
    ("A.", None, None, "unrecognized"),
    ("", None, None, "unrecognized"),
])
def test_parse_generation_distinguishes_strict_content_and_diagnostic_recovery(text, strict, semantic, kind):
    assert _parse(text) == {"strict_position": strict, "semantic_position": semantic, "semantic_kind": kind}


def test_generation_requires_eos_and_never_interprets_hidden_labels():
    assert _parse("Fox", completed=False) == {
        "strict_position": None, "semantic_position": None, "semantic_kind": "incomplete",
    }
    for text in ("A", "A: fox"):
        assert _parse(text, display="bullets")["semantic_kind"] == "unrecognized"
    assert _parse("FOX", display="unlabeled", labels=None)["semantic_position"] == 0
    assert _parse("A", display="bullets", answers=("A", "bear"))["semantic_position"] == 0


def test_ambiguity_is_distinct_from_conflicting_composite_and_strict_exactness():
    assert _parse("animal", aliases=(("animal",), ("animal",)))["semantic_kind"] == "ambiguous"
    assert _parse("A: animal", aliases=(("animal",), ("animal",)))["semantic_kind"] == "ambiguous"
    assert _parse("A", labels=("A", "A"))["semantic_kind"] == "ambiguous"
    assert _parse("A", answers=("fox", "A")) == {
        "strict_position": 1, "semantic_position": None, "semantic_kind": "ambiguous",
    }


def test_parsed_positions_follow_visible_option_reversal():
    assert _parse("B: fox", answers=("brown bear", "red fox"), aliases=(("Bear",), ("Fox",))) == {
        "strict_position": None, "semantic_position": 1, "semantic_kind": "label_content",
    }
    with pytest.raises(ValueError, match="two entries"):
        _parse("A", labels=("A",))
    with pytest.raises(ValueError, match="sequence of forms"):
        _parse("A", aliases=("fox", "bear"))


def _score(tokens, total):
    return {"answer_token_ids": tokens, "answer_tokens": len(tokens),
            "sum_log_probability": total, "mean_log_probability": total / len(tokens),
            "terminated_sum_log_probability": total - 10}


def test_prior_calibration_uses_own_lengths_and_reverses_with_both_pairs():
    conditional = [_score((1,), -3), _score((7, 9), -5)]
    prior = [_score([1], -5), _score([7, 9], -2)]
    expected = [
        {"answer_token_ids": [1], "answer_tokens": 1, "sum_log_probability": 2, "mean_log_probability": 2},
        {"answer_token_ids": [7, 9], "answer_tokens": 2, "sum_log_probability": -3, "mean_log_probability": -1.5},
    ]
    assert prior_subtracted_scores(conditional, prior) == expected
    assert prior_subtracted_scores(conditional[::-1], prior[::-1]) == expected[::-1]


@pytest.mark.parametrize(("alpha", "expected"), [(0, -3), (0.5, -0.5), (1.5, 4.5)])
def test_prior_calibration_scales_the_prior(alpha, expected):
    conditional = [_score([1], -3), _score([2], -4)]
    prior = [_score([1], -5), _score([2], -6)]
    assert prior_subtracted_scores(conditional, prior, alpha=alpha)[0]["sum_log_probability"] == expected


@pytest.mark.parametrize("alpha", [math.inf, math.nan, -1, True, "1"])
def test_prior_calibration_refuses_invalid_weights(alpha):
    scores = [_score([1], -3), _score([2], -4)]
    with pytest.raises(ValueError, match="alpha must be finite and nonnegative"):
        prior_subtracted_scores(scores, scores, alpha=alpha)


def test_prior_calibration_detects_equal_length_swaps_and_contextual_tokenization_changes():
    conditional = [_score([1, 4], -3), _score([5, 9], -2)]
    with pytest.raises(ValueError, match="differ or are out of order"):
        prior_subtracted_scores(conditional, conditional[::-1])
    with pytest.raises(ValueError, match="differ or are out of order"):
        prior_subtracted_scores(conditional, [_score([1, 3], -3), conditional[1]])
    with pytest.raises(ValueError, match="distinct answer token IDs"):
        prior_subtracted_scores([conditional[0]] * 2, [conditional[0]] * 2)


@pytest.mark.parametrize(("update", "message"), [
    ({"answer_tokens": 2}, "matching positive length"),
    ({"answer_token_ids": None}, "exact answer token IDs"),
    ({"sum_log_probability": math.inf}, "finite and nonpositive"),
    ({"mean_log_probability": math.nan}, "finite and nonpositive"),
    ({"sum_log_probability": 1}, "finite and nonpositive"),
    ({"mean_log_probability": -20}, "does not match"),
])
def test_prior_calibration_refuses_invalid_scores(update, message):
    conditional = [_score([1], -3), _score([2], -5)]
    prior = [{**conditional[0], **update}, conditional[1]]
    with pytest.raises(ValueError, match=message):
        prior_subtracted_scores(conditional, prior)
