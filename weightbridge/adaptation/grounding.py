"""Deterministic response parsing and contextual score calibration."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
import re


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _two_options(values, field):
    if isinstance(values, str) or not isinstance(values, Sequence) or len(values) != 2:
        raise ValueError(f"{field} must contain two entries in printed order")
    return values


def _option_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Options and aliases must be nonempty text")
    return _normalize(text)


def _matches(text, choices):
    return [position for position, forms in enumerate(choices) if text in forms]


def _diagnostic(text, content, labels):
    content_matches = _matches(text, content)
    exact_labels = _matches(text, [{label} for label in labels])
    composites = []
    for position, label in enumerate(labels):
        match = re.fullmatch(rf"{re.escape(label)}(?:\s*[.):]\s*|\s+[-–—]\s+|\s+)(.+)", text)
        if match:
            body_matches = _matches(match[1], content)
            if body_matches:
                composites.append((position, body_matches))

    if (len(content_matches) > 1 or len(exact_labels) > 1 or len(composites) > 1
            or any(len(matches) > 1 for _, matches in composites)):
        return None, "ambiguous"
    if any(position != matches[0] for position, matches in composites):
        return None, "conflict"
    positions = {*content_matches, *exact_labels, *(position for position, _ in composites)}
    if len(positions) > 1:
        return None, "ambiguous"
    if content_matches:
        return content_matches[0], "content"
    if exact_labels:
        return exact_labels[0], "label"
    if composites:
        return composites[0][0], "label_content"
    return None, "unrecognized"


def parse_generation(text: str, *, display: str, labels: Sequence[str] | None = None,
                     answers: Sequence[str], aliases: Sequence[Sequence[str]], completed: bool) -> dict:
    """Parse EOS-completed output against exact normalized options and aliases.

    Strict content compliance is measured separately from diagnostic recovery
    of a printed label or an agreeing label and answer. Positions are zero-based
    printed positions. Diagnostic recovery never interprets hidden labels.
    """
    if not isinstance(text, str) or type(completed) is not bool:
        raise ValueError("Generation must supply text and boolean completion")
    if display in ("labels", "labeled"):
        visible_labels = tuple(_option_text(label) for label in _two_options(labels, "labels"))
    elif display in ("bullets", "unlabeled"):
        visible_labels = ()
    else:
        raise ValueError("Display must specify labels or bullets")
    content = []
    for answer, forms in zip(_two_options(answers, "answers"), _two_options(aliases, "aliases"), strict=True):
        if isinstance(forms, str) or not isinstance(forms, Sequence):
            raise ValueError("Aliases must contain a sequence of forms for each answer")
        content.append({_option_text(form) for form in (answer, *forms)})
    if not completed:
        return {"strict_position": None, "semantic_position": None, "semantic_kind": "incomplete"}
    normalized = _normalize(text)
    strict = _matches(normalized, content)
    semantic_position, semantic_kind = _diagnostic(normalized, content, visible_labels)
    return {"strict_position": strict[0] if len(strict) == 1 else None,
            "semantic_position": semantic_position, "semantic_kind": semantic_kind}


def _finite_real(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _answer_score(score: Mapping) -> tuple[tuple[int, ...], float]:
    if not isinstance(score, Mapping):
        raise ValueError("Each option score must be a mapping")
    token_ids, length = score.get("answer_token_ids"), score.get("answer_tokens")
    if (not isinstance(token_ids, (list, tuple)) or type(length) is not int or length < 1
            or length != len(token_ids) or any(type(token) is not int or token < 0 for token in token_ids)):
        raise ValueError("Option score requires exact answer token IDs and matching positive length")
    total, mean = score.get("sum_log_probability"), score.get("mean_log_probability")
    if any(not _finite_real(value) or value > 0 for value in (total, mean)):
        raise ValueError("Raw answer log probabilities must be finite and nonpositive")
    if not math.isclose(mean, total / length, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("Raw mean log probability does not match the sum and answer length")
    return tuple(token_ids), float(total)


def prior_subtracted_scores(conditional: Sequence[Mapping], prior: Sequence[Mapping], *,
                            alpha: float = 1.0) -> list[dict]:
    """Subtract alpha times matched neutral-prior log likelihoods, excluding EOS.

    Both pairs must be in the same printed order with identical answer-token
    sequences across contexts. Mean calibration divides each calibrated sum
    by that option's own continuation length.
    """
    if not _finite_real(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    alpha = float(alpha)
    result = []
    for conditional_option, prior_option in zip(_two_options(conditional, "conditional scores"),
                                                _two_options(prior, "prior scores"), strict=True):
        conditional_ids, conditional_sum = _answer_score(conditional_option)
        prior_ids, prior_sum = _answer_score(prior_option)
        if conditional_ids != prior_ids:
            raise ValueError("Conditional and prior answer token IDs differ or are out of order")
        calibrated_sum = conditional_sum - alpha * prior_sum
        if not math.isfinite(calibrated_sum):
            raise ValueError("Calibrated log probability must be finite")
        result.append({"answer_token_ids": list(conditional_ids), "answer_tokens": len(conditional_ids),
                       "sum_log_probability": calibrated_sum,
                       "mean_log_probability": calibrated_sum / len(conditional_ids)})
    if result[0]["answer_token_ids"] == result[1]["answer_token_ids"]:
        raise ValueError("Calibrated options must encode distinct answer token IDs")
    return result
