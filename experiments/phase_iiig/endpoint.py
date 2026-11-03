"""Strict first-line integer measurement for explicit bare answer prefixes."""
from __future__ import annotations

from copy import copy
import re
import time

from weightbridge.adaptation.content_scoring import generate_responses, prepare_continuations
from weightbridge.provenance import peak_rss_bytes


_INTEGER = re.compile(r"[ \t]*(-?(?:0|[1-9][0-9]{0,8}))[ \t]*")


class BareTokenizer:
    """Keep the native token IDs and EOS while using the supplied text verbatim."""

    chat_template = None

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


def bare_runner(runner):
    view = copy(runner)
    view.tokenizer = BareTokenizer(runner.tokenizer)
    return view


def parse_integer(text, stop_reason):
    """Require EOS or an observed line boundary before scoring the integer."""
    first_line, newline, _ = text.partition("\n")
    if not newline and stop_reason != "eos":
        return None
    if newline:
        first_line = first_line.removesuffix("\r")
    match = _INTEGER.fullmatch(first_line)
    return str(int(match[1])) if match else None


def answer_pair(runner, prompt, correct, incorrect, *, max_input_tokens):
    """Find one common, token-boundary-safe surface for a contrast pair."""
    if correct == incorrect:
        raise ValueError("Adaptation contrast requires distinct integers")
    tokenizer = BareTokenizer(runner.tokenizer)
    preferred = ("", " ") if prompt[-1:].isspace() else (" ", "")
    for spacing in preferred:
        answers = tuple(spacing + answer for answer in (correct, incorrect))
        try:
            prepare_continuations(tokenizer, [prompt], [answers], max_input_tokens=max_input_tokens)
        except ValueError:
            continue
        return answers
    raise ValueError("Integer contrast changes the frozen prompt boundary")


def measure(runner, rows, settings):
    wall, cpu = time.perf_counter(), time.process_time()
    outputs, resources = generate_responses(
        bare_runner(runner), [row["prompt"] for row in rows],
        max_input_tokens=settings["max_input_tokens"], max_new_tokens=settings["max_new_tokens"],
        batch_size=settings["batch_size"],
    )
    measured = []
    for row, output in zip(rows, outputs, strict=True):
        exact = parse_integer(output["text"], output["stop_reason"])
        measured.append({**row, **output, "exact": exact})
    return {"rows": measured, "resources": {**resources,
            "wall_seconds": time.perf_counter() - wall,
            "cpu_seconds": time.process_time() - cpu, "peak_rss_bytes": peak_rss_bytes()}}
