"""Text-level repetition signals for diagnostic evaluation.

The two signals mirror the thresholds documented in the public Experiment 8
mathqa evaluator: a repeated final word block, and dense repeated four-word
phrases in the final 200 words. They are observable heuristics, not semantic
proof that a model is reasoning in a loop.
"""
from __future__ import annotations

from collections import Counter


def loop_signals(text: str) -> dict[str, bool]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    words = text.split()[:-1]  # avoid treating a token-cap cut inside a word as a loop boundary
    count = len(words)
    verbatim = False
    max_period, min_repeats, min_span = 200, 3, 40
    for period in range(1, max_period + 1):
        repeats = max(min_repeats, (min_span + period - 1) // period)
        if repeats * period > count:
            break
        block = words[count - period:]
        if all(words[count - (index + 1) * period:count - index * period] == block
               for index in range(1, repeats)):
            verbatim = True
            break

    tail = text.split()[-200:]
    width = 4
    ngrams = [tuple(tail[index:index + width]) for index in range(max(0, len(tail) - width + 1))]
    if not ngrams:
        repetitive = False
    else:
        counts = Counter(ngrams)
        repeated_positions = sum(counts[gram] >= 3 for gram in ngrams)
        repetitive = repeated_positions / len(ngrams) >= 0.5
    return {"verbatim_loop_end": verbatim, "repetitive_end": repetitive}
