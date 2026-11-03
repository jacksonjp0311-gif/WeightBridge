"""Build a small menu of one-scalar toy MLP edits from practice margins."""
from __future__ import annotations

import numpy as np

from .coordinate_search import disconnected_coordinate


TIE_MARGIN = 1e-10


def _interior_trials(roots, limit):
    roots = roots[np.isfinite(roots) & (roots > -limit) & (roots < limit)]
    edges = np.unique(np.r_[-limit, roots, limit])
    midpoints = edges[:-1] + (edges[1:] - edges[:-1]) / 2
    return np.unique(np.r_[-limit, midpoints, limit])


def rank_key(candidate):
    return (-candidate["practice_correct"], -candidate["minimum_signed_margin"],
            abs(candidate["delta"]), candidate["tensor"], candidate["index"], candidate["delta"])


def breakpoint_menu(model, features, labels, policy):
    """Return at most one improving candidate per permitted tensor.

    For each scalar, the logit margin is linear or a shifted tanh in the
    scalar delta. Evaluate interval interiors and cap endpoints, avoiding
    ambiguous near ties. The broker independently scores the selected patch.
    """
    x, y = np.asarray(features, dtype=np.float64), np.asarray(labels)
    if (x.ndim != 2 or x.shape[1] != 2 or y.shape != (len(x),)
            or not len(y) or not np.isfinite(x).all() or not np.isin(y, (0, 1)).all()):
        raise ValueError("invalid toy practice arrays")
    limit = float(policy["max_abs_delta"])
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("invalid scalar delta cap")
    tensors = model.tensors
    z = x @ tensors["input_weight"] + tensors["hidden_bias"]
    hidden = np.tanh(z)
    out = tensors["output_weight"]
    logits = hidden @ out + tensors["output_bias"]
    margin = logits[:, 1] - logits[:, 0]
    correct = int(np.count_nonzero(logits.argmax(axis=1) == y))
    signed = 2 * y.astype(np.int64) - 1
    blocked = model.logit_connectivity()["single_edits_with_no_logit_path"]
    menu = []
    screened = 0
    for tensor in policy["allow_tensors"]:
        best = None
        for index in np.ndindex(tensors[tensor].shape):
            if disconnected_coordinate(blocked, tensor, list(index)):
                continue
            if tensor in ("input_weight", "hidden_bias"):
                unit = index[-1]
                scale = float(out[unit, 1] - out[unit, 0])
                if scale == 0:
                    continue
                other = margin - scale * hidden[:, unit]
                slope = x[:, index[0]] if tensor == "input_weight" else np.ones(len(x))
                target = -other / scale
                has_root = (slope != 0) & (np.abs(target) < 1)
                roots = (np.arctanh(target[has_root]) - z[has_root, unit]) / slope[has_root]
                deltas = _interior_trials(roots, limit)
                margins = other + scale * np.tanh(z[:, unit] + deltas[:, None] * slope)
            else:
                sign = 1 if index[-1] == 1 else -1
                slope = sign * (hidden[:, index[0]] if tensor == "output_weight" else np.ones(len(x)))
                active = slope != 0
                deltas = _interior_trials(-margin[active] / slope[active], limit)
                margins = margin + deltas[:, None] * slope
            screened += len(deltas)
            scores = np.count_nonzero((margins > 0) == y, axis=1)
            for k in np.flatnonzero((scores > correct) & (np.min(np.abs(margins), axis=1) > TIE_MARGIN)):
                candidate = {
                    "tensor": tensor, "index": list(index), "delta": float(deltas[k]),
                    "practice_correct": int(scores[k]),
                    "minimum_signed_margin": float(np.min(margins[k] * signed)),
                }
                if best is None or rank_key(candidate) < rank_key(best):
                    best = candidate
        if best is not None:
            menu.append(best)
    return {"baseline_correct": correct, "practice_n": len(y),
            "analytical_deltas_scored": screened, "candidates": menu}
