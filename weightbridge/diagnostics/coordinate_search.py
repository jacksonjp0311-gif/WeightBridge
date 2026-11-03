"""Deterministic, practice-only search schedule for the toy MLP."""

import numpy as np


def disconnected_coordinate(blocked, tensor, index):
    """Return whether the proposed scalar is in the exact connectivity mask."""
    return (isinstance(tensor, str) and isinstance(index, list)
            and all(type(item) is int for item in index)
            and index in blocked.get(tensor, []))


def stratified_proposals(tensors, blocked, policy, seed):
    """Select one eligible coordinate per allowed tensor and try both signs."""
    rng = np.random.default_rng(seed)
    magnitude = min(3.0, policy["max_abs_delta"])
    for tensor in policy["allow_tensors"]:
        eligible = [list(index) for index in np.ndindex(tuple(tensors[tensor]["shape"]))
                    if not disconnected_coordinate(blocked, tensor, list(index))]
        if not eligible:
            continue
        index = eligible[int(rng.integers(len(eligible)))]
        for delta in (-magnitude, magnitude):
            yield {"tensor": tensor, "index": index, "delta": delta,
                   "reason": "Fixed coordinate search with two signed perturbations per tensor."}
