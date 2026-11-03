"""Create ordinary patch proposals from a small labeled toy practice split."""
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from ..core import EditRejected, _check_policy, file_hash
from .breakpoint_search import breakpoint_menu, rank_key
from .toy_mlp import ToyMLP


def _check_npz(path):
    if Path(path).suffix != ".npz":
        raise EditRejected("breakpoint-menu requires NPZ inputs")
    with ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 65_536:
            raise EditRejected("breakpoint-menu input exceeds the toy archive limit")


def propose_breakpoints(base_path, practice_path, policy, *, best_patch=False):
    allow, _, fraction, _ = _check_policy(policy)
    if not allow <= set(ToyMLP.tensor_names) or fraction < 1 / 12:
        raise EditRejected("policy must authorize a scalar edit on the toy checkpoint")
    for path in (base_path, practice_path):
        _check_npz(path)
    base_sha = file_hash(base_path)
    practice_sha = file_hash(practice_path)
    with np.load(base_path, allow_pickle=False) as base:
        tensors = {name: base[name] for name in base.files}
    if any(tensor.dtype != np.dtype("float64") for tensor in tensors.values()):
        raise EditRejected("breakpoint-menu requires a float64 toy checkpoint")
    with np.load(practice_path, allow_pickle=False) as practice:
        if set(practice.files) != {"features", "labels"}:
            raise EditRejected("practice must contain features and labels for 1 to 256 examples")
        features = practice["features"]
        if features.ndim != 2 or not 1 <= features.shape[0] <= 256:
            raise EditRejected("practice must contain features and labels for 1 to 256 examples")
        menu = breakpoint_menu(ToyMLP(tensors), features, practice["labels"], policy)
    if file_hash(base_path) != base_sha or file_hash(practice_path) != practice_sha:
        raise EditRejected("base or practice file changed during candidate generation")
    ranked = sorted(menu["candidates"], key=rank_key)
    candidates = []
    for candidate in ranked:
        patch = {
            "schema": "weightbridge.patch.v1", "base_sha256": base_sha,
            "reason": "Practice margin breakpoint search found an improving scalar edit.",
            "evidence_ids": [f"practice-sha256:{practice_sha}"],
            "changes": [{"tensor": candidate["tensor"], "index": candidate["index"],
                         "op": "add", "value": candidate["delta"]}],
        }
        candidates.append({**candidate, "patch": patch})
    if best_patch:
        if not candidates:
            raise EditRejected("no practice-improving breakpoint candidate")
        return candidates[0]["patch"]
    return {
        "schema": "weightbridge.breakpoint_menu.v1",
        "base_sha256": base_sha, "practice_sha256": practice_sha,
        "baseline_correct": menu["baseline_correct"], "practice_n": menu["practice_n"],
        "analytical_deltas_scored": menu["analytical_deltas_scored"], "candidates": candidates,
    }
