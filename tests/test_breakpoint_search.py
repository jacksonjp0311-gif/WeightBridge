import json
import subprocess
import sys

import numpy as np

from weightbridge.diagnostics.breakpoint_search import breakpoint_menu, rank_key
from weightbridge.diagnostics.toy_mlp import ToyMLP


def model_with_slope(slope, bias=(0, 0)):
    return ToyMLP({"input_weight": [[slope, 0], [0, 2]], "hidden_bias": [-1, -1],
                   "output_weight": [[-3, 3], [0, 0]], "output_bias": bias})


def test_breakpoint_menu_repairs_reversed_input_with_practice_only():
    model = model_with_slope(-2)
    x = np.array([[0.1, 0.1], [0.4, 0.9], [0.6, 0.1], [0.9, 0.9]])
    y = np.array([0, 0, 1, 1])
    policy = {"allow_tensors": list(model.tensor_names), "max_abs_delta": 8}
    menu = breakpoint_menu(model, x, y, policy)
    best = min(menu["candidates"], key=rank_key)
    assert (best["tensor"], best["index"], best["practice_correct"]) == ("input_weight", [0, 0], 4)
    for candidate in menu["candidates"]:
        weights = {name: tensor.copy() for name, tensor in model.tensors.items()}
        weights[candidate["tensor"]][tuple(candidate["index"])] += candidate["delta"]
        correct = np.count_nonzero(ToyMLP(weights).forward(x).argmax(axis=1) == y)
        assert correct == candidate["practice_correct"]
        assert abs(candidate["delta"]) <= 8
    assert model.tensors["input_weight"][0, 0] == -2


def test_breakpoint_menu_checks_linear_output_bias_and_respects_allowlist():
    model = model_with_slope(2, (0, 4.5))
    x = np.array([[0.1, 0.5], [0.4, 0.5], [0.6, 0.5], [0.9, 0.5]])
    y = np.array([0, 0, 1, 1])
    menu = breakpoint_menu(model, x, y, {"allow_tensors": ["output_bias"], "max_abs_delta": 8})
    assert menu["baseline_correct"] < 4
    assert len(menu["candidates"]) == 1
    assert menu["candidates"][0]["practice_correct"] == 4
    assert menu["candidates"][0]["tensor"] == "output_bias"
    clean = breakpoint_menu(model_with_slope(2), x, y,
                           {"allow_tensors": ["output_bias"], "max_abs_delta": 8})
    assert clean["baseline_correct"] == 4 and clean["candidates"] == []


def test_breakpoint_cli_emits_a_patch_the_host_can_apply(tmp_path):
    base, practice, policy = (tmp_path / name for name in ("base.npz", "practice.npz", "policy.json"))
    patch, candidate = tmp_path / "patch.json", tmp_path / "candidate.npz"
    model_with_slope(-2).save(base)
    x = np.array([[0.1, 0.1], [0.4, 0.9], [0.6, 0.1], [0.9, 0.9]])
    y = np.array([0, 0, 1, 1])
    np.savez_compressed(practice, features=x, labels=y)
    policy.write_text(json.dumps({"schema": "weightbridge.policy.v1",
                                  "allow_tensors": ["input_weight"], "max_entries": 1,
                                  "max_fraction": 1 / 12, "max_abs_delta": 8}), encoding="utf-8")
    cli = [sys.executable, "-m", "weightbridge"]
    subprocess.run(cli + ["breakpoint-menu", "--base", str(base), "--practice", str(practice),
                          "--policy", str(policy), "--best-patch", "--out", str(patch)],
                   check=True, capture_output=True, timeout=30)
    subprocess.run(cli + ["apply", "--base", str(base), "--patch", str(patch),
                          "--policy", str(policy), "--out", str(candidate)],
                   check=True, capture_output=True, timeout=30)
    np.testing.assert_array_equal(ToyMLP.load(candidate).forward(x).argmax(axis=1), y)
    assert ToyMLP.load(base).tensors["input_weight"][0, 0] == -2
