from types import SimpleNamespace

import numpy as np
import pytest

from weightbridge.adaptation import paired_update
from weightbridge.adaptation.endpoint_update import _tensor_bytes
from weightbridge.adaptation.generative_fork import fork_statistics


@pytest.fixture
def fixed_fixture(monkeypatch):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 2, bias=False).eval()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0., 0., 0.], [-1., -1., 1.]]))
    raw = _tensor_bytes(model.weight, torch)
    monkeypatch.setattr(paired_update, "_region", lambda *args: (SimpleNamespace(shape=(2, 3)), model, raw))
    features = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 0., -3.]])
    monkeypatch.setattr(paired_update, "next_token_logits",
        lambda runner, prefixes: runner.model(features[[prefix[0] for prefix in prefixes]]))
    plan = SimpleNamespace(checkpoint_sha256="parent", region=SimpleNamespace(tensor_name="weight"), max_columns=3,
        limits=SimpleNamespace(max_forward_calls=12, max_backward_calls=12,
                               max_memory_bytes=1 << 40, max_update_entries=6))
    forks = tuple(SimpleNamespace(status="semantic", replay_status="matched", checkpoint_sha256="parent",
        prompt_id=f"p{i}", prefix_ids=(i,), accepted_token_ids=(1,), actual_token_id=0) for i in range(3))
    return torch, raw, SimpleNamespace(model=model), plan, forks, features


def fit(runner, plan, forks, **kwargs):
    return paired_update.fit_paired_basis(None, runner, plan, forks, {"a": "close-short", "b": "close-long"},
        columns=(0, 1, 2), right_factor=np.array([1., 0., 0.]), item_by_prompt={"p0": "a", "p1": "a", "p2": "b"},
        batch_size=3, **kwargs)


def test_fixed_geometry_and_margin_orientation_unchanged_by_wording_multiplicity(fixed_fixture):
    torch, raw, runner, plan, forks, features = fixed_fixture
    first = fit(runner, plan, (forks[0], forks[2]))
    basis, gradients, evidence = fit(runner, plan, forks)
    np.testing.assert_array_equal(first[0].right, [1., 0., 0.])
    np.testing.assert_array_equal(first[0].right, basis.right)
    np.testing.assert_allclose(first[1], gradients)
    assert not evidence["geometry_reestimated"]
    assert evidence["resources"]["backward_calls"] == 3
    assert evidence["resources"]["forward_calls"] == 1
    assert evidence["item_prompt_ids"] == {"a": ["p0", "p1"], "b": ["p2"]}
    assert np.allclose(gradients[0], basis.target_gradients[:2].mean(axis=0))
    logits = runner.model(features[[0, 2]])
    stats = fork_statistics(logits, [(1,), (1,)])
    for index in range(2):
        derivative, = torch.autograd.grad(stats.decision_margins[index], runner.model.weight, retain_graph=True)
        np.testing.assert_allclose(derivative.numpy() @ basis.right, gradients[index])
    assert _tensor_bytes(runner.model.weight, torch) == raw


def test_fixed_basis_refuses_geometry_mismatch_and_budget(fixed_fixture):
    _, _, runner, plan, forks, _ = fixed_fixture
    with pytest.raises(ValueError, match="geometry"):
        paired_update.fit_paired_basis(None, runner, plan, forks, {"a": "short", "b": "long"},
            columns=(0,), right_factor=[2.], item_by_prompt={"p0": "a", "p1": "a", "p2": "b"})
    plan.limits.max_backward_calls = 2
    with pytest.raises(ValueError, match="compute budget"):
        fit(runner, plan, forks)
    forks[0].checkpoint_sha256 = "other"
    with pytest.raises(ValueError, match="another checkpoint"):
        fit(runner, plan, forks)


def test_interrupted_vjp_restores_requires_grad_and_parent(fixed_fixture, monkeypatch):
    torch, raw, runner, plan, forks, _ = fixed_fixture
    runner.model.weight.requires_grad_(False)

    def interrupted(*args):
        raise RuntimeError("interrupted gradient measurement")

    monkeypatch.setattr(paired_update, "next_token_logits", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        fit(runner, plan, forks)
    assert not runner.model.weight.requires_grad
    assert _tensor_bytes(runner.model.weight, torch) == raw


def test_corrected_comparison_forks_cannot_be_labeled_as_errors(fixed_fixture):
    _, _, runner, plan, forks, _ = fixed_fixture
    forks[0].replay_status = "mismatch"
    with pytest.raises(ValueError, match="replay-qualified"):
        fit(runner, plan, forks)


def test_eligibility_keeps_correct_presentations_as_obligations_and_requires_wording_pair(monkeypatch):
    from experiments.phase_iiin import run_paired

    rows, forks = [], []
    for item, correct in (("a", False), ("b", True)):
        for order in range(2):
            for wording in range(2):
                prompt = f"{item}-{order}-{wording}"
                rows.append({"id": prompt, "group": "target", "comparison_id": item, "pair_id": item + "-family",
                             "tier": "negative-close", "truth_precision": "shorter", "wording": wording,
                             "correct": correct})
                if not correct:
                    forks.append({"prompt_id": prompt, "question_id": f"{item}-{order}",
                                  "status": "semantic", "replay_status": "matched"})
    monkeypatch.setattr(run_paired, "_restore_fork", lambda row: SimpleNamespace(**row))
    contract = SimpleNamespace(required_wordings=(0, 1))
    records, eligible, _, _ = run_paired._eligible_comparisons({"rows": rows}, {"forks": forks}, contract)
    assert records[0].eligible and len(eligible["a"]) == 4
    assert not records[1].eligible
    assert len(records[1].parent_correct_prompt_ids) == 4
    forks.pop()
    records, eligible, _, _ = run_paired._eligible_comparisons({"rows": rows}, {"forks": forks}, contract)
    assert len(eligible["a"]) == 2  # A lone wording never supplies a second independent row.


def test_pinned_reference_geometry_checks_tensor_identity_before_runtime():
    from experiments.phase_iiin import run_paired

    config = run_paired.read_json(run_paired.PROTOCOL)
    archived = run_paired.read_json(run_paired.ROOT / config["fixed_geometry"]["fit_report_path"])
    record = run_paired.read_json(run_paired.ROOT / "results/phase-iii-m/factorial-v1/trial/arm-C2.json")["plan"]
    region = record["region"]
    plan = SimpleNamespace(checkpoint_sha256=config["checkpoint_sha256"], region=SimpleNamespace(
        tensor_name=region["tensor_name"], dtype=region["dtype"], shape=tuple(region["shape"])))
    columns, right, evidence = run_paired._reference_geometry(config, plan)
    assert list(columns) == archived["detail"]["columns"]
    assert right.tolist() == archived["detail"]["right_factor"]
    assert evidence["right_factor_reused_exactly"]
    plan.region.tensor_name = "different.weight"
    with pytest.raises(ValueError, match="source checkpoint"):
        run_paired._reference_geometry(config, plan)
