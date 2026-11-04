from types import SimpleNamespace

import numpy as np
import pytest

from weightbridge.adaptation import factorial_update
from weightbridge.adaptation.endpoint_update import _tensor_bytes
from weightbridge.adaptation.generative_fork import fork_statistics


@pytest.fixture
def factorial_fixture(monkeypatch):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 2, bias=False).eval()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0., 0., 0.], [-1., -1., 1.]]))
    raw = _tensor_bytes(model.weight, torch)
    monkeypatch.setattr(factorial_update, "_region",
                        lambda *args: (SimpleNamespace(shape=(2, 3)), model, raw))
    features = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 0., -3.]])
    monkeypatch.setattr(factorial_update, "next_token_logits",
                        lambda runner, prefixes: runner.model(features[[prefix[0] for prefix in prefixes]]))
    runner = SimpleNamespace(model=model)
    plan = SimpleNamespace(checkpoint_sha256="parent", region=SimpleNamespace(tensor_name="weight"),
                           max_columns=3, limits=SimpleNamespace(max_forward_calls=12, max_backward_calls=12,
                                                               max_memory_bytes=1 << 40, max_rank=1,
                                                               max_update_entries=6))
    forks = tuple(SimpleNamespace(status="semantic", replay_status="matched", checkpoint_sha256="parent",
                                  prompt_id=f"p{i}", question_id=f"order-{i}", prefix_ids=(i,),
                                  accepted_token_ids=(1,), actual_token_id=0) for i in range(3))
    return torch, raw, runner, plan, forks, features


def test_factorial_objectives_use_full_vocabulary_and_set_equivalence():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[1., 1., 0.]], dtype=torch.float64, requires_grad=True)
    stats = fork_statistics(logits, [(1, 0)])
    expected_ce = torch.logsumexp(logits[0], 0) - torch.logsumexp(logits[0, :2], 0)
    assert stats.cross_entropy.item() == pytest.approx(expected_ce.item())
    improving_ce, = torch.autograd.grad(-stats.cross_entropy.sum(), logits, retain_graph=True)
    improving_margin, = torch.autograd.grad(stats.decision_margins.sum(), logits)
    assert np.allclose(improving_ce.detach().numpy(), [[.0776812017, .0776812017, -.1553624035]])
    assert np.array_equal(improving_margin.numpy(), [[1., 0., -1.]])
    # Native max uses the first vocabulary index on ties, regardless of set ordering.
    second = fork_statistics(logits.detach(), [(0, 1)])
    assert second.cross_entropy.item() == stats.cross_entropy.item()
    assert second.decision_margins.item() == 1.


def test_equal_comparison_basis_is_invariant_to_duplicate_wording(factorial_fixture):
    torch, raw, runner, plan, forks, features = factorial_fixture
    groups = {"a": "short", "b": "long"}
    items = {"p0": "a", "p1": "a", "p2": "b"}
    first = factorial_update.fit_factorial_basis(None, runner, plan, (forks[0], forks[2]), groups,
                                                 item_by_prompt=items, batch_size=3)
    repeated = factorial_update.fit_factorial_basis(None, runner, plan, forks, groups,
                                                    item_by_prompt=items, batch_size=3)
    basis, ce_rows, margin_rows, receipt = repeated
    assert np.allclose(first[0].right, basis.right)
    assert np.allclose(first[1], ce_rows)
    assert np.allclose(first[2], margin_rows)
    assert receipt["item_weight"] == .5
    assert receipt["resources"]["forward_calls"] == 2
    assert receipt["resources"]["backward_calls"] == 7
    assert receipt["item_prompt_ids"] == {"a": ["p0", "p1"], "b": ["p2"]}
    assert np.allclose(margin_rows[0], basis.target_gradients[:2].mean(axis=0))
    assert np.array_equal(margin_rows[1], basis.target_gradients[2])
    assert receipt["projected_gradients"]["CE"] == ce_rows.tolist()
    assert receipt["projected_gradients"]["MARGIN"] == margin_rows.tolist()
    assert receipt["full_gradient_agreement"]["exact"]
    assert np.linalg.norm(basis.right) == pytest.approx(1.)
    assert _tensor_bytes(runner.model.weight, torch) == raw
    # Check actual objective derivatives against the shared right factor.
    logits = runner.model(features[[0, 2]])
    stats = fork_statistics(logits, [(1,), (1,)])
    for index in range(2):
        ce, = torch.autograd.grad(-stats.cross_entropy[index], runner.model.weight, retain_graph=True)
        margin, = torch.autograd.grad(stats.decision_margins[index], runner.model.weight, retain_graph=True)
        assert np.allclose(ce.detach().numpy()[:, basis.columns] @ basis.right, ce_rows[index])
        assert np.allclose(margin.detach().numpy()[:, basis.columns] @ basis.right, margin_rows[index])


def test_budget_and_checkpoint_refusal_precede_runtime(factorial_fixture, monkeypatch):
    _, _, runner, plan, forks, _ = factorial_fixture

    def no_forward(*args):
        raise AssertionError("Invalid evidence must be rejected before runtime")

    monkeypatch.setattr(factorial_update, "next_token_logits", no_forward)
    with pytest.raises(ValueError, match="named group"):
        factorial_update.fit_factorial_basis(None, runner, plan, forks, {})
    plan.limits.max_backward_calls = 6
    with pytest.raises(ValueError, match="compute budget"):
        factorial_update.fit_factorial_basis(None, runner, plan, forks, {"a": "short", "b": "long"},
                                             item_by_prompt={"p0": "a", "p1": "a", "p2": "b"})
    forks[0].checkpoint_sha256 = "different"
    with pytest.raises(ValueError, match="another checkpoint"):
        factorial_update.fit_factorial_basis(None, runner, plan, forks, {})


def test_failed_gradient_replay_restores_flags_and_parent(factorial_fixture, monkeypatch):
    torch, raw, runner, plan, forks, _ = factorial_fixture
    runner.model.weight.requires_grad_(False)
    calls = 0
    original = factorial_update.next_token_logits

    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Interrupted second pass")
        return original(*args)

    monkeypatch.setattr(factorial_update, "next_token_logits", interrupted)
    with pytest.raises(RuntimeError, match="Interrupted"):
        factorial_update.fit_factorial_basis(None, runner, plan, forks, {"a": "short", "b": "long"},
                                             item_by_prompt={"p0": "a", "p1": "a", "p2": "b"}, batch_size=3)
    assert not runner.model.weight.requires_grad
    assert _tensor_bytes(runner.model.weight, torch) == raw


def test_fork_token_subsets_are_explicit_and_keep_full_accepted_probability(monkeypatch):
    torch = pytest.importorskip("torch")
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [0] if text != "\n" else [1],
                                decode=lambda tokens, **kwargs: "0" if tokens == [0] else "\n", eos_token_id=2)
    runner = SimpleNamespace(tokenizer=tokenizer)
    logits = torch.tensor([[0., 1., 2., 3.]])
    monkeypatch.setattr(factorial_update, "next_token_logits", lambda *args: logits)
    forks = (SimpleNamespace(prompt_id="p", prefix_ids=(3,), accepted_token_ids=(1, 2), actual_token_id=3),)
    result = factorial_update.measure_fork_token_classes(runner, forks, max_forward_calls=1,
                                                        max_memory_bytes=1 << 40)
    probabilities = logits.double().softmax(-1)[0]
    row = result["rows"][0]
    assert row["accepted_probability"] == pytest.approx(float(probabilities[1:3].sum()))
    assert row["termination_subset_probability"] == pytest.approx(float(probabilities[1:3].sum()))
    assert row["numeric_subset_probability"] == pytest.approx(float(probabilities[0]))
    assert result["resources"]["forward_calls"] == 1


def test_multitoken_termination_fragments_are_not_counted_as_complete(monkeypatch):
    torch = pytest.importorskip("torch")
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [0, 1],
                                decode=lambda tokens, **kwargs: "fragment", eos_token_id=2)
    monkeypatch.setattr(factorial_update, "next_token_logits", lambda *args: torch.tensor([[0., 1., 2., 3.]]))
    fork = SimpleNamespace(prompt_id="p", prefix_ids=(3,), accepted_token_ids=(2,), actual_token_id=3)
    result = factorial_update.measure_fork_token_classes(SimpleNamespace(tokenizer=tokenizer), (fork,),
                                                         max_forward_calls=1, max_memory_bytes=1 << 40)
    assert result["numeric_token_ids"] == []
    assert result["termination_token_ids"] == [2]
    assert result["rows"][0]["numeric_subset_probability"] == 0.
