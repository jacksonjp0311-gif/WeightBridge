from types import SimpleNamespace

import numpy as np
import pytest

from weightbridge.adaptation import semantic_update
from weightbridge.adaptation.endpoint_update import _tensor_bytes


@pytest.fixture
def semantic_fixture(monkeypatch):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 2, bias=False).eval()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0., 0., 0.], [-1., -1., 1.]]))
    raw = _tensor_bytes(model.weight, torch)
    shapes = SimpleNamespace(shape=(2, 3))
    monkeypatch.setattr(semantic_update, "_region", lambda *args: (shapes, model, raw))
    features = torch.tensor([[1., 0., 0.], [0., 2., 0.], [0., 0., -3.]])
    monkeypatch.setattr(semantic_update, "next_token_logits",
                        lambda runner, prefixes: runner.model(features[[prefix[0] for prefix in prefixes]]))
    runner = SimpleNamespace(model=model)
    plan = SimpleNamespace(checkpoint_sha256="parent", region=SimpleNamespace(tensor_name="weight"),
                           max_columns=3, limits=SimpleNamespace(max_forward_calls=12, max_backward_calls=12,
                                                               max_memory_bytes=1 << 40, max_rank=1,
                                                               max_update_entries=6))
    forks = tuple(SimpleNamespace(status="semantic", replay_status="matched", checkpoint_sha256="parent",
                                  prompt_id=f"p{i}", question_id=f"order-{i}", prefix_ids=(i,),
                                  accepted_token_ids=(1,), actual_token_id=0) for i in range(3))
    return torch, raw, runner, plan, forks


def test_fit_keeps_right_factor_shared_and_makes_presentations_one_vote(semantic_fixture):
    torch, raw, runner, plan, forks = semantic_fixture
    groups = {"a": "short", "b": "long"}
    items = {"p0": "a", "p1": "a", "p2": "b"}
    basis, gradients, receipt = semantic_update.fit_semantic_basis(
        None, runner, plan, forks, groups, item_by_prompt=items, batch_size=3,
    )
    assert receipt["item_prompt_ids"] == {"a": ["p0", "p1"], "b": ["p2"]}
    assert receipt["resources"]["forward_calls"] == 2
    assert receipt["resources"]["backward_calls"] == 5
    assert np.allclose(gradients[0], basis.target_gradients[:2].mean(axis=0))
    assert np.array_equal(gradients[1], basis.target_gradients[2])
    assert np.linalg.norm(basis.right) == pytest.approx(1.)
    ce = receipt["ce_item_gradient_agreement"]
    assert ce["exact"]
    assert ce["item_cosine"][0][1] == pytest.approx(0.)
    assert np.array_equal(basis.target_margins, [-1., -2., -3.])
    assert _tensor_bytes(runner.model.weight, torch) == raw


def test_budget_and_item_mapping_fail_before_forward(semantic_fixture, monkeypatch):
    _, _, runner, plan, forks = semantic_fixture

    def no_forward(*args):
        raise AssertionError("Validation must happen before evaluating the model")

    monkeypatch.setattr(semantic_update, "next_token_logits", no_forward)
    with pytest.raises(ValueError, match="named group"):
        semantic_update.fit_semantic_basis(None, runner, plan, forks, {})
    plan.limits.max_backward_calls = 4
    with pytest.raises(ValueError, match="compute budget"):
        semantic_update.fit_semantic_basis(None, runner, plan, forks, {"a": "x", "b": "y"},
                                          item_by_prompt={"p0": "a", "p1": "a", "p2": "b"})


def test_option_scores_keep_tokenization_and_greedy_endpoint_separate(monkeypatch):
    runner = SimpleNamespace(tokenizer=object())
    contract = SimpleNamespace(prompt_representation="native", max_input_tokens=40, batch_size=2,
                               question=lambda row: SimpleNamespace(answer="-1.2", allowed_answers=("-1.2", "-1.24")),
                               parse=lambda text, stop: SimpleNamespace(answer=text))
    row = {"id": "p", "prompt": "frozen prefix", "operands": ["-1.2", "-1.24"],
           "exact": "-1.2", "valid": True}

    def prepare(tokenizer, prompts, options, *, max_input_tokens):
        assert tokenizer is runner.tokenizer
        assert prompts == ["frozen prefix"]
        assert options == [("-1.2", "-1.24")]
        return (SimpleNamespace(answer_ids=(4, 1)), SimpleNamespace(answer_ids=(4, 1, 9)))

    def score(runner, rows, *, batch_size):
        return [{"sum_log_probability": -1.5, "mean_log_probability": -.75},
                {"sum_log_probability": -1.8, "mean_log_probability": -.6}], {"padded_input_token_positions": 16}

    monkeypatch.setattr(semantic_update, "prepare_continuations", prepare)
    monkeypatch.setattr(semantic_update, "score_continuations", score)
    preferences, receipt = semantic_update.score_teacher_forced_options(
        runner, [row], contract, max_forward_calls=1, max_memory_bytes=1 << 40,
    )
    assert preferences == {"p": "-1.2"}
    [result] = receipt["rows"]
    assert result["sum_matches_greedy"] and not result["mean_matches_greedy"]
    assert result["options"][1]["answer_token_ids"] == [4, 1, 9]
    assert receipt["resources"]["forward_calls"] == 1
