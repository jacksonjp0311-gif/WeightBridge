from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from weightbridge.inference.probes import activation_summary, attention_summary, zero_module_output


def test_activation_summary_returns_bounded_read_only_stats():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))
    values = torch.tensor([[1.0, 2.0]])
    output_before = model(values).clone()
    summary = activation_summary(model, ["0"], lambda: model(values), max_elements_per_module=4)
    assert summary["0"]["shape"] == [1, 2]
    assert summary["0"]["captured"] is True
    assert torch.equal(model(values), output_before)


def test_zero_ablation_restores_hook_after_scope_and_failure():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))
    values = torch.tensor([[1.0, 2.0]])
    with zero_module_output(model, "0", max_elements=2):
        assert torch.count_nonzero(model(values)) == 0
    assert torch.equal(model(values), values)
    with pytest.raises(ValueError, match="exceeds"):
        with zero_module_output(model, "0", max_elements=1):
            model(values)
    assert torch.equal(model(values), values)


class ToyAttentionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=2, num_attention_heads=2)

    def forward(self, input_ids, output_attentions=False, use_cache=True, return_dict=True):
        attention = torch.full((1, 2, input_ids.shape[1], input_ids.shape[1]),
                               1 / input_ids.shape[1], dtype=torch.float32)
        return SimpleNamespace(attentions=(attention, attention) if output_attentions else None)


def test_attention_summary_enforces_quadratic_budget_and_returns_aggregates():
    model = ToyAttentionModel()
    inputs = {"input_ids": torch.ones((1, 4), dtype=torch.long)}
    summary = attention_summary(model, inputs, [1], max_tokens=8, max_attention_elements=64)
    layer = summary["attention_layers"]["1"]
    assert layer["heads"] == 2
    assert layer["mean_entropy_nats"] == pytest.approx(1.386294, rel=1e-5)
    assert layer["mean_max_attention"] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="hard element budget"):
        attention_summary(model, inputs, [1], max_tokens=8, max_attention_elements=16)
