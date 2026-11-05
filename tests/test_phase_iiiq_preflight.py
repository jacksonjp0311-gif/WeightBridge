"""Small inventory fixtures; no model capability claim follows from them."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from experiments.phase_iiiq_baseline.preflight import text_affine_inventory


class FixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList()
        for _ in range(2):
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            layer.mlp.down_proj = torch.nn.Linear(3, 3, bias=False)
            layer.self_attn = torch.nn.Module()
            layer.self_attn.q_proj = torch.nn.Linear(3, 3, bias=False)
            self.model.language_model.layers.append(layer)
        self.model.visual = torch.nn.Sequential(torch.nn.Linear(3, 3))
        self.lm_head = torch.nn.Linear(3, 3, bias=False)
        self.embedding = torch.nn.Embedding(3, 3)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head


def test_actual_linear_type_and_explicit_text_roles_define_inventory():
    model = FixtureModel()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    result = text_affine_inventory(model)
    assert result["allowed_count"] == 4
    assert result["counts_by_role"] == {"mlp": 2, "self_attn": 2}
    assert result["counts_by_layer"] == {"0": 2, "1": 2}
    assert not any("visual" in value or "lm_head" in value for value in result["explicit_module_allowlist"])
    assert result["discovery_is_training_or_efficacy"] is False
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_tied_embedding_or_output_head_alias_is_never_an_adaptation_target():
    model = FixtureModel()
    model.model.language_model.layers[0].self_attn.q_proj.weight = model.embedding.weight
    model.model.language_model.layers[1].mlp.down_proj.weight = model.lm_head.weight
    result = text_affine_inventory(model)
    assert result["allowed_count"] == 2
    tied = [row for row in result["excluded"] if row["reason"] == "embedding_or_head_weight_alias"]
    assert len(tied) == 2


def test_extra_nested_or_unrecognized_modules_do_not_silently_expand_allowlist():
    model = FixtureModel()
    layer = model.model.language_model.layers[0]
    layer.unrecognized = torch.nn.Linear(3, 3)
    layer.mlp.nested = torch.nn.Sequential(torch.nn.Linear(3, 3))
    layer.linear_attn = torch.nn.Module()
    layer.linear_attn.out_proj = torch.nn.Linear(3, 3)
    result = text_affine_inventory(model)
    assert result["allowed_count"] == 5
    assert result["counts_by_role"]["linear_attn"] == 1
    assert not any("unrecognized" in name or "nested" in name for name in result["explicit_module_allowlist"])
