import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from weightbridge.adaptation import bic
from weightbridge.adaptation.bic_math import fit_local_quadratic, select_coefficients
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches.huggingface import apply_safetensors_patch


def test_mixed_probe_recovers_dormant_direction_with_zero_gradient():
    target = 0.2
    radius = math.sqrt(2 * target)

    def objective(z):
        return (z[0] * z[1] - target) ** 2

    linear = fit_local_quadratic(objective, objective(np.zeros(2)), radius=radius, interaction=False)
    mixed = fit_local_quadratic(objective, objective(np.zeros(2)), radius=radius, interaction=True)
    np.testing.assert_array_equal(linear.gradient, np.zeros(2))
    np.testing.assert_array_equal(select_coefficients(linear, radius=radius), np.zeros(2))
    np.testing.assert_allclose(mixed.hessian, [[0, -0.4], [-0.4, 0]], atol=1e-12)
    assert objective(select_coefficients(mixed, radius=radius)) < 1e-24
    assert (len(linear.probes), len(mixed.probes)) == (4, 8)


def test_bic_exports_the_exact_rounded_model_measured_in_practice(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    tensors = pytest.importorskip("safetensors.torch")
    names = [f"model.language_model.layers.{i}.mlp.down_proj.weight" for i in range(2)]
    inference = {"device": "cpu", "dtype": "bfloat16", "cpu_threads": 1, "batch_size": 2, "max_input_tokens": 8}
    suite = {"schema": "weightbridge.hf_edit_suite.v1", "choices": ["A", "B"],
             "target": [{"id": "a", "text": "positive", "label": "A"},
                        {"id": "b", "text": "negative", "label": "B"}],
             "sentinel": [{"id": "s", "text": "zero", "label": "A"}]}
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")
    tensors.save_file({name: torch.eye(2, dtype=torch.bfloat16) for name in names}, str(base / "model.safetensors"))

    class Tokenizer:
        pad_token_id = 3

        def encode(self, choice, **kwargs):
            return ["AB".index(choice)]

        def __call__(self, prompts, **kwargs):
            values = {"positive": 1, "negative": -1, "zero": 0}
            return {"input_ids": torch.tensor([[values[prompt]] for prompt in prompts])}

    class Model(torch.nn.Module):
        def __init__(self, weights):
            super().__init__()
            self.projections = torch.nn.ModuleList(
                [torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16) for _ in names]
            )
            self.output = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
            with torch.no_grad():
                self.output.weight.copy_(torch.eye(2))
                for name, layer in zip(names, self.projections, strict=True):
                    layer.weight.copy_(weights[name])

        def get_submodule(self, target):
            return self.projections[names.index(target + ".weight")]

        def get_output_embeddings(self):
            return self.output

        def forward(self, input_ids, use_cache, logits_to_keep):
            features = input_ids.to(torch.bfloat16).unsqueeze(-1).repeat(1, 1, 2)
            for layer in self.projections:
                features = layer(features)
            return SimpleNamespace(logits=self.output(features))

    opened = []

    def load(model_path, **kwargs):
        model = Model(tensors.load_file(str(Path(model_path) / "model.safetensors")))
        opened.append(model)
        return SimpleNamespace(model=model, tokenizer=Tokenizer(), device="cpu")

    monkeypatch.setattr(bic.HFTextRunner, "from_pretrained", load)
    patch, diagnostics = bic.propose_bic(base, suite, names, inference, interaction=True,
                                        radius=0.25, max_entries=4)
    assert diagnostics["disposition"] == "PROPOSED"
    assert diagnostics["requested_frobenius_norm"] <= 0.25 + 1e-12
    assert diagnostics["touched_entries"] == 4
    assert diagnostics["resources"]["forward_batches"] == 20
    assert diagnostics["resources"]["backward_batches"] == 2
    assert diagnostics["observed_objective"] < diagnostics["origin_objective"]
    assert patch["base_sha256"] == load_checkpoint(base).checkpoint_sha256()
    for layer in opened[0].projections:
        torch.testing.assert_close(layer.weight, torch.eye(2, dtype=torch.bfloat16))
        assert not layer._forward_hooks and not layer._forward_pre_hooks

    candidate = tmp_path / "candidate"
    policy = {"schema": "weightbridge.policy.v1", "allow_tensors": names, "max_entries": 4,
              "max_fraction": 1.0, "max_abs_delta": 1.0, "max_tensor_bytes": 1024}
    receipt = apply_safetensors_patch(base, patch, policy, candidate)
    assert receipt["changed_entries"] == diagnostics["changed_entries"]
    rows = suite["target"] + suite["sentinel"]
    scored = bic.score_binary_margins(candidate, rows, inference, jacobian_tensor=names[1])
    assert [row["margin"] for row in scored["rows"]] == [row["observed_margin"] for row in diagnostics["rows"]]
    assert all(row["embedding_jacobian_cosine"] == pytest.approx(1.0) for row in scored["rows"])
