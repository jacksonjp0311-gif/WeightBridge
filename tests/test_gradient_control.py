from types import SimpleNamespace

import numpy as np
import pytest

from weightbridge.adaptation import gradient_control


TENSOR = "model.language_model.layers.0.mlp.down_proj.weight"
SUITE = {"schema": "weightbridge.hf_edit_suite.v1", "choices": ["A", "B"],
         "target": [{"id": "a", "text": "positive", "label": "A"},
                    {"id": "b", "text": "negative", "label": "B"}],
         "sentinel": [{"id": "s", "text": "zero", "label": "A"}]}
INFERENCE = {"device": "cpu", "dtype": "float32", "cpu_threads": 1, "batch_size": 2, "max_input_tokens": 8}


def test_gradient_control_rejects_unbounded_steps_and_nonpractice_fields():
    with pytest.raises(ValueError, match="steps"):
        gradient_control.fit_gradient_control("unused", SUITE, {}, INFERENCE, steps=9)
    with pytest.raises(ValueError, match="practice-only"):
        gradient_control.fit_gradient_control("unused", {**SUITE, "holdout": []}, {}, INFERENCE)


def test_gradient_control_fits_frozen_rank_one_scope(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")

    class Tokenizer:
        pad_token_id = 3

        def encode(self, choice, **kwargs):
            return ["AB".index(choice)]

        def __call__(self, prompts, **kwargs):
            return {"input_ids": torch.tensor([[{"positive": 1, "negative": -1, "zero": 0}[p]] for p in prompts])}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(2, 2, bias=False)
            torch.nn.init.zeros_(self.projection.weight)

        def get_submodule(self, target):
            assert target + ".weight" == TENSOR
            return self.projection

        def forward(self, input_ids, **kwargs):
            features = torch.stack((input_ids.float(), torch.zeros_like(input_ids, dtype=torch.float32)), dim=-1)
            return SimpleNamespace(logits=self.projection(features))

    model = Model()
    monkeypatch.setattr(gradient_control.HFTextRunner, "from_pretrained",
                        lambda *args, **kwargs: SimpleNamespace(model=model, tokenizer=Tokenizer(), device="cpu"))
    checkpoint = SimpleNamespace(root=tmp_path, config={"model_type": "qwen3_5"},
                                 tensors={TENSOR: SimpleNamespace(shape=(2, 2))},
                                 checkpoint_sha256=lambda: "a" * 64, tensor_sha256=lambda _: "b" * 64)
    monkeypatch.setattr(gradient_control, "load_checkpoint", lambda _: checkpoint)
    reference = {"schema": "weightbridge.patch.v1", "base_sha256": "a" * 64, "tensor_sha256": {TENSOR: "b" * 64},
                 "slices": [{"tensor": TENSOR, "axis": "columns", "indices": [0], "op": "add", "values": [[0.01, 0.01]]}]}

    patch = gradient_control.fit_gradient_control(tmp_path, SUITE, reference, INFERENCE, seed=12)
    assert patch == gradient_control.fit_gradient_control(tmp_path, SUITE, reference, INFERENCE, seed=12)
    assert patch["base_sha256"] == reference["base_sha256"]
    assert patch["tensor_sha256"] == reference["tensor_sha256"]
    assert patch["slices"][0]["indices"] == [0]
    values = patch["slices"][0]["values"][0]
    assert values[0] > values[1]
    assert np.linalg.norm(values) <= np.linalg.norm(reference["slices"][0]["values"]) * (1 + 1e-14)
    assert torch.count_nonzero(model.projection.weight) == 0
    assert not model.projection._forward_hooks
