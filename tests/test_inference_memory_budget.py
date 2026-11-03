from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import weightbridge.inference.huggingface as inference
from weightbridge.inference import HFTextRunner, InferenceConfig


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return "prompt"

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1]], dtype=torch.long)}

    def decode(self, tokens, **kwargs):
        return "partial"


class TinyModel:
    def parameters(self):
        return iter(())

    def generate(self, **kwargs):
        assert kwargs["stopping_criteria"](torch.tensor([[1]], dtype=torch.long), None)
        return torch.tensor([[1, 3]], dtype=torch.long)


def test_generation_stops_when_host_rss_budget_is_exceeded(monkeypatch):
    monkeypatch.setattr(inference, "_current_rss_bytes", lambda: 512 * 1024**2)
    runner = HFTextRunner(TinyModel(), TinyTokenizer(), device="cpu", model_id="tiny")
    result = runner.generate("hello", InferenceConfig(max_new_tokens=8, max_rss_bytes=256 * 1024**2))
    assert result.finish_reason == "memory_limit"
    assert result.generated_tokens == 1
    assert result.configuration["max_rss_bytes"] == 256 * 1024**2
