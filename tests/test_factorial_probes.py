from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.phase_iiim.runtime_termination import termination_bias, _token_set
from weightbridge.adaptation import factorial_probe
from weightbridge.adaptation.endpoint_update import _tensor_bytes


def _space():
    module = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[1., -.5], [.25, -1.]], dtype=torch.bfloat16))
    raw = _tensor_bytes(module.weight, torch)
    source = module.weight.detach().float().numpy().astype(np.float64)
    return factorial_probe.StoredAffineSpace(module, (0, 1), np.array([1., 0.]),
                                             "BF16", source, raw, .025)


def test_stored_preview_records_bfloat16_noop_and_refuses_spoofed_live_payload():
    space = _space()
    original = _tensor_bytes(space.module.weight, torch)
    noop = space.preview(np.array([1e-8, 1e-8]))
    assert noop["changed_entries"] == 0 and noop["realized_norm"] == 0
    np.testing.assert_array_equal(noop["delta"], np.zeros((2, 2)))
    assert _tensor_bytes(space.module.weight, torch) == original
    with pytest.raises(ValueError, match="stored preview budget"):
        space.preview([.1, 0])
    actual = space.preview([.01, .01])
    assert actual["changed_entries"] > 0 and actual["realized_norm"] <= .025
    with pytest.raises(ValueError, match="stored identity"):
        space.probe_arguments({**actual, "tensor_sha256": "wrong"})
    with pytest.raises(ValueError, match="stored update policy"):
        space.probe_arguments({**actual, "stored": space.source + 1})


def test_live_stored_probe_restores_exact_parent_after_measurement_failure(monkeypatch):
    space = _space()
    original = _tensor_bytes(space.module.weight, torch)
    preview = space.preview([.01, .01])

    def fail(*_args, **_kwargs):
        assert _tensor_bytes(space.module.weight, torch) != original
        raise RuntimeError("deliberate measurement failure")

    monkeypatch.setattr(factorial_probe, "measure_update_margins", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        space.margins(None, None, (), preview, batch_size=2, max_memory_bytes=1 << 20)
    assert _tensor_bytes(space.module.weight, torch) == original


class _Runtime(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((2, 4)))
        self.last_output = None

    def forward(self):
        self.last_output = SimpleNamespace(logits=self.weight.unsqueeze(0))
        return self.last_output


def test_runtime_bias_clones_logits_and_removes_hook_on_exception_without_parameter_changes():
    model = _Runtime()
    original = model.weight.detach().clone()
    with pytest.raises(RuntimeError, match="decode failed"):
        with termination_bias(model, (1, 3), .125) as counts:
            output = model()
            assert output is not model.last_output
            torch.testing.assert_close(model.last_output.logits, original.unsqueeze(0))
            torch.testing.assert_close(output.logits[..., [1, 3]], torch.full((1, 2, 2), 1.125))
            assert counts["forward_calls"] == 1 and counts["logit_positions"] == 2
            raise RuntimeError("decode failed")
    assert not model._forward_hooks
    torch.testing.assert_close(model.weight, original)
    torch.testing.assert_close(model().logits, original.unsqueeze(0))
    with termination_bias(model, (1, 3), -.125):
        torch.testing.assert_close(model().logits[..., [1, 3]], torch.full((1, 2, 2), .875))
    assert not model._forward_hooks


def test_unsupported_runtime_logits_and_invalid_native_termination_tokens_are_explicit():
    model = torch.nn.Identity()
    with pytest.raises(ValueError, match="native next-token logits"):
        with termination_bias(model, (1,), .125):
            model(torch.ones(2, 3))
    assert not model._forward_hooks
    tokenizer = SimpleNamespace(eos_token_id=4, encode=lambda *_args, **_kwargs: [2, 3],
                                decode=lambda ids, **_kwargs: str(ids))
    tokens, evidence = _token_set(tokenizer)
    assert tokens == (4,) and not evidence["newline_bias_applied"]
    tokenizer.encode = lambda *_args, **_kwargs: [2]
    assert _token_set(tokenizer)[0] == (4, 2)
    tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="native EOS"):
        _token_set(tokenizer)


def test_runtime_hook_preserves_transformers_output_mapping_and_attribute_semantics():
    outputs = pytest.importorskip("transformers.modeling_outputs")

    class Native(torch.nn.Module):
        def forward(self):
            return outputs.CausalLMOutput(logits=torch.ones((1, 1, 5), dtype=torch.bfloat16))

    model = Native()
    with termination_bias(model, (2, 4), .125):
        result = model()
        assert result.logits[0, 0, 2].item() == 1.125
        assert result["logits"][0, 0, 4].item() == 1.125
    assert not model._forward_hooks
