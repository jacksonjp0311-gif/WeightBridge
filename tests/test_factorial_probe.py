import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from weightbridge.adaptation import factorial_probe
from weightbridge.adaptation.endpoint_update import _tensor_bytes
from weightbridge.patches.huggingface import _to_float64


@pytest.fixture
def stored_space():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.5, -.25], [.75, .125]], dtype=torch.bfloat16))
    parent = _tensor_bytes(model.weight, torch)
    source = _to_float64(parent, "BF16").reshape((2, 2))
    space = factorial_probe.StoredAffineSpace(model, (0,), np.array([1.]), "BF16", source, parent, .025)
    return torch, model, parent, space


def test_preview_is_exact_bf16_and_rejects_a_tampered_stored_identity(stored_space):
    torch, model, parent, space = stored_space
    noop = space.preview(np.array([1e-6, -1e-6]))
    assert noop["changed_entries"] == 0 and noop["realized_norm"] == 0.
    assert noop["tensor_sha256"] == hashlib.sha256(parent).hexdigest()
    changed = space.preview(np.array([.01, -.01]))
    assert 0 < changed["realized_norm"] <= .025
    assert changed["changed_entries"] == 2
    assert _tensor_bytes(model.weight, torch) == parent
    changed["tensor_sha256"] = "different"
    with pytest.raises(ValueError, match="stored identity"):
        space.probe_arguments(changed)
    assert _tensor_bytes(model.weight, torch) == parent


def test_failed_numerical_remeasurement_restores_parent_tensor_exactly(stored_space, monkeypatch):
    torch, model, parent, space = stored_space
    preview = space.preview(np.array([.01, -.01]))

    def failed_measure(*args, **kwargs):
        assert hashlib.sha256(_tensor_bytes(model.weight, torch)).hexdigest() == preview["tensor_sha256"]
        raise RuntimeError("Measured numerical probe interrupted")

    monkeypatch.setattr(factorial_probe, "measure_update_margins", failed_measure)
    with pytest.raises(RuntimeError, match="interrupted"):
        space.margins(SimpleNamespace(model=model), object(), (), preview, batch_size=8, max_memory_bytes=4096)
    assert _tensor_bytes(model.weight, torch) == parent
    with pytest.raises(ValueError, match="preview budget"):
        space.preview(np.array([.1, -.1]))
