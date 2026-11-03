from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from weightbridge.adaptation import universal_update
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches.huggingface import apply_safetensors_patch


@pytest.mark.parametrize("method", ["direct_low_rank", "constrained", "gradient"])
def test_bound_update_rounds_then_restores_runtime_and_materializes(tmp_path, monkeypatch, method):
    torch = pytest.importorskip("torch")
    save = pytest.importorskip("safetensors.numpy").save_file
    base = tmp_path / "base.safetensors"
    save({"affine.weight": np.zeros((2, 2), np.float32)}, str(base))
    checkpoint = load_checkpoint(base)
    model = torch.nn.Module()
    model.affine = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.affine.weight.zero_()
    runner = SimpleNamespace(model=model, tokenizer=object(), device="cpu")
    adapter = SimpleNamespace(checkpoint=checkpoint)
    plan = SimpleNamespace(checkpoint_sha256=checkpoint.checkpoint_sha256(), method=method,
                           region=SimpleNamespace(tensor_name="affine.weight", module_name="affine"),
                           limits=SimpleNamespace(max_forward_calls=12, max_backward_calls=8,
                                                  max_memory_bytes=16 * 1024**3, max_update_entries=4,
                                                  max_update_norm=.25, max_rank=1, max_candidates=2))
    rows = [{"prompt": "first", "answers": ("true", "false"), "preserve": False},
            {"prompt": "second", "answers": ("true", "false"), "preserve": True}]
    monkeypatch.setattr(universal_update, "prepare_continuations", lambda *args, **kwargs: tuple(range(4)))

    def score(_runner, batch):
        margins = (model.affine(torch.tensor([1., 0.]))[0] - .08,
                   model.affine(torch.tensor([0., 1.]))[0] + 1.)
        return tuple((margins[index // 2] if index % 2 == 0 else torch.tensor(0.)).reshape(1)
                     for index in batch)

    monkeypatch.setattr(universal_update, "score_batch", score)
    patch, stats = universal_update.fit_candidate(adapter, runner, plan, rows)
    assert patch is not None
    assert torch.count_nonzero(model.affine.weight) == 0 and model.affine.weight.requires_grad
    policy = {"schema": "weightbridge.policy.v1", "allow_tensors": ["affine.weight"],
              "max_entries": 4, "max_fraction": 1., "max_abs_delta": .25}
    output = Path(tmp_path / "candidate.safetensors")
    receipt = apply_safetensors_patch(base, patch, policy, output)
    assert receipt["candidate_sha256"] == load_checkpoint(output).checkpoint_sha256()
    assert load_checkpoint(base).checkpoint_sha256() == plan.checkpoint_sha256
    if method != "direct_low_rank":
        assert stats["selected"]["corrections"] == 1
        assert stats["selected"]["preservation_stable"]
