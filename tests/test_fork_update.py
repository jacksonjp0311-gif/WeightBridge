"""Exercise the numerical fork backend in stored precision on a causal fixture."""
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from weightbridge.adaptation.content_scoring import Continuation
from weightbridge.adaptation.endpoint_update import _tensor_bytes
from weightbridge.adaptation.fork_update import (
    AffineUpdate, _rank_one, fit_lora, prepare_probe_reference, probe_update, scan_fork_regions,
)
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches.huggingface import apply_safetensors_patch


torch = pytest.importorskip("torch")


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.affine = torch.nn.Linear(6, 6, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.affine.weight.fill_(.125)
            self.affine.weight[2, 1] = .50
            self.affine.weight[3, 1] = .75
            self.affine.weight[2, 4] = .75
            self.affine.weight[0, 2] = .75

    def forward(self, input_ids, attention_mask, use_cache=False, logits_to_keep=1):
        states = torch.nn.functional.one_hot(input_ids[:, -logits_to_keep:], num_classes=6).to(torch.bfloat16)
        return SimpleNamespace(logits=self.affine(states))


def fixture():
    model = SmallModel().eval()
    raw = _tensor_bytes(model.affine.weight, torch)
    digest = sha256(raw).hexdigest()
    tokenizer = SimpleNamespace(eos_token_id=0, pad_token_id=5)
    runner = SimpleNamespace(model=model, tokenizer=tokenizer, device="cpu")
    ref = SimpleNamespace(shape=(6, 6), dtype="BF16", nbytes=len(raw))
    checkpoint = SimpleNamespace(checkpoint_sha256=lambda: digest, read_raw=lambda *a, **k: raw,
                                 tensors={"affine.weight": ref})
    limits = SimpleNamespace(max_update_norm=2., max_memory_bytes=2**40, max_update_entries=36,
                             max_rank=4, max_forward_calls=32, max_backward_calls=32)
    region = SimpleNamespace(tensor_name="affine.weight", module_name="affine", dtype="BF16", shape=(6, 6),
                             layer_index=1)
    plan = SimpleNamespace(region=region, checkpoint_sha256=digest, limits=limits, max_columns=6,
                           method="first_order")
    fork = SimpleNamespace(prefix_ids=(1,), accepted_token_ids=(2,), actual_token_id=3, status="semantic",
                           checkpoint_sha256=digest, replay_status="matched")
    protected = SimpleNamespace(prompt_token_ids=(4,), token_ids=(2, 0))
    return SimpleNamespace(checkpoint=checkpoint), runner, plan, fork, protected, raw


def test_rank_one_projection_preserves_anchor_and_reports_unreachable_case():
    target = np.array([[-1., 0.], [-1., 0.]])
    protect = np.array([[-1., 0.], [0., 0.]])
    updates, evidence = _rank_one(target, -target, [protect], [.01], .1, 0., 1)
    assert updates["protected"].rank == 1
    assert evidence["rank_one_output_factor_reachability"] == pytest.approx(2**-.5)
    assert evidence["unconstrained"]["feasible_radius"] < .1
    assert evidence["protected"]["predicted_protected_margins"] == pytest.approx([.01])
    updates, evidence = _rank_one(target, -target, [target], [.01], .1, 0., 1)
    assert "protected" not in updates
    assert evidence["protected_failure"] == "sampled_preservation_conflict"


def test_scan_and_endpoint_probe_replay_with_bf16_noop_and_exact_restore():
    adapter, runner, plan, fork, trajectory, raw = fixture()
    selected, updates, scan = scan_fork_regions(adapter, runner, [plan], [fork], [trajectory], radius=.3,
                                               max_constraints=1)
    assert selected == 1
    assert scan["resources"]["backward_calls"] == 3
    assert scan["parent_trajectories"][0]["replay_argmax_ids"] == [2, 0]
    update = updates[1]["fork"]
    seen = set()
    reference = prepare_probe_reference(adapter, runner, [fork], [trajectory], max_memory_bytes=2**40)

    def measure():
        return {"guard": {"passed": False, "target_item_gain": 0}, "measurement": {}}

    def evaluate_failure():
        assert _tensor_bytes(runner.model.affine.weight, torch) != raw
        runner.model.affine.weight[:, 4] = 0
        raise RuntimeError("endpoint failed")

    kwargs = dict(seen=seen, forks=[fork], trajectories=[trajectory], reference=reference)
    patch, noop = probe_update(adapter, runner, plan, update, radius=1e-10, evaluate_candidate=measure, **kwargs)
    assert patch is None and noop["reason"] == "no_stored_change"
    with pytest.raises(RuntimeError, match="endpoint failed"):
        probe_update(adapter, runner, plan, update, radius=.3, evaluate_candidate=evaluate_failure, **kwargs)
    assert _tensor_bytes(runner.model.affine.weight, torch) == raw
    assert not seen
    patch, trial = probe_update(adapter, runner, plan, update, radius=.3, evaluate_candidate=measure, **kwargs)
    assert patch is None and trial["reason"] == "endpoint_rejected"
    assert trial["fork"]["accepted_next_token_count"] == 1
    assert trial["trajectory"]["new_nonpositive_margin_positions"] == 0
    assert trial["restored_exactly"] is True
    patch, duplicate = probe_update(adapter, runner, plan, update, radius=.3, evaluate_candidate=measure, **kwargs)
    assert patch is None and duplicate["reason"] == "duplicate_stored_candidate"
    assert _tensor_bytes(runner.model.affine.weight, torch) == raw


def test_conventional_lora_trains_full_affine_factors_and_leaves_parent_exact(tmp_path):
    _, runner, plan, _, _, raw = fixture()
    save_file = pytest.importorskip("safetensors.torch").save_file
    base, output = tmp_path / "base.safetensors", tmp_path / "edited.safetensors"
    save_file(runner.model.state_dict(), str(base))
    adapter = SimpleNamespace(checkpoint=load_checkpoint(base))
    plan.checkpoint_sha256 = adapter.checkpoint.checkpoint_sha256()
    paths = (Continuation((1,), (2,), 0, True), Continuation((4,), (2,), 0, True))
    update, result = fit_lora(adapter, runner, plan, paths, rank=4, steps=3, lr=.02)
    assert isinstance(update, AffineUpdate) and update.rank == 4 and update.columns == tuple(range(6))
    assert result["trainable_parameters"] == 48
    assert result["resources"]["forward_calls"] == result["resources"]["backward_calls"] == 3
    assert result["mean_token_nll_by_step"][-1] < result["mean_token_nll_by_step"][0]
    evaluate = lambda: {"guard": {"passed": True, "target_item_gain": 1}, "measurement": {}}
    patch, trial = probe_update(adapter, runner, plan, update, radius=None, seen=set(), evaluate_candidate=evaluate)
    receipt = apply_safetensors_patch(base, patch, {"schema": "weightbridge.policy.v1",
        "allow_tensors": ["affine.weight"], "max_entries": 36, "max_fraction": 1., "max_abs_delta": 1.,
        "max_rank": 4}, output)
    assert receipt["tensor_receipts"]["affine.weight"]["candidate_sha256"] == trial["candidate_tensor_sha256"]
    assert _tensor_bytes(runner.model.affine.weight, torch) == raw
