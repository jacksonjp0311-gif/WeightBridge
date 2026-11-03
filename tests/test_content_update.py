import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from weightbridge.adaptation import bic
from weightbridge.adaptation.content_scoring import prepare_continuations, score_continuations
from weightbridge.adaptation.content_update import propose_content_update
from weightbridge.core import EditRejected
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches.huggingface import apply_safetensors_patch


NAME = "model.language_model.layers.0.mlp.down_proj.weight"
SECOND_NAME = "model.language_model.layers.1.mlp.down_proj.weight"
INFERENCE = {"device": "cpu", "dtype": "bfloat16", "cpu_threads": 1,
             "batch_size": 4, "max_input_tokens": 8}
POLICY = {"schema": "weightbridge.policy.v1", "allow_tensors": [NAME], "max_entries": 1,
          "max_fraction": 1.0, "max_abs_delta": 1.0}
TARGETS = [{"id": "target", "prompt": "t", "answers": ["xx", "yy"], "correct": 0}]
PRESERVATION = [{"id": "old", "prompt": "s", "answers": ["xx", "yy"], "correct": 0}]


@pytest.fixture
def toy_checkpoint(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    tensors = pytest.importorskip("safetensors.torch")
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")
    initial = torch.tensor([[-0.125]], dtype=torch.bfloat16)
    tensors.save_file({NAME: initial, SECOND_NAME: torch.ones_like(initial)}, str(base / "model.safetensors"))

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 5

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return ["_tsxy".index(character) for character in text]

    class Model(torch.nn.Module):
        def __init__(self, weights):
            super().__init__()
            self.projection = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
            self.later = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
            with torch.no_grad():
                self.projection.weight.copy_(weights[NAME])
                self.later.weight.copy_(weights[SECOND_NAME])

        def get_submodule(self, target):
            return {NAME: self.projection, SECOND_NAME: self.later}[target + ".weight"]

        def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
            seen_target = ((input_ids == 1) & attention_mask.bool()).cumsum(dim=1) > 0
            answer_input = (input_ids == 3) | (input_ids == 4)
            # The first answer token is unaffected. Only its continuation
            # supplies a score derivative in the target examples.
            feature = (seen_target & answer_input).to(torch.bfloat16).unsqueeze(-1)
            activation = self.later(self.projection(feature))
            zero = torch.zeros_like(activation)
            logits = torch.cat([zero, zero, zero, activation, -activation, zero], dim=-1)
            return SimpleNamespace(logits=logits[:, -logits_to_keep:])

    opened = []

    def load(model_path, **kwargs):
        model = Model(tensors.load_file(str(Path(model_path) / "model.safetensors")))
        opened.append(model)
        return SimpleNamespace(model=model, tokenizer=Tokenizer(), device="cpu")

    monkeypatch.setattr(bic.HFTextRunner, "from_pretrained", load)
    return base, initial, opened, load


def test_content_gradient_produces_exact_rounded_checkpoint(toy_checkpoint, tmp_path):
    base, initial, opened, load = toy_checkpoint
    patch, report = propose_content_update(base, TARGETS, PRESERVATION, [NAME], INFERENCE,
                                           policy=POLICY, scoring="mean", radius=0.375)
    assert report["disposition"] == "PROPOSED"
    assert report["selected_probe"]["target_gain"] == 1
    assert report["selected_probe"]["preservation_flips"] == 0
    assert report["basis"][0]["target_loss_gradient_norm"] > 0
    assert report["resources"]["backward_batches"] == 1
    assert report["resources"]["forward_batches"] == 3
    assert report["realized_frobenius_norm"] <= 0.375
    assert patch["base_sha256"] == load_checkpoint(base).checkpoint_sha256()
    torch = pytest.importorskip("torch")
    torch.testing.assert_close(opened[0].projection.weight, initial)
    assert not opened[0].projection._forward_hooks and not opened[0].projection._forward_pre_hooks

    candidate = tmp_path / "candidate"
    receipt = apply_safetensors_patch(base, patch, POLICY, candidate)
    assert receipt["changed_entries"] == report["changed_entries"] == 1
    restarted = load(candidate)
    rows = TARGETS + PRESERVATION
    pairs = prepare_continuations(restarted.tokenizer, [row["prompt"] for row in rows],
                                  [row["answers"] for row in rows], max_input_tokens=8)
    scores, _ = score_continuations(restarted, pairs, batch_size=4)
    means = [score["mean_log_probability"] for score in scores]
    assert [means[0] - means[1], means[2] - means[3]] == report["selected_probe"]["observed_margins"]
    repeated, _ = propose_content_update(base, TARGETS, PRESERVATION, [NAME], INFERENCE,
                                          policy=POLICY, scoring="mean", radius=0.375)
    assert repeated == patch


def test_guard_rejects_lower_loss_that_flips_preservation(toy_checkpoint):
    base, _, _, _ = toy_checkpoint
    conflicting = [{**TARGETS[0], "id": "prior"}]
    guarded, report = propose_content_update(base, TARGETS, conflicting, [NAME], INFERENCE,
                                             policy=POLICY, scoring="mean", radius=0.375)
    assert guarded is None
    assert report["disposition"] == "NO_ELIGIBLE_MEASURED_PROBE"
    assert all(probe["preservation_flips"] == 1 for probe in report["probes"])
    unguarded, report = propose_content_update(base, TARGETS, conflicting, [NAME], INFERENCE,
                                               policy=POLICY, scoring="mean", radius=0.375, guard=False)
    assert unguarded is not None
    assert report["selected_probe"]["preservation_flips"] == 1


def test_rounded_noop_and_prefit_policy_refusal(toy_checkpoint):
    base, _, opened, _ = toy_checkpoint
    with pytest.raises(EditRejected, match="outside host policy"):
        propose_content_update(base, TARGETS, PRESERVATION, [NAME], INFERENCE, scoring="mean",
                               policy={**POLICY, "allow_tensors": ["other"]})
    with pytest.raises(ValueError, match="batch budget"):
        propose_content_update(base, TARGETS, PRESERVATION, [NAME], INFERENCE, scoring="mean",
                               policy=POLICY, max_forward_batches=2)
    assert not opened
    patch, report = propose_content_update(base, TARGETS, PRESERVATION, [NAME], INFERENCE, scoring="mean",
                                           policy=POLICY, radius=1e-6)
    assert patch is None
    assert [probe["disposition"] for probe in report["probes"]] == ["PRECISION_LOSS", "PRECISION_LOSS"]
    assert report["resources"]["forward_batches"] == 1


def test_two_layer_probes_use_both_functional_directions_and_restore_weights(toy_checkpoint):
    base, initial, opened, _ = toy_checkpoint
    names = [NAME, SECOND_NAME]
    policy = {**POLICY, "allow_tensors": names, "max_entries": 2}
    _, report = propose_content_update(base, TARGETS, PRESERVATION, names, INFERENCE,
                                       policy=policy, scoring="sum", radius=0.375)
    assert [entry["tensor"] for entry in report["basis"]] == names
    assert len(report["probe_schedule"]) == 6
    assert any(all(probe["coefficients"]) and "observed_margins" in probe for probe in report["probes"])
    assert report["resources"]["backward_batches"] == 1
    torch = pytest.importorskip("torch")
    torch.testing.assert_close(opened[0].projection.weight, initial)
    torch.testing.assert_close(opened[0].later.weight, torch.ones_like(initial))
