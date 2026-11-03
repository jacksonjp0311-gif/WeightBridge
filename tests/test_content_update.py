import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from weightbridge.adaptation import bic
from weightbridge.adaptation.content_scoring import prepare_continuations, score_continuations
from weightbridge.adaptation.content_update import propose_content_update
from weightbridge.adaptation.grounding import prior_subtracted_scores
from weightbridge.core import EditRejected
from weightbridge.inference import InferenceError
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


PRIOR_TARGETS = [{**TARGETS[0], "neutral_prompt": "tt"}]
PRIOR_PRESERVATION = [{**PRESERVATION[0], "neutral_prompt": "ss"}]


def _independent_prior_margins(runner, rows, alpha):
    task, neutral = [], []
    for field, destination in (("prompt", task), ("neutral_prompt", neutral)):
        continuations = prepare_continuations(runner.tokenizer, [row[field] for row in rows],
                                              [row["answers"] for row in rows], max_input_tokens=8)
        scores, _ = score_continuations(runner, continuations, batch_size=4)
        for score, continuation in zip(scores, continuations, strict=True):
            destination.append({**score, "answer_token_ids": continuation.answer_ids})
    calibrated_margins = []
    for index in range(0, len(task), 2):
        first, second = prior_subtracted_scores(task[index:index + 2], neutral[index:index + 2], alpha=alpha)
        calibrated_margins.append(first["sum_log_probability"] - second["sum_log_probability"])
    raw_margins = {}
    for name, scores in (("raw_task_margins", task), ("raw_neutral_margins", neutral)):
        raw_margins[name] = [first["sum_log_probability"] - second["sum_log_probability"]
                             for first, second in zip(scores[::2], scores[1::2], strict=True)]
    return calibrated_margins, raw_margins


def test_prior_uses_both_gradients_and_matches_saved_bf16_replay(toy_checkpoint, tmp_path):
    base, initial, opened, load = toy_checkpoint
    alpha = 0.5
    patch, report = propose_content_update(base, PRIOR_TARGETS, PRIOR_PRESERVATION, [NAME], INFERENCE,
                                           policy=POLICY, scoring="prior_subtracted_sum", prior_alpha=alpha,
                                           radius=0.375, max_forward_batches=6, max_backward_batches=1)
    assert report["disposition"] == "PROPOSED"
    assert report["continuations_per_row"] == 4
    assert report["prior_alpha"] == alpha
    assert report["resources"]["forward_batches"] == 6
    assert report["resources"]["forward_sequences"] == 24
    assert report["resources"]["backward_batches"] == 1
    assert report["selected_probe"]["target_gain"] == 1
    assert report["selected_probe"]["preservation_flips"] == 0

    torch = pytest.importorskip("torch")
    torch.testing.assert_close(opened[0].projection.weight, initial)
    assert not opened[0].projection._forward_hooks and not opened[0].projection._forward_pre_hooks
    rows = PRIOR_TARGETS + PRIOR_PRESERVATION
    source_runner = load(base)
    margins, raw = _independent_prior_margins(source_runner, rows, alpha)
    assert report["baseline_margins"] == margins
    assert report["baseline_raw_margins"] == raw

    step = 1 / 64
    losses = []
    for sign in (-1, 1):
        with torch.no_grad():
            source_runner.model.projection.weight.copy_(initial + sign * step)
        margins, _ = _independent_prior_margins(source_runner, rows, alpha)
        losses.append(math.log1p(math.exp(-margins[0])))
    derivative = (losses[1] - losses[0]) / (2 * step)
    # Task and neutral projections have identical behavior in the fixture.
    # The calibrated derivative must include their opposing contributions.
    assert derivative == pytest.approx(-1 / (1 + math.exp(-0.125)), rel=0.002)
    assert report["basis"][0]["target_loss_gradient_norm"] == pytest.approx(abs(derivative), rel=0.02)

    candidate = tmp_path / "prior-candidate"
    receipt = apply_safetensors_patch(base, patch, POLICY, candidate)
    assert receipt["changed_entries"] == report["changed_entries"] == 1
    margins, raw = _independent_prior_margins(load(candidate), rows, alpha)
    probe = report["selected_probe"]
    assert probe["observed_margins"] == margins
    assert all(probe[field] == values for field, values in raw.items())
    assert probe["raw_neutral_margins"][0] != report["baseline_raw_margins"]["raw_neutral_margins"][0]
    repeated, _ = propose_content_update(base, PRIOR_TARGETS, PRIOR_PRESERVATION, [NAME], INFERENCE,
                                          policy=POLICY, scoring="prior_subtracted_sum", prior_alpha=alpha,
                                          radius=0.375)
    assert repeated == patch


def test_prior_guard_rejects_calibrated_preservation_regression(toy_checkpoint):
    base, _, _, _ = toy_checkpoint
    conflicting = [{**PRIOR_TARGETS[0], "id": "prior"}]
    patch, report = propose_content_update(base, PRIOR_TARGETS, conflicting, [NAME], INFERENCE,
                                           policy=POLICY, scoring="prior_subtracted_sum", prior_alpha=0.5,
                                           radius=0.375)
    assert patch is None
    assert report["disposition"] == "NO_ELIGIBLE_MEASURED_PROBE"
    assert all(probe["target_gain"] == 1 and probe["preservation_flips"] == 1 for probe in report["probes"])


def test_prior_configuration_is_refused_before_loading_the_model(toy_checkpoint):
    base, _, opened, _ = toy_checkpoint
    args = (base, PRIOR_TARGETS, PRIOR_PRESERVATION, [NAME], INFERENCE)
    mode = {"scoring": "prior_subtracted_sum", "policy": POLICY}
    with pytest.raises(ValueError, match="batch budget"):
        propose_content_update(*args, **mode, prior_alpha=0.5, max_forward_batches=5)
    with pytest.raises(ValueError, match="measured guard and alpha"):
        propose_content_update(*args, **mode, prior_alpha=0.5, guard=False)
    with pytest.raises(ValueError, match="measured guard and alpha"):
        propose_content_update(*args, **mode, prior_alpha=0.75)
    with pytest.raises(ValueError, match="four-continuation"):
        propose_content_update(*args[:-1], {**INFERENCE, "batch_size": 2}, **mode, prior_alpha=0.5)
    with pytest.raises(ValueError, match="distinct neutral prompt"):
        propose_content_update(base, [{**PRIOR_TARGETS[0], "neutral_prompt": "t"}], *args[2:], **mode,
                               prior_alpha=0.5)
    with pytest.raises(ValueError, match="only allowed"):
        propose_content_update(base, TARGETS, PRESERVATION, [NAME], INFERENCE, policy=POLICY,
                               scoring="sum", prior_alpha=0.5)
    assert not opened


def test_prior_refuses_contextually_different_answer_tokens(toy_checkpoint, monkeypatch):
    base, _, _, load = toy_checkpoint

    def incompatible(model_path, **kwargs):
        runner = load(model_path, **kwargs)
        original = runner.tokenizer.encode

        def encode(text, **kwargs):
            tokens = original(text, **kwargs)
            if text == "ttxx":
                tokens[-1] = 4
            return tokens

        runner.tokenizer.encode = encode
        return runner

    monkeypatch.setattr(bic.HFTextRunner, "from_pretrained", incompatible)
    with pytest.raises(InferenceError, match="matching answer token IDs"):
        propose_content_update(base, PRIOR_TARGETS, PRIOR_PRESERVATION, [NAME], INFERENCE,
                               policy=POLICY, scoring="prior_subtracted_sum", prior_alpha=0.5)
