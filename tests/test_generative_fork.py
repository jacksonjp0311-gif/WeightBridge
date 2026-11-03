import math
from types import SimpleNamespace

import pytest

from weightbridge.adaptation.behavior_endpoint import decimal_endpoint
from weightbridge.adaptation.generative_fork import (
    diagnose_fork, fork_loss, fork_statistics, next_token_logits, protected_trajectory,
    trajectory_diagnostics, trajectory_logits, with_fork_logits,
)


class NativeTokenizer:
    eos_token_id = 128
    pad_token_id = 129
    chat_template = None

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return list(map(ord, text))

    def decode(self, ids, skip_special_tokens):
        assert not skip_special_tokens
        return "".join(chr(i) if i < 128 else "<eos>" if i == 128 else "<pad>" for i in ids)


class BoundaryMergeTokenizer(NativeTokenizer):
    def encode(self, text, add_special_tokens):
        ids = super().encode(text, add_special_tokens)
        marker = text.find("Answer:1")
        if marker >= 0:
            ids[marker + 6:marker + 8] = [130]
        return ids


def endpoint(max_new_tokens=12):
    return decimal_endpoint({"max_input_tokens": 512, "max_new_tokens": max_new_tokens, "batch_size": 2})


def measured(runner, left, right, answer, text, *, stop="eos"):
    prompt = f"Choose the larger value.\nValues: {left} or {right}\nAnswer:"
    tokenizer = runner.tokenizer
    ids = tokenizer.encode(text, add_special_tokens=False)
    if stop == "eos":
        ids.append(tokenizer.eos_token_id)
    return {"id": "item/0", "item": "item", "group": "target", "family": "larger-decimal",
            "prompt": prompt, "operands": (left, right), "answer": answer, "text": text,
            "stop_reason": stop, "generated_token_ids": ids, "prefix_tokens": len(prompt)}


def test_first_token_diagnosis_and_longest_shared_negative_prefix():
    runner = SimpleNamespace(tokenizer=NativeTokenizer(), model=object())
    wrong_sign = measured(runner, "0.25", "-1.25", "0.25", "-1.25")
    first = diagnose_fork(runner, wrong_sign, endpoint(), checkpoint_sha256="x", answer_surfaces=("0.25",))
    assert first.status == "semantic"
    assert first.shared_token_ids == ()
    assert first.actual_token_id == ord("-")
    assert first.accepted_token_ids == (ord("0"),)
    assert first.prefix_ids == tuple(map(ord, wrong_sign["prompt"]))
    assert first.to_dict()["prompt"] == wrong_sign["prompt"]

    wrong_precision = measured(runner, "-1.35", "-1.250", "-1.250", " -1.35")
    later = diagnose_fork(runner, wrong_precision, endpoint(), checkpoint_sha256="x")
    assert later.original_status == "incorrect"
    assert later.correct_answer == "-1.25"
    assert later.shared_token_ids == tuple(map(ord, " -1."))
    assert later.actual_token_id == ord("3")
    assert later.accepted_token_ids == (ord("2"),)
    assert later.preferred_continuation == tuple(map(ord, " -1.25\n"))


def test_equivalent_correct_generation_never_creates_a_fork():
    runner = SimpleNamespace(tokenizer=NativeTokenizer(), model=object())
    equivalent = measured(runner, "-0.25", "-0.50", "-0.25", "\t-.2500")
    fork = diagnose_fork(runner, equivalent, endpoint(), checkpoint_sha256="x", answer_surfaces=("-0.25",))
    assert fork.status == "accepted" and fork.original_correct
    assert fork.accepted_token_ids == () and fork.actual_token_id is None
    assert not any(path == fork.generated_token_ids for path in fork.accepted_continuations)
    with pytest.raises(ValueError, match="accepted semantic answer"):
        diagnose_fork(runner, equivalent, endpoint(), checkpoint_sha256="x", answer_surfaces=("-0.50",))


def test_native_prompt_boundary_must_survive_joint_tokenization():
    runner = SimpleNamespace(tokenizer=BoundaryMergeTokenizer(), model=object())
    wrong = measured(runner, "1", "0", "1", "0")
    rejected = diagnose_fork(runner, wrong, endpoint(), checkpoint_sha256="x", answer_surfaces=("1",))
    assert rejected.status == "no_aligned_continuation" and rejected.unaligned_surfaces == ("1",)
    assert rejected.actual_token_id is None
    chosen = diagnose_fork(runner, wrong, endpoint(), checkpoint_sha256="x", answer_surfaces=("1", " 1"))
    assert chosen.status == "semantic" and chosen.unaligned_surfaces == ("1",)
    assert chosen.accepted_token_ids == (ord(" "),)
    assert chosen.preferred_continuation == tuple(map(ord, " 1\n"))


def test_trajectories_stop_on_first_newline_or_eos_and_refuse_truncation():
    runner = SimpleNamespace(tokenizer=NativeTokenizer(), model=object())
    complete_text = "-0.25\njunk"
    row = measured(runner, "-0.25", "-0.5", "-0.25", complete_text, stop="token_budget")
    path = protected_trajectory(runner, row, endpoint(len(complete_text)))
    assert path.token_ids == tuple(map(ord, "-0.25\n")) and path.completion == "boundary"

    eos_row = measured(runner, "-0.25", "-0.5", "-0.25", "-0.25")
    eos_path = protected_trajectory(runner, eos_row, endpoint())
    assert eos_path.token_ids[-1] == runner.tokenizer.eos_token_id and eos_path.completion == "eos"
    capped = measured(runner, "-0.25", "-0.5", "-0.25", "-0.25", stop="token_budget")
    with pytest.raises(ValueError, match="complete and semantically correct"):
        protected_trajectory(runner, capped, endpoint(5))
    diagnostic = diagnose_fork(runner, capped, endpoint(5), checkpoint_sha256="x")
    assert diagnostic.status == "truncated" and diagnostic.actual_token_id is None

    altered = {**eos_row, "text": "-0.50"}
    with pytest.raises(ValueError, match="native token transcript"):
        protected_trajectory(runner, altered, endpoint())


def test_full_vocabulary_set_ce_rival_margin_and_replay_disagreement():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float64, requires_grad=True)
    scores = fork_statistics(logits, [(1, 2)], actual_token_ids=[0])
    expected_mass = (math.exp(1) + math.exp(2)) / sum(math.exp(value) for value in range(4))
    assert scores.cross_entropy.item() == pytest.approx(-math.log(expected_mass))
    assert scores.strongest_rival_ids.tolist() == [3] and scores.decision_margins.tolist() == [-1]
    assert scores.actual_margin_loss.item() < scores.margin_loss.item()
    fork_loss(logits, [(1, 2)]).backward()
    assert logits.grad[0, 3] > 0 and logits.grad[0, 2] < 0

    runner = SimpleNamespace(tokenizer=NativeTokenizer(), model=object())
    wrong = measured(runner, "1", "0", "1", "0")
    fork = diagnose_fork(runner, wrong, endpoint(), checkpoint_sha256="x", answer_surfaces=("1",))
    replay = torch.zeros(130)
    replay[ord("1")] = 3
    replay[ord("0")] = 2
    preferred = with_fork_logits(fork, replay, runner.tokenizer)
    assert preferred.replay_status == "accepted_already_preferred"
    replay[ord("0")] = 4
    observed = with_fork_logits(fork, replay, runner.tokenizer)
    assert observed.replay_status == "matched"
    assert observed.replay["actual"]["token_text"] == "0"
    assert observed.replay["top"][0]["token_id"] == ord("0")


def test_batched_causal_logits_align_to_each_trajectory_token_and_keep_gradients():
    torch = pytest.importorskip("torch")

    class PositionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))

        def forward(self, input_ids, attention_mask, position_ids, use_cache, logits_to_keep):
            assert not use_cache
            assert (position_ids[attention_mask == 0] == 0).all()
            positions = position_ids[:, -logits_to_keep:].float()
            logits = torch.stack((positions, positions + 1), dim=-1)
            return SimpleNamespace(logits=logits * self.scale)

    runner = SimpleNamespace(tokenizer=NativeTokenizer(), model=PositionModel(), device="cpu")
    paths = [SimpleNamespace(prompt_token_ids=(9, 2), token_ids=(1, 1)),
             SimpleNamespace(prompt_token_ids=(9, 2, 3, 2), token_ids=(1, 1, 1))]
    logits = trajectory_logits(runner, paths)
    assert logits[0][:, 0].tolist() == [1, 2]
    assert logits[1][:, 0].tolist() == [3, 4, 5]
    values = next_token_logits(runner, [(9, 2), (9, 2, 3, 2)])
    assert values[:, 0].tolist() == [1, 3]
    values.sum().backward()
    assert runner.model.scale.grad.item() == 10

    row = measured(runner, "1", "0", "1", "1")
    trajectory = protected_trajectory(runner, row, endpoint())
    raw = torch.zeros(2, 130)
    for index, token in enumerate(trajectory.token_ids):
        raw[index, token] = 1.0
    assert trajectory_diagnostics(trajectory, raw, runner.tokenizer)["all_strictly_preserved"]
    raw[-1, 1] = 2.0
    assert not trajectory_diagnostics(trajectory, raw, runner.tokenizer)["all_replayed"]
