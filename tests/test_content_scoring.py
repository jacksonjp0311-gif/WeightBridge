from types import SimpleNamespace

import pytest

from weightbridge.adaptation.content_scoring import (
    generate_responses,
    prepare_continuations,
    score_batch,
    score_continuations,
)


class ToyTokenizer:
    pad_token_id = None
    eos_token_id = 0
    vocabulary = "#pqABC"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [self.vocabulary.index(character) for character in text]

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join(self.vocabulary[index] for index in ids)


def test_requires_a_stable_boundary_and_uses_contextual_answer_ids():
    class ContextTokenizer(ToyTokenizer):
        def encode(self, text, *, add_special_tokens):
            if text == "pA":
                return [1, 4]
            if text == "qA":
                return [5]
            return super().encode(text, add_special_tokens=add_special_tokens)

    tokenizer = ContextTokenizer()
    first, second = prepare_continuations(tokenizer, ["p"], [["A", "C"]], max_input_tokens=2)
    assert first.prefix_ids == (1,)
    assert first.answer_ids == (4,)
    assert first.joint_tokenization_matches is False
    assert second.answer_ids == (5,)
    assert second.joint_tokenization_matches is True
    with pytest.raises(ValueError, match="frozen prompt boundary"):
        prepare_continuations(tokenizer, ["q"], [["A", "C"]], max_input_tokens=2)
    with pytest.raises(ValueError, match="same continuation tokens"):
        prepare_continuations(tokenizer, ["p"], [["A", "B"]], max_input_tokens=2)
    with pytest.raises(ValueError, match="exceed max_input_tokens"):
        prepare_continuations(tokenizer, ["p"], [["A", "C"]], max_input_tokens=2, include_eos=True)


def test_padding_causal_alignment_and_eos_match_manual_token_scores():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transitions = torch.nn.Parameter(torch.arange(36, dtype=torch.float32).reshape(6, 6) / 20)
            self.last_mask = None

        def forward(self, input_ids, attention_mask, position_ids, use_cache, logits_to_keep):
            assert use_cache is False
            self.last_mask = attention_mask.tolist()
            shifts = torch.arange(6, dtype=torch.float32) / 10
            logits = self.transitions[input_ids] + position_ids.unsqueeze(-1) * shifts
            return SimpleNamespace(logits=logits[:, -logits_to_keep:])

    runner = SimpleNamespace(model=Model(), tokenizer=ToyTokenizer(), device="cpu")
    prompts, answers = ["p", "pq"], [["A", "BC"], ["B", "A"]]
    rows = prepare_continuations(runner.tokenizer, prompts, answers, max_input_tokens=4, include_eos=True)
    batched, resources = score_continuations(runner, rows, batch_size=4)
    assert runner.model.last_mask == [[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    assert resources == {"forward_batches": 1, "forward_examples": 4, "padded_input_token_positions": 16}
    individually, _ = score_continuations(runner, rows, batch_size=1)
    assert individually == batched
    answer_only = prepare_continuations(runner.tokenizer, prompts, answers, max_input_tokens=4)
    unterminated, _ = score_continuations(runner, answer_only, batch_size=4)
    for row, result, non_eos in zip(rows, batched, unterminated, strict=True):
        manual = []
        for position in range(len(row.prefix_ids) - 1, len(row.input_ids) - 1):
            previous, next_token = row.input_ids[position:position + 2]
            distribution = (runner.model.transitions[previous] + position * torch.arange(6) / 10).log_softmax(0)
            manual.append(float(distribution[next_token].detach()))
        assert result["answer_tokens"] == len(manual) - 1
        assert result["sum_log_probability"] == pytest.approx(sum(manual[:-1]))
        assert result["mean_log_probability"] == pytest.approx(sum(manual[:-1]) / (len(manual) - 1))
        assert result["eos_log_probability"] == pytest.approx(manual[-1])
        assert result["terminated_sum_log_probability"] == pytest.approx(sum(manual))
        assert non_eos["sum_log_probability"] == pytest.approx(result["sum_log_probability"])
        assert non_eos["eos_log_probability"] is None
    first, second = score_batch(runner, rows[:2])
    (first[:-1].mean() - second[:-1].mean()).backward()
    assert torch.count_nonzero(runner.model.transitions.grad) > 0


def test_greedy_responses_separate_eos_completion_from_budget_and_batch_padding():
    torch = pytest.importorskip("torch")

    class Model:
        def generate(self, *, input_ids, attention_mask, do_sample, num_beams, max_new_tokens, **kwargs):
            assert do_sample is False and num_beams == 1 and max_new_tokens == 3
            assert kwargs["eos_token_id"] == kwargs["pad_token_id"] == 0
            assert attention_mask.tolist() == [[0, 1], [1, 1]]
            return torch.cat([input_ids, torch.tensor([[3, 0, 0], [4, 5, 3]])], dim=1)

    runner = SimpleNamespace(model=Model(), tokenizer=ToyTokenizer(), device="cpu")
    actual, resources = generate_responses(runner, ["p", "pq"], max_input_tokens=2, max_new_tokens=3, batch_size=2)
    assert actual == [
        {"text": "A", "stop_reason": "eos", "generated_token_ids": [3, 0], "prefix_tokens": 1},
        {"text": "BCA", "stop_reason": "token_budget", "generated_token_ids": [4, 5, 3], "prefix_tokens": 2},
    ]
    assert resources == {"generate_batches": 1, "generated_examples": 2,
                         "padded_prompt_token_positions": 4, "batched_decode_steps": 3}
