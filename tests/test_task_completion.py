"""Mixed decimal/JSON native labels, masks and effective-batch gradients."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
torch = pytest.importorskip('torch')

from weightbridge.adaptation.completion_training import collate_completions, completion_loss, prepare_completion
from weightbridge.adaptation.task_completion import (
    canonical_json_answer, completion_accumulation, native_tool_prefix,
    prepare_task_completion, prepare_tool_completion, validate_native_terminator,
)


SPEC = {'schema': 'weightbridge.native_tool_completion.v1', 'end_token_id': 255, 'end_token_surface': '<end>'}


class Tokenizer:
    chat_template = 'native'
    all_special_ids = [255]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
        return '<user>' + messages[0]['content'] + '<assistant>'

    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            if text.startswith('<end>'):
                result.append(255)
                text = text[5:]
            else:
                result.append(ord(text[0]) + 1)
                text = text[1:]
        return result

    def decode(self, tokens, skip_special_tokens=False):
        return ''.join('<end>' if token == 255 else chr(token - 1) for token in tokens)


class Bare(Tokenizer):
    chat_template = None


class Causal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(256, 6)
        self.head = torch.nn.Linear(6, 256)

    def forward(self, input_ids, attention_mask, use_cache=False, position_ids=None, logits_to_keep=0):
        hidden = (self.embedding(input_ids) * attention_mask[:, :, None]).cumsum(1)
        if position_ids is not None:
            hidden = hidden + .01 * position_ids[:, :, None]
        if isinstance(logits_to_keep, torch.Tensor):
            hidden = hidden[:, logits_to_keep]
        elif logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        return SimpleNamespace(logits=self.head(hidden))


def mixed_rows():
    tool = Tokenizer()
    return tuple(row for _ in range(4) for row in (
        prepare_completion(Bare(), 'Compare: -2 and -3. Answer:', '-2'),
        prepare_tool_completion(tool, 'Use lookup on x.', {'tool': 'lookup', 'arguments': {'id': 'long-id-007'}}, SPEC),
        prepare_completion(Bare(), 'Copy: 1.255. Answer:', '1.255'),
        prepare_tool_completion(tool, 'No tool.', {'tool': None}, SPEC)))


def test_native_full_json_and_declared_end_are_supervised_without_prompt_changes():
    tok = Tokenizer()
    row = prepare_tool_completion(tok, 'Lookup x.', {'tool': 'lookup', 'arguments': {'id': 'x'}}, SPEC)
    assert tok.decode(row.prefix_ids) == native_tool_prefix(tok, 'Lookup x.')
    assert row.completion_surface == '{"arguments":{"id":"x"},"tool":"lookup"}<end>'
    assert row.target_ids[-1] == 255 and row.target_ids.count(255) == 1
    batch = collate_completions((mixed_rows()[0], row), 0)
    assert batch['labels'][1, -1] == 255
    assert torch.all(batch['labels'][1, :-len(row.target_ids)] == -100)
    assert torch.all(batch['labels'][batch['attention_mask'] == 0] == -100)


@pytest.mark.parametrize('mode', ['all', 'tail', 'positions'])
@pytest.mark.parametrize('weighted', [False, True])
def test_four_microbatches_equal_one_mixed_length_effective_batch(mode, weighted):
    torch.manual_seed(606118)
    model = Causal()
    rows = mixed_rows()
    if weighted:
        rows = tuple(replace(row, training_weight=1. + index % 3) for index, row in enumerate(rows))
    full, _ = completion_loss(model, collate_completions(rows, 0), logits_mode='all')
    expected = torch.autograd.grad(full, tuple(model.parameters()))
    actual, work = completion_accumulation(model, rows, micro_batch_size=4, pad_token_id=0, logits_mode=mode,
        categories=['decimal' if i % 2 == 0 else 'tool' for i in range(len(rows))])
    assert actual == pytest.approx(float(full.detach()), rel=1e-6)
    assert len(work) == 4 and sum(r['backward_calls'] for r in work) == 4
    assert sum(r['microbatch_weight'] for r in work) == pytest.approx(1.)
    numerator = sum(values['weighted_token_nll_sum'] for r in work for values in r['by_category'].values())
    denominator = sum(values['weighted_supervised_tokens'] for r in work for values in r['by_category'].values())
    assert numerator / denominator == pytest.approx(actual, rel=1e-6)
    assert all(torch.allclose(p.grad, want, atol=2e-5, rtol=1e-5) for p, want in zip(model.parameters(), expected))


def test_tool_schema_and_native_boundary_are_not_guessed():
    with pytest.raises(ValueError, match='native chat'):
        prepare_tool_completion(Bare(), 'Q', {'tool': None}, SPEC)
    with pytest.raises(ValueError, match='terminator'):
        validate_native_terminator(Tokenizer(), {**SPEC, 'end_token_id': 254})
    with pytest.raises(ValueError):
        prepare_tool_completion(Tokenizer(), 'Q', {'tool': '<end>'}, SPEC)
    class Changed(Tokenizer):
        def encode(self, text, **kwargs):
            ids = super().encode(text, **kwargs)
            if text.startswith('<user>') and text.endswith('<end>'):
                ids[0] = 7
            return ids
    with pytest.raises(ValueError, match='native prefix'):
        prepare_tool_completion(Changed(), 'Q', {'tool': None}, SPEC)


def test_dispatch_keeps_historical_decimal_wrapper_and_refuses_unknown_task():
    limits = {'max_input_tokens': 512, 'max_new_tokens': 96}
    row = {'task': 'decimal', 'prompt': 'Answer:', 'answer': '-01.20'}
    got = prepare_task_completion(Tokenizer(), Bare(), row, SPEC, tool_limits=limits, decimal_limits=limits)
    assert got == prepare_completion(Bare(), 'Answer:', '-01.20')
    assert got.completion_surface == ' -1.2\n'
    tool = {'task': 'tool', 'prompt': 'Q', 'gold_action': {'tool': None}, 'completion': '{"tool":null}'}
    assert prepare_task_completion(Tokenizer(), Bare(), tool, SPEC, tool_limits=limits, decimal_limits=limits).target_ids[-1] == 255
    with pytest.raises(ValueError, match='differs'):
        prepare_task_completion(Tokenizer(), Bare(), {**tool, 'completion': '{}'}, SPEC, tool_limits=limits, decimal_limits=limits)
    with pytest.raises(ValueError, match='Unknown supervised task'):
        prepare_task_completion(Tokenizer(), Bare(), {**row, 'task': 'untrusted'}, SPEC, tool_limits=limits, decimal_limits=limits)
    with pytest.raises(ValueError):
        canonical_json_answer({'bad': float('nan')})
