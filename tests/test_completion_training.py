"""Completion-only labels and efficient causal-logit equivalence fixtures."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
torch = pytest.importorskip('torch')
from weightbridge.adaptation.behavior_endpoint import endpoint_runner
from weightbridge.adaptation.completion_training import prepare_completion, collate_completions, completion_loss


class Tokenizer:
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        return [ord(character) + 1 for character in text]

    def decode(self, tokens, skip_special_tokens=False):
        return ''.join(chr(token - 1) for token in tokens)


class Causal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(256, 6)
        self.head = torch.nn.Linear(6, 256)

    def forward(self, input_ids, attention_mask, use_cache=False, position_ids=None, logits_to_keep=0):
        hidden = (self.embedding(input_ids) * attention_mask[:, :, None]).cumsum(1)
        if position_ids is not None: hidden = hidden + .01 * position_ids[:, :, None]
        if isinstance(logits_to_keep, torch.Tensor): hidden = hidden[:, logits_to_keep]
        elif logits_to_keep: hidden = hidden[:, -logits_to_keep:]
        self.output_head_positions = hidden.shape[1]
        return SimpleNamespace(logits=self.head(hidden))


def prepared():
    tokenizer = Tokenizer()
    return (prepare_completion(tokenizer, 'Q: ', '-02.500'), prepare_completion(tokenizer, 'Longer Q: ', '3'))


def test_native_canonical_completion_prompt_and_padding_are_masked():
    rows = prepared()
    assert rows[0].completion_surface == ' -2.5\n' and rows[0].canonical_answer == '-2.5'
    batch = collate_completions(rows, pad_token_id=0)
    for index, row in enumerate(rows):
        assert batch['labels'][index, -len(row.target_ids):].tolist() == list(row.target_ids)
        assert torch.all(batch['labels'][index, :-len(row.target_ids)] == -100)
    assert torch.all(batch['labels'][batch['attention_mask'] == 0] == -100)


def test_bare_endpoint_tokenizer_must_be_used_instead_of_implicitly_changing_native_rendering():
    class Chat(Tokenizer):
        chat_template = 'configured'
        def apply_chat_template(self, messages, **kwargs):
            return '<user>' + messages[0]['content'] + '<assistant>'
    native = Chat()
    prompt = 'Answer:'
    first = prepare_completion(native, prompt, '-1')
    runner = endpoint_runner(SimpleNamespace(tokenizer=native), SimpleNamespace(prompt_representation='bare_prefill'))
    second = prepare_completion(runner.tokenizer, prompt, '-1')
    assert native.decode(first.prefix_ids) == '<user>Answer:<assistant>'
    assert native.decode(second.prefix_ids) == prompt
    assert first.rendered_prompt_sha256 != second.rendered_prompt_sha256


def test_joint_retokenization_cannot_silently_change_prompt_boundary():
    class BoundaryChanging(Tokenizer):
        def encode(self, text, **kwargs):
            ids = super().encode(text)
            if 'Q: ' in text and len(text) > 3: ids[1] = 999
            return ids
    with pytest.raises(ValueError, match='boundary'):
        prepare_completion(BoundaryChanging(), 'Q: ', '-1')


@pytest.mark.parametrize('mode', ['tail', 'positions'])
def test_selected_output_head_logits_match_full_loss_and_parameter_gradients(mode):
    torch.manual_seed(831)
    model = Causal()
    batch = collate_completions(prepared(), pad_token_id=0)
    all_loss, all_stats = completion_loss(model, batch, logits_mode='all')
    all_gradients = torch.autograd.grad(all_loss, tuple(model.parameters()))
    fast_loss, fast_stats = completion_loss(model, batch, logits_mode=mode)
    fast_gradients = torch.autograd.grad(fast_loss, tuple(model.parameters()))
    assert torch.allclose(all_loss, fast_loss, rtol=1e-6, atol=1e-6)
    assert all(torch.allclose(a, b, rtol=1e-5, atol=1e-5) for a, b in zip(all_gradients, fast_gradients))
    assert fast_stats['returned_logit_positions_per_row'] < all_stats['returned_logit_positions_per_row']
    assert fast_stats['supervised_tokens'] == sum(len(row.target_ids) for row in prepared())


def test_explicit_causal_shift_matches_reference_cross_entropy():
    model = Causal()
    batch = collate_completions(prepared(), pad_token_id=0)
    loss, _ = completion_loss(model, batch, logits_mode='all')
    positions = (batch['attention_mask'].cumsum(-1) - 1).masked_fill(batch['attention_mask'] == 0, 0)
    logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], position_ids=positions).logits
    reference = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 256), batch['labels'][:, 1:].reshape(-1), ignore_index=-100)
    assert torch.allclose(loss, reference)


def test_padding_is_not_a_target_and_unequal_lengths_remain_equivalent():
    model = Causal()
    rows = prepared()
    losses = [completion_loss(model, collate_completions((row,), 0), logits_mode='all')[0] for row in rows]
    together, _ = completion_loss(model, collate_completions(rows, 0), logits_mode='tail')
    expected = sum(loss * len(row.target_ids) for loss, row in zip(losses, rows)) / sum(len(row.target_ids) for row in rows)
    assert torch.allclose(together, expected, atol=1e-6)


def test_invalid_supervision_and_noncanonical_answers_refuse():
    with pytest.raises(ValueError): prepare_completion(Tokenizer(), 'Q', 'nan')
    with pytest.raises(ValueError): prepare_completion(Tokenizer(), 'Q', '-1', training_weight=0)
    batch = collate_completions(prepared(), 0)
    batch['labels'][batch['attention_mask'] == 0] = 1
    with pytest.raises(ValueError, match='padding'):
        completion_loss(Causal(), batch)


def test_prompt_label_leak_is_refused_against_explicit_native_boundary():
    batch = collate_completions(prepared(), 0)
    index = int(batch['completion_start_positions'][0]) - 1
    batch['labels'][0, index] = batch['input_ids'][0, index]
    with pytest.raises(ValueError, match='native prompt/completion boundary'):
        completion_loss(Causal(), batch)
