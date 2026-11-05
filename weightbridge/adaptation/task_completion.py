"""Versioned native tool completions beside the unchanged decimal SFT path.

This module defines tokens and loss accounting, never a tool executor. The
host supplies the answer and explicitly pins the native completion boundary.
"""
from __future__ import annotations

import hashlib
import json
import math

from .completion_training import SupervisedCompletion, prepare_completion, collate_completions, completion_loss
from .content_scoring import build_prefix


def native_tool_prefix(tokenizer, prompt):
    if not getattr(tokenizer, 'chat_template', None):
        raise ValueError('Tool supervision requires the pinned native chat template')
    return build_prefix(tokenizer, prompt)


def validate_native_terminator(tokenizer, spec):
    """Validate the declared end token rather than guessing an EOS convention."""
    if spec.get('schema') != 'weightbridge.native_tool_completion.v1':
        raise ValueError('Unknown native tool completion version')
    token, surface = spec.get('end_token_id'), spec.get('end_token_surface')
    if (type(token) is not int or token < 0 or not isinstance(surface, str) or not surface
            or tokenizer.encode(surface, add_special_tokens=False) != [token]
            or tokenizer.decode([token], skip_special_tokens=False) != surface
            or token not in getattr(tokenizer, 'all_special_ids', ())):
        raise ValueError('Declared native terminator does not match the tokenizer')
    return token, surface


def canonical_json_answer(action):
    if not isinstance(action, dict) or not action or any(not isinstance(k, str) for k in action):
        raise ValueError('Tool completion requires a nonempty JSON object')
    return json.dumps(action, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def prepare_tool_completion(tokenizer, prompt, action, spec, *, training_weight=1.,
                            max_input_tokens=512, max_target_tokens=96):
    token, end = validate_native_terminator(tokenizer, spec)
    if (not isinstance(training_weight, (float, int)) or isinstance(training_weight, bool)
            or not math.isfinite(training_weight) or training_weight <= 0
            or type(max_input_tokens) is not int or max_input_tokens < 1
            or type(max_target_tokens) is not int or max_target_tokens < 1):
        raise ValueError('Native completion needs positive weights and bounded token counts')
    answer = canonical_json_answer(action)
    rendered = native_tool_prefix(tokenizer, prompt)
    surface = answer + end
    prefix = tuple(tokenizer.encode(rendered, add_special_tokens=False))
    full = tuple(tokenizer.encode(rendered + surface, add_special_tokens=False))
    target = full[len(prefix):]
    if (not 1 <= len(prefix) <= max_input_tokens or full[:len(prefix)] != prefix
            or not 1 <= len(target) <= max_target_tokens or target[-1] != token
            or token in target[:-1] or any(type(t) is not int or t < 0 for t in full)
            or any(t in getattr(tokenizer, 'all_special_ids', ()) for t in target[:-1])
            or tokenizer.decode(full, skip_special_tokens=False) != rendered + surface):
        raise ValueError('Tool completion changes native prefix, contains special tokens or exceeds its token budget')
    return SupervisedCompletion(prefix, target, answer, surface,
        hashlib.sha256(rendered.encode()).hexdigest(), float(training_weight))


def prepare_task_completion(native_tokenizer, decimal_tokenizer, row, tool_spec, *,
                            tool_limits, decimal_limits):
    """Explicit dispatch: decimal rows retain the historical endpoint verbatim."""
    task = row.get('task')
    weight = row.get('training_weight', 1.)
    if task == 'decimal':
        return prepare_completion(decimal_tokenizer, row['prompt'], row['answer'], leading_space=True,
            training_weight=weight, max_input_tokens=decimal_limits['max_input_tokens'],
            max_target_tokens=decimal_limits['max_new_tokens'])
    if task == 'tool':
        if row.get('completion') != canonical_json_answer(row['gold_action']):
            raise ValueError('Tool row answer text differs from its explicit gold action')
        return prepare_tool_completion(native_tokenizer, row['prompt'], row['gold_action'], tool_spec,
            training_weight=weight, max_input_tokens=tool_limits['max_input_tokens'],
            max_target_tokens=tool_limits['max_new_tokens'])
    raise ValueError('Unknown supervised task; no implicit fallback to decimal or tool semantics')


def _category_loss(model, batch, categories, logits_mode):
    """Reuse the exact existing loss, observing its one forward for accounting."""
    import torch
    observed = []
    handle = model.register_forward_hook(lambda _module, _args, output: observed.append(output.logits))
    try:
        loss, work = completion_loss(model, batch, logits_mode=logits_mode)
    finally:
        handle.remove()
    if len(observed) != 1:
        raise ValueError('Category accounting requires exactly one model forward')
    with torch.no_grad():
        logits = observed[0]
        positions = torch.tensor(work['supervised_prediction_positions'], device=logits.device)
        if work['effective_logits_mode'] == 'positions':
            selected = logits
        else:
            offset = work['full_sequence_length'] - logits.shape[1]
            selected = logits[:, positions - offset]
        labels = batch['labels'][:, 1:][:, positions]
        values = torch.nn.functional.cross_entropy(selected.float().reshape(-1, selected.shape[-1]),
            labels.reshape(-1), ignore_index=-100, reduction='none').reshape(labels.shape)
        result = {}
        for index, category in enumerate(categories):
            if not isinstance(category, str) or not category:
                raise ValueError('Loss categories must be explicit nonempty labels')
            record = result.setdefault(category, {'examples': 0, 'supervised_tokens': 0,
                'weighted_supervised_tokens': 0., 'weighted_token_nll_sum': 0.})
            count = int((labels[index] != -100).sum())
            weight = float(batch['training_weights'][index])
            record['examples'] += 1
            record['supervised_tokens'] += count
            record['weighted_supervised_tokens'] += count * weight
            record['weighted_token_nll_sum'] += float(values[index].sum()) * weight
    return loss, {**work, 'by_category': result}


def completion_accumulation(model, rows, *, micro_batch_size, pad_token_id, device='cpu', logits_mode='tail',
                            categories=None):
    """One effective batch: sum weighted token losses / all answer-token weight.

Calls backward for each microbatch but never zeroes gradients, clips or steps
the optimizer. This equals one full-batch causal CE even with unequal answer
lengths and explicit positive per-row weights.
"""
    rows = tuple(rows)
    if not rows or type(micro_batch_size) is not int or not 1 <= micro_batch_size <= 64:
        raise ValueError('Accumulation needs a positive bounded microbatch and nonempty rows')
    if categories is not None and len(categories) != len(rows):
        raise ValueError('Category and completion rows differ')
    denominator = sum(len(row.target_ids) * row.training_weight for row in rows)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError('Effective batch has invalid answer-token weight')
    value, work_rows = 0., []
    for offset in range(0, len(rows), micro_batch_size):
        subset = rows[offset:offset + micro_batch_size]
        batch = collate_completions(subset, pad_token_id=pad_token_id, device=device)
        if categories is None:
            loss, work = completion_loss(model, batch, logits_mode=logits_mode)
        else:
            loss, work = _category_loss(model, batch, categories[offset:offset + micro_batch_size], logits_mode)
        fraction = work['weighted_supervised_tokens'] / denominator
        (loss * fraction).backward()
        value += float(loss.detach()) * fraction
        work_rows.append({**work, 'backward_calls': 1, 'examples_seen': len(subset),
            'effective_batch_denominator': denominator, 'microbatch_weight': fraction})
    return value, work_rows
