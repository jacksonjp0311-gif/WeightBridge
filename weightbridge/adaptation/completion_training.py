"""Native completion-only causal labels for the supervised LoRA baseline."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import math

from .behavior_endpoint import canonical_decimal
from .content_scoring import build_prefix


@dataclass(frozen=True)
class SupervisedCompletion:
    prefix_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    canonical_answer: str
    completion_surface: str
    rendered_prompt_sha256: str
    training_weight: float = 1.

    @property
    def input_ids(self):
        return self.prefix_ids + self.target_ids


def prepare_completion(tokenizer, prompt, answer, *, leading_space=True, training_weight=1.,
                       max_input_tokens=128, max_target_tokens=32):
    """Use the caller's frozen tokenizer representation.

    A bare-prefill endpoint must pass ``endpoint_runner(...).tokenizer``. This
    helper intentionally does not override native chat-template semantics.
    """
    canonical = canonical_decimal(answer)
    if (canonical is None or type(leading_space) is not bool or not isinstance(prompt, str) or not prompt
            or not isinstance(training_weight, (int, float)) or isinstance(training_weight, bool)
            or not math.isfinite(training_weight) or training_weight <= 0
            or type(max_input_tokens) is not int or max_input_tokens < 1
            or type(max_target_tokens) is not int or max_target_tokens < 1):
        raise ValueError("Supervised completion requires a canonical decimal and bounded authorized prompt")
    rendered = build_prefix(tokenizer, prompt)
    surface = (' ' if leading_space else '') + canonical + '\n'
    prefix = tuple(tokenizer.encode(rendered, add_special_tokens=False))
    full = tuple(tokenizer.encode(rendered + surface, add_special_tokens=False))
    if (not 1 <= len(prefix) <= max_input_tokens or full[:len(prefix)] != prefix
            or not 1 <= len(full) - len(prefix) <= max_target_tokens
            or any(type(token) is not int or token < 0 for token in full)):
        raise ValueError("Completion tokenization changes the native prompt boundary or exceeds its token budget")
    if tokenizer.decode(full, skip_special_tokens=False) != rendered + surface:
        raise ValueError("Native completion tokens do not reconstruct the declared prompt and answer")
    return SupervisedCompletion(prefix, full[len(prefix):], canonical, surface,
        hashlib.sha256(rendered.encode('utf-8')).hexdigest(), float(training_weight))


def collate_completions(rows, pad_token_id, device='cpu'):
    import torch
    rows = tuple(rows)
    if (not rows or len(rows) > 64 or type(pad_token_id) is not int or pad_token_id < 0
            or any(not row.prefix_ids or not row.target_ids or not 0 < row.training_weight < float('inf') for row in rows)):
        raise ValueError("Completion batches need bounded rows, native tokens and positive weights")
    width = max(len(row.input_ids) for row in rows)
    inputs = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    attention, labels = torch.zeros_like(inputs), torch.full_like(inputs, -100)
    for index, row in enumerate(rows):
        start = width - len(row.input_ids)
        inputs[index, start:] = torch.tensor(row.input_ids, dtype=torch.long)
        attention[index, start:] = 1
        labels[index, start + len(row.prefix_ids):] = torch.tensor(row.target_ids, dtype=torch.long)
    return {'input_ids': inputs.to(device), 'attention_mask': attention.to(device), 'labels': labels.to(device),
        'training_weights': torch.tensor([row.training_weight for row in rows], device=device, dtype=torch.float32),
        'completion_start_positions': torch.tensor([width - len(row.target_ids) for row in rows], device=device),
        'target_token_counts': torch.tensor([len(row.target_ids) for row in rows], device=device)}


def completion_loss(model, batch, *, logits_mode='tail'):
    """Explicit causal shift with optional output-head work restricted to targets.

    Models must independently qualify loss/gradient equivalence before an
    efficient mode is used for a scientific training trial.
    """
    import torch
    if logits_mode not in {'all', 'tail', 'positions'}:
        raise ValueError("Unknown supervised logits mode")
    inputs, mask, labels = (batch[name] for name in ('input_ids', 'attention_mask', 'labels'))
    if (inputs.ndim != 2 or inputs.shape != mask.shape or inputs.shape != labels.shape
            or inputs.shape[1] < 2 or not torch.any(labels[:, 1:] != -100)
            or inputs.dtype != torch.long or labels.dtype != torch.long
            or torch.any(labels[mask == 0] != -100) or torch.any(labels[:, 0] != -100)
            or torch.any((mask != 0) & (mask != 1))
            or torch.any(labels[labels != -100] != inputs[labels != -100])):
        raise ValueError("Supervised labels must preserve prompt/padding masks and causal targets")
    starts, counts_per_row = batch.get('completion_start_positions'), batch.get('target_token_counts')
    if starts is not None or counts_per_row is not None:
        if (starts is None or counts_per_row is None or starts.shape != (len(inputs),)
                or counts_per_row.shape != (len(inputs),) or torch.any(counts_per_row <= 0)
                or torch.any(starts != inputs.shape[1] - counts_per_row)
                or torch.any((labels != -100) != (torch.arange(inputs.shape[1], device=inputs.device)[None, :] >= starts[:, None]))):
            raise ValueError('Supervised labels disagree with the declared native prompt/completion boundary')
    wanted = labels[:, 1:]
    positions = torch.nonzero((wanted != -100).any(dim=0), as_tuple=False).reshape(-1)
    parameters = inspect.signature(model.forward).parameters
    kwargs = {'input_ids': inputs, 'attention_mask': mask, 'use_cache': False}
    if 'position_ids' in parameters:
        p = mask.cumsum(-1) - 1
        kwargs['position_ids'] = p.masked_fill(mask == 0, 0)
    field = 'logits_to_keep' if 'logits_to_keep' in parameters else 'num_logits_to_keep' if 'num_logits_to_keep' in parameters else None
    mode = logits_mode if field or logits_mode == 'all' else 'all_fallback_unsupported'
    start = 0
    if field and logits_mode == 'tail':
        keep = inputs.shape[1] - int(positions[0])
        start = inputs.shape[1] - keep
        kwargs[field] = keep
    elif field and logits_mode == 'positions':
        kwargs[field] = positions
    logits = model(**kwargs).logits
    if logits.ndim != 3 or logits.shape[0] != len(inputs):
        raise ValueError("Model returned invalid supervised causal logits")
    if mode == 'positions':
        if logits.shape[1] != len(positions):
            raise ValueError("Model does not support exact supervised logits positions")
        selected = logits
    elif mode == 'tail':
        if logits.shape[1] != inputs.shape[1] - start:
            raise ValueError("Model did not honor bounded supervised tail logits")
        selected = logits[:, positions - start]
    else:
        if logits.shape[1] != inputs.shape[1]:
            raise ValueError("Model returned truncated full supervised logits")
        selected = logits[:, positions]
    target = wanted[:, positions]
    counts = target != -100
    weights = batch.get('training_weights', torch.ones(len(inputs), device=inputs.device))
    if (weights.shape != (len(inputs),) or not torch.isfinite(weights).all() or torch.any(weights <= 0)):
        raise ValueError("Completion training weights must be finite and positive per row")
    losses = torch.nn.functional.cross_entropy(selected.float().reshape(-1, selected.shape[-1]),
        target.reshape(-1), ignore_index=-100, reduction='none').reshape(target.shape)
    weighted = counts * weights[:, None]
    denominator = weighted.sum()
    loss = (losses * weights[:, None]).sum() / denominator
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite supervised completion loss")
    return loss, {'supervised_tokens': int(counts.sum()), 'weighted_supervised_tokens': float(denominator),
        'padded_input_token_positions': int(inputs.numel()), 'unmasked_input_tokens': int(mask.sum()),
        'supervised_prediction_positions': positions.detach().cpu().tolist(), 'causal_shift': 'logits[t] predicts labels[t+1]',
        'requested_logits_mode': logits_mode, 'effective_logits_mode': mode,
        'returned_logit_positions_per_row': logits.shape[1], 'full_sequence_length': inputs.shape[1],
        'forward_calls': 1, 'backward_calls': 0}
