"""Measure fixed causal answer continuations and greedy responses.

Teacher-forced scores use a joint tokenization that preserves the rendered
prompt's tokens. Generation runs separately from that scoring path.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import inspect

from ..inference import InferenceError


@dataclass(frozen=True)
class Continuation:
    prefix_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]
    eos_token_id: int | None
    joint_tokenization_matches: bool

    @property
    def scored_ids(self) -> tuple[int, ...]:
        return self.answer_ids if self.eos_token_id is None else (*self.answer_ids, self.eos_token_id)

    @property
    def input_ids(self) -> tuple[int, ...]:
        return self.prefix_ids + self.scored_ids


def build_prefix(tokenizer, prompt: str) -> str:
    """Render one text-only prompt, with the generation prefix frozen."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be nonempty text")
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
    return prompt


def prepare_continuations(tokenizer, prompts: Sequence[str], answers: Sequence[Sequence[str]], *,
                          max_input_tokens: int, include_eos: bool = False) -> tuple[Continuation, ...]:
    """Encode a two-option contrast for every prompt in row and option order."""
    _positive_int(max_input_tokens, "max_input_tokens")
    if type(include_eos) is not bool:
        raise ValueError("include_eos must be boolean")
    eos = getattr(tokenizer, "eos_token_id", None)
    if include_eos and eos is None:
        raise ValueError("EOS scoring requires an EOS token")
    completions = []
    for prompt, options in zip(prompts, answers, strict=True):
        if (isinstance(options, str) or len(options) != 2
                or any(not isinstance(answer, str) or not answer.strip() for answer in options)
                or options[0] == options[1]):
            raise ValueError("Every prompt requires two distinct nonempty answer strings")
        rendered = build_prefix(tokenizer, prompt)
        prefix = tuple(tokenizer.encode(rendered, add_special_tokens=False))
        if not prefix:
            raise ValueError("Rendered prompt encodes to no tokens")
        prepared_options = []
        for answer in options:
            joint = tuple(tokenizer.encode(rendered + answer, add_special_tokens=False))
            if joint[:len(prefix)] != prefix:
                raise ValueError("Joint tokenization changes the frozen prompt boundary")
            suffix = joint[len(prefix):]
            if not suffix or eos in suffix:
                raise ValueError("Answer must encode to nonempty tokens without EOS")
            separate = tuple(tokenizer.encode(answer, add_special_tokens=False))
            continuation = Continuation(prefix, suffix, eos if include_eos else None, separate == suffix)
            if len(continuation.input_ids) > max_input_tokens:
                raise ValueError("Prompt and answer exceed max_input_tokens")
            prepared_options.append(continuation)
        if prepared_options[0].answer_ids == prepared_options[1].answer_ids:
            raise ValueError("Answer strings map to the same continuation tokens")
        completions.extend(prepared_options)
    if not completions:
        raise ValueError("At least one prompt is required")
    return tuple(completions)


def _positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise InferenceError("Install WeightBridge inference extras to score model responses") from exc
    return torch


def _pad_id(tokenizer) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    if pad is None:
        raise ValueError("Scoring requires a padding token or EOS fallback")
    return pad


def _left_pad(torch, sequences, pad, device):
    width = max(map(len, sequences))
    input_ids = torch.full((len(sequences), width), pad, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for index, sequence in enumerate(sequences):
        input_ids[index, -len(sequence):] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[index, -len(sequence):] = 1
    return input_ids.to(device), attention_mask.to(device)


def score_batch(runner, continuations: Sequence[Continuation]):
    """Return differentiable answer log-probabilities followed by optional EOS.

    In left-padded coordinates a continuation of length n uses logit
    positions [-n-1:-1]. The caller controls the autograd context.
    """
    if not continuations:
        raise ValueError("At least one continuation is required")
    torch = _torch()
    input_ids, attention_mask = _left_pad(torch, [row.input_ids for row in continuations],
                                        _pad_id(runner.tokenizer), runner.device)
    forward = inspect.signature(runner.model.forward).parameters
    keep = max(len(row.scored_ids) for row in continuations) + 1
    kwargs = {"use_cache": False}
    if "position_ids" in forward:
        # Give valid tokens the same positions with or without left padding.
        positions = attention_mask.cumsum(-1) - 1
        kwargs["position_ids"] = positions.masked_fill(attention_mask == 0, 0)
    if "logits_to_keep" in forward:
        kwargs["logits_to_keep"] = keep
    elif "num_logits_to_keep" in forward:
        kwargs["num_logits_to_keep"] = keep
    logits = runner.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs).logits
    if logits.ndim != 3 or logits.shape[0] != len(continuations) or logits.shape[1] < keep:
        raise InferenceError("Model returned insufficient causal logits for the continuation")
    scores = []
    for index, row in enumerate(continuations):
        scored_ids = row.scored_ids
        answer_logits = logits[index, -len(scored_ids) - 1:-1].float()
        wanted = torch.tensor(scored_ids, device=logits.device).unsqueeze(1)
        log_probs = answer_logits.gather(1, wanted).squeeze(1) - answer_logits.logsumexp(dim=1)
        if not torch.isfinite(log_probs).all().item():
            raise InferenceError("Model returned nonfinite continuation log probabilities")
        scores.append(log_probs)
    return tuple(scores)


def score_continuations(runner, continuations: Sequence[Continuation], *, batch_size: int):
    """Return summed, length-normalized and optional EOS likelihood scores."""
    _positive_int(batch_size, "batch_size")
    if not continuations:
        raise ValueError("At least one continuation is required")
    torch = _torch()
    results = []
    forward_batches = padded_positions = 0
    with torch.inference_mode():
        for start in range(0, len(continuations), batch_size):
            batch = continuations[start:start + batch_size]
            scores = score_batch(runner, batch)
            forward_batches += 1
            padded_positions += len(batch) * max(len(row.input_ids) for row in batch)
            for row, log_probs in zip(batch, scores, strict=True):
                answer = log_probs[:len(row.answer_ids)]
                results.append({
                    "prefix_tokens": len(row.prefix_ids), "answer_tokens": len(row.answer_ids),
                    "sum_log_probability": float(answer.sum()), "mean_log_probability": float(answer.mean()),
                    "eos_log_probability": float(log_probs[-1]) if row.eos_token_id is not None else None,
                    "terminated_sum_log_probability": float(log_probs.sum()) if row.eos_token_id is not None else None,
                    "joint_tokenization_matches": row.joint_tokenization_matches,
                })
    return results, {"forward_batches": forward_batches, "forward_examples": len(continuations),
                     "padded_input_token_positions": padded_positions}


def generate_responses(runner, prompts: Sequence[str], *, max_input_tokens: int, max_new_tokens: int,
                       batch_size: int):
    """Generate greedily from the same rendered prefix used for scoring."""
    _positive_int(max_input_tokens, "max_input_tokens")
    _positive_int(max_new_tokens, "max_new_tokens")
    _positive_int(batch_size, "batch_size")
    torch = _torch()
    tokenizer = runner.tokenizer
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        raise ValueError("Generation completion measurement requires an EOS token")
    pad = _pad_id(tokenizer)
    encoded = [tuple(tokenizer.encode(build_prefix(tokenizer, prompt), add_special_tokens=False))
               for prompt in prompts]
    if not encoded or any(not 1 <= len(prefix) <= max_input_tokens for prefix in encoded):
        raise ValueError("Generation requires prompts within max_input_tokens")
    results = []
    batches = padded_positions = decode_steps = 0
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            batch = encoded[start:start + batch_size]
            input_ids, attention_mask = _left_pad(torch, batch, pad, runner.device)
            generated = runner.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, do_sample=False, num_beams=1,
                max_new_tokens=max_new_tokens, use_cache=True, pad_token_id=pad, eos_token_id=eos,
                return_dict_in_generate=False,
            )
            if (generated.ndim != 2 or generated.shape[0] != len(batch)
                    or not torch.equal(generated[:, :input_ids.shape[1]], input_ids)):
                raise InferenceError("Model returned invalid greedy sequences")
            continuation = generated[:, input_ids.shape[1]:]
            if not 1 <= continuation.shape[1] <= max_new_tokens:
                raise InferenceError("Model returned an invalid number of generated tokens")
            batches += 1
            padded_positions += input_ids.numel()
            decode_steps += continuation.shape[1]
            for prefix, ids in zip(batch, continuation.tolist(), strict=True):
                ended = eos in ids
                if not ended and len(ids) != max_new_tokens:
                    raise InferenceError("Model stopped before EOS or the generation budget")
                text_ids = ids[:ids.index(eos)] if ended else ids
                results.append({
                    "text": tokenizer.decode(text_ids, skip_special_tokens=False),
                    "stop_reason": "eos" if ended else "token_budget",
                    "generated_token_ids": text_ids + ([eos] if ended else []),
                    "prefix_tokens": len(prefix),
                })
    return results, {"generate_batches": batches, "generated_examples": len(results),
                     "padded_prompt_token_positions": padded_positions, "batched_decode_steps": decode_steps}
