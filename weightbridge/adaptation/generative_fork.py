"""Bounded diagnoses of actual greedy decisions and their protected prefixes."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
import inspect

from .behavior_endpoint import BehavioralEndpointContract, canonical_decimal, endpoint_runner
from .content_scoring import _left_pad, _pad_id, _torch, build_prefix


@dataclass(frozen=True)
class GenerativeFork:
    """A first divergence from a finite, semantically validated set of continuations."""

    checkpoint_sha256: str
    model_id: str
    question_id: str
    prompt_id: str
    prompt: str
    rendered_prompt_sha256: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    original_text: str
    original_answer: str | None
    correct_answer: str
    original_status: str
    original_completed: bool
    original_valid: bool
    original_correct: bool
    status: str
    shared_token_ids: tuple[int, ...]
    accepted_token_ids: tuple[int, ...]
    actual_token_id: int | None
    accepted_continuations: tuple[tuple[int, ...], ...]
    preferred_continuation: tuple[int, ...]
    tokenizer_class: str
    decoding: dict
    unaligned_surfaces: tuple[str, ...]
    replay_status: str = "unmeasured"
    replay: dict | None = None

    @property
    def prefix_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + self.shared_token_ids

    def to_dict(self) -> dict:
        return {"schema": "weightbridge.generative_fork.v1", **asdict(self)}


@dataclass(frozen=True)
class ProtectedTrajectory:
    prompt_id: str
    prompt_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    completion: str

    def to_dict(self) -> dict:
        return {"schema": "weightbridge.protected_trajectory.v1", **asdict(self)}


def _encode(tokenizer, text: str) -> tuple[int, ...]:
    return tuple(tokenizer.encode(text, add_special_tokens=False))


def _transcript(runner, row: Mapping, contract: BehavioralEndpointContract):
    tokenizer = endpoint_runner(runner, contract).tokenizer
    rendered = build_prefix(tokenizer, row["prompt"])
    prefix = _encode(tokenizer, rendered)
    if (not prefix or len(prefix) > contract.max_input_tokens
            or row.get("prefix_tokens", len(prefix)) != len(prefix)):
        raise ValueError("Generated trace does not match the frozen prompt token boundary")
    eos = getattr(tokenizer, "eos_token_id", None)
    if type(eos) is not int or eos < 0:
        raise ValueError("The frozen endpoint requires a native EOS token")
    ids = tuple(row["generated_token_ids"])
    if not 1 <= len(ids) <= contract.max_new_tokens or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("Generated trace has invalid native token IDs or exceeds the budget")
    stop = row["stop_reason"]
    if (stop == "eos" and ids[-1] == eos and eos not in ids[:-1]):
        decoded = tokenizer.decode(ids[:-1], skip_special_tokens=False)
    elif stop == "token_budget" and eos not in ids and len(ids) == contract.max_new_tokens:
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
    else:
        raise ValueError("Generated token IDs do not agree with the frozen stopping rule")
    if decoded != row["text"]:
        raise ValueError("Generated text does not match its native token transcript")
    return tokenizer, rendered, prefix, ids, eos


def _completion(tokenizer, ids: tuple[int, ...], eos: int, contract: BehavioralEndpointContract):
    for end in range(1, len(ids) + 1):
        stopped = ids[end - 1] == eos
        decoded = tokenizer.decode(ids[:end - stopped], skip_special_tokens=False)
        parsed = contract.parse(decoded, "eos" if stopped else "token_budget")
        if parsed.completed:
            return ids[:end], parsed.answer, "eos" if stopped else "boundary"
    return None


def _answer_surfaces(row: Mapping, answer: str, contract: BehavioralEndpointContract) -> tuple[str, ...]:
    variants = [answer]
    if contract.endpoint_id == "weightbridge.decimal_first_line.v1":
        variants.extend(value for value in row.get("operands", ()) if canonical_decimal(value) == answer)
        if "." in answer:
            variants.append(answer + "0")
            if answer.startswith(("0.", "-0.")):
                variants.append(answer.replace("0.", ".", 1))
        else:
            variants.append(answer + ".0")
    return tuple(dict.fromkeys(surface for value in variants
                               for surface in (" " + value, value)
                               if contract.parse(surface, "eos").answer == answer))


def _aligned_continuations(tokenizer, rendered, prefix, eos, surfaces, answer, contract):
    continuations, unaligned = [], []
    for surface in surfaces:
        if not isinstance(surface, str) or not surface.strip() or len(surface) > 256:
            raise ValueError("Accepted answer surfaces must be bounded, nonempty text")
        parsed = contract.parse(surface, "eos")
        if not parsed.completed or parsed.answer != answer:
            raise ValueError("Declared target continuation does not match the accepted semantic answer")
        aligned = False
        for body, suffix in ((surface + "\n", ()), (surface, (eos,))):
            joined = _encode(tokenizer, rendered + body)
            if joined[:len(prefix)] != prefix:
                continue
            body_tokens = joined[len(prefix):]
            candidate = body_tokens + suffix
            if not body_tokens or eos in body_tokens or len(candidate) > contract.max_new_tokens:
                continue
            completed = _completion(tokenizer, candidate, eos, contract)
            if completed is None or completed[1] != answer:
                continue
            continuations.append(completed[0])
            aligned = True
        if not aligned:
            unaligned.append(surface)
    return tuple(dict.fromkeys(continuations)), tuple(unaligned)


def _shared_length(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def diagnose_fork(runner, row: Mapping, contract: BehavioralEndpointContract, *,
                  checkpoint_sha256: str, answer_surfaces: Sequence[str] | None = None,
                  model_id: str | None = None) -> GenerativeFork:
    """Align a frozen generation to the longest matching accepted native prefix.

    Semantic correctness is adjudicated before comparing tokens. An already
    accepted response never supplies an adaptation target, even if its native
    tokens differ from all enumerated continuation spellings.
    """
    tokenizer, rendered, prefix, ids, eos = _transcript(runner, row, contract)
    question = contract.question(row)
    parsed = contract.parse(row["text"], row["stop_reason"])
    valid = (parsed.completed and parsed.answer is not None
             and (question.allowed_answers is None or parsed.answer in question.allowed_answers))
    correct = valid and parsed.answer == question.answer
    original_status = ("correct" if correct else "incorrect" if valid else "incomplete" if not parsed.completed
                       else "invalid_grammar" if parsed.answer is None else "out_of_option")
    for name, value in {"status": original_status, "exact": parsed.answer, "completed": parsed.completed,
                        "valid": valid, "correct": correct}.items():
        if name in row and row[name] != value:
            raise ValueError(f"Measured endpoint row disagrees with verified {name}")
    continuations, unaligned = (), ()
    status, shared, wanted, actual, preferred = "accepted", (), (), None, ()
    if answer_surfaces is not None and (isinstance(answer_surfaces, str) or not 1 <= len(answer_surfaces) <= 16):
        raise ValueError("Supply 1 to 16 explicitly accepted answer surfaces")
    if not correct or answer_surfaces is not None:
        surfaces = answer_surfaces if answer_surfaces is not None else _answer_surfaces(row, question.answer, contract)
        continuations, unaligned = _aligned_continuations(tokenizer, rendered, prefix, eos, surfaces,
                                                          question.answer, contract)
    if not correct:
        if not parsed.completed and contract.parse(row["text"], "eos").answer == question.answer:
            status, shared = "truncated", ids
        elif not continuations:
            status = "no_aligned_continuation"
        else:
            observed_completion = _completion(tokenizer, ids, eos, contract)
            observed = observed_completion[0] if observed_completion is not None else ids
            shared_count = max(_shared_length(observed, candidate) for candidate in continuations)
            shared = observed[:shared_count]
            aligned = tuple(candidate for candidate in continuations if candidate[:shared_count] == shared
                            and len(candidate) > shared_count)
            if not aligned:
                raise ValueError("An incorrect response unexpectedly matches a validated complete target")
            preferred = aligned[0]
            wanted = tuple(sorted({candidate[shared_count] for candidate in aligned}))
            actual = observed[shared_count] if shared_count < len(observed) else None
            status = ("truncated" if actual is None else "semantic" if parsed.answer is not None else "syntactic")
    decoding = {**contract.manifest()["decode"], "endpoint_id": contract.endpoint_id,
                "use_cache": True, "pad_token_id": _pad_id(tokenizer), "eos_token_id": eos,
                "stop_reason": row["stop_reason"], "prompt_representation": contract.prompt_representation}
    return GenerativeFork(checkpoint_sha256, model_id or type(runner.model).__name__, row["item"], row["id"],
                          row["prompt"], hashlib.sha256(rendered.encode()).hexdigest(), prefix, ids, row["text"],
                          parsed.answer, question.answer, original_status, parsed.completed, valid, correct, status,
                          shared, wanted, actual, continuations, preferred, type(runner.tokenizer).__name__,
                          decoding, unaligned)


def protected_trajectory(runner, row: Mapping, contract: BehavioralEndpointContract) -> ProtectedTrajectory:
    """Bind the completed, accepted path, excluding any generation after completion."""
    tokenizer, _, prefix, ids, eos = _transcript(runner, row, contract)
    observed = _completion(tokenizer, ids, eos, contract)
    if observed is None or observed[1] != contract.question(row).answer:
        raise ValueError("Preservation requires a complete and semantically correct parent trajectory")
    return ProtectedTrajectory(row["id"], prefix, observed[0], observed[2])


def _causal_logits(runner, sequences: Sequence[Sequence[int]], *, keep: int):
    if not sequences or any(not sequence or any(type(token) is not int or token < 0 for token in sequence)
                            for sequence in sequences):
        raise ValueError("Expected nonempty causal prefixes with native token IDs")
    torch = _torch()
    ids, mask = _left_pad(torch, sequences, _pad_id(runner.tokenizer), runner.device)
    parameters = inspect.signature(runner.model.forward).parameters
    kwargs = {"use_cache": False}
    if "position_ids" in parameters:
        kwargs["position_ids"] = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = keep
    elif "num_logits_to_keep" in parameters:
        kwargs["num_logits_to_keep"] = keep
    logits = runner.model(input_ids=ids, attention_mask=mask, **kwargs).logits
    if logits.ndim != 3 or logits.shape[0] != len(sequences) or logits.shape[1] < keep:
        raise ValueError("Runtime returned insufficient causal logits")
    return logits[:, -keep:].float()


def next_token_logits(runner, sequences: Sequence[Sequence[int]]):
    """Return full vocabulary next-token logits without detaching the graph."""
    return _causal_logits(runner, sequences, keep=1)[:, 0]


def trajectory_logits(runner, trajectories: Sequence[ProtectedTrajectory]):
    """Score every observed completion decision with one causal forward batch."""
    if not trajectories or any(not path.token_ids for path in trajectories):
        raise ValueError("Expected nonempty observed completion trajectories")
    logits = _causal_logits(runner, [path.prompt_token_ids + path.token_ids[:-1] for path in trajectories],
                            keep=max(len(path.token_ids) for path in trajectories))
    return tuple(logits[i, -len(path.token_ids):] for i, path in enumerate(trajectories))


@dataclass(frozen=True)
class ForkStatistics:
    cross_entropy: object
    margin_loss: object
    actual_margin_loss: object | None
    strongest_rival_ids: object
    predicted_ids: object
    decision_margins: object


def fork_statistics(logits, accepted_token_sets: Sequence[Sequence[int]], *,
                    actual_token_ids: Sequence[int] | None = None) -> ForkStatistics:
    """Compute set cross-entropy, strongest-rival loss and true argmax margins."""
    torch = _torch()
    if logits.ndim != 2 or not len(logits) or len(logits) != len(accepted_token_sets):
        raise ValueError("Expected one full-vocabulary vector per accepted token set")
    z = logits if logits.dtype in (torch.float32, torch.float64) else logits.float()
    if not torch.isfinite(z).all().item():
        raise ValueError("Fork scores require finite full-vocabulary logits")
    mask = torch.zeros_like(z, dtype=torch.bool)
    for row, accepted in enumerate(accepted_token_sets):
        if (not 0 < len(accepted) < z.shape[1] or len(set(accepted)) != len(accepted)
                or any(type(token) is not int or not 0 <= token < z.shape[1] for token in accepted)):
            raise ValueError("Each accepted token set must be a proper nonempty vocabulary subset")
        mask[row, list(accepted)] = True
    accepted_logits = z.masked_fill(~mask, -torch.inf)
    accepted_mass = accepted_logits.logsumexp(dim=1)
    rival, rival_ids = z.masked_fill(mask, -torch.inf).max(dim=1)
    actual_loss = None
    if actual_token_ids is not None:
        if (len(actual_token_ids) != len(z)
                or any(type(token) is not int or not 0 <= token < z.shape[1] or token in accepted
                       for token, accepted in zip(actual_token_ids, accepted_token_sets, strict=True))):
            raise ValueError("Each actual rival must lie outside its accepted token set")
        actual = z.gather(1, torch.tensor(actual_token_ids, device=z.device).view(-1, 1)).squeeze(1)
        actual_loss = torch.nn.functional.softplus(actual - accepted_mass)
    return ForkStatistics(z.logsumexp(dim=1) - accepted_mass,
                          torch.nn.functional.softplus(rival - accepted_mass), actual_loss,
                          rival_ids, z.argmax(dim=1), accepted_logits.max(dim=1).values - rival)


def fork_losses(logits, accepted_token_sets: Sequence[Sequence[int]], *, objective: str = "cross_entropy",
                actual_token_ids: Sequence[int] | None = None):
    scores = fork_statistics(logits, accepted_token_sets, actual_token_ids=actual_token_ids)
    if objective == "cross_entropy":
        return scores.cross_entropy
    if objective == "margin":
        return scores.margin_loss
    if objective == "actual_margin" and scores.actual_margin_loss is not None:
        return scores.actual_margin_loss
    raise ValueError("Unknown fork objective or missing actual rivals")


def fork_loss(logits, accepted_token_sets: Sequence[Sequence[int]], *, objective: str = "cross_entropy",
              actual_token_ids: Sequence[int] | None = None):
    return fork_losses(logits, accepted_token_sets, objective=objective, actual_token_ids=actual_token_ids).mean()


def _token_score(tokenizer, log_probs, token):
    return {"token_id": token, "token_text": tokenizer.decode([token], skip_special_tokens=False),
            "log_probability": float(log_probs[token])}


def with_fork_logits(fork: GenerativeFork, logits, tokenizer, *, top_k: int = 5) -> GenerativeFork:
    """Compare uncached replay against the observed greedy action before fitting."""
    if fork.status not in {"semantic", "syntactic"} or fork.actual_token_id is None:
        raise ValueError("Fork logits require an observed, actionable divergence")
    if logits.ndim != 1 or type(top_k) is not int or not 1 <= top_k <= 32:
        raise ValueError("Expected one full-vocabulary vector and a bounded top-k")
    scores = fork_statistics(logits.unsqueeze(0), [fork.accepted_token_ids], actual_token_ids=[fork.actual_token_id])
    chosen, rival = int(scores.predicted_ids.item()), int(scores.strongest_rival_ids.item())
    status = ("matched" if chosen == fork.actual_token_id else "accepted_already_preferred"
              if chosen in fork.accepted_token_ids else "greedy_replay_mismatch")
    log_probs = logits.detach().float().log_softmax(dim=0)
    top = log_probs.topk(min(top_k, len(log_probs))).indices.tolist()
    return replace(fork, replay_status=status, replay={
        "predicted_token_id": chosen, "strongest_rival_token_id": rival,
        "decision_margin": float(scores.decision_margins.item()), "cross_entropy": float(scores.cross_entropy.item()),
        "margin_loss": float(scores.margin_loss.item()), "actual_margin_loss": float(scores.actual_margin_loss.item()),
        "top": [_token_score(tokenizer, log_probs, token) for token in top],
        "accepted": [_token_score(tokenizer, log_probs, token) for token in fork.accepted_token_ids],
        "actual": _token_score(tokenizer, log_probs, fork.actual_token_id),
    })


def trajectory_diagnostics(trajectory: ProtectedTrajectory, logits, tokenizer) -> dict:
    """Report strict per-decision margins and agreement with cached generation."""
    scores = fork_statistics(logits, [(token,) for token in trajectory.token_ids])
    margins = scores.decision_margins.detach().tolist()
    rivals = scores.strongest_rival_ids.detach().tolist()
    predictions = scores.predicted_ids.detach().tolist()
    return {**trajectory.to_dict(), "all_replayed": predictions == list(trajectory.token_ids),
            "all_strictly_preserved": all(margin > 0 for margin in margins),
            "decisions": [{"position": position, "token_id": token,
                           "token_text": tokenizer.decode([token], skip_special_tokens=False),
                           "predicted_token_id": predicted, "strongest_rival_token_id": rival, "margin": margin}
                          for position, (token, predicted, rival, margin) in enumerate(
                              zip(trajectory.token_ids, predictions, rivals, margins, strict=True))]}
