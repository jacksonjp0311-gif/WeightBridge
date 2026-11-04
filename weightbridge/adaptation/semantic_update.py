"""Item-level affine sensitivities for a shared-right-factor semantic study."""

from __future__ import annotations

from collections import defaultdict
import time

import numpy as np

from .behavior_endpoint import endpoint_runner
from .content_scoring import prepare_continuations, score_continuations
from .counterexample_update import ForkBasis
from .endpoint_update import _tensor_bytes
from .fork_update import _batches, _check_fork_identity, _check_memory, _only_train, _region, _resources, _usable_forks
from .generative_fork import fork_statistics, next_token_logits


def _column_sketch(matrix, buckets, signs, width):
    """Hash input coordinates, retaining the output axis for each bucket."""
    sketch = np.zeros((matrix.shape[0], width), dtype=np.float64)
    for bucket in range(width):
        chosen = buckets == bucket
        sketch[:, bucket] = matrix[:, chosen] @ signs[chosen]
    return sketch.ravel()


def _cosines(rows):
    rows = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1)
    units = np.divide(rows, norms[:, None], out=np.zeros_like(rows), where=norms[:, None] > 0)
    cosines = np.clip(units @ units.T, -1., 1.)
    return [[float(value) if norms[i] and norms[j] else None for j, value in enumerate(row)]
            for i, row in enumerate(cosines)]


def _group_cosines(vectors, groups):
    names = tuple(dict.fromkeys(groups))
    directions = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(directions, axis=1)
    directions = np.divide(directions, norms[:, None], out=np.zeros_like(directions), where=norms[:, None] > 0)
    means = [directions[[value == group for value in groups]].mean(axis=0) for group in names]
    return {"groups": list(names), "counts": [groups.count(group) for group in names], "cosine": _cosines(means)}


def _item_batches(items, batch_size):
    """Pack full items together so one backward pass measures their joint loss."""
    packed, count = [], 0
    for item, presentations in items.items():
        if len(presentations) > batch_size:
            raise ValueError("A semantic item's presentations must fit in one gradient batch")
        if count + len(presentations) > batch_size:
            yield packed
            packed, count = [], 0
        packed.append((item, presentations))
        count += len(presentations)
    if packed:
        yield packed


def fit_semantic_basis(adapter, runner, plan, forks, groups_by_item, *, item_by_prompt=None, batch_size=8,
                       sketch_seed=20261103, max_forks=64):
    """Fit the historical CE basis and item-weighted projected margin gradients.

    Only one full affine gradient and the CE accumulator are retained at a time.
    The historical basis uses the unweighted mean over fitted presentations.
    Each consensus item is the arithmetic mean of its own presentation VJPs.
    """
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    supplied = tuple(forks)
    forks = _usable_forks(supplied)
    limits = plan.limits
    _check_fork_identity(supplied, plan.checkpoint_sha256)
    if (type(max_forks) is not int or not 1 <= max_forks <= 64
            or not 1 <= len(forks) <= max_forks
            or len({fork.prompt_id for fork in supplied}) != len(supplied)):
        raise ValueError("Expected bounded distinct replay-matched semantic forks")
    # The existing runtime cap also validates batch_size.
    margin_batches = tuple(_batches(forks, batch_size))
    items = defaultdict(list)
    item_ids = {}
    for fork in forks:
        item = fork.question_id if item_by_prompt is None else item_by_prompt.get(fork.prompt_id)
        group = groups_by_item.get(item)
        if not isinstance(item, str) or not item or not isinstance(group, str) or not group:
            raise ValueError("Every fitted fork needs a semantic item and a named group")
        item_ids[fork.prompt_id] = item
        items[item].append(fork)
    ce_batches = tuple(_item_batches(items, batch_size))
    if (len(ce_batches) + len(margin_batches) > limits.max_forward_calls
            or len(items) + len(forks) > limits.max_backward_calls):
        raise ValueError("Item gradient collection exceeds the authorized compute budget")
    _check_memory(limits.max_memory_bytes)
    ref, module, raw = _region(adapter, runner, plan, torch)
    width = min(16, ref.shape[1], plan.max_columns, limits.max_update_entries // ref.shape[0])
    if width < 1 or limits.max_rank < 1:
        raise ValueError("Semantic adaptation requires authorized selected affine columns")
    sketch_width = min(32, ref.shape[1])
    rng = np.random.default_rng(sketch_seed)
    # Assign every input coordinate once, balancing bucket sizes. Small layers
    # therefore get exact cosines. Wider layers produce a single seeded estimate.
    buckets = rng.permutation(ref.shape[1]) % sketch_width
    signs = rng.choice(np.array([-1., 1.], dtype=np.float32), ref.shape[1])
    forwards, backwards = 0, 0

    def forward(batch):
        nonlocal forwards
        forwards += 1
        logits = next_token_logits(runner, [fork.prefix_ids for fork in batch])
        result = fork_statistics(logits, [fork.accepted_token_ids for fork in batch],
                                 actual_token_ids=[fork.actual_token_id for fork in batch])
        if result.predicted_ids.detach().tolist() != [fork.actual_token_id for fork in batch]:
            raise ValueError("Semantic fork gradient replay disagreed with the original greedy token")
        _check_memory(limits.max_memory_bytes)
        return result

    def derivative(value, retain_graph):
        nonlocal backwards
        backwards += 1
        gradient, = torch.autograd.grad(value, (module.weight,), retain_graph=retain_graph)
        _check_memory(limits.max_memory_bytes)
        return gradient.detach()

    ce_norms, sketches = [], []
    margins, slopes = [], []
    singular = None
    try:
        with _only_train(runner.model, (module.weight,)):
            mean = np.zeros(ref.shape, dtype=np.float64)
            for packed in ce_batches:
                stats = forward([fork for _, presentations in packed for fork in presentations])
                offset = 0
                for index, (_, presentations) in enumerate(packed):
                    count = len(presentations)
                    gradient = derivative(stats.cross_entropy[offset:offset + count].mean(),
                                          index + 1 < len(packed)).float().cpu().numpy()
                    if not np.isfinite(gradient).all():
                        raise ValueError("Nonfinite full-region semantic gradient")
                    mean += (count / len(forks)) * gradient
                    ce_norms.append(float(np.sqrt(np.einsum("ij,ij->", gradient, gradient, dtype=np.float64))))
                    sketches.append(_column_sketch(gradient, buckets, signs, sketch_width))
                    offset += count
            column_norms = np.linalg.norm(mean, axis=0)
            columns = tuple(sorted(np.argsort(-column_norms, kind="stable")[:width].tolist()))
            if float(np.linalg.norm(mean[:, columns])) > 1e-12:
                left, singular, right = np.linalg.svd(-mean[:, columns], full_matrices=False)
                preferred, right_factor = left[:, 0], right[0]
                for batch in margin_batches:
                    stats = forward(batch)
                    margins.extend(stats.decision_margins.detach().tolist())
                    for index, value in enumerate(stats.decision_margins):
                        gradient = derivative(value, index + 1 < len(batch))
                        projected = gradient[:, columns].float().cpu().numpy().astype(np.float64) @ right_factor
                        if not np.isfinite(projected).all():
                            raise ValueError("Nonfinite projected corrective margin gradient")
                        slopes.append(projected)
    finally:
        if _tensor_bytes(module.weight, torch) != raw:
            raise RuntimeError("Semantic gradient collection changed the live parent")
    groups = tuple(groups_by_item[item] for item in items)
    evidence = {
        "schema": "weightbridge.semantic_rank_one_basis.v1",
        "resources": _resources(wall, cpu, forwards, backwards),
        "item_ids": list(items), "item_groups": list(groups),
        "item_prompt_ids": {item: [fork.prompt_id for fork in presentations] for item, presentations in items.items()},
        "excluded_fork_prompt_ids": [fork.prompt_id for fork in supplied if fork.prompt_id not in item_ids],
        "historical_basis_formulation": "presentation-weighted mean fork CE from streamed item VJPs",
        "ce_item_gradient_norms": ce_norms,
        "ce_item_gradient_agreement": {
            "method": "signed_input_column_sketch", "width": sketch_width, "seed": sketch_seed,
            "exact": sketch_width == ref.shape[1], "item_cosine": _cosines(sketches),
            "group_cosine": _group_cosines(sketches, groups),
        },
        "historical_target_gradient_norm": float(np.linalg.norm(mean)),
    }
    if singular is None:
        return None, None, {**evidence, "reason": "no_finite_target_direction"}
    gradients = np.asarray(slopes)
    item_rows = [item_ids[fork.prompt_id] for fork in forks]
    item_gradients = np.stack([gradients[[row == item for row in item_rows]].mean(axis=0) for item in items])
    basis = ForkBasis(plan.checkpoint_sha256, plan.region.tensor_name, columns, preferred, right_factor,
                      tuple(fork.prompt_id for fork in forks), tuple(item_rows), np.asarray(margins), gradients)
    return basis, item_gradients, {
        **evidence, "reason": "measured", "columns": list(columns), "right_factor": right_factor.tolist(),
        "top_selected_singular_value": float(singular[0]),
        "projected_margin_gradient_agreement": {
            "exact": True, "item_cosine": _cosines(item_gradients),
            "group_cosine": _group_cosines(item_gradients, groups),
            "item_gradient_norms": np.linalg.norm(item_gradients, axis=1).tolist(),
        },
        "item_historical_slopes": (item_gradients @ preferred).tolist(),
        "presentation_historical_slopes": (gradients @ preferred).tolist(),
        "presentation_parent_margins": margins,
        "presentation_prompt_ids": list(basis.target_ids), "presentation_items": list(basis.target_items),
    }


def score_teacher_forced_options(runner, rows, contract, *, max_forward_calls, max_memory_bytes):
    """Compare displayed answer likelihoods as diagnostics under the same frozen prefix."""
    wall, cpu = time.perf_counter(), time.process_time()
    runner = endpoint_runner(runner, contract)
    records, prepared = {}, {}
    for row in rows:
        question = contract.question(row)
        operands = tuple(row["operands"])
        if question.allowed_answers is None or len(operands) != 2:
            continue
        canonical = tuple(contract.parse(value, "eos").answer for value in operands)
        if (len(set(canonical)) != 2 or set(canonical) != set(question.allowed_answers)
                or row["id"] in records):
            raise ValueError("Teacher-forced options must correspond to distinct, validated displayed answers")
        record = {"prompt_id": row["id"], "displayed": list(operands), "canonical": list(canonical),
                  "true_answer": question.answer, "greedy_answer": row.get("exact"),
                  "greedy_valid": row.get("valid"), "sum_preference": None, "mean_preference": None}
        records[row["id"]] = record
        try:
            continuations = prepare_continuations(runner.tokenizer, [row["prompt"]], [operands],
                                                 max_input_tokens=contract.max_input_tokens)
        except ValueError as exc:
            record.update(status="unaligned_or_over_budget", reason=str(exc))
        else:
            prepared[row["id"]] = continuations
    sequences = tuple(sequence for options in prepared.values() for sequence in options)
    batches = tuple(_batches(sequences, contract.batch_size))
    if len(batches) > max_forward_calls:
        raise ValueError("Teacher-forced option diagnostic exceeds the authorized forward budget")
    measurements, padded_positions = [], 0
    _check_memory(max_memory_bytes)
    for batch in batches:
        scored, work = score_continuations(runner, batch, batch_size=contract.batch_size)
        measurements.extend(scored)
        padded_positions += work["padded_input_token_positions"]
        _check_memory(max_memory_bytes)
    for index, (prompt_id, continuations) in enumerate(prepared.items()):
        record = records[prompt_id]
        scores = measurements[index * 2:index * 2 + 2]
        record.update(status="measured",
                      options=[{"answer_token_ids": list(option.answer_ids), **score}
                               for option, score in zip(continuations, scores, strict=True)])
        for metric in ("sum", "mean"):
            first, second = (score[f"{metric}_log_probability"] for score in scores)
            preference = None if first == second else record["canonical"][int(second > first)]
            record[f"{metric}_preference"] = preference
            record[f"{metric}_matches_greedy"] = (preference == record["greedy_answer"]
                                                   if preference is not None and record["greedy_valid"] else None)
    return {key: row["sum_preference"] for key, row in records.items()}, {
        "schema": "weightbridge.semantic_teacher_forced_options.v1",
        "comparison": "displayed answer surfaces; no EOS or newline appended; generation remains authoritative",
        "rows": list(records.values()),
        "resources": _resources(wall, cpu, len(batches), 0, padded_input_token_positions=padded_positions),
    }
