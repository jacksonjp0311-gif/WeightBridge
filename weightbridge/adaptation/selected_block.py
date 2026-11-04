"""Native decision derivatives retained only in an authorized affine block.

These are parent-model measurements, not weight proposals. Context-only rows
reproduce diagnostic execution but never enter a corrective objective.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from .counterexample_update import _prefix
from .endpoint_update import _tensor_bytes
from .fork_update import _batches, _check_fork_identity, _check_memory, _only_train, _region, _resources, _usable_forks
from .generative_fork import fork_statistics, next_token_logits
from .jspace_update import _known_anchors
from .paired_update import _replay_contexts


class SelectedBlockReplayError(ValueError):
    """Required parent decisions did not reproduce in their native context."""

    def __init__(self, evidence):
        super().__init__("Selected-block margin replay disagrees with the required parent decision")
        self.evidence = evidence


def block_hash(block):
    """Bind shape and little-endian float64 values independently of NPZ packing."""
    values = np.ascontiguousarray(block, dtype="<f8")
    return hashlib.sha256(str(values.shape).encode("ascii") + b"\0" + values.tobytes()).hexdigest()


def _prefix_hash(prefix):
    return hashlib.sha256(json.dumps(list(prefix), separators=(",", ":")).encode("ascii")).hexdigest()


def _open(adapter, runner, plan, columns, count, torch):
    ref, module, raw = _region(adapter, runner, plan, torch)
    columns = tuple(columns)
    limits = plan.limits
    if (not columns or any(type(column) is not int for column in columns)
            or tuple(sorted(set(columns))) != columns
            or columns[0] < 0 or columns[-1] >= ref.shape[1]
            or len(columns) > min(16, plan.max_columns) or ref.shape[0] > 1024
            or len(columns) * ref.shape[0] > limits.max_update_entries
            or not 0 <= count <= min(64, limits.max_backward_calls)
            or count * ref.shape[0] * len(columns) * 8 > limits.max_memory_bytes // 8):
        raise ValueError("Selected-block derivatives exceed the authorized region or memory/compute budget")
    _check_memory(limits.max_memory_bytes)
    identity = {"checkpoint_sha256": plan.checkpoint_sha256,
        "tensor_name": plan.region.tensor_name, "source_tensor_sha256": hashlib.sha256(raw).hexdigest(),
        "source_tensor_shape": list(ref.shape), "source_dtype": getattr(ref, "dtype", str(module.weight.dtype)),
        "columns": list(columns), "block_shape": [ref.shape[0], len(columns)],
        "objective": "positive_full_vocabulary_greedy_decision_margin",
        "gradient_orientation": "positive derivative of accepted maximum minus strongest rejected maximum",
        "retained_full_parameter_gradients": False}
    return columns, ref, module, raw, identity


def _derivative(value, module, columns, retain_graph, torch):
    full, = torch.autograd.grad(value, (module.weight,), retain_graph=retain_graph)
    # The runtime may transiently construct a full parameter VJP, but only this
    # copied selected block crosses the operation boundary or enters storage.
    selected = full.detach()[:, columns].float().cpu().numpy().astype(np.float64, copy=True)
    del full
    if not np.isfinite(selected).all():
        raise ValueError("Selected-block derivative contains nonfinite values")
    return selected


def _restored(module, raw, torch):
    if _tensor_bytes(module.weight, torch) != raw:
        raise RuntimeError("Selected-block derivative collection changed the live parent tensor")


def collect_target_blocks(adapter, runner, plan, forks, *, columns, replay_batches,
                          metadata_by_prompt=None, batch_size=8):
    """Return ``(blocks, evidence)`` in the supplied required-fork order.

    Every selected fork must replay the original greedy wrong token. Its full
    authorized column block is retained; no averaging, projection or example
    normalization is applied. All context rows remain visible in provenance.
    """
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    forks = tuple(forks)
    _check_fork_identity(forks, plan.checkpoint_sha256)
    if (not forks or len(forks) != len(_usable_forks(forks))
            or len({fork.prompt_id for fork in forks}) != len(forks)):
        raise ValueError("Expected distinct replay-qualified required target forks")
    batches = _replay_contexts({"required": forks}, replay_batches, batch_size,
                               plan.checkpoint_sha256)["required"]
    if len(batches) > plan.limits.max_forward_calls:
        raise ValueError("Selected target contexts exceed the explicit forward budget")
    columns, ref, module, raw, identity = _open(adapter, runner, plan, columns, len(forks), torch)
    metadata_by_prompt = metadata_by_prompt or {}
    required = {fork.prompt_id for fork in forks}
    values, records, packing = {}, {}, []
    forwards = backwards = 0
    try:
        with _only_train(runner.model, (module.weight,)):
            for batch in batches:
                stats = fork_statistics(next_token_logits(runner, [fork.prefix_ids for fork in batch]),
                    [fork.accepted_token_ids for fork in batch],
                    actual_token_ids=[fork.actual_token_id for fork in batch])
                forwards += 1
                winners = stats.predicted_ids.detach().tolist()
                rivals = stats.strongest_rival_ids.detach().tolist()
                margins = stats.decision_margins.detach().tolist()
                context = {"prompt_ids": [fork.prompt_id for fork in batch],
                    "native_prefix_ids": [list(fork.prefix_ids) for fork in batch],
                    "prefix_lengths": [len(fork.prefix_ids) for fork in batch],
                    "selected_prompt_ids": [fork.prompt_id for fork in batch if fork.prompt_id in required],
                    "original_token_ids": [fork.actual_token_id for fork in batch],
                    "replay_token_ids": winners}
                packing.append(context)
                indices = [i for i, fork in enumerate(batch) if fork.prompt_id in required]
                bad = [batch[i].prompt_id for i in indices
                       if winners[i] != batch[i].actual_token_id
                       or winners[i] in batch[i].accepted_token_ids or margins[i] > 0]
                if bad:
                    raise SelectedBlockReplayError({**identity, "reason": "required_target_context_replay_mismatch",
                        "mismatched_prompt_ids": bad, "packing": packing,
                        "resources": _resources(wall, cpu, forwards, backwards)})
                for number, i in enumerate(indices):
                    fork = batch[i]
                    block = _derivative(stats.decision_margins[i], module, columns,
                                        number + 1 < len(indices), torch)
                    backwards += 1
                    values[fork.prompt_id] = block
                    records[fork.prompt_id] = {"prompt_id": fork.prompt_id,
                        "model_id": getattr(fork, "model_id", None),
                        "question_id": getattr(fork, "question_id", None),
                        "metadata": metadata_by_prompt.get(fork.prompt_id, {}),
                        "native_prefix_ids": list(fork.prefix_ids), "native_prefix_sha256": _prefix_hash(fork.prefix_ids),
                        "accepted_token_ids": list(fork.accepted_token_ids),
                        "actual_token_id": fork.actual_token_id, "strongest_rejected_token_id": rivals[i],
                        "parent_margin": float(margins[i]), "replay_argmax_token_id": winners[i],
                        "original_context_index": len(packing) - 1,
                        "gradient_norm": float(np.linalg.norm(block)), "gradient_block_sha256": block_hash(block)}
                    _check_memory(plan.limits.max_memory_bytes)
    finally:
        _restored(module, raw, torch)
    blocks = np.stack([values[fork.prompt_id] for fork in forks])
    return blocks, {"schema": "weightbridge.selected_target_blocks.v1", **identity,
        "rows": [records[fork.prompt_id] for fork in forks],
        "block_array_sha256": block_hash(blocks), "packing": packing,
        "context_only_rows_have_supervision_weight": False,
        "parent_restored_exactly": True, "resources": _resources(wall, cpu, forwards, backwards)}


def collect_preservation_blocks(adapter, runner, plan, pool, anchors, *, columns, batch_size=8):
    """Return full selected blocks for every supplied protected native prefix.

    The exact batched prefix order is retained in the evidence. A nonpositive
    margin or changed parent token refuses the entire required collection;
    no protected row is silently omitted.
    """
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    anchors = _known_anchors(pool, anchors)
    batches = tuple(_batches(anchors, batch_size))
    if len(batches) > plan.limits.max_forward_calls:
        raise ValueError("Selected preservation contexts exceed the explicit forward budget")
    for anchor in anchors:
        path = pool.trajectories.get(anchor.prompt_id)
        if (path is None or not 0 <= anchor.position < len(path.token_ids)
                or path.token_ids[anchor.position] != anchor.token_id):
            raise ValueError("Selected preservation anchor differs from its parent trajectory")
    columns, ref, module, raw, identity = _open(adapter, runner, plan, columns, len(anchors), torch)
    blocks, records, packing = [], [], []
    forwards = backwards = 0
    try:
        with _only_train(runner.model, (module.weight,)):
            for batch in batches:
                prefixes = [_prefix(pool, anchor) for anchor in batch]
                stats = fork_statistics(next_token_logits(runner, prefixes), [(anchor.token_id,) for anchor in batch])
                forwards += 1
                winners = stats.predicted_ids.detach().tolist()
                rivals = stats.strongest_rival_ids.detach().tolist()
                margins = stats.decision_margins.detach().tolist()
                context = {"prompt_ids": [anchor.prompt_id for anchor in batch],
                    "positions": [anchor.position for anchor in batch],
                    "native_prefix_ids": [list(prefix) for prefix in prefixes],
                    "original_token_ids": [anchor.token_id for anchor in batch], "replay_token_ids": winners}
                packing.append(context)
                bad = [list(anchor.key) for anchor, margin, winner in zip(batch, margins, winners, strict=True)
                       if margin <= 0 or winner != anchor.token_id]
                if bad:
                    raise SelectedBlockReplayError({**identity, "reason": "required_preservation_context_replay_mismatch",
                        "mismatched_decisions": bad, "packing": packing,
                        "resources": _resources(wall, cpu, forwards, backwards)})
                for i, anchor in enumerate(batch):
                    block = _derivative(stats.decision_margins[i], module, columns, i + 1 < len(batch), torch)
                    backwards += 1
                    blocks.append(block)
                    records.append({"prompt_id": anchor.prompt_id, "item": anchor.item, "group": anchor.group,
                        "wording": anchor.wording, "position": anchor.position,
                        "native_prefix_ids": list(prefixes[i]), "native_prefix_sha256": _prefix_hash(prefixes[i]),
                        "accepted_token_ids": [anchor.token_id], "actual_token_id": anchor.token_id,
                        "strongest_rejected_token_id": rivals[i], "parent_margin": float(margins[i]),
                        "pool_parent_margin": anchor.margin, "replay_argmax_token_id": winners[i],
                        "original_context_index": len(packing) - 1,
                        "gradient_norm": float(np.linalg.norm(block)), "gradient_block_sha256": block_hash(block)})
                    _check_memory(plan.limits.max_memory_bytes)
    finally:
        _restored(module, raw, torch)
    array = np.stack(blocks) if blocks else np.empty((0, ref.shape[0], len(columns)))
    return array, {"schema": "weightbridge.selected_preservation_blocks.v1", **identity,
        "rows": records, "block_array_sha256": block_hash(array), "packing": packing,
        "parent_restored_exactly": True, "resources": _resources(wall, cpu, forwards, backwards)}
