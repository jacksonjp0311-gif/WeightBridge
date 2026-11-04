"""Comparison-weighted corrective VJPs in a previously audited affine geometry."""
from __future__ import annotations

from collections import defaultdict
import time

import numpy as np

from .counterexample_update import ForkBasis
from .endpoint_update import _tensor_bytes
from .fork_update import _batches, _check_fork_identity, _check_memory, _only_train, _region, _resources, _usable_forks
from .generative_fork import fork_statistics, next_token_logits
from .semantic_update import _column_sketch, _cosines, _group_cosines, _item_batches


def fit_paired_basis(adapter, runner, plan, forks, groups_by_item, *, columns, right_factor,
                     item_by_prompt, batch_size=8, max_forks=64, sketch_seed=20261105,
                     metadata_by_prompt=None):
    """Average each comparison once without deriving a new right factor.

    The positive full-vocabulary greedy decision-margin VJP is computed for each
    authorized replay-qualified fork. Only one full affine item accumulator is
    retained. Projected per-fork rows support individual feasibility reporting.
    """
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    supplied = tuple(forks)
    forks = _usable_forks(supplied)
    _check_fork_identity(supplied, plan.checkpoint_sha256)
    if (type(max_forks) is not int or not 1 <= max_forks <= 64 or not 1 <= len(forks) <= max_forks
            or len(forks) != len(supplied) or len({fork.prompt_id for fork in supplied}) != len(supplied)):
        raise ValueError("Expected bounded distinct replay-qualified paired comparison forks")
    tuple(_batches(forks, batch_size))
    items = defaultdict(list)
    for fork in forks:
        item = item_by_prompt.get(fork.prompt_id)
        if not isinstance(item, str) or not item or not groups_by_item.get(item):
            raise ValueError("Every fork needs a comparison identity and semantic group")
        items[item].append(fork)
    packed_batches = tuple(_item_batches(items, batch_size))
    limits = plan.limits
    if len(packed_batches) > limits.max_forward_calls or len(forks) > limits.max_backward_calls:
        raise ValueError("Paired gradient collection exceeds the authorized compute budget")
    ref, module, raw = _region(adapter, runner, plan, torch)
    columns = tuple(columns)
    right = np.asarray(right_factor, dtype=np.float64)
    if (tuple(sorted(set(columns))) != columns or not columns or columns[0] < 0
            or columns[-1] >= ref.shape[1] or len(columns) > plan.max_columns
            or len(columns) * ref.shape[0] > limits.max_update_entries
            or right.shape != (len(columns),) or not np.isfinite(right).all()
            or not np.isclose(np.linalg.norm(right), 1., atol=1e-10, rtol=1e-10)):
        raise ValueError("Pinned rank-one geometry does not match the authorized affine region")
    sketch_width = min(32, ref.shape[1])
    rng = np.random.default_rng(sketch_seed)
    buckets = rng.permutation(ref.shape[1]) % sketch_width
    signs = rng.choice(np.array([-1., 1.], dtype=np.float32), ref.shape[1])
    rows, norms, sketches, slopes, margins, ordered_forks = [], [], [], [], [], []
    forwards = backwards = 0
    try:
        with _only_train(runner.model, (module.weight,)):
            for packed in packed_batches:
                batch = [fork for _, presentations in packed for fork in presentations]
                stats = fork_statistics(next_token_logits(runner, [fork.prefix_ids for fork in batch]),
                    [fork.accepted_token_ids for fork in batch], actual_token_ids=[fork.actual_token_id for fork in batch])
                forwards += 1
                if stats.predicted_ids.detach().tolist() != [fork.actual_token_id for fork in batch]:
                    raise ValueError("Paired margin replay disagrees with the observed greedy decision")
                offset = 0
                for _, presentations in packed:
                    item_gradient = np.zeros(ref.shape, dtype=np.float64)
                    for local_index, fork in enumerate(presentations):
                        index = offset + local_index
                        value, = torch.autograd.grad(stats.decision_margins[index], (module.weight,),
                                                     retain_graph=index + 1 < len(batch))
                        backwards += 1
                        value = value.detach().float().cpu().numpy().astype(np.float64)
                        if not np.isfinite(value).all():
                            raise ValueError("Nonfinite paired affine margin gradient")
                        item_gradient += value / len(presentations)
                        slopes.append(value[:, columns] @ right)
                        margins.append(float(stats.decision_margins[index].detach()))
                        ordered_forks.append(fork)
                        _check_memory(limits.max_memory_bytes)
                    rows.append(item_gradient[:, columns] @ right)
                    norms.append(float(np.linalg.norm(item_gradient)))
                    sketches.append(_column_sketch(item_gradient, buckets, signs, sketch_width))
                    offset += len(presentations)
    finally:
        if _tensor_bytes(module.weight, torch) != raw:
            raise RuntimeError("Paired gradient collection changed the live parent tensor")
    names, rows = tuple(items), np.asarray(rows)
    groups = tuple(groups_by_item[item] for item in names)
    preferred = rows.mean(axis=0)
    norm = float(np.linalg.norm(preferred))
    preferred = preferred / norm if norm > 1e-12 else np.zeros_like(preferred)
    metadata_by_prompt = metadata_by_prompt or {}
    item_margins = []
    cursor = 0
    for presentations in items.values():
        item_margins.append(float(np.mean(margins[cursor:cursor + len(presentations)])))
        cursor += len(presentations)
    evidence = {
        "schema": "weightbridge.paired_fixed_basis.v1", "objective": "positive_full_vocabulary_greedy_margin",
        "scaling": "RAW", "item_ids": list(names), "item_groups": list(groups),
        "item_prompt_ids": {item: [fork.prompt_id for fork in members] for item, members in items.items()},
        "gradient_unit": "Arithmetic mean over eligible replay-qualified presentations within each comparison",
        "columns": list(columns), "right_factor": right.tolist(), "geometry_reestimated": False,
        "full_gradient_norms": norms, "projected_gradients": rows.tolist(),
        "projected_norms": np.linalg.norm(rows, axis=1).tolist(),
        "comparison_parent_margins": item_margins,
        "projected_item_cosines": _cosines(rows),
        "full_gradient_agreement": {"method": "signed_input_column_sketch", "width": sketch_width,
            "seed": sketch_seed, "exact": sketch_width == ref.shape[1], "item_cosines": _cosines(sketches),
            "groups": _group_cosines(sketches, groups)},
        "forks": [{"prompt_id": fork.prompt_id, "comparison_id": item_by_prompt[fork.prompt_id],
            "accepted_token_ids": list(fork.accepted_token_ids), "actual_token_id": fork.actual_token_id,
            "parent_margin": margin, "projected_margin_gradient": slope.tolist(),
            "metadata": metadata_by_prompt.get(fork.prompt_id, {})}
            for fork, margin, slope in zip(ordered_forks, margins, slopes, strict=True)],
        "resources": _resources(wall, cpu, forwards, backwards),
    }
    basis = ForkBasis(plan.checkpoint_sha256, plan.region.tensor_name, columns, preferred, right,
        tuple(fork.prompt_id for fork in ordered_forks),
        tuple(item_by_prompt[fork.prompt_id] for fork in ordered_forks), np.asarray(margins), np.asarray(slopes))
    return basis, rows, evidence
