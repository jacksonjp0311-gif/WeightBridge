"""Equal-comparison CE and decision-margin VJPs in one authorized rank-one space."""
from __future__ import annotations

from collections import defaultdict
import time

import numpy as np

from .counterexample_update import ForkBasis
from .endpoint_update import _tensor_bytes
from .fork_update import _batches, _check_fork_identity, _check_memory, _only_train, _region, _resources, _usable_forks
from .generative_fork import fork_statistics, next_token_logits
from .semantic_update import _column_sketch, _cosines, _group_cosines, _item_batches


def fit_factorial_basis(adapter, runner, plan, forks, groups_by_item, *, item_by_prompt=None,
                        batch_size=8, max_forks=32, sketch_seed=20261104, metadata_by_prompt=None):
    """Measure improving CE and margin rows after fixing columns and right factor.

    Every comparison contributes one equally weighted average to the shared CE
    basis. The second pass returns one projected CE and margin row per comparison.
    Individual margin VJPs remain available for per-fork feasibility diagnostics.
    """
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    supplied = tuple(forks)
    forks = _usable_forks(supplied)
    limits = plan.limits
    _check_fork_identity(supplied, plan.checkpoint_sha256)
    if (type(max_forks) is not int or not 1 <= max_forks <= 32
            or not 1 <= len(forks) <= max_forks
            or len({fork.prompt_id for fork in supplied}) != len(supplied)):
        raise ValueError("Expected bounded distinct replay-matched comparison forks")
    tuple(_batches(forks, batch_size))
    items, item_ids = defaultdict(list), {}
    for fork in forks:
        item = fork.question_id if item_by_prompt is None else item_by_prompt.get(fork.prompt_id)
        group = groups_by_item.get(item)
        if not isinstance(item, str) or not item or not isinstance(group, str) or not group:
            raise ValueError("Every fitted comparison needs an item identity and named group")
        items[item].append(fork)
        item_ids[fork.prompt_id] = item
    packed_batches = tuple(_item_batches(items, batch_size))
    if (2 * len(packed_batches) > limits.max_forward_calls
            or 2 * len(items) + len(forks) > limits.max_backward_calls):
        raise ValueError("Factorial gradient collection exceeds the authorized compute budget")
    _check_memory(limits.max_memory_bytes)
    ref, module, raw = _region(adapter, runner, plan, torch)
    width = min(16, ref.shape[1], plan.max_columns, limits.max_update_entries // ref.shape[0])
    if width < 1 or limits.max_rank < 1:
        raise ValueError("Factorial gradients require authorized selected affine columns")
    sketch_width = min(32, ref.shape[1])
    rng = np.random.default_rng(sketch_seed)
    buckets = rng.permutation(ref.shape[1]) % sketch_width
    signs = rng.choice(np.array([-1., 1.], dtype=np.float32), ref.shape[1])
    forwards = backwards = 0

    def forward(batch):
        nonlocal forwards
        forwards += 1
        logits = next_token_logits(runner, [fork.prefix_ids for fork in batch])
        stats = fork_statistics(logits, [fork.accepted_token_ids for fork in batch],
                                actual_token_ids=[fork.actual_token_id for fork in batch])
        if stats.predicted_ids.detach().tolist() != [fork.actual_token_id for fork in batch]:
            raise ValueError("Factorial gradient replay disagrees with the observed greedy token")
        _check_memory(limits.max_memory_bytes)
        return stats

    def derivative(objective, retain_graph):
        nonlocal backwards
        backwards += 1
        value, = torch.autograd.grad(objective, (module.weight,), retain_graph=retain_graph)
        _check_memory(limits.max_memory_bytes)
        value = value.detach().float().cpu().numpy().astype(np.float64)
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite factorial affine gradient")
        return value

    ce_norms, margin_norms, ce_sketches, margin_sketches = [], [], [], []
    ce_rows, margin_rows, fork_slopes, margins = [], [], [], []
    singular = None
    try:
        with _only_train(runner.model, (module.weight,)):
            mean = np.zeros(ref.shape, dtype=np.float64)
            for packed in packed_batches:
                stats = forward([fork for _, presentations in packed for fork in presentations])
                offset = 0
                for index, (_, presentations) in enumerate(packed):
                    count = len(presentations)
                    improving = derivative(-stats.cross_entropy[offset:offset + count].mean(),
                                            index + 1 < len(packed))
                    mean += improving / len(items)
                    ce_norms.append(float(np.linalg.norm(improving)))
                    ce_sketches.append(_column_sketch(improving, buckets, signs, sketch_width))
                    offset += count
            column_norms = np.linalg.norm(mean, axis=0)
            columns = tuple(sorted(np.argsort(-column_norms, kind="stable")[:width].tolist()))
            if float(np.linalg.norm(mean[:, columns])) > 1e-12:
                left, singular, right = np.linalg.svd(mean[:, columns], full_matrices=False)
                preferred, right_factor = left[:, 0], right[0]
                for packed in packed_batches:
                    stats = forward([fork for _, presentations in packed for fork in presentations])
                    offset = 0
                    for item_index, (_, presentations) in enumerate(packed):
                        count = len(presentations)
                        ce_gradient = derivative(-stats.cross_entropy[offset:offset + count].mean(), True)
                        ce_rows.append(ce_gradient[:, columns] @ right_factor)
                        item_margin = np.zeros(ref.shape, dtype=np.float64)
                        for local_index, _ in enumerate(presentations):
                            index = offset + local_index
                            last = item_index + 1 == len(packed) and local_index + 1 == count
                            gradient = derivative(stats.decision_margins[index], not last)
                            item_margin += gradient / count
                            fork_slopes.append(gradient[:, columns] @ right_factor)
                            margins.append(float(stats.decision_margins[index].detach()))
                        margin_rows.append(item_margin[:, columns] @ right_factor)
                        margin_norms.append(float(np.linalg.norm(item_margin)))
                        margin_sketches.append(_column_sketch(item_margin, buckets, signs, sketch_width))
                        offset += count
    finally:
        if _tensor_bytes(module.weight, torch) != raw:
            raise RuntimeError("Factorial gradient collection changed the live parent tensor")

    names = tuple(items)
    groups = tuple(groups_by_item[item] for item in names)
    evidence = {
        "schema": "weightbridge.factorial_rank_one_basis.v1",
        "reason": "measured" if singular is not None else "no_finite_shared_ce_direction",
        "item_ids": list(names), "item_groups": list(groups),
        "item_prompt_ids": {item: [fork.prompt_id for fork in members] for item, members in items.items()},
        "shared_basis": "equally weighted comparison means of improving full-vocabulary fork CE",
        "item_weight": 1. / len(items),
        "full_gradient_norms": {"CE": ce_norms, "MARGIN": margin_norms},
        "full_gradient_agreement": {
            "method": "signed_input_column_sketch", "width": sketch_width, "seed": sketch_seed,
            "exact": sketch_width == ref.shape[1],
            "CE": {"item_cosines": _cosines(ce_sketches), "groups": _group_cosines(ce_sketches, groups)},
            "MARGIN": {"item_cosines": _cosines(margin_sketches), "groups": _group_cosines(margin_sketches, groups)}
                      if margin_sketches else None,
        },
        "shared_full_ce_gradient_norm": float(np.linalg.norm(mean)),
        "excluded_fork_prompt_ids": [fork.prompt_id for fork in supplied if fork.prompt_id not in item_ids],
        "resources": _resources(wall, cpu, forwards, backwards),
    }
    if singular is None:
        return None, None, None, evidence
    ce_rows, margin_rows = np.asarray(ce_rows), np.asarray(margin_rows)
    ordered_forks = tuple(fork for presentations in items.values() for fork in presentations)
    metadata_by_prompt = metadata_by_prompt or {}
    evidence.update({
        "columns": list(columns), "right_factor": right_factor.tolist(),
        "top_selected_singular_value": float(singular[0]),
        "projected_gradients": {"CE": ce_rows.tolist(), "MARGIN": margin_rows.tolist()},
        "projected_norms": {"CE": np.linalg.norm(ce_rows, axis=1).tolist(),
                            "MARGIN": np.linalg.norm(margin_rows, axis=1).tolist()},
        "projected_item_cosines": {"CE": _cosines(ce_rows), "MARGIN": _cosines(margin_rows)},
        "forks": [{"prompt_id": fork.prompt_id, "comparison_id": item_ids[fork.prompt_id],
                   "accepted_token_ids": list(fork.accepted_token_ids), "actual_token_id": fork.actual_token_id,
                   "parent_margin": margin, "projected_margin_gradient": slope.tolist(),
                   "metadata": metadata_by_prompt.get(fork.prompt_id, {})}
                  for fork, margin, slope in zip(ordered_forks, margins, fork_slopes, strict=True)],
    })
    basis = ForkBasis(plan.checkpoint_sha256, plan.region.tensor_name, columns, preferred, right_factor,
                      tuple(fork.prompt_id for fork in ordered_forks), tuple(item_ids[fork.prompt_id] for fork in ordered_forks),
                      np.asarray(margins), np.asarray(fork_slopes))
    return basis, ce_rows, margin_rows, evidence


def measure_fork_token_classes(runner, forks, *, batch_size=8, max_forward_calls, max_memory_bytes):
    """Measure accepted mass and explicit native digit/termination token subsets."""
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    forks = tuple(forks)
    batches = tuple(_batches(forks, batch_size))
    if not 1 <= len(forks) <= 32 or len(batches) > max_forward_calls:
        raise ValueError("Fork token diagnostic exceeds its bounded evidence budget")
    tokenizer = runner.tokenizer
    numeric = set()
    numeric_sequences = {}
    for text in (*map(str, range(10)), "."):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        numeric_sequences[text] = list(tokens)
        if len(tokens) == 1 and tokenizer.decode(tokens, skip_special_tokens=False).strip() == text:
            numeric.add(tokens[0])
    newline = tokenizer.encode("\n", add_special_tokens=False)
    termination = {token for token in newline
                   if "\n" in tokenizer.decode([token], skip_special_tokens=False)}
    eos = getattr(tokenizer, "eos_token_id", None)
    termination.update(eos if isinstance(eos, (tuple, list)) else (() if eos is None else (eos,)))
    rows = []
    with torch.inference_mode():
        for batch in batches:
            logits = next_token_logits(runner, [fork.prefix_ids for fork in batch])
            log_probs = logits.double().log_softmax(-1)
            if any(token < 0 or token >= logits.shape[1] for token in numeric | termination):
                raise ValueError("Tokenizer diagnostic token lies outside the model vocabulary")
            for fork, log_prob in zip(batch, log_probs, strict=True):
                wanted = torch.tensor(fork.accepted_token_ids, device=log_prob.device)
                rows.append({"prompt_id": fork.prompt_id, "actual_token_id": fork.actual_token_id,
                             "accepted_token_ids": list(fork.accepted_token_ids),
                             "accepted_probability": float(log_prob[wanted].logsumexp(0).exp()),
                             "numeric_subset_probability": float(log_prob[list(sorted(numeric))].exp().sum()),
                             "termination_subset_probability": float(log_prob[list(sorted(termination))].exp().sum()),
                             "greedy_token_id": int(log_prob.argmax()),
                             "accepted_top_token": int(wanted[log_prob[wanted].argmax()])})
            _check_memory(max_memory_bytes)
    return {"schema": "weightbridge.fork_token_classes.v1", "rows": rows,
            "numeric_token_ids": sorted(numeric), "termination_token_ids": sorted(termination),
            "numeric_surface_tokenizations": numeric_sequences, "newline_tokenization": list(newline),
            "scope": "Native standalone digit and decimal-point tokens versus newline/EOS tokens. This bounded subset does not exhaust every vocabulary token that can extend a number.",
            "resources": _resources(wall, cpu, len(batches), 0)}
