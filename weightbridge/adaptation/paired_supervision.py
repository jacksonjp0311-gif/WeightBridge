"""Comparison-budgeted supervision allocation in a fixed model-specific subspace.

The numerical helpers operate only on host-authorized development evidence. They
do not evaluate candidates, authorize data access, or promote checkpoints.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite

import numpy as np

from .preservation_geometry import solve_inequalities


GROUP_ORDER = ("close-short", "close-long", "far-short", "far-long")


@dataclass(frozen=True)
class ComparisonEvidence:
    comparison_id: str
    family_id: str
    group: str
    eligible_prompt_ids: tuple[str, ...]
    parent_correct_prompt_ids: tuple[str, ...] = ()

    def __post_init__(self):
        ids = (*self.eligible_prompt_ids, *self.parent_correct_prompt_ids)
        if (not self.comparison_id or not isinstance(self.comparison_id, str)
                or not self.family_id or not isinstance(self.family_id, str)
                or self.group not in GROUP_ORDER
                or any(not isinstance(value, str) or not value for value in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("Comparison evidence needs disjoint qualified errors and parent-correct obligations")

    @property
    def eligible(self):
        """An order's two wording variants must already have qualified together."""
        return len(self.eligible_prompt_ids) >= 2


@dataclass(frozen=True)
class SupervisionSelection:
    arm_items: dict[str, tuple[str, ...]]
    evidence: dict

    def record(self):
        return self.evidence


def select_supervision(records, *, one_sided_count=8, paired_family_count=4):
    """Freeze deterministic allocation using error qualification, never validation.

    One-sided allocation round-robins semantic groups across distinct families.
    Paired allocation picks equally many close/far families, ranking each by its
    weakest qualified member's fork count and then its stable family identifier.
    Both strategies give each selected comparison exactly one raw mean gradient.
    """
    records = tuple(records)
    if (not records or any(not isinstance(row, ComparisonEvidence) for row in records)
            or len({row.comparison_id for row in records}) != len(records)
            or type(one_sided_count) is not int or one_sided_count <= 0 or one_sided_count % 4
            or type(paired_family_count) is not int or paired_family_count <= 0 or paired_family_count % 2
            or one_sided_count != 2 * paired_family_count):
        raise ValueError("Matched allocation needs unique comparison records and balanced equal comparison budgets")
    by_family, by_group = defaultdict(list), defaultdict(list)
    prompt_ids = [prompt for row in records for prompt in (*row.eligible_prompt_ids, *row.parent_correct_prompt_ids)]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("A presentation cannot belong to multiple comparison records")
    for row in records:
        by_family[row.family_id].append(row)
        if row.eligible:
            by_group[row.group].append(row)
    if any(len(rows) > 2 for rows in by_family.values()):
        raise ValueError("A counterfactual family has at most two opposing comparisons")
    for rows in by_family.values():
        if len(rows) == 2 and (len({row.group.split('-')[0] for row in rows}) != 1
                              or {row.group.split('-')[1] for row in rows} != {"short", "long"}):
            raise ValueError("Opposing family members must share distance and cover both answer-length outcomes")
    for rows in by_group.values():
        rows.sort(key=lambda row: (-len(row.eligible_prompt_ids), row.comparison_id))
    chosen, families = [], set()
    for _ in range(one_sided_count // 4):
        for group in GROUP_ORDER:
            row = next((row for row in by_group[group] if row.family_id not in families), None)
            if row is not None:
                chosen.append(row)
                families.add(row.family_id)
    paired = []
    for distance in ("close", "far"):
        candidates = [rows for rows in by_family.values() if len(rows) == 2
                      and all(row.eligible and row.group.startswith(distance + "-") for row in rows)]
        candidates.sort(key=lambda rows: (-min(len(row.eligible_prompt_ids) for row in rows), rows[0].family_id))
        for rows in candidates[:paired_family_count // 2]:
            paired.extend(sorted(rows, key=lambda row: GROUP_ORDER.index(row.group)))
    available = {row.comparison_id: row for row in records}
    arm_items = {"A": tuple(row.comparison_id for row in chosen),
                 "B": tuple(row.comparison_id for row in paired),
                 "C": tuple(row.comparison_id for row in paired)}
    enough_a = len(chosen) == one_sided_count
    enough_b = len(paired) == one_sided_count
    evidence = {
        "schema": "weightbridge.paired_supervision_selection.v1",
        "status": "READY" if enough_a and enough_b else "HOLD",
        "reason": "qualified_equal_comparison_allocations" if enough_a and enough_b else
                  "insufficient_one_sided_family_diversity" if not enough_a else "insufficient_dual_error_families",
        "selection_rule": "group round-robin; descending qualified fork count; stable identity ties",
        "paired_rule": "equal close/far family count; descending weakest-member fork count; stable family ties",
        "comparison_budget": one_sided_count, "paired_family_budget": paired_family_count,
        "arms": {arm: {"comparison_ids": list(ids), "comparison_count": len(ids),
                       "family_ids": list(dict.fromkeys(available[item].family_id for item in ids)),
                       "family_count": len({available[item].family_id for item in ids}),
                       "group_counts": dict(Counter(available[item].group for item in ids)),
                       "comparison_weight": 1 / len(ids) if ids else None}
                 for arm, ids in arm_items.items()},
        "population": [{"comparison_id": row.comparison_id, "family_id": row.family_id, "group": row.group,
                        "eligible_error_prompt_ids": list(row.eligible_prompt_ids),
                        "parent_correct_preservation_prompt_ids": list(row.parent_correct_prompt_ids),
                        "eligible": row.eligible} for row in records],
        "causal_limitation": "A has more families; the contrast compares supervision allocation, not pairing alone",
    }
    return SupervisionSelection(arm_items, evidence)


def _rows(item_ids, gradients):
    ids = tuple(item_ids)
    values = np.asarray(gradients, dtype=np.float64)
    if (not ids or len(set(ids)) != len(ids) or any(not isinstance(item, str) or not item for item in ids)
            or values.ndim != 2 or values.shape[0] != len(ids) or not values.shape[1]
            or not np.isfinite(values).all() or not np.isfinite(np.linalg.norm(values, axis=1)).all()):
        raise ValueError("Expected one finite corrective row per unique comparison")
    return ids, values


def mean_direction(item_ids, gradients):
    """Average already-averaged comparison gradients; never duplicate a wording."""
    _, values = _rows(item_ids, gradients)
    return values.mean(axis=0)


def pair_gradient_diagnostics(item_ids, family_ids, groups, gradients, *, margins=None):
    ids, rows = _rows(item_ids, gradients)
    families, labels = tuple(family_ids), tuple(groups)
    if (len(families) != len(ids) or len(labels) != len(ids)
            or any(not isinstance(value, str) or not value for value in (*families, *labels))):
        raise ValueError("Every comparison needs a family and semantic group")
    baseline = None if margins is None else np.asarray(margins, dtype=np.float64)
    if baseline is not None and (baseline.shape != (len(ids),) or not np.isfinite(baseline).all()):
        raise ValueError("Expected one finite parent margin per comparison")
    aggregate = rows.mean(axis=0)
    aggregate_norm = float(np.linalg.norm(aggregate))
    slopes = rows @ (aggregate / aggregate_norm) if aggregate_norm else np.zeros(len(ids))
    norms = np.linalg.norm(rows, axis=1)
    members = defaultdict(list)
    for index, family in enumerate(families):
        members[family].append(index)
    if any(len(indices) > 2 for indices in members.values()):
        raise ValueError("A counterfactual family has at most two comparison gradients")
    pair_rows = {}
    for family, indices in members.items():
        record = {"comparison_ids": [ids[index] for index in indices],
                  "groups": [labels[index] for index in indices],
                  "gradient_norms": norms[indices].tolist(),
                  "mean_direction_slopes": slopes[indices].tolist(),
                  "paired": len(indices) == 2}
        if baseline is not None:
            record["parent_comparison_margins"] = baseline[indices].tolist()
        if len(indices) == 2:
            first, second = indices
            denominator = float(norms[first] * norms[second])
            cosine = float(np.clip(rows[first] @ rows[second] / denominator, -1, 1)) if denominator else None
            record.update(cosine=cosine,
                norm_ratio_first_to_second=float(norms[first] / norms[second]) if norms[second] else None,
                norm_imbalance_ratio=float(max(norms[indices]) / min(norms[indices])) if min(norms[indices]) else None,
                relationship="zero_gradient" if cosine is None else "compatible" if cosine > .1 else
                             "conflicting" if cosine < -.1 else "weakly_coupled",
                both_mean_slopes_positive=bool(np.all(slopes[indices] > 0)))
        pair_rows[family] = record
    return {"schema": "weightbridge.paired_gradient_diagnostics.v1", "families": pair_rows,
            "comparison_ids": list(ids), "comparison_groups": list(labels),
            "mean_direction": aggregate.tolist(), "mean_direction_norm": aggregate_norm,
            "classification_cosine_thresholds": {"conflicting_below": -.1, "compatible_above": .1},
            "scope": "Local fixed-right corrective derivatives; cosine is not a global feasibility proof"}


@dataclass(frozen=True)
class PairDirection:
    update: np.ndarray | None
    evidence: dict

    def record(self):
        return self.evidence


def solve_pair_maximin(item_ids, family_ids, gradients, parent_margins,
                       preservation_gradients, protected_margins, *, radius, floor=0.,
                       max_search_steps=16, max_projection_iterations=256, tolerance=1e-9):
    """Find a bounded witness maximizing the weakest absolute predicted margin.

    Each raw row averages a comparison's replay-qualified forks. Its intercept is
    the matching average parent margin, not a proof that every individual fork is
    corrected. Full stored-precision generation remains authoritative. A failed
    bounded halfspace projection is never advertised as exact infeasibility.
    """
    ids, rows = _rows(item_ids, gradients)
    families = tuple(family_ids)
    counts = Counter(families)
    prior = np.asarray(parent_margins, dtype=np.float64)
    protected = np.asarray(preservation_gradients, dtype=np.float64)
    if protected.shape == (0,):
        protected = protected.reshape(0, rows.shape[1])
    margins = np.asarray(protected_margins, dtype=np.float64)
    bounds = np.asarray(floor, dtype=np.float64)
    if bounds.ndim == 0:
        bounds = np.full(len(margins), float(bounds))
    if (len(families) != len(ids) or any(not isinstance(value, str) or not value for value in families)
            or any(count != 2 for count in counts.values())
            or prior.shape != (len(ids),) or not np.isfinite(prior).all()
            or protected.ndim != 2 or protected.shape[1] != rows.shape[1]
            or margins.shape != (len(protected),) or bounds.shape != margins.shape
            or not np.isfinite(protected).all() or not np.isfinite(margins).all()
            or not np.isfinite(bounds).all() or np.any(margins < 0) or np.any(bounds < 0)
            or not isfinite(radius) or radius <= 0 or not isfinite(tolerance) or tolerance <= 0
            or type(max_search_steps) is not int or not 1 <= max_search_steps <= 64
            or type(max_projection_iterations) is not int or not 1 <= max_projection_iterations <= 4096):
        raise ValueError("Pair maximin needs paired finite rows, shared preservation coordinates and bounded budgets")
    settings = {"radius": radius, "max_iterations": max_projection_iterations, "tolerance": tolerance}
    zero = np.zeros(rows.shape[1])
    baseline = solve_inequalities(zero, protected, margins, floor=bounds, **settings)
    evidence = {"schema": "weightbridge.paired_maximin.v1", "comparison_ids": list(ids),
                "family_ids": list(families), "radius": radius, "parent_margins": prior.tolist(),
                "objective": "max minimum absolute predicted comparison margin; raw rows",
                "projection_budget": max_projection_iterations, "search_budget": max_search_steps,
                "preservation_baseline": baseline.record(), "search_steps": [],
                "bounded_search_is_infeasibility_certificate": False}
    if baseline.update is None:
        return PairDirection(None, {**evidence, "status": "no_preservation_witness"})
    ordinary = solve_inequalities(rows.mean(axis=0), protected, margins, floor=bounds, **settings)
    witnesses = [baseline.update] + ([ordinary.update] if ordinary.update is not None else [])
    best = max(witnesses, key=lambda point: float(np.min(prior + rows @ point)))
    lower = float(np.min(prior + rows @ best))
    # Each individual margin has this Cauchy-Schwarz upper bound in the ball.
    upper = float(np.min(prior + radius * np.linalg.norm(rows, axis=1)))
    search_upper = max(lower, upper)
    joint_rows = np.vstack((protected, rows))
    joint_margins = np.concatenate((margins, prior))
    for _ in range(max_search_steps):
        if search_upper - lower <= tolerance:
            break
        level = (lower + search_upper) / 2
        fit = solve_inequalities(zero, joint_rows, joint_margins,
            floor=np.concatenate((bounds, np.full(len(rows), level))), **settings)
        observed = float(np.min(prior + rows @ fit.update)) if fit.update is not None else None
        evidence["search_steps"].append({"requested_minimum_margin": level, "projection": fit.record(),
                                         "witnessed_minimum_margin": observed})
        if observed is not None and observed > lower + tolerance:
            best, lower = fit.update.copy(), observed
        else:
            search_upper = level
    effects = rows @ best
    predictions = prior + effects
    norm = float(np.linalg.norm(best))
    evidence.update(status="positive_joint_margin_witness" if lower > tolerance else
                           "bounded_nonpositive_margin_witness",
                    update_norm=norm, witnessed_minimum_margin=lower,
                    analytic_individual_upper_bound=upper, numerical_search_ceiling=search_upper,
                    predicted_margins=dict(zip(ids, predictions.tolist(), strict=True)),
                    predicted_margin_changes=dict(zip(ids, effects.tolist(), strict=True)),
                    individual_directional_slopes=dict(zip(ids, (effects / norm).tolist(), strict=True)) if norm else None,
                    protected_predicted_margins=(margins + protected @ best).tolist(),
                    protected_floor=bounds.tolist(),
                    active_preservation_constraint_count=int(np.count_nonzero(
                        margins + protected @ best - bounds <= tolerance)),
                    limiting_comparison_ids=[item for item, value in zip(ids, predictions, strict=True)
                                             if value <= lower + tolerance],
                    all_predicted_comparisons_correct=bool(np.all(predictions > 0)))
    return PairDirection(best.copy(), evidence)
