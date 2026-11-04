"""Keep local answers, four-form comparisons and eight-form rule families distinct."""
from experiments.phase_iiil.semantic_diagnostics import compare_semantic, summarize


def compare(parent, candidate, *, min_comparison_gain=1, min_family_gain=0, min_repeatable_corrections=0):
    gate = compare_semantic(parent, candidate, min_comparison_gain=min_comparison_gain,
                            min_repeatable_corrections=min_repeatable_corrections)
    before, after = summarize(parent), summarize(candidate)
    completed_before = {name for name, family in before["families"].items() if family["status"] == "both_correct"}
    completed_after = {name for name, family in after["families"].items() if family["status"] == "both_correct"}
    corrected = [row["id"] for row, edited in zip(parent["rows"], candidate["rows"], strict=True)
                 if row["group"] == "target" and not row["correct"] and edited["correct"]]
    gate.update(local_corrected_target_presentations=corrected,
                parent_complete_families=len(completed_before), candidate_complete_families=len(completed_after),
                complete_family_gain=len(completed_after) - len(completed_before),
                newly_completed_families=sorted(completed_after - completed_before),
                lost_complete_families=sorted(completed_before - completed_after))
    gate["passed"] = bool(gate["passed"] and len(completed_after) - len(completed_before) >= min_family_gain)
    return gate
