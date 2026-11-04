from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.phase_iiio import analyze, execute, verification as verify
from experiments.phase_iiio.archive_trial import FIT_FILES, WORKER_FILES
from weightbridge.adaptation.joint_feasibility import solve_joint_feasibility
from weightbridge.core import file_hash, json_dump_atomic


def test_joint_system_dual_certificate_replays_and_tampered_bound_refused():
    system = {"target_rows": [[1., 0.], [0., 2.]], "target_parent_margins": [-.01, -.02],
        "target_floor": .0625, "preservation_rows": [], "preservation_parent_margins": [],
        "preservation_floor": .125, "radius": .025, "max_sweeps": 128, "tolerance": 1e-8,
        "target_ids": ["a", "b"], "preservation_ids": []}
    record = solve_joint_feasibility(**system).record()
    joint = {"target_only": record, "systems": {"target_only": system}}
    analyze.verify_joint_systems(joint)
    changed = deepcopy(joint)
    changed["target_only"]["certified_norm_lower_bound"] /= 2
    with pytest.raises(ValueError, match="does not replay"):
        analyze.verify_joint_systems(changed)


def _nonconverged_system():
    return {"target_rows": [[1., 0.], [-.8, .6]], "target_parent_margins": [-.01, -.01],
        "target_floor": .0625, "preservation_rows": [], "preservation_parent_margins": [],
        "preservation_floor": .125, "radius": .025, "max_sweeps": 1, "tolerance": 1e-12,
        "target_ids": ["a", "b"], "preservation_ids": []}


def test_nonconverged_cauchy_ratio_bound_can_exceed_dual_objective_bound():
    system = _nonconverged_system()
    record = solve_joint_feasibility(**system).record()
    assert not record["converged"]
    certificate = record["certificate"]
    assert certificate["dual_direction_norm_lower_bound"] > certificate["dual_objective_norm_lower_bound"] + .01
    assert record["certified_norm_lower_bound"] == certificate["dual_direction_norm_lower_bound"]
    analyze.verify_joint_systems({"target_only": record, "systems": {"target_only": system}})
    changed = deepcopy(record)
    changed["certificate"]["dual_direction_norm_lower_bound"] *= .5
    with pytest.raises(ValueError, match="does not replay"):
        analyze.verify_joint_systems({"target_only": changed, "systems": {"target_only": system}})


@pytest.mark.parametrize("field", ["dual_direction_numerator", "dual_direction_denominator",
                                 "dual_objective_norm_lower_bound", "original_dual_multipliers"])
def test_bound_arithmetic_is_checked_independently_of_solver_replay(monkeypatch, field):
    system = _nonconverged_system()
    changed = solve_joint_feasibility(**system).record()
    if field == "original_dual_multipliers":
        changed["certificate"][field][0] *= 1.5
    else:
        changed["certificate"][field] *= 1.5
    # Simulate a self-consistent replay implementation bug; the independent
    # arithmetic must still reject its unearned certificate component.
    monkeypatch.setattr(analyze, "solve_joint_feasibility", lambda **_: SimpleNamespace(record=lambda: changed))
    with pytest.raises(ValueError, match="does not follow"):
        analyze.verify_joint_systems({"target_only": changed, "systems": {"target_only": system}})


def test_every_individual_native_target_is_required_not_average():
    basis = {"forks": [{"prompt_id": name, "parent_margin": -.1, "projected_margin_gradient": [1.],
        "accepted_token_ids": [1]} for name in ("a", "b")]}
    native = {"targets": {"rows": [{"prompt_id": name, "measured_margin": margin,
        "accepted_token_ids": [1], "accepted_token_selected": token == 1, "greedy_token_id": token}
        for name, margin, token in (("a", .3, 1), ("b", -.01, 2))]}}
    result = analyze.validate_native_targets(native, basis, np.array([.2]), .0625)
    assert not result["passed"] and result["accepted_native_forks"] == 1
    native["targets"]["rows"].pop()
    with pytest.raises(ValueError, match="omitted"):
        analyze.validate_native_targets(native, basis, np.array([.2]), .0625)


def test_eight_outcomes_preserve_unmeasured_behavior_as_none():
    rows = [{"id": f"{variant}-{order}-{wording}", "comparison_id": f"c{variant}", "counterfactual": variant,
        "order": order, "wording": wording, "group": "target", "pair_id": "f", "correct": False,
        "exact": "-2", "answer": "-1"} for variant in (0, 1) for order in (0, 1) for wording in (0, 1)]
    result = analyze.response_matrix({"rows": rows}, None, {}, None, None, ["f"])
    assert len(result["f"]["rows"]) == 8
    assert result["f"]["candidate_complete"] is None
    assert all(row["candidate_correct"] is None and not row["behavior_measured"] for row in result["f"]["rows"])


def test_capacity_recommendation_distinguishes_local_norm_and_precision_failure():
    local = analyze.capacity_decision({"target_only": {"status": "CERTIFIED_NORM_BOUND_EXCEEDED"}}, {}, {})
    assert local["next_experiment"] == "optimized_right_factor_rank_one"
    precision = analyze.capacity_decision({"target_only": {"within_radius": True}},
        {"C1": {"native_targets": {"passed": False}}}, {})
    assert precision["next_experiment"] == "precision_aware_synthesis"
    assert not precision["production_authority"] and not precision["lineage_authority"]


def test_public_allowlists_exclude_unscored_private_and_b_generation_arms():
    assert FIT_FILES.fullmatch("native-parent-targets.json")
    assert FIT_FILES.fullmatch("joint-feasibility.json")
    assert FIT_FILES.fullmatch("arm-C2.json")
    assert not FIT_FILES.fullmatch("arm-B2.json")
    assert not WORKER_FILES.fullmatch("verification/hidden-base.json")
    assert not FIT_FILES.fullmatch("selection.json")


def test_execution_interruption_records_whole_process_and_refuses_overwrite(tmp_path, monkeypatch):
    def fail(base, material, out):
        out.mkdir()
        raise RuntimeError("bounded fixture interruption")

    monkeypatch.setattr(execute, "run", fail)
    out, receipt = tmp_path / "trial", tmp_path / "execution.json"
    with pytest.raises(RuntimeError):
        execute.execute(tmp_path / "base", tmp_path / "material", out, receipt)
    saved = json.loads(receipt.read_bytes())
    assert saved["status"] == "INTERRUPTED" and saved["error"]["type"] == "RuntimeError"
    assert saved["resources"]["cpu_seconds"] >= 0
    with pytest.raises(FileExistsError):
        execute.execute(tmp_path / "base", tmp_path / "material", out, receipt)
