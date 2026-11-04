from types import SimpleNamespace

import numpy as np
import pytest

from weightbridge.adaptation import joint_update
from weightbridge.adaptation.endpoint_update import _tensor_bytes
from weightbridge.adaptation.factorial_probe import StoredAffineSpace


@pytest.fixture
def native_fixture(monkeypatch):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.01, 0.], [0., .01]], dtype=torch.bfloat16))
    parent_bytes = _tensor_bytes(model.weight, torch)
    features = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.bfloat16)
    predict = lambda runner, prefixes: runner.model(features[[prefix[0] for prefix in prefixes]]).float()
    monkeypatch.setattr(joint_update, "next_token_logits", predict)
    from weightbridge.adaptation import jspace_update
    monkeypatch.setattr(jspace_update, "next_token_logits", predict)
    monkeypatch.setattr(joint_update, "_first_divergences", lambda *args: [])
    selected = SimpleNamespace(prompt_id="wrong", prefix_ids=(0,), accepted_token_ids=(1,), actual_token_id=0,
        shared_token_ids=(), status="semantic", replay_status="matched", checkpoint_sha256="parent")
    context = SimpleNamespace(prompt_id="context", prefix_ids=(1,), accepted_token_ids=(0,), actual_token_id=1)
    anchor = SimpleNamespace(prompt_id="protected", position=0, token_id=1, key=("protected", 0),
        item="control", group="positive", wording=0)
    pool = SimpleNamespace(eligible_anchors=(anchor,), trajectories={"protected": SimpleNamespace(
        prompt_token_ids=(1,), token_ids=(1,))})
    space = StoredAffineSpace(model, (0,), np.array([1.]), "BF16",
        model.weight.detach().float().double().numpy(), parent_bytes, .025)
    preview = space.preview(np.array([-.01, .01]))
    plan = SimpleNamespace(checkpoint_sha256="parent", method="constrained", region=SimpleNamespace(
        tensor_name="weight", layer_index=0), limits=SimpleNamespace(max_update_norm=.025,
        max_update_entries=2, max_forward_calls=8, max_memory_bytes=1 << 40))
    runner = SimpleNamespace(model=model, tokenizer=SimpleNamespace())
    return torch, runner, space, preview, plan, (selected,), ((selected, context),), pool, (anchor,), parent_bytes


def test_native_measurement_preserves_context_order_and_accepted_alternatives(native_fixture):
    _, runner, _, _, _, forks, contexts, _, _, _ = native_fixture
    result = joint_update.measure_native_forks(runner, forks, contexts,
        max_forward_calls=1, max_memory_bytes=1 << 40)
    assert result["resources"]["forward_calls"] == 1
    assert result["original_context_batches"][0]["prompt_ids"] == ["wrong", "context"]
    assert len(result["rows"]) == 1
    assert not result["rows"][0]["accepted_token_selected"]
    assert result["rows"][0]["strongest_rejected_token_id"] == 0
    assert not result["context_only_rows_have_target_weight"]


def test_native_context_omission_and_budget_refused_before_measurement(native_fixture, monkeypatch):
    _, runner, _, _, _, forks, contexts, _, _, _ = native_fixture
    monkeypatch.setattr(joint_update, "next_token_logits", lambda *args: pytest.fail("Must refuse before inference"))
    with pytest.raises(ValueError, match="omitted"):
        joint_update.measure_native_forks(runner, forks, ((contexts[0][1],),), max_forward_calls=1, max_memory_bytes=1 << 40)
    with pytest.raises(ValueError, match="forward budget"):
        joint_update.measure_native_forks(runner, forks, contexts, max_forward_calls=0, max_memory_bytes=1 << 40)


def test_stored_native_target_and_protected_measurement_restores_parent(native_fixture):
    torch, runner, space, preview, _, forks, contexts, pool, anchors, raw = native_fixture
    result = joint_update.measure_stored_joint(space, preview, runner, forks, contexts, pool, anchors,
        target_floor=.005, preserve_floor=.005, batch_size=2, max_forward_calls=8, max_memory_bytes=1 << 40)
    assert result["passed"] and result["restored_exactly"]
    assert result["targets"]["rows"][0]["greedy_token_id"] == 1
    assert result["resources"]["forward_calls"] == 2
    assert _tensor_bytes(runner.model.weight, torch) == raw


def test_full_generation_is_blocked_when_any_individual_margin_fails(native_fixture):
    torch, runner, space, preview, plan, forks, contexts, pool, anchors, raw = native_fixture
    patch, result = joint_update.probe_joint_generation(space, preview, runner, plan, forks, contexts, pool, anchors,
        target_floor=.5, preserve_floor=.005, batch_size=2, seen=set(),
        evaluate_candidate=lambda: pytest.fail("No full decode after failed numerical gate"), baseline_endpoint={})
    assert patch is None and result["reason"] == "stored_joint_margin_refusal"
    assert _tensor_bytes(runner.model.weight, torch) == raw


def test_generation_exception_restores_exact_source_and_cannot_mark_candidate_seen(native_fixture):
    torch, runner, space, preview, plan, forks, contexts, pool, anchors, raw = native_fixture
    seen = set()

    def interrupted():
        assert _tensor_bytes(runner.model.weight, torch) != raw
        raise RuntimeError("generation interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        joint_update.probe_joint_generation(space, preview, runner, plan, forks, contexts, pool, anchors,
            target_floor=.005, preserve_floor=.005, batch_size=2, seen=seen,
            evaluate_candidate=interrupted, baseline_endpoint={})
    assert not seen
    assert _tensor_bytes(runner.model.weight, torch) == raw


def test_generation_pass_produces_existing_patch_and_duplicate_has_no_inference(native_fixture, monkeypatch):
    torch, runner, space, preview, plan, forks, contexts, pool, anchors, raw = native_fixture
    seen = set()
    evaluate = lambda: {"measurement": {}, "guard": {"passed": True, "target_item_gain": 1,
        "prohibited_regressions": [], "lost_valid_presentations": []}}
    patch, result = joint_update.probe_joint_generation(space, preview, runner, plan, forks, contexts, pool, anchors,
        target_floor=.005, preserve_floor=.005, batch_size=2, seen=seen, evaluate_candidate=evaluate, baseline_endpoint={})
    assert patch["schema"] == "weightbridge.patch.v1"
    assert result["reason"] == "endpoint_pass"
    monkeypatch.setattr(joint_update, "measure_stored_joint", lambda *args, **kwargs: pytest.fail("Duplicate must not remeasure"))
    patch, repeated = joint_update.probe_joint_generation(space, preview, runner, plan, forks, contexts, pool, anchors,
        target_floor=.005, preserve_floor=.005, batch_size=2, seen=seen, evaluate_candidate=evaluate, baseline_endpoint={})
    assert patch is None and repeated["reason"] == "duplicate_stored_candidate"
    assert _tensor_bytes(runner.model.weight, torch) == raw


def test_explicit_budget_refuses_before_more_calls_and_never_raises_its_limit():
    from experiments.phase_iiio.run_joint import _BudgetExhausted, _ForwardBudget

    budget = _ForwardBudget(4, 2)
    budget.reserve(2)
    with pytest.raises(_BudgetExhausted):
        budget.reserve(1)
    assert budget.used == budget.limit == 4


def test_passing_practice_guard_does_not_hide_fitted_completion_failure(native_fixture):
    _, runner, space, preview, plan, forks, contexts, pool, anchors, _ = native_fixture
    evaluate = lambda: {"measurement": {"rows": [{"id": "wrong", "correct": False}]},
        "guard": {"passed": True, "target_item_gain": 1, "prohibited_regressions": [], "lost_valid_presentations": []}}
    patch, result = joint_update.probe_joint_generation(space, preview, runner, plan, forks, contexts, pool, anchors,
        target_floor=.005, preserve_floor=.005, batch_size=2, seen=set(), evaluate_candidate=evaluate, baseline_endpoint={})
    assert patch is not None  # The independently frozen practice gate is unchanged.
    assert result["practice_admission"]
    assert result["mechanism_status"] == "FORK_CORRECTED_COMPLETION_WRONG"
    assert result["fitted_error_responses_still_wrong"] == ["wrong"]


def test_family_selection_requires_every_wrong_presentation_not_just_a_wording_pair(monkeypatch):
    from experiments.phase_iiio import run_joint
    from weightbridge.adaptation.paired_supervision import ComparisonEvidence

    rows, records = [], []
    for family, distance in (("close-family", "close"), ("far-family", "far")):
        for variant, length in enumerate(("short", "long")):
            item = f"{family}/{variant}"
            ids = tuple(f"{item}/{order}/{wording}" for order in range(2) for wording in range(2))
            records.append(ComparisonEvidence(item, family, distance + "-" + length, ids))
            rows.extend({"id": prompt, "pair_id": family, "group": "target", "valid": True, "correct": False} for prompt in ids)
    monkeypatch.setattr(run_joint, "_eligible_comparisons", lambda *args: (records, {}, {}, {}))
    names, _, _, _, evidence = run_joint._select_families({"rows": rows}, {}, None)
    assert evidence["status"] == "READY" and len(names) == 4
    first = records[0]
    records[0] = ComparisonEvidence(first.comparison_id, first.family_id, first.group, first.eligible_prompt_ids[:-1])
    _, _, _, _, evidence = run_joint._select_families({"rows": rows}, {}, None)
    assert evidence["status"] == "HOLD"
    assert evidence["population"][0]["uncovered_error_prompt_ids"] == [first.eligible_prompt_ids[-1]]


def test_final_predicted_target_floor_is_not_relaxed_by_solver_tolerance(native_fixture, monkeypatch):
    from experiments.phase_iiio import run_joint

    _, runner, space, _, plan, forks, contexts, pool, anchors, _ = native_fixture
    config = run_joint.read_json(run_joint.PROTOCOL)
    config["probe_design"]["target_margin"] = .01
    config["probe_design"]["norm_search_steps"] = 2
    basis = SimpleNamespace(target_ids=("wrong",), target_margins=np.array([0.]),
                            target_gradients=np.array([[1., 0.]]))
    selected = (SimpleNamespace(gradient=np.zeros(2), margin=.01),)
    shared = (pool, SimpleNamespace(anchors=anchors), selected, None, SimpleNamespace(anchors=()), {}, {}, .005)
    monkeypatch.setattr(run_joint, "solve_robust", lambda *args, **kwargs: SimpleNamespace(
        update=np.array([.01 - 5e-9, 0.]), record=lambda: {"status": "fixture_tolerant_witness"}))
    monkeypatch.setattr(space, "preview", lambda *args: pytest.fail("Below-floor witness must not reach BF16 execution"))
    budget = run_joint._ForwardBudget(20, 0)
    best, result = run_joint._numeric_cell(space, runner, plan, basis, forks, contexts, np.array([1., 0.]),
        shared, .020, config, "A", budget)
    assert best is None and result["norm_match"]["status"] == "UNMATCHED"
    assert result["norm_match"]["attempts"][0]["reason"] == "individual_linear_margin_refusal"
    assert budget.used == 0


@pytest.mark.parametrize("status", ["INFEASIBLE_ZERO_ROW", "INFEASIBLE_OPPOSED_CONSTRAINTS",
                                    "CERTIFIED_RADIUS_SEPARATOR", "CERTIFIED_NORM_BOUND_EXCEEDED"])
def test_certificate_failure_classes_distinguish_target_and_preservation_conflict(status):
    from experiments.phase_iiio.run_joint import _failure_classification

    record = {"status": status, "certified_norm_lower_bound": None}
    assert _failure_classification(record, arm="C", target_only_feasible=False, maximum_norm=.025) == "NORM_BOUND_FAILURE"
    assert _failure_classification(record, arm="C", target_only_feasible=True, maximum_norm=.025) == "TARGET_PRESERVATION_CONFLICT"
    assert _failure_classification({"status": "BOUNDED_NO_WITNESS"}, arm="C",
                                  target_only_feasible=True, maximum_norm=.025) == "NUMERICAL_INCONCLUSIVE"
