"""Small deterministic derivative fixtures; these are not model research data."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from weightbridge.adaptation import selected_block
from weightbridge.adaptation.endpoint_update import _tensor_bytes
from weightbridge.adaptation.preservation_geometry import DecisionAnchor


@pytest.fixture
def fixture(monkeypatch):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(4, 3, bias=False).eval()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2., 0., 0., 1.], [0., 0., 1., 0.], [-1., 0., -1., 0.]]))
    features = torch.tensor([[1., 0., 0., 0.], [1., 0., 1., 0.],
                             [0., 0., 0., 1.], [0., 1., 0., 0.]])
    raw = _tensor_bytes(model.weight, torch)
    monkeypatch.setattr(selected_block, "_region", lambda *args:
        (SimpleNamespace(shape=(3, 4), dtype="F32"), model, raw))
    monkeypatch.setattr(selected_block, "next_token_logits", lambda runner, prefixes:
        runner.model(features[[prefix[0] for prefix in prefixes]]))
    plan = SimpleNamespace(checkpoint_sha256="parent", region=SimpleNamespace(tensor_name="weight"),
        max_columns=4, limits=SimpleNamespace(max_forward_calls=12, max_backward_calls=12,
        max_memory_bytes=1 << 40, max_update_entries=12))
    forks = tuple(SimpleNamespace(status="semantic", replay_status="matched", checkpoint_sha256="parent",
        prompt_id=f"p{i}", question_id=f"q{i}", prefix_ids=(i,), accepted_token_ids=(1,),
        actual_token_id=0) for i in range(4))
    anchors = tuple(DecisionAnchor(f"item{i}", f"s{i}", "target", i % 2, 0, 1., 0) for i in range(3))
    pool = SimpleNamespace(eligible_anchors=anchors,
        trajectories={anchor.prompt_id: SimpleNamespace(prompt_token_ids=(i,), token_ids=(0,))
                      for i, anchor in enumerate(anchors)})
    return torch, SimpleNamespace(model=model), plan, forks, pool, anchors, features, raw


def target(fixture, *, indices=(0, 1), contexts=None, **kwargs):
    _, runner, plan, forks, _, _, _, _ = fixture
    return selected_block.collect_target_blocks(None, runner, plan, tuple(forks[i] for i in indices),
        columns=(0, 2), replay_batches=(forks,) if contexts is None else contexts,
        batch_size=4, metadata_by_prompt={f"p{i}": {"pair_id": f"family{i // 2}",
            "comparison_id": f"comparison{i}", "wording": i % 2, "order": i // 2} for i in indices}, **kwargs)


def test_full_selected_block_retained_with_correct_margin_orientation_and_provenance(fixture):
    torch, runner, _, _, _, _, features, raw = fixture
    blocks, evidence = target(fixture)
    assert blocks.shape == (2, 3, 2) and blocks.dtype == np.float64
    for i in range(2):
        row = features[i, [0, 2]].numpy()
        np.testing.assert_array_equal(blocks[i], np.stack((-row, row, np.zeros(2))))
        record = evidence["rows"][i]
        assert record["gradient_block_sha256"] == selected_block.block_hash(blocks[i])
        assert record["strongest_rejected_token_id"] == 0
        assert record["native_prefix_ids"] == [i] and record["accepted_token_ids"] == [1]
        assert record["metadata"]["comparison_id"] == f"comparison{i}"
    assert evidence["block_array_sha256"] == selected_block.block_hash(blocks)
    assert evidence["resources"]["forward_calls"] == 1
    assert evidence["resources"]["backward_calls"] == 2
    assert not evidence["retained_full_parameter_gradients"]
    assert not evidence["context_only_rows_have_supervision_weight"]
    assert _tensor_bytes(runner.model.weight, torch) == raw


def test_original_context_rows_are_present_but_not_supervised(fixture, monkeypatch):
    torch, runner, _, forks, _, _, _, _ = fixture
    original = selected_block.next_token_logits

    def packing_sensitive(runtime, prefixes):
        values = original(runtime, prefixes)
        return values if len(prefixes) == 4 else values + torch.tensor([0., 9., 0.])

    monkeypatch.setattr(selected_block, "next_token_logits", packing_sensitive)
    blocks, evidence = target(fixture, indices=(0, 2))
    assert len(blocks) == 2 and len(evidence["rows"]) == 2
    assert evidence["packing"][0]["prompt_ids"] == ["p0", "p1", "p2", "p3"]
    assert evidence["packing"][0]["selected_prompt_ids"] == ["p0", "p2"]
    with pytest.raises(selected_block.SelectedBlockReplayError):
        target(fixture, indices=(0, 2), contexts=((forks[0], forks[2]),))


def test_changed_native_prefix_or_omitted_context_is_rejected_before_forward(fixture, monkeypatch):
    _, _, _, forks, _, _, _, _ = fixture
    monkeypatch.setattr(selected_block, "next_token_logits", lambda *a: pytest.fail("No inference after invalid context"))
    with pytest.raises(ValueError, match="omitted"):
        target(fixture, contexts=((forks[0],),))
    changed = SimpleNamespace(**vars(forks[0]))
    changed.prefix_ids = (9,)
    with pytest.raises(ValueError, match="omitted"):
        target(fixture, contexts=((changed, *forks[1:]),))


def test_replay_refusal_preserves_parent_flags_and_partial_work(fixture, monkeypatch):
    torch, runner, _, _, _, _, _, raw = fixture
    runner.model.weight.requires_grad_(False)
    original = selected_block.next_token_logits
    monkeypatch.setattr(selected_block, "next_token_logits", lambda runtime, prefixes:
        original(runtime, prefixes) + torch.tensor([0., 9., 0.]))
    with pytest.raises(selected_block.SelectedBlockReplayError) as error:
        target(fixture)
    assert error.value.evidence["mismatched_prompt_ids"] == ["p0", "p1"]
    assert error.value.evidence["resources"]["backward_calls"] == 0
    assert error.value.evidence["resources"]["forward_calls"] == 1
    assert not runner.model.weight.requires_grad
    assert _tensor_bytes(runner.model.weight, torch) == raw


def test_derivative_exception_preserves_source_and_training_flags(fixture, monkeypatch):
    torch, runner, _, _, _, _, _, raw = fixture
    runner.model.weight.requires_grad_(False)

    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted VJP")

    monkeypatch.setattr(selected_block, "_derivative", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        target(fixture)
    assert not runner.model.weight.requires_grad
    assert _tensor_bytes(runner.model.weight, torch) == raw


@pytest.mark.parametrize("columns", [(2, 0), (0, 0), (0, 4), (True, 2), ()])
def test_invalid_authorized_column_map_refused(fixture, columns):
    _, runner, plan, forks, _, _, _, _ = fixture
    with pytest.raises(ValueError, match="authorized"):
        selected_block.collect_target_blocks(None, runner, plan, forks[:2], columns=columns,
            replay_batches=(forks,), batch_size=4)


def test_gradient_budget_and_checkpoint_identity_refused(fixture):
    _, _, plan, forks, _, _, _, _ = fixture
    plan.limits.max_backward_calls = 1
    with pytest.raises(ValueError, match="budget"):
        target(fixture)
    plan.limits.max_backward_calls = 12
    forks[0].checkpoint_sha256 = "different"
    with pytest.raises(ValueError, match="another checkpoint"):
        target(fixture)


def test_preservation_blocks_keep_all_required_decisions_and_individual_prefixes(fixture):
    torch, runner, plan, _, pool, anchors, features, raw = fixture
    blocks, evidence = selected_block.collect_preservation_blocks(None, runner, plan, pool, anchors,
        columns=(0, 2), batch_size=2)
    assert blocks.shape == (3, 3, 2)
    for i in range(3):
        row = features[i, [0, 2]].numpy()
        np.testing.assert_array_equal(blocks[i], np.stack((row, -row, np.zeros(2))))
        assert evidence["rows"][i]["native_prefix_ids"] == [i]
        assert evidence["rows"][i]["accepted_token_ids"] == [0]
        assert evidence["rows"][i]["parent_margin"] > 0
    assert evidence["resources"]["forward_calls"] == 2
    assert evidence["resources"]["backward_calls"] == 3
    assert _tensor_bytes(runner.model.weight, torch) == raw


def test_required_protected_token_failure_cannot_be_silently_dropped(fixture, monkeypatch):
    torch, runner, plan, _, pool, anchors, _, raw = fixture
    original = selected_block.next_token_logits
    monkeypatch.setattr(selected_block, "next_token_logits", lambda runtime, prefixes:
        original(runtime, prefixes) + torch.tensor([0., 9., 0.]))
    with pytest.raises(selected_block.SelectedBlockReplayError) as error:
        selected_block.collect_preservation_blocks(None, runner, plan, pool, anchors, columns=(0, 2))
    assert len(error.value.evidence["mismatched_decisions"]) == 3
    assert error.value.evidence["resources"]["backward_calls"] == 0
    assert _tensor_bytes(runner.model.weight, torch) == raw


def test_truncated_or_unknown_protection_refused_before_inference(fixture, monkeypatch):
    _, runner, plan, _, pool, anchors, _, _ = fixture
    monkeypatch.setattr(selected_block, "next_token_logits", lambda *a: pytest.fail("Bad path must refuse before inference"))
    pool.trajectories["s0"].token_ids = ()
    with pytest.raises(ValueError, match="trajectory"):
        selected_block.collect_preservation_blocks(None, runner, plan, pool, anchors, columns=(0, 2))
    with pytest.raises(ValueError, match="distinct eligible"):
        selected_block.collect_preservation_blocks(None, runner, plan, pool,
            (replace(anchors[0], token_id=1),), columns=(0, 2))


def test_empty_preservation_shape_and_deterministic_block_identity(fixture):
    _, runner, plan, _, pool, _, _, _ = fixture
    blocks, evidence = selected_block.collect_preservation_blocks(None, runner, plan, pool, (), columns=(0, 2))
    assert blocks.shape == (0, 3, 2) and evidence["resources"]["forward_calls"] == 0
    a = np.arange(6.).reshape(2, 3)
    assert selected_block.block_hash(a) == selected_block.block_hash(a.astype('>f8'))
    assert selected_block.block_hash(a) != selected_block.block_hash(a.reshape(3, 2))
