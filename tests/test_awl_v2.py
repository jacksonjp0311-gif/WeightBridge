import pytest

from weightbridge.adaptation import awl_v2


TENSORS = [f"model.language_model.layers.{layer}.mlp.down_proj.weight" for layer in (23, 21)]
POLICY = {"schema": "weightbridge.policy.v1", "allow_tensors": TENSORS, "max_entries": 65536,
          "max_fraction": 0.00008, "max_abs_delta": 8.0}


def intent(method="activation_ridge", preserve=None):
    return {"schema": "weightbridge.awl.v2", "objective": "binary_choice_margin",
            "target": TENSORS[:1] if method == "activation_ridge" else TENSORS,
            "preserve": ["word-order"] if preserve is None else preserve,
            "representation": "rank_one_slices", "optimize": {"method": method, "radius": 0.03}}


def practice():
    return {"schema": "weightbridge.hf_edit_suite.v1", "choices": ["A", "B"],
            "target": [{"id": "target", "text": "Target practice example", "label": "A"}],
            "sentinel": [{"id": "preservation", "text": "Prior practice example", "label": "B",
                          "family": "word-order"}]}


def test_task_aware_dispatch_preserves_authorized_practice(monkeypatch):
    data = practice()
    calls = []

    def solver(base, suite, tensor, strength, inference):
        calls.append((suite, tensor, strength))
        return {"schema": "weightbridge.patch.v1"}

    monkeypatch.setattr(awl_v2, "propose_direction", solver)
    patch, receipt = awl_v2.compile_awl_v2("base", data, intent(), policy=POLICY, inference={"batch_size": 8},
                                          authorized_preservation_families=("word-order",), max_radius=0.03,
                                          max_probe_batches=1)
    assert patch["schema"] == "weightbridge.patch.v1"
    assert calls == [(data, TENSORS[0], 0.03)]
    assert receipt["preservation_families"] == ["word-order"]


@pytest.mark.parametrize("kind, match", [("family", "outside the host practice scope"),
                                          ("undeclared", "must match"), ("probe", "forward batch budget"),
                                          ("holdout", "practice suite"), ("tensor", "outside the host tensor")])
def test_host_rejects_expanded_scope_before_model_access(kind, match):
    document, data, policy, families = intent(), practice(), POLICY, ("word-order",)
    if kind == "family":
        families = ()
    elif kind == "undeclared":
        document["preserve"] = []
    elif kind == "probe":
        document = intent("bic_interaction")
    elif kind == "holdout":
        data["holdout"] = []
    elif kind == "tensor":
        policy = {**POLICY, "allow_tensors": TENSORS[1:]}
    with pytest.raises(ValueError, match=match):
        awl_v2.compile_awl_v2("unavailable", data, document, policy=policy, inference={"batch_size": 8},
                              authorized_preservation_families=families, max_radius=1, max_probe_batches=1)
