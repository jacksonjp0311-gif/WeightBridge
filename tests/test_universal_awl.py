from copy import deepcopy
from dataclasses import replace
import json

import pytest

from weightbridge.adaptation.meta_controller import Action, EvidenceState
from weightbridge.adaptation.model_adapter import AccessLevel, Capability, EditableRegion, ModelCapabilityContract
from weightbridge.adaptation.universal_awl import (
    HostAuthorization,
    ModelEvidence,
    ModelPredictions,
    compile_awl_v3,
    parse_awl_v3,
    recommend_awl_v3,
)


def _document():
    return {
        "schema": "weightbridge.awl.v3",
        "objective": {"family": "parity", "metric": "exact_answer", "min_gain": 1},
        "preserve": ["sign"],
        "risk": {"max_regressions": 0, "max_harm_probability": .12},
        "limits": {"max_forward_calls": 128, "max_backward_calls": 16, "max_memory_bytes": 10 * 1024**3,
                   "max_update_entries": 64, "max_update_norm": 1., "max_rank": 1, "max_candidates": 2},
        "data": {"practice": "parity-practice-v1", "evaluation": "parity-eval-v1"},
        "methods": ["first_order", "constrained", "gradient", "direct_low_rank"],
        "verify": {"fresh_evaluation": True, "checkpoint_reload": True, "compare_baselines": True, "persistent": True},
    }


def _contract(family, architecture, sha, shape, last_layer):
    regions = tuple(
        EditableRegion("ffn_output", f"network.{family}.{i}.weight", f"network.{family}.{i}", i, shape, "BF16")
        for i in (0, last_layer)
    )
    unavailable = {Capability.TRAIN_ADAPTER, Capability.COMPUTE_JVP_VJP, Capability.CONFIGURE_RUNTIME}
    return ModelCapabilityContract(
        family=family, architecture=architecture, access_level=AccessLevel.EDITABLE,
        checkpoint_sha256=sha * 64, serialization="safetensors", storage_dtypes=("BF16",),
        tokenizer_class="SyntheticTokenizer", quantized=False, capabilities=frozenset(Capability) - unavailable,
        editable_regions=regions, checkpoint_bytes=1024,
    )


def _host(intent, contract):
    return HostAuthorization(contract.checkpoint_sha256, intent.methods,
                             tuple(region.tensor_name for region in contract.editable_regions), ("parity", "sign"),
                             intent.practice_data_id, intent.evaluation_data_id, intent.limits)


def test_same_intent_plans_different_affine_geometries():
    intent = parse_awl_v3(_document())
    qwen = _contract("qwen3_5", "Qwen3_5ForConditionalGeneration", "a", (16, 40), 23)
    llama = _contract("llama", "LlamaForCausalLM", "b", (8, 20), 29)
    qwen_plan = compile_awl_v3(intent, qwen, _host(intent, qwen), method="constrained")
    llama_plan = compile_awl_v3(intent, llama, _host(intent, llama), method="constrained")
    assert qwen_plan.intent == llama_plan.intent == intent
    assert qwen_plan.region.layer_index == 23 and llama_plan.region.layer_index == 29
    assert qwen_plan.region.tensor_name != llama_plan.region.tensor_name
    assert qwen_plan.max_columns == 4 and llama_plan.max_columns == 8
    assert json.loads(json.dumps(llama_plan.record()))["region"]["shape"] == [8, 20]


@pytest.mark.parametrize(("key", "value"), [
    ("tensor", "model.layers.1.weight"), ("data", {"practice": "../public", "evaluation": "sealed"}),
    ("methods", ["exec"]), ("verify", {"fresh_evaluation": True, "checkpoint_reload": True,
                                      "compare_baselines": True, "persistent": False}),
    ("risk", {"max_regressions": False, "max_harm_probability": .12}),
])
def test_parser_refuses_executable_or_out_of_contract_intent(key, value):
    document = _document()
    document[key] = value
    with pytest.raises(ValueError):
        parse_awl_v3(document)


def test_compiler_refuses_host_and_runtime_mismatches():
    intent = parse_awl_v3(_document())
    contract = _contract("llama", "LlamaForCausalLM", "a", (8, 20), 29)
    host = _host(intent, contract)
    with pytest.raises(ValueError, match="checkpoint differs"):
        compile_awl_v3(intent, contract, replace(host, checkpoint_sha256="f" * 64))
    with pytest.raises(ValueError, match="outside the host scope"):
        compile_awl_v3(intent, contract, replace(host, evaluation_data_id="another-holdout"))
    with pytest.raises(ValueError, match="memory"):
        compile_awl_v3(intent, replace(contract, checkpoint_bytes=11 * 1024**3), host)
    limited_host = replace(host, limits=replace(host.limits, max_update_entries=63))
    with pytest.raises(ValueError, match="exceed host authorization"):
        compile_awl_v3(intent, contract, limited_host)
    inference_only = replace(contract, checkpoint_sha256=None, capabilities=frozenset({Capability.EVALUATE_BEHAVIOR}))
    with pytest.raises(ValueError, match="modify_weights"):
        compile_awl_v3(intent, inference_only, host)
    no_gradient = replace(contract, capabilities=contract.capabilities - {Capability.COMPUTE_GRADIENTS})
    with pytest.raises(ValueError, match="supported model capabilities"):
        compile_awl_v3(intent, no_gradient, host, method="gradient")
    probe = compile_awl_v3(intent, no_gradient, host, method="direct_low_rank")
    assert probe.eligible_methods == ("direct_low_rank",)


def test_selector_binds_evidence_and_risk_model_to_checkpoint_and_limits_methods():
    intent = parse_awl_v3(_document())
    contract = _contract("llama", "LlamaForCausalLM", "a", (8, 20), 29)
    host = replace(_host(intent, contract), allowed_methods=("constrained", "gradient"))
    plan = compile_awl_v3(intent, contract, host)
    state = EvidenceState(True, 3, 10., True, curvature=.8)
    evidence = ModelEvidence(plan.family, plan.checkpoint_sha256, state)
    assert recommend_awl_v3(plan, evidence).action == Action.CONSTRAINED
    assert recommend_awl_v3(plan, replace(evidence, state=replace(state, measurement_valid=False))).action == Action.MEASURE
    with pytest.raises(ValueError, match="evidence belongs"):
        recommend_awl_v3(plan, replace(evidence, family="qwen3_5"))
    p = ModelPredictions(plan.family, plan.checkpoint_sha256, ((.99, .0), (.8, .2), (.99, .0)))
    assert recommend_awl_v3(plan, evidence, p).action == Action.HOLD
    with pytest.raises(ValueError, match="risk predictions belong"):
        recommend_awl_v3(plan, evidence, replace(p, checkpoint_sha256="b" * 64))


def test_parser_copies_untrusted_input_arrays():
    document = _document()
    original = deepcopy(document)
    intent = parse_awl_v3(document)
    document["preserve"].append("program")
    document["methods"].clear()
    assert intent == parse_awl_v3(original)
