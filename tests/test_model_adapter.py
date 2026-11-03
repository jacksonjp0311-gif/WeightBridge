import json

import numpy as np
import pytest

from weightbridge.adaptation.model_adapter import (
    AccessLevel, Capability, CapabilityError, InferenceOnlyAdapter, LlamaAdapter, select_hf_adapter,
)
from weightbridge.inference.huggingface import HFTextRunner


def _checkpoint(tmp_path, config):
    save_file = pytest.importorskip("safetensors.numpy").save_file
    save_file({"weight": np.eye(2, dtype=np.float32)}, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def test_metadata_contract_does_not_grant_edits_for_unknown_architecture(tmp_path):
    path = _checkpoint(tmp_path, {"model_type": "unregistered", "architectures": ["OtherLM"]})
    adapter = select_hf_adapter(path)
    contract = adapter.discover_model()
    assert contract.access_level is AccessLevel.RESTRICTED
    assert contract.capabilities == {Capability.READ_WEIGHTS, Capability.INSPECT_TENSORS}
    assert contract.checkpoint_sha256 == adapter.checkpoint.checkpoint_sha256()
    assert contract.checkpoint_bytes == (path / "model.safetensors").stat().st_size
    assert adapter.inspect_parameters()[0].shape == (2, 2)
    with pytest.raises(CapabilityError, match="registered"):
        adapter.load_runner()


def test_quantization_metadata_refuses_unvalidated_modification(tmp_path):
    path = _checkpoint(tmp_path, {"model_type": "llama", "architectures": ["LlamaForCausalLM"],
                                  "quantization_config": {"quant_method": "awq"}})
    adapter = select_hf_adapter(path)
    contract = adapter.discover_model()
    assert isinstance(adapter, LlamaAdapter)
    assert contract.quantized
    assert Capability.MODIFY_WEIGHTS not in contract.capabilities
    with pytest.raises(CapabilityError, match="Quantized"):
        adapter.load_runner()


def test_unsupported_file_and_explicit_inference_only_access(tmp_path):
    unsupported = tmp_path / "weights.gguf"
    unsupported.touch()
    with pytest.raises(CapabilityError, match="safetensors"):
        select_hf_adapter(unsupported)
    adapter = InferenceOnlyAdapter("remote", "text_endpoint", str.upper)
    contract = adapter.discover_model()
    assert contract.access_level is AccessLevel.INFERENCE_ONLY
    assert contract.checkpoint_sha256 is None
    assert contract.capabilities == {Capability.EVALUATE_BEHAVIOR}
    assert adapter.evaluate_behavior("test") == "TEST"
    with pytest.raises(CapabilityError, match="checkpoint"):
        adapter.inspect_parameters()


def test_llama_role_discovery_checks_actual_affine_runtime_and_gradient(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors.torch")
    config = transformers.LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                       num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    model = transformers.LlamaForCausalLM(config).eval()
    model.save_pretrained(tmp_path)

    class Tokenizer:
        def encode(self, text):
            return [0, 1]

        def decode(self, tokens):
            return str(tokens)

    adapter = select_hf_adapter(tmp_path)
    runner = HFTextRunner(model, Tokenizer(), device="cpu", model_id=str(tmp_path))
    metadata = adapter.discover_model()
    assert Capability.MODIFY_WEIGHTS not in metadata.capabilities
    contract = adapter.discover_model(runner)
    assert contract.access_level is AccessLevel.EDITABLE
    assert [region.layer_index for region in contract.editable_regions] == [0, 1]
    region = contract.editable_regions[-1]
    assert region.role == "ffn_output"
    assert region.tensor_name == "model.layers.1.mlp.down_proj.weight"
    assert region.shape == (16, 32)
    assert region.dtype == "F32"
    assert Capability.COMPUTE_GRADIENTS in contract.capabilities
    assert contract.runtime.gradient_probe_tensor == region.tensor_name
    assert contract.runtime.forward_calls == contract.runtime.backward_calls == 1
    assert json.loads(json.dumps(contract.to_dict()))["capabilities"].count("modify_weights") == 1
    readonly = adapter.discover_model(runner, probe_gradients=False)
    assert Capability.COMPUTE_GRADIENTS not in readonly.capabilities
    assert readonly.runtime.backward_calls == 0
    runner.model_id = str(tmp_path / "different")
    with pytest.raises(CapabilityError, match="identities"):
        adapter.discover_model(runner)
    runner.model_id = str(tmp_path)
    with torch.no_grad():
        model.model.layers[1].mlp.down_proj.weight.zero_()
    mismatched = adapter.discover_model(runner, probe_gradients=False)
    assert [region.layer_index for region in mismatched.editable_regions] == [0]
    model.model.layers[1].mlp.down_proj.weight = model.model.layers[0].mlp.down_proj.weight
    shared = adapter.discover_model(runner)
    assert shared.access_level is AccessLevel.RESTRICTED
    assert not shared.editable_regions
    assert Capability.MODIFY_WEIGHTS not in shared.capabilities
