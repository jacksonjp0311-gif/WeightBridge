"""Discover and validate capabilities of supported local model runtimes."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
from pathlib import Path
import time
from typing import Callable, Protocol

from ..inference.huggingface import HFTextRunner, InferenceConfig
from ..inspection.safetensors import Checkpoint, SafetensorsError, TensorRef, load_checkpoint
from ..provenance.resources import peak_rss_bytes


class Capability(str, Enum):
    READ_WEIGHTS = "read_weights"
    INSPECT_TENSORS = "inspect_tensors"
    MODIFY_WEIGHTS = "modify_weights"
    TRAIN_ADAPTER = "train_adapter"
    COMPUTE_GRADIENTS = "compute_gradients"
    COMPUTE_JVP_VJP = "compute_jvp_vjp"
    EVALUATE_BEHAVIOR = "evaluate_behavior"
    SAVE_CHECKPOINT = "save_checkpoint"
    RELOAD_CHECKPOINT = "reload_checkpoint"
    RESTORE_ORIGINAL = "restore_original"
    CONFIGURE_RUNTIME = "configure_runtime"


class AccessLevel(str, Enum):
    EDITABLE = "editable_open_weight"
    RESTRICTED = "restricted_weight"
    INFERENCE_ONLY = "inference_only"


class CapabilityError(ValueError):
    """The requested operation is unavailable for the supplied model."""


@dataclass(frozen=True)
class EditableRegion:
    role: str
    tensor_name: str
    module_name: str
    layer_index: int
    shape: tuple[int, int]
    dtype: str


@dataclass(frozen=True)
class RuntimeEvidence:
    device: str
    torch_version: str
    transformers_version: str
    model_load_seconds: float
    peak_rss_bytes: int | None
    gradient_probe_tensor: str | None
    forward_calls: int
    backward_calls: int
    probe_wall_seconds: float
    probe_cpu_seconds: float


@dataclass(frozen=True)
class ModelCapabilityContract:
    family: str
    architecture: str
    access_level: AccessLevel
    checkpoint_sha256: str | None
    serialization: str | None
    storage_dtypes: tuple[str, ...]
    tokenizer_class: str | None
    quantized: bool
    capabilities: frozenset[Capability]
    editable_regions: tuple[EditableRegion, ...] = ()
    limitations: tuple[str, ...] = ()
    runtime: RuntimeEvidence | None = None
    checkpoint_bytes: int | None = None
    schema: str = "weightbridge.model_capabilities.v1"

    def to_dict(self) -> dict:
        result = asdict(self)
        result["access_level"] = self.access_level.value
        result["capabilities"] = sorted(capability.value for capability in self.capabilities)
        return result


class ModelAdapter(Protocol):
    """The local checkpoint and inference surface used by adaptation compilers."""

    checkpoint: Checkpoint

    def discover_model(self, runner: HFTextRunner | None = None, *,
                       probe_gradients: bool = True) -> ModelCapabilityContract: ...
    def inspect_parameters(self) -> tuple[TensorRef, ...]: ...
    def discover_editable_modules(self, runner: HFTextRunner) -> tuple[EditableRegion, ...]: ...
    def load_runner(self, *, device: str = "cpu", dtype: str = "auto",
                    max_rss_bytes: int = 8 * 1024**3) -> HFTextRunner: ...


_STORAGE_DTYPES = {"F16", "BF16", "F32", "F64"}
_BASE_CAPABILITIES = frozenset({Capability.READ_WEIGHTS, Capability.INSPECT_TENSORS})
_EDIT_CAPABILITIES = frozenset({Capability.MODIFY_WEIGHTS, Capability.SAVE_CHECKPOINT,
                                Capability.RELOAD_CHECKPOINT, Capability.RESTORE_ORIGINAL})


def _declared_quantization(checkpoint: Checkpoint) -> bool:
    config = checkpoint.config
    text = config.get("text_config")
    configs = (config, text) if isinstance(text, dict) else (config,)
    return (
        any(mapping.get(key) is not None for mapping in configs
            for key in ("quantization_config", "compression_config", "quant_method"))
        or any(ref.dtype.startswith("F8") or set(ref.name.split(".")) & {
            "qweight", "qzeros", "quant_state", "weight_packed",
        } for ref in checkpoint.tensors.values())
    )


class HFModelAdapter:
    """Metadata-only HF inspection, specialized by an explicit dense model family."""

    model_types: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ()
    implementation = ""
    mlp_class = ""

    def __init__(self, checkpoint: Checkpoint):
        self.checkpoint = checkpoint
        config = checkpoint.config
        architecture = config.get("architectures")
        self.architecture = (architecture[0] if isinstance(architecture, list) and len(architecture) == 1
                             and isinstance(architecture[0], str) else "unknown")
        family = config.get("model_type")
        self.family = family if isinstance(family, str) else "unknown"

    def inspect_parameters(self) -> tuple[TensorRef, ...]:
        return tuple(self.checkpoint.tensors.values())

    def _restriction(self) -> str | None:
        checkpoint = self.checkpoint
        if _declared_quantization(checkpoint):
            return "Quantized or compressed weights require a separately validated backend."
        if self.architecture not in self.architectures:
            return "No direct adaptation adapter is registered for this model architecture."
        if (not checkpoint.root.is_dir()
                or (checkpoint.index_path is None and checkpoint.shards[0].name != "model.safetensors")
                or (checkpoint.index_path is not None
                    and checkpoint.index_path.name != "model.safetensors.index.json")):
            return "The local Hugging Face adapter requires a standard safetensors directory."
        if any(ref.dtype not in _STORAGE_DTYPES for ref in checkpoint.tensors.values()):
            return "The dense adapter requires supported floating-point tensor storage."
        if any(path.suffix.lower() in {".bin", ".pt", ".pth", ".gguf"} for path in checkpoint.assets):
            return "The adapter requires an unambiguous safetensors-only weights layout."
        return None

    def load_runner(self, *, device: str = "cpu", dtype: str = "auto",
                    max_rss_bytes: int = 8 * 1024**3) -> HFTextRunner:
        restriction = self._restriction()
        if restriction:
            raise CapabilityError(restriction)
        return HFTextRunner.from_pretrained(str(self.checkpoint.root), device=device, dtype=dtype,
                                            local_files_only=True, max_rss_bytes=max_rss_bytes)

    def _is_implementation(self, instance, class_name: str) -> bool:
        actual = type(instance)
        return actual.__module__ == self.implementation and actual.__name__ == class_name

    def _validate_runner(self, runner: HFTextRunner) -> None:
        if (Path(runner.model_id).resolve() != self.checkpoint.root
                or not self._is_implementation(runner.model, self.architecture)
                or getattr(runner.model.config, "model_type", None) != self.family):
            raise CapabilityError("Loaded model and source checkpoint identities differ.")
        if (not callable(getattr(runner.model, "generate", None))
                or not callable(getattr(runner.tokenizer, "encode", None))
                or not callable(getattr(runner.tokenizer, "decode", None))):
            raise CapabilityError("Loaded runtime lacks supported text input or generation.")
        if any(getattr(runner.model, flag, False) for flag in ("is_quantized", "is_loaded_in_4bit",
                                                              "is_loaded_in_8bit")):
            raise CapabilityError("The loaded runtime uses unvalidated quantized weights.")

    def _decoder(self, model):
        return model.model

    def discover_editable_modules(self, runner: HFTextRunner) -> tuple[EditableRegion, ...]:
        if self._restriction():
            return ()
        self._validate_runner(runner)
        import torch

        decoder = self._decoder(runner.model)
        hidden = decoder.config.hidden_size
        intermediate = decoder.config.intermediate_size
        if hidden == intermediate:
            return ()
        names = {id(module): name for name, module in runner.model.named_modules()}
        aliases = Counter(parameter.data_ptr() for _, parameter in
                          runner.model.named_parameters(remove_duplicate=False))
        dtypes = {torch.float16: "F16", torch.bfloat16: "BF16", torch.float32: "F32", torch.float64: "F64"}
        regions = []
        for layer_index, block in enumerate(decoder.layers):
            mlps = [child for child in block.children() if self._is_implementation(child, self.mlp_class)]
            if len(mlps) != 1:
                continue
            linears = [child for child in mlps[0].children() if type(child) is torch.nn.Linear]
            contractions = [linear for linear in linears if
                            (linear.in_features, linear.out_features) == (intermediate, hidden)]
            expansions = [linear for linear in linears if
                          (linear.in_features, linear.out_features) == (hidden, intermediate)]
            if len(contractions) != 1 or len(expansions) != 2:
                continue
            module = contractions[0]
            module_name = names[id(module)]
            tensor_name = f"{module_name}.weight"
            weight = module.weight
            ref = self.checkpoint.tensors.get(tensor_name)
            if (ref is not None and ref.shape == (hidden, intermediate) and ref.shape == tuple(weight.shape)
                    and ref.dtype == dtypes.get(weight.dtype) and aliases[weight.data_ptr()] == 1
                    and weight.device.type != "meta"):
                runtime_bytes = weight.detach().cpu().contiguous().view(torch.uint8).numpy()
                if hashlib.sha256(runtime_bytes).hexdigest() == self.checkpoint.tensor_sha256(tensor_name):
                    regions.append(EditableRegion("ffn_output", tensor_name, module_name, layer_index,
                                                  (hidden, intermediate), ref.dtype))
        return tuple(regions)

    @staticmethod
    def _runtime(runner, regions, probe_gradients):
        import torch
        import transformers

        tested = None
        forward_calls = backward_calls = 0
        started, cpu = time.perf_counter(), time.process_time()
        if regions and probe_gradients:
            selected = regions[-1]
            parameter = runner.model.get_parameter(selected.tensor_name)
            if parameter.requires_grad:
                try:
                    with torch.enable_grad():
                        ids = torch.tensor([[0, 1]], device=parameter.device)
                        forward_calls = 1
                        output = runner.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                        backward_calls = 1
                        gradient, = torch.autograd.grad(output.logits[0, -1, 0].float(), parameter)
                        if bool(torch.isfinite(gradient).all()) and bool(torch.count_nonzero(gradient)):
                            tested = selected.tensor_name
                except (RuntimeError, NotImplementedError, ValueError):
                    # A failed runtime probe removes gradient capability from this receipt.
                    pass
        return RuntimeEvidence(
            runner.device, str(torch.__version__), transformers.__version__, runner.model_load_seconds,
            peak_rss_bytes(), tested, forward_calls, backward_calls,
            time.perf_counter() - started, time.process_time() - cpu,
        )

    def discover_model(self, runner: HFTextRunner | None = None, *,
                       probe_gradients: bool = True) -> ModelCapabilityContract:
        checkpoint = self.checkpoint
        capabilities = _BASE_CAPABILITIES
        regions = ()
        runtime = None
        tokenizer = checkpoint.tokenizer_config.get("tokenizer_class")
        limitations = []
        restriction = self._restriction()
        if restriction:
            limitations.append(restriction)
        elif runner is None:
            limitations.append("Editable regions require validation against a loaded runtime.")
        else:
            regions = self.discover_editable_modules(runner)
            tokenizer = type(runner.tokenizer).__name__
            capabilities |= {Capability.EVALUATE_BEHAVIOR, Capability.CONFIGURE_RUNTIME}
            runtime = self._runtime(runner, regions, probe_gradients)
            if regions:
                capabilities |= _EDIT_CAPABILITIES
                if runtime.gradient_probe_tensor:
                    capabilities |= {Capability.COMPUTE_GRADIENTS}
                else:
                    limitations.append("No finite, nonzero model gradient was verified for this receipt.")
            else:
                limitations.append("No supported affine region matched stored shape, dtype and untied weight bytes.")
        return ModelCapabilityContract(
            self.family, self.architecture, AccessLevel.EDITABLE if regions else AccessLevel.RESTRICTED,
            checkpoint.checkpoint_sha256(), "safetensors", tuple(sorted({ref.dtype for ref in checkpoint.tensors.values()})),
            tokenizer if isinstance(tokenizer, str) else None, _declared_quantization(checkpoint),
            capabilities, regions, tuple(limitations), runtime, sum(path.stat().st_size for path in checkpoint.shards),
        )

    def evaluate_behavior(self, runner: HFTextRunner, prompt: str, config: InferenceConfig):
        restriction = self._restriction()
        if restriction:
            raise CapabilityError(restriction)
        self._validate_runner(runner)
        return runner.generate(prompt, config)


class Qwen35Adapter(HFModelAdapter):
    model_types = ("qwen3_5", "qwen3_5_text")
    architectures = ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")
    implementation = "transformers.models.qwen3_5.modeling_qwen3_5"
    mlp_class = "Qwen3_5MLP"

    def _decoder(self, model):
        if self.architecture == "Qwen3_5ForConditionalGeneration":
            return model.model.language_model
        return model.model


class LlamaAdapter(HFModelAdapter):
    model_types = ("llama",)
    architectures = ("LlamaForCausalLM",)
    implementation = "transformers.models.llama.modeling_llama"
    mlp_class = "LlamaMLP"


def select_hf_adapter(path: str | Path) -> HFModelAdapter:
    """Inspect supported safetensors metadata and select a concrete family adapter."""
    try:
        checkpoint = load_checkpoint(path)
    except SafetensorsError as exc:
        raise CapabilityError(str(exc)) from exc
    family = checkpoint.config.get("model_type")
    for adapter in (Qwen35Adapter, LlamaAdapter):
        if family in adapter.model_types:
            return adapter(checkpoint)
    return HFModelAdapter(checkpoint)


class InferenceOnlyAdapter:
    """Explicit evaluator access to a runtime without local weight authority."""

    def __init__(self, family: str, architecture: str, evaluate: Callable[[str], str]):
        if not family or not architecture or not callable(evaluate):
            raise CapabilityError("Inference-only access requires an identified evaluator.")
        self.family = family
        self.architecture = architecture
        self._evaluate = evaluate

    def discover_model(self) -> ModelCapabilityContract:
        return ModelCapabilityContract(
            self.family, self.architecture, AccessLevel.INFERENCE_ONLY, None, None, (), None, False,
            frozenset({Capability.EVALUATE_BEHAVIOR}), limitations=("No checkpoint or weight access.",),
        )

    def evaluate_behavior(self, prompt: str) -> str:
        return self._evaluate(prompt)

    def inspect_parameters(self):
        raise CapabilityError("Parameter inspection requires checkpoint access.")
