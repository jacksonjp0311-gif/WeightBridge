"""Optional Hugging Face inference isolated from checkpoint patching code.

The module never writes model parameters. It loads a local model in a separate
inference process and enforces input, output, sampling and wall-time budgets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import time
from typing import Any
import platform

from ..provenance.resources import current_rss_bytes as _current_rss_bytes
from ..provenance.resources import peak_rss_bytes as _peak_rss_bytes


class InferenceError(RuntimeError):
    """Inference request or optional runtime is invalid."""


@dataclass(frozen=True)
class InferenceConfig:
    max_new_tokens: int = 256
    max_input_tokens: int = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5
    seed: int = 0
    thinking_enabled: bool = True
    timeout_seconds: float = 600.0
    deterministic_algorithms: bool = False
    cpu_threads: int = 0
    max_rss_bytes: int = 8 * 1024**3

    def validate(self, *, max_generation_budget: int = 32768,
                 max_input_budget: int = 65536, max_timeout_seconds: float = 3600.0) -> None:
        if type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= max_generation_budget:
            raise InferenceError("max_new_tokens exceeds host budget")
        if type(self.max_input_tokens) is not int or not 1 <= self.max_input_tokens <= max_input_budget:
            raise InferenceError("max_input_tokens exceeds host budget")
        if type(self.seed) is not int or self.seed < 0:
            raise InferenceError("seed must be a nonnegative integer")
        if type(self.top_k) is not int or self.top_k < 0:
            raise InferenceError("top_k must be a nonnegative integer")
        if type(self.cpu_threads) is not int or not 0 <= self.cpu_threads <= 256:
            raise InferenceError("cpu_threads must be 0 (runtime default) or in [1, 256]")
        if type(self.max_rss_bytes) is not int or not 256 * 1024**2 <= self.max_rss_bytes <= 128 * 1024**3:
            raise InferenceError("max_rss_bytes must be in [256 MiB, 128 GiB]")
        if type(self.thinking_enabled) is not bool or type(self.deterministic_algorithms) is not bool:
            raise InferenceError("thinking and deterministic flags must be booleans")
        if type(self.temperature) not in {int, float} or not 0 < self.temperature <= 5:
            raise InferenceError("temperature must be in (0, 5]")
        if type(self.top_p) not in {int, float} or not 0 < self.top_p <= 1:
            raise InferenceError("top_p must be in (0, 1]")
        if type(self.presence_penalty) not in {int, float} or not -5 <= self.presence_penalty <= 5:
            raise InferenceError("presence_penalty must be in [-5, 5]")
        if type(self.timeout_seconds) not in {int, float} or not 0 < self.timeout_seconds <= max_timeout_seconds:
            raise InferenceError("timeout_seconds exceeds host budget")


@dataclass(frozen=True)
class GenerationRecord:
    prompt_sha256: str
    prompt_tokens: int
    generated_tokens: int
    text: str
    finish_reason: str
    elapsed_seconds: float
    tokens_per_second: float
    model_load_seconds: float
    model_parameter_count: int
    runtime: dict[str, str]
    peak_rss_bytes: int | None
    peak_vram_bytes: int | None
    seed: int
    configuration: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HFTextRunner:
    """Read-only local Hugging Face text generation with fixed budgets."""

    def __init__(self, model: Any, tokenizer: Any, *, device: str, model_id: str,
                 model_load_seconds: float = 0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_id = model_id
        self.model_load_seconds = model_load_seconds
        self.parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())

    @classmethod
    def from_pretrained(cls, model_id: str, *, device: str = "auto", dtype: str = "auto",
                        local_files_only: bool = True, max_rss_bytes: int = 8 * 1024**3) -> "HFTextRunner":
        try:
            import torch
            import transformers
            from transformers import AutoConfig, AutoTokenizer
        except ImportError as exc:
            raise InferenceError("Install WeightBridge inference extras to load Transformers models") from exc
        if type(max_rss_bytes) is not int or not 256 * 1024**2 <= max_rss_bytes <= 128 * 1024**3:
            raise InferenceError("max_rss_bytes must be in [256 MiB, 128 GiB]")
        try:
            local_path = Path(model_id)
            if local_path.exists():
                if local_path.is_file():
                    checkpoint_bytes = local_path.stat().st_size
                else:
                    checkpoint_bytes = sum(path.stat().st_size for pattern in
                                           ("*.safetensors", "pytorch_model*.bin", "*.pt")
                                           for path in local_path.glob(pattern) if path.is_file())
                if checkpoint_bytes > max_rss_bytes:
                    raise InferenceError("Model checkpoint files exceed max_rss_bytes preflight budget")
            load_start = time.perf_counter()
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=False, local_files_only=local_files_only)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False, local_files_only=local_files_only)
            architecture = " ".join(getattr(config, "architectures", []) or [])
            if "ConditionalGeneration" in architecture:
                from transformers import AutoModelForImageTextToText
                model_class = AutoModelForImageTextToText
            else:
                from transformers import AutoModelForCausalLM
                model_class = AutoModelForCausalLM
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype_map = {"auto": "auto", "float32": torch.float32, "float16": torch.float16,
                         "bfloat16": torch.bfloat16}
            if dtype not in dtype_map:
                raise InferenceError("dtype must be auto, float32, float16 or bfloat16")
            kwargs: dict[str, Any] = {"trust_remote_code": False, "local_files_only": local_files_only}
            dtype_argument = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
            if dtype_map[dtype] == "auto":
                kwargs[dtype_argument] = "auto"
            else:
                kwargs[dtype_argument] = dtype_map[dtype]
            model = model_class.from_pretrained(model_id, **kwargs)
            model.to(device)
            model.eval()
            observed_rss = _current_rss_bytes()
            if observed_rss is not None and observed_rss > max_rss_bytes:
                del model
                raise InferenceError("Loaded model exceeds max_rss_bytes host budget")
            return cls(model, tokenizer, device=device, model_id=model_id,
                       model_load_seconds=time.perf_counter() - load_start)
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"Could not load model {model_id}: {type(exc).__name__}: {exc}") from exc

    def generate(self, prompt: str, config: InferenceConfig = InferenceConfig()) -> GenerationRecord:
        config.validate()
        if not isinstance(prompt, str) or not prompt.strip():
            raise InferenceError("prompt must be nonempty text")
        try:
            import torch
            from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
        except ImportError as exc:
            raise InferenceError("Transformers and PyTorch are required for generation") from exc

        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                encoded_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=config.thinking_enabled,
                )
            except TypeError:
                encoded_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                                      add_generation_prompt=True)
        else:
            encoded_prompt = prompt
        inputs = self.tokenizer(encoded_prompt, return_tensors="pt", add_special_tokens=False)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        if prompt_tokens > config.max_input_tokens:
            raise InferenceError("Prompt exceeds max_input_tokens")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        class PresencePenalty(LogitsProcessor):
            def __call__(self, input_ids, scores):
                # Match the published harness's processor: tokens present in
                # both prompt and generated context receive one presence penalty.
                for row in range(input_ids.shape[0]):
                    ids = torch.unique(input_ids[row])
                    scores[row, ids] -= float(config.presence_penalty)
                return scores

        start = time.perf_counter()
        deadline = start + float(config.timeout_seconds)
        stop_reason: dict[str, str | None] = {"reason": None}

        class DeadlineCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                if time.perf_counter() >= deadline:
                    stop_reason["reason"] = "timeout"
                    return True
                rss = _current_rss_bytes()
                if rss is not None and rss > config.max_rss_bytes:
                    stop_reason["reason"] = "memory_limit"
                    return True
                return False

        generator = torch.Generator(device=self.device)
        generator.manual_seed(config.seed)
        cuda_devices = []
        if self.device.startswith("cuda"):
            cuda_devices = [torch.device(self.device).index or torch.cuda.current_device()]
            torch.cuda.reset_peak_memory_stats()
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        old_threads = torch.get_num_threads()
        if config.cpu_threads:
            torch.set_num_threads(config.cpu_threads)
        generation_kwargs = {
            **inputs,
            "max_new_tokens": config.max_new_tokens,
            "do_sample": True,
            "temperature": float(config.temperature),
            "top_p": float(config.top_p),
            "top_k": config.top_k,
            "logits_processor": LogitsProcessorList([PresencePenalty()]),
            "stopping_criteria": StoppingCriteriaList([DeadlineCriteria()]),
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None)
                or getattr(self.tokenizer, "eos_token_id", None),
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
            "generator": generator,
        }
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(config.seed)
                if cuda_devices:
                    torch.cuda.manual_seed_all(config.seed)
                torch.use_deterministic_algorithms(config.deterministic_algorithms, warn_only=True)
                try:
                    with torch.inference_mode():
                        output = self.model.generate(**generation_kwargs)
                except (TypeError, ValueError) as exc:
                    if "generator" not in str(exc) or not ("not used by the model" in str(exc) or "unexpected keyword" in str(exc)):
                        raise
                    generation_kwargs.pop("generator")
                    with torch.inference_mode():
                        output = self.model.generate(**generation_kwargs)
        except Exception as exc:
            raise InferenceError(f"Generation failed: {type(exc).__name__}: {exc}") from exc
        finally:
            torch.use_deterministic_algorithms(old_deterministic, warn_only=True)
            if config.cpu_threads:
                torch.set_num_threads(old_threads)
        elapsed = time.perf_counter() - start
        generated_ids = output[0, prompt_tokens:]
        count = int(generated_ids.numel())
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        saw_eos = bool(count and eos_id is not None and int(generated_ids[-1]) == eos_id)
        timed_out = stop_reason["reason"] == "timeout" and count < config.max_new_tokens and not saw_eos
        reason = "eos" if saw_eos else ("memory_limit" if stop_reason["reason"] == "memory_limit" else
                                        "timeout" if timed_out else
                                        "max_new_tokens" if count >= config.max_new_tokens else "stopped")
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        vram = int(torch.cuda.max_memory_allocated()) if self.device.startswith("cuda") else None
        return GenerationRecord(
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            prompt_tokens=prompt_tokens, generated_tokens=count, text=text, finish_reason=reason,
            elapsed_seconds=elapsed, tokens_per_second=count / elapsed if elapsed else 0.0,
            model_load_seconds=self.model_load_seconds, model_parameter_count=self.parameter_count,
            peak_rss_bytes=_peak_rss_bytes(), peak_vram_bytes=vram, seed=config.seed,
            configuration={**asdict(config), "device": self.device, "model_id": self.model_id,
                           "deterministic_algorithms": config.deterministic_algorithms},
            runtime={"python": platform.python_version(), "platform": platform.platform(),
                     "torch": str(torch.__version__), "transformers": __import__("transformers").__version__},
        )
