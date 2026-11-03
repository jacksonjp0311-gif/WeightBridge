"""Bounded read-only activation summaries and reversible activation ablations."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


def _tensor_leaves(value: Any):
    if hasattr(value, "detach") and hasattr(value, "numel"):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_leaves(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_leaves(item)


def activation_summary(model: Any, module_names: list[str], forward: Any, *,
                       max_modules: int = 16, max_elements_per_module: int = 1_000_000) -> dict[str, Any]:
    """Capture summary statistics only; activations are never returned or retained."""
    if (type(max_modules) is not int or not 1 <= max_modules <= 16
            or type(max_elements_per_module) is not int or not 1 <= max_elements_per_module <= 1_000_000
            or not isinstance(module_names, list) or not 1 <= len(module_names) <= max_modules):
        raise ValueError("module_names exceeds probe budget")
    modules = dict(model.named_modules())
    if len(set(module_names)) != len(module_names) or any(name not in modules for name in module_names):
        raise ValueError("Unknown or duplicate probe module")
    summaries: dict[str, Any] = {}
    handles = []
    def make_hook(name: str):
        def hook(_module, _inputs, output):
            leaves = list(_tensor_leaves(output))
            if not leaves:
                summaries[name] = {"captured": False, "reason": "no tensor output"}
                return
            tensor = leaves[0].detach()
            if tensor.numel() > max_elements_per_module:
                summaries[name] = {"captured": False, "reason": "activation exceeds element budget"}
                return
            import torch
            value = tensor.float()
            summaries[name] = {
                "captured": True, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "mean": float(value.mean().item()), "std": float(value.std(unbiased=False).item()),
                "l2_norm": float(torch.linalg.vector_norm(value).item()),
                "max_abs": float(value.abs().max().item()) if tensor.numel() else 0.0,
            }
        return hook
    try:
        for name in module_names:
            handles.append(modules[name].register_forward_hook(make_hook(name)))
        forward()
    finally:
        for handle in handles:
            handle.remove()
    return summaries


def attention_summary(model: Any, model_inputs: dict[str, Any], layer_indices: list[int], *,
                      max_tokens: int = 64, max_attention_elements: int = 4_000_000) -> dict[str, Any]:
    """Summarize explicitly returned attention weights under a quadratic budget.

    This asks the model for attention weights only on a short bounded input and
    returns per-layer aggregate entropy and concentration, never full matrices.
    Some architectures/backends do not expose attention weights; those calls
    fail explicitly instead of silently substituting another probe.
    """
    import torch
    if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
        raise ValueError("max_tokens must be in [1, 512]")
    input_ids = model_inputs.get("input_ids")
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("attention probe requires one input_ids sequence")
    sequence_length = int(input_ids.shape[1])
    if not 1 <= sequence_length <= max_tokens:
        raise ValueError("attention probe input exceeds max_tokens")
    if not isinstance(layer_indices, list) or not layer_indices or len(set(layer_indices)) != len(layer_indices):
        raise ValueError("layer_indices must be nonempty and unique")
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    num_layers = getattr(text_config, "num_hidden_layers", None)
    num_heads = getattr(text_config, "num_attention_heads", None)
    if type(num_layers) is not int or type(num_heads) is not int or num_layers < 1 or num_heads < 1:
        raise ValueError("model config does not expose bounded attention dimensions")
    if any(type(layer) is not int or not 0 <= layer < num_layers for layer in layer_indices):
        raise ValueError("attention layer index is outside model config")
    estimated_elements = num_layers * num_heads * sequence_length * sequence_length
    if type(max_attention_elements) is not int or not 1 <= max_attention_elements <= 4_000_000 or estimated_elements > max_attention_elements:
        raise ValueError("attention request exceeds the hard element budget")
    try:
        with torch.inference_mode():
            output = model(**model_inputs, output_attentions=True, use_cache=False, return_dict=True)
    except Exception as exc:
        raise ValueError(f"model cannot provide attention weights for this input: {type(exc).__name__}: {exc}") from exc
    attentions = getattr(output, "attentions", None)
    if not attentions or len(attentions) != num_layers:
        raise ValueError("model returned no complete attention tuple")
    result: dict[str, Any] = {"input_tokens": sequence_length, "attention_layers": {}}
    for layer in layer_indices:
        tensor = attentions[layer]
        if tensor is None or tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError(f"attention weights unavailable or malformed at layer {layer}")
        if tensor.numel() > max_attention_elements:
            raise ValueError(f"selected attention layer {layer} exceeds the output budget")
        values = tensor[0].float()
        if not torch.isfinite(values).all():
            raise ValueError(f"attention layer {layer} contains nonfinite values")
        probs = values.clamp_min(0)
        sums = probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probs.dtype).tiny)
        probs = probs / sums
        entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(dim=-1)
        result["attention_layers"][str(layer)] = {
            "shape": list(tensor.shape), "heads": int(tensor.shape[1]),
            "mean_entropy_nats": float(entropy.mean().item()),
            "mean_max_attention": float(probs.max(dim=-1).values.mean().item()),
            "mean_diagonal_attention": float(probs.diagonal(dim1=-2, dim2=-1).mean().item()),
        }
    del attentions, output
    return result


@contextmanager
def zero_module_output(model: Any, module_name: str, *, max_elements: int = 1_000_000) -> Iterator[None]:
    """Temporarily zero one module output for an ablation; always removes its hook."""
    modules = dict(model.named_modules())
    if type(max_elements) is not int or not 1 <= max_elements <= 1_000_000:
        raise ValueError("max_elements must be in [1, 1000000]")
    if module_name not in modules:
        raise ValueError(f"Unknown ablation module: {module_name}")
    def zero(_module, _inputs, output):
        leaves = list(_tensor_leaves(output))
        if not leaves or leaves[0].numel() > max_elements:
            raise ValueError("Ablation output exceeds activation budget")
        first = leaves[0]
        try:
            import torch
            if isinstance(output, tuple):
                return (torch.zeros_like(first), *output[1:])
            if isinstance(output, list):
                return [torch.zeros_like(first), *output[1:]]
            if isinstance(output, dict):
                key = next(key for key, value in output.items() if value is first)
                copied = dict(output)
                copied[key] = torch.zeros_like(first)
                return copied
            return torch.zeros_like(first)
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for activation ablation") from exc
    handle = modules[module_name].register_forward_hook(zero)
    try:
        yield
    finally:
        handle.remove()
