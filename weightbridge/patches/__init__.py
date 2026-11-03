"""Host-controlled immutable safetensors patching."""

from .huggingface import apply_safetensors_patch

__all__ = ["apply_safetensors_patch"]
