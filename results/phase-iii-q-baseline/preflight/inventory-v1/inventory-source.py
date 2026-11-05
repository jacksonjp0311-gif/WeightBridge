"""Bounded real-runtime inventory and training-cost preflight, never efficacy.

This entry point does not read study development, selection or hidden data.
Every output is a fresh write-once receipt. The pinned source model remains
immutable; any temporary training adapters are discarded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import sys
import time

from weightbridge.adaptation.model_adapter import select_hf_adapter
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.provenance import peak_rss_bytes


PINNED_SHA256 = "7d4532858cf839d05ec34da088114df47238322f86f5e6e5c6693594a21a03b8"
_TEXT_AFFINE = re.compile(r"model\.language_model\.layers\.(\d+)\.(self_attn|linear_attn|mlp)\.[^.]+")


def text_affine_inventory(model):
    """Inventory actual dense linears in explicit text attention/MLP regions.

    These model-specific names describe this preflight, not universal adapter
    semantics. Embeddings, heads, vision, tied aliases and other module types
    never enter the declared allowlist.
    """
    import torch

    tied = set()
    for getter in ("get_input_embeddings", "get_output_embeddings"):
        module = getattr(model, getter, lambda: None)()
        if module is not None and getattr(module, "weight", None) is not None:
            tied.add(module.weight.data_ptr())
    allowed, excluded = [], []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        match = _TEXT_AFFINE.fullmatch(name)
        reason = ("outside_explicit_text_attention_mlp" if match is None else
                  "embedding_or_head_weight_alias" if module.weight.data_ptr() in tied else
                  "non_matrix_or_nonfloating_weight" if module.weight.ndim != 2 or not module.weight.is_floating_point() else None)
        row = {"name": name, "class": type(module).__name__, "weight_shape": list(module.weight.shape),
            "weight_dtype": str(module.weight.dtype), "device": str(module.weight.device),
            "has_bias": module.bias is not None}
        if reason:
            excluded.append({**row, "reason": reason})
        else:
            allowed.append({**row, "layer": int(match.group(1)), "role": match.group(2)})
    return {"schema": "weightbridge.phase_iiiq_baseline.affine_inventory.v1",
        "explicit_module_allowlist": [row["name"] for row in allowed], "allowed": allowed, "excluded": excluded,
        "allowed_count": len(allowed), "excluded_linear_count": len(excluded),
        "counts_by_role": dict(sorted(Counter(row["role"] for row in allowed).items())),
        "counts_by_layer": dict(sorted(Counter(str(row["layer"]) for row in allowed).items())),
        "discovery_is_training_or_efficacy": False}


def environment():
    import psutil
    import torch

    names = ("torch", "transformers", "tokenizers", "numpy", "safetensors", "psutil", "peft")
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
        "physical_cpu_cores": psutil.cpu_count(logical=False), "logical_cpu_cores": psutil.cpu_count(),
        "system_ram_bytes": psutil.virtual_memory().total, "cuda_available": torch.cuda.is_available(),
        "cpu_threads": 6, "blas_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "replacement_downloads": 0, "external_compute": False}


def inventory(base, out):
    import torch

    if out.exists():
        raise FileExistsError("Preflight output is write-once")
    out.mkdir(parents=True)
    wall, cpu = time.perf_counter(), time.process_time()
    adapter = select_hf_adapter(base)
    before = adapter.checkpoint.checkpoint_sha256()
    if before != PINNED_SHA256:
        raise ValueError("Preflight source checkpoint differs from the required parent")
    json_dump_atomic(out / "environment.json", environment())
    previous = torch.get_num_threads()
    torch.set_num_threads(6)
    runner = None
    try:
        runner = adapter.load_runner(device="cpu", dtype="auto", max_rss_bytes=24*1024**3)
        found = text_affine_inventory(runner.model)
        found.update(checkpoint_sha256=before, model_class=type(runner.model).__name__,
            tokenizer_class=type(runner.tokenizer).__name__, parameter_count=runner.parameter_count,
            model_load_seconds=runner.model_load_seconds,
            parameter_dtype_counts=dict(Counter(str(p.dtype) for p in runner.model.parameters())))
        json_dump_atomic(out / "architecture-inventory.json", found)
        after = adapter.checkpoint.checkpoint_sha256()
        if after != before:
            raise RuntimeError("Read-only inventory changed the source checkpoint")
        receipt = {"schema": "weightbridge.phase_iiiq_baseline.inventory_receipt.v1", "status": "PASS",
            "checkpoint_sha256": before, "checkpoint_identity_after": after,
            "source_sha256": file_hash(__file__), "inventory_sha256": file_hash(out / "architecture-inventory.json"),
            "environment_sha256": file_hash(out / "environment.json"), "model_forward_calls": 0, "model_backward_calls": 0,
            "resources": {"wall_seconds": time.perf_counter()-wall, "cpu_seconds": time.process_time()-cpu,
                          "peak_rss_bytes": peak_rss_bytes()},
            "scope": "Actual pretrained runtime inventory only. No task scoring or parameter update."}
        json_dump_atomic(out / "inventory-receipt.json", receipt)
        return receipt
    finally:
        runner = None
        torch.set_num_threads(previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory",))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(inventory(args.base, args.out), flush=True)
