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


def _parameter_digest(parameters):
    """Hash original parameter bytes in bounded chunks, independent of wrappers."""
    import hashlib
    import torch
    digest = hashlib.sha256()
    for parameter in parameters:
        tensor = parameter.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        values = tensor.view(torch.uint8).reshape(-1)
        for start in range(0, values.numel(), 8*1024**2):
            digest.update(values[start:start+8*1024**2].numpy().tobytes())
    return digest.hexdigest()


def timing(base, out, *, bare=False):
    """Time at most six real F/B microbatches; no efficacy endpoint is scored."""
    import hashlib
    import torch
    from decimal import Decimal
    from experiments.phase_iiig.roster_decimal_targeted import _prompt
    from weightbridge.adaptation.supervised_lora import SupervisedLoRA, discover_lora_targets
    from weightbridge.adaptation.completion_training import prepare_completion, collate_completions, completion_loss
    from weightbridge.adaptation.behavior_endpoint import endpoint_runner
    from experiments.phase_iiiq_baseline.measurement import endpoint

    if out.exists():
        raise FileExistsError("Preflight timing evidence is write-once")
    out.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    (out / "preflight-source.py").write_bytes(source)
    started, cpu = time.perf_counter(), time.process_time()
    previous = torch.get_num_threads()
    torch.set_num_threads(6)
    memory_limit = 24*1024**3
    error, record, runner, session = None, None, None, None
    forwards = backwards = optimizer_steps = 0
    maximum_forwards, maximum_backwards = (4, 2) if bare else (8, 8)
    maximum_wall = 110 if bare else 280
    adapter = select_hf_adapter(base)
    before = adapter.checkpoint.checkpoint_sha256()
    parameters, live_before, live_after = (), None, None
    def budget():
        if time.perf_counter()-started > maximum_wall:
            raise TimeoutError("Bounded training-cost preflight exceeded its prospective wall limit")
        if (peak_rss_bytes() or 0) > memory_limit:
            raise MemoryError("Training-cost preflight exceeded24GiB")
        if forwards > maximum_forwards or backwards > maximum_backwards:
            raise RuntimeError("Training-cost preflight exceeds its prospective F/B limit")
    try:
        if before != PINNED_SHA256:
            raise ValueError("Timing parent differs from the pinned checkpoint")
        runner = adapter.load_runner(device="cpu", dtype="auto", max_rss_bytes=memory_limit)
        parameters = tuple(runner.model.parameters())
        live_before = _parameter_digest(parameters)
        targets, target_inventory = discover_lora_targets(adapter, runner, expected_layers=24)
        own_inventory = text_affine_inventory(runner.model)
        json_dump_atomic(out / "target-inventory.json", target_inventory)
        json_dump_atomic(out / "actual-linears.json", own_inventory)
        operands = (("-1000.0811", "-1000.1"), ("-1000.1811", "-1000.1"),
                    ("-800.0811", "-801.1"), ("-802.1811", "-801.1"))
        raw_rows = [{"id": f"timing-only/{i}", "operands": list(values),
            "prompt": _prompt("larger-decimal", values, i % 2),
            "answer": max(values, key=Decimal)} for i, values in enumerate(operands)]
        json_dump_atomic(out / "timing-inputs.json", {"scope": "Cost-only artificial preflight text, not a scored study roster or efficacy measurement.", "rows": raw_rows})
        training_tokenizer = (endpoint_runner(runner, endpoint({"max_input_tokens": 128,
            "max_new_tokens": 16, "batch_size": 4})).tokenizer if bare else runner.tokenizer)
        prepared = [prepare_completion(training_tokenizer, row["prompt"], row["answer"], leading_space=True,
                    max_input_tokens=128, max_target_tokens=16) for row in raw_rows]
        target_lengths = [len(row.target_ids) for row in prepared]
        json_dump_atomic(out / "tokenized-inputs.json", {"prompt_representation": "bare" if bare else "native_chat_template",
            "rows": [{"id": row["id"], "prefix_ids": list(encoded.prefix_ids), "target_ids": list(encoded.target_ids),
                "rendered_prompt_sha256": encoded.rendered_prompt_sha256}
                for row, encoded in zip(raw_rows, prepared, strict=True)]})
        comparison_batch = collate_completions(prepared[:2], pad_token_id=runner.tokenizer.pad_token_id,
                                               device=runner.device)
        with torch.inference_mode():
            full_loss, full_counts = completion_loss(runner.model, comparison_batch, logits_mode="all")
            forwards += 1
            tail_loss, tail_counts = completion_loss(runner.model, comparison_batch, logits_mode="tail")
            forwards += 1
        native_equivalence = {"full_loss": float(full_loss), "tail_loss": float(tail_loss),
            "absolute_difference": abs(float(full_loss)-float(tail_loss)),
            "full_counts": full_counts, "tail_counts": tail_counts,
            "scope": "Native numerical comparison only; no generated-answer accuracy or model selection."}
        del full_loss, tail_loss, comparison_batch
        json_dump_atomic(out / "native-logit-comparison.json", native_equivalence)
        batch_records = []
        for microbatch in ((4,) if bare else (2, 4)):
            budget()
            session = SupervisedLoRA(runner.model, targets, checkpoint_sha256=before,
                                     rank=8, alpha=16, dropout=0, seed=472019)
            session.attach()
            trainable = tuple(session.parameters())
            trainable_before = _parameter_digest(trainable)
            optimizer = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=0.)
            batch = collate_completions(prepared[:microbatch], pad_token_id=runner.tokenizer.pad_token_id,
                                        device=runner.device)
            input_shape = list(batch["input_ids"].shape)
            if input_shape[1] > 128:
                raise ValueError("Cost-only preflight tokens exceed128")
            runner.model.train()
            samples = []
            def microstep(stage, divisor=1.):
                nonlocal forwards, backwards
                budget()
                wall0, cpu0 = time.perf_counter(), time.process_time()
                loss, counts = completion_loss(runner.model, batch, logits_mode="tail")
                forwards += 1
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite completion training loss during timing")
                forward_wall, forward_cpu = time.perf_counter()-wall0, time.process_time()-cpu0
                wall0, cpu0 = time.perf_counter(), time.process_time()
                (loss/divisor).backward()
                backwards += 1
                samples.append({"stage": stage, "forward_wall_seconds": forward_wall,
                    "forward_cpu_seconds": forward_cpu, "backward_wall_seconds": time.perf_counter()-wall0,
                    "backward_cpu_seconds": time.process_time()-cpu0, "counts": counts})
                del loss
                budget()
            def optimize():
                nonlocal optimizer_steps
                wall0, cpu0 = time.perf_counter(), time.process_time()
                torch.nn.utils.clip_grad_norm_(trainable, 1.)
                optimizer.step()
                optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
                return {"wall_seconds": time.perf_counter()-wall0, "cpu_seconds": time.process_time()-cpu0}
            optimizer.zero_grad(set_to_none=True)
            microstep("warmup")
            warmup_optimizer = optimize()
            timed_count = 1 if bare else 2
            for _ in range(timed_count):
                microstep("timed", divisor=float(timed_count))
            measured_optimizer = optimize()
            session.assert_base_unchanged()
            finite = all(torch.isfinite(value).all().item() for value in trainable)
            trainable_after = _parameter_digest(trainable)
            timed = [value for value in samples if value["stage"] == "timed"]
            mean_wall = sum(value["forward_wall_seconds"]+value["backward_wall_seconds"] for value in timed)/len(timed)
            mean_cpu = sum(value["forward_cpu_seconds"]+value["backward_cpu_seconds"] for value in timed)/len(timed)
            macro_wall = (16/microbatch)*mean_wall+measured_optimizer["wall_seconds"]
            macro_cpu = (16/microbatch)*mean_cpu+measured_optimizer["cpu_seconds"]
            row = {"microbatch_size": microbatch, "input_shape": input_shape,
                "trainable_parameter_count": sum(value.numel() for value in trainable),
                "LoRA_rank": 8, "LoRA_alpha": 16, "dropout": 0, "seed": 472019,
                "samples": samples, "warmup_optimizer": warmup_optimizer,
                "measured_optimizer": measured_optimizer, "trainable_parameters_finite": finite,
                "temporary_adapter_hash_before": trainable_before, "temporary_adapter_hash_after": trainable_after,
                "estimated_effective_batch16_step_wall_seconds": macro_wall,
                "estimated_effective_batch16_step_cpu_seconds": macro_cpu,
                "estimated_three_epoch384step_run_wall_seconds": 384*macro_wall,
                "estimated_three_epoch384step_run_cpu_seconds": 384*macro_cpu,
                "estimate_scope": f"Extrapolation from{timed_count} timed F/B microbatches plus one optimizer update; excludes full-training reload, checkpoint, evaluation and orchestration costs. Not an accuracy estimate."}
            batch_records.append(row)
            print({"microbatch": microbatch, "timed_mean_fb_wall_seconds": mean_wall,
                "estimated_effective_batch16_step_wall_seconds": macro_wall,
                "estimated384step_wall_seconds": 384*macro_wall}, flush=True)
            session.detach()
            session = None
            runner.model.eval()
            del optimizer, trainable, batch
        live_after = _parameter_digest(parameters)
        if live_after != live_before:
            raise RuntimeError("Cost preflight changed original live parameter bytes")
        after = adapter.checkpoint.checkpoint_sha256()
        if after != before:
            raise RuntimeError("Cost preflight changed original checkpoint identity")
        record = {"schema": "weightbridge.phase_iiiq_baseline.training_cost.v1", "status": "PASS",
            "source_checkpoint_sha256": before, "checkpoint_identity_after": after,
            "original_live_parameter_digest_before": live_before, "original_live_parameter_digest_after": live_after,
            "target_modules": len(targets), "batch_measurements": batch_records,
            "native_full_vs_tail": native_equivalence,
            "native_target_token_lengths": target_lengths,
            "maximum_native_target_tokens": max(target_lengths),
            "prompt_representation": "bare" if bare else "native_chat_template",
            "maximum_forward_calls": maximum_forwards, "maximum_backward_calls": maximum_backwards,
            "maximum_wall_seconds": maximum_wall,
            "model_forward_calls": forwards, "model_backward_calls": backwards, "optimizer_steps": optimizer_steps,
            "generation_calls": 0, "efficacy_evaluations": 0, "retained_adapter_candidates": 0,
            "input_sha256": file_hash(out / "timing-inputs.json"),
            "tokenized_inputs_sha256": file_hash(out / "tokenized-inputs.json"),
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "training_implementation_sha256": {name: file_hash(Path(__file__).resolve().parents[2]/name) for name in (
                "weightbridge/adaptation/supervised_lora.py", "weightbridge/adaptation/completion_training.py")},
            "resources": {"wall_seconds": time.perf_counter()-started, "cpu_seconds": time.process_time()-cpu,
                          "peak_rss_bytes": peak_rss_bytes()},
            "scope": "Real pinned-model supervised training cost only on four disclosed timing inputs. No practice, development, selection or hidden efficacy decision; temporary adapters discarded."}
        json_dump_atomic(out / "training-cost.json", record)
        return record
    except BaseException as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
        raise
    finally:
        if session is not None:
            session.detach()
        torch.set_num_threads(previous)
        json_dump_atomic(out / "execution.json", {"status": "COMPLETED" if record is not None else "INTERRUPTED",
            "error": error, "source_sha256": hashlib.sha256(source).hexdigest(),
            "model_forward_calls": forwards, "model_backward_calls": backwards,
            "optimizer_steps": optimizer_steps, "source_checkpoint_sha256": before,
            "resources": {"wall_seconds": time.perf_counter()-started, "cpu_seconds": time.process_time()-cpu,
                          "peak_rss_bytes": peak_rss_bytes()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "timing", "timing-bare"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print((inventory(args.base, args.out) if args.command == "inventory" else
           timing(args.base, args.out, bare=args.command == "timing-bare")), flush=True)
