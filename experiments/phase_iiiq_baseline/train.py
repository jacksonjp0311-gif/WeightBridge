"""Bounded standard AdamW SFT. This worker receives training-only material."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import math
from pathlib import Path
import random
import time

from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.provenance import peak_rss_bytes
from weightbridge.adaptation.training_policy import TrainingAuthorization
from .integrity import PARENT, read, verify_sources


@contextmanager
def runtime(base, protocol):
    import torch
    from weightbridge.adaptation.model_adapter import select_hf_adapter
    adapter = select_hf_adapter(base)
    if adapter.checkpoint.checkpoint_sha256() != PARENT:
        raise ValueError("Training parent checkpoint is not the original pinned model")
    previous = torch.get_num_threads()
    torch.set_num_threads(protocol["inference"]["cpu_threads"])
    runner = adapter.load_runner(device="cpu", dtype="auto",
        max_rss_bytes=protocol["resources"]["max_memory_bytes"])
    try:
        runner.model.eval()
        yield adapter, runner
    finally:
        if adapter.checkpoint.checkpoint_sha256() != PARENT:
            raise RuntimeError("Original checkpoint bytes changed")
        torch.set_num_threads(previous)


def data_order(count, seed, epochs):
    result = []
    for epoch in range(epochs):
        indices = list(range(count))
        random.Random(f"{seed}/epoch-{epoch}").shuffle(indices)
        result.extend(indices)
    return result


def learning_rate_factor(step, total, warmup):
    return (step + 1) / warmup if warmup and step < warmup else (total - step) / max(1, total - warmup)


def run(base, protocol_path, authorization_path, bundle_path, out, *, run_index, smoke=False, smoke_gate=None):
    import torch
    from weightbridge.adaptation.supervised_lora import SupervisedLoRA, discover_lora_targets
    from weightbridge.adaptation.completion_training import prepare_completion, collate_completions, completion_loss
    from weightbridge.adaptation.behavior_endpoint import endpoint_runner
    from .measurement import endpoint

    protocol, bundle = read(protocol_path), read(bundle_path)
    verify_sources(protocol["source_sha256"])
    auth = TrainingAuthorization.from_dict(read(authorization_path))
    if file_hash(bundle_path) != auth.training_bundle_sha256:
        raise ValueError("Training material is not the exact host-authorized bundle")
    if not smoke:
        if (smoke_gate is None or auth.smoke_gate_sha256 is None
                or file_hash(smoke_gate) != auth.smoke_gate_sha256):
            raise ValueError("Main training requires the exact host-admitted pipeline smoke")
        admitted = read(smoke_gate)
        if not admitted.get("passed") or admitted.get("protocol_sha256") != file_hash(protocol_path):
            raise ValueError("Main training refused after an unqualified or foreign smoke")
    if bundle["protocol_sha256"] != file_hash(protocol_path) or set(bundle) != {
            "schema", "protocol_sha256", "rows", "manifest_sha256", "split_ids"}:
        raise ValueError("Training worker received an unbound or non-training bundle")
    if not set(bundle["split_ids"]) <= ({"smoke"} if smoke else {"train", "replay"}):
        raise ValueError("Training worker may not receive evaluation populations")
    settings = protocol["smoke_training"] if smoke else protocol["training"]
    rate = settings["learning_rates"][run_index]
    seed, steps = settings["seed"], settings["optimizer_steps"]
    checkpoint_steps = tuple(settings["checkpoint_steps"])
    if (not checkpoint_steps or checkpoint_steps[-1] != steps
            or len(checkpoint_steps) > auth.max_development_checkpoints_per_run
            or tuple(sorted(set(checkpoint_steps))) != checkpoint_steps):
        raise ValueError("Checkpoint schedule is not prospectively bounded")
    out = Path(out)
    if out.resolve().is_relative_to(Path(base).resolve()):
        raise ValueError("Training artifacts must stay outside the immutable original directory")
    out.mkdir(parents=True, exist_ok=False)
    wall, cpu = time.perf_counter(), time.process_time()
    torch.manual_seed(seed)
    session = None
    failed = False
    counts = {"optimizer_steps": 0, "forward_calls": 0, "backward_calls": 0,
              "answer_tokens": 0, "input_tokens": 0, "padded_input_token_positions": 0,
              "examples_seen": 0}
    identity = {"protocol_sha256": file_hash(protocol_path), "authorization_sha256": file_hash(authorization_path),
                "bundle_sha256": file_hash(bundle_path), "manifest_sha256": bundle["manifest_sha256"],
                "parent_sha256": PARENT, "run_index": run_index, "smoke": smoke}
    json_dump_atomic(out / "started.json", {**identity, "schema": "weightbridge.supervised_run.v1",
                     "utc": datetime.now(timezone.utc).isoformat()})
    try:
        with runtime(base, protocol) as (adapter, runner):
            targets, inventory = discover_lora_targets(adapter, runner)
            session = SupervisedLoRA(runner.model, targets, checkpoint_sha256=PARENT,
                rank=auth.rank, alpha=auth.alpha, dropout=auth.dropout, seed=seed).attach()
            runner.model.train()
            parameters = tuple(session.parameters())
            trainable = sum(p.numel() for p in parameters)
            auth.validate_plan(parent_sha256=PARENT, module_names=tuple(t.module_name for t in targets),
                trainable_parameters=trainable, rank=session.rank, alpha=session.alpha,
                dropout=session.dropout, learning_rate=rate, steps=steps, run_index=run_index)
            if any(p.requires_grad for p in runner.model.parameters()) or not all(p.requires_grad for p in parameters):
                raise RuntimeError("Unexpected base or adapter gradient flags")
            optimizer = torch.optim.AdamW(parameters, lr=rate, betas=tuple(settings["betas"]),
                eps=settings["epsilon"], weight_decay=settings["weight_decay"])
            tokenizer = endpoint_runner(runner, endpoint(protocol["inference"])).tokenizer
            rows = [prepare_completion(tokenizer, row["prompt"], row["answer"], leading_space=True,
                    max_input_tokens=protocol["inference"]["max_input_tokens"],
                    max_target_tokens=protocol["inference"]["max_new_tokens"]) for row in bundle["rows"]]
            initial = [p.detach().clone() for p in parameters]
            gradient_steps = {name: 0 for name, _ in session.adapters.named_parameters()}
            batch_size, micro = settings["effective_batch_size"], settings["micro_batch_size"]
            order = data_order(len(rows), seed, settings["epochs"])
            if len(order) < steps * batch_size or batch_size % micro:
                raise ValueError("Frozen example/accumulation schedule is inconsistent")
            warmup = math.ceil(steps * settings["warmup_fraction"])
            losses, checkpoints = [], []
            json_dump_atomic(out / "inventory.json", {**inventory,
                "trainable_parameters": trainable, "total_model_parameters": runner.parameter_count,
                "trainable_fraction": trainable / runner.parameter_count,
                "trainable_tensors": [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype)}
                                     for n, p in session.adapters.named_parameters()],
                "training_examples": len(rows), "unique_answer_tokens": sum(len(r.target_ids) for r in rows),
                "max_target_tokens": max(len(r.target_ids) for r in rows),
                "data_order_sha256": __import__("hashlib").sha256(bytes(str(order), "ascii")).hexdigest()})
            for step in range(steps):
                selected = [rows[index] for index in order[step * batch_size:(step + 1) * batch_size]]
                tokens = sum(len(row.target_ids) for row in selected)
                optimizer.zero_grad(set_to_none=True)
                current_lr = rate * learning_rate_factor(step, steps, warmup)
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
                loss_value = 0.
                step_wall, step_cpu = time.perf_counter(), time.process_time()
                for first in range(0, len(selected), micro):
                    subset = selected[first:first + micro]
                    batch = collate_completions(subset, pad_token_id=runner.tokenizer.pad_token_id
                                               if runner.tokenizer.pad_token_id is not None else runner.tokenizer.eos_token_id,
                                               device=runner.device)
                    loss, work = completion_loss(runner.model, batch, logits_mode=settings["logits_mode"])
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite supervised loss")
                    weight = work["supervised_tokens"] / tokens
                    (loss * weight).backward()
                    loss_value += float(loss.detach()) * weight
                    counts["forward_calls"] += 1
                    counts["backward_calls"] += 1
                    counts["answer_tokens"] += work["supervised_tokens"]
                    counts["input_tokens"] += sum(len(row.input_ids) for row in subset)
                    counts["padded_input_token_positions"] += work["padded_input_token_positions"]
                    counts["examples_seen"] += len(subset)
                    del batch, loss
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                    raise RuntimeError("Missing or nonfinite adapter gradient")
                for name, parameter in session.adapters.named_parameters():
                    gradient_steps[name] += int(bool(torch.count_nonzero(parameter.grad)))
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_clip"]))
                if not math.isfinite(grad_norm):
                    raise RuntimeError("Nonfinite adapter gradient norm")
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in parameters):
                    raise RuntimeError("Nonfinite adapter parameter update")
                counts["optimizer_steps"] += 1
                session.assert_base_unchanged()
                usage = {"steps": counts["optimizer_steps"], "answer_tokens": counts["answer_tokens"],
                         "input_tokens": counts["input_tokens"], "wall_seconds": time.perf_counter() - wall,
                         "cpu_seconds": time.process_time() - cpu, "rss_bytes": peak_rss_bytes()}
                auth.check_usage(**usage)
                record = {"step": step + 1, "loss": loss_value, "learning_rate": current_lr,
                    "gradient_norm_before_clip": grad_norm, "step_wall_seconds": time.perf_counter() - step_wall,
                    "step_cpu_seconds": time.process_time() - step_cpu, "counts": dict(counts), "usage": usage}
                if step == 0 or (step + 1) % settings["norm_measurement_interval"] == 0:
                    # ||BA||_F^2 = tr((B^T B)(A A^T)); this never materializes model-scale deltas.
                    with torch.no_grad():
                        square = sum(float(torch.sum((part.B.double().T @ part.B.double())
                                        * (part.A.double() @ part.A.double().T))) * part.scale**2
                                     for part in session.adapters.values())
                    record["unrounded_effective_update_norm"] = math.sqrt(max(0., square))
                json_dump_atomic(out / "steps" / f"{step+1:04d}.json", record)
                losses.append(record)
                print(f"step={step+1}/{steps} loss={loss_value:.5f} wall={usage['wall_seconds']:.1f}s", flush=True)
                if step + 1 in checkpoint_steps:
                    checkpoint = out / "checkpoints" / f"step-{step+1:04d}"
                    checkpoint.mkdir(parents=True, exist_ok=False)
                    metadata = {**identity, "step": step + 1}
                    saved = session.save(checkpoint / "adapter.safetensors", metadata=metadata)
                    statistics = session.effective_statistics()
                    json_dump_atomic(checkpoint / "receipt.json", {**saved, "step": step + 1,
                        "effective_update": statistics, "training_loss": loss_value, "counts": dict(counts)})
                    # Safe local recovery artifact, bound by a cryptographic receipt; never loaded from untrusted input.
                    torch.save({"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                                "step": step + 1, "counts": dict(counts)}, checkpoint / "optimizer.pt")
                    json_dump_atomic(checkpoint / "recovery.json", {"optimizer_sha256": file_hash(checkpoint / "optimizer.pt"),
                        "adapter_sha256": saved["sha256"], **metadata,
                        "recovery_policy": "Incomplete runs cannot qualify; recovery must retain all prior attempt costs."})
                    checkpoints.append(str(checkpoint))
            changed = [{"name": name, "changed_entries": int(torch.count_nonzero(value.detach() - parent)),
                        "norm": float(torch.linalg.vector_norm((value.detach()-parent).double())),
                        "nonzero_gradient_steps": gradient_steps[name]}
                       for (name, value), parent in zip(session.adapters.named_parameters(), initial, strict=True)]
            session.assert_base_unchanged(hash_check=True)
            verify_sources(protocol["source_sha256"])
            outcome = {"schema": "weightbridge.supervised_run_completed.v1", **identity, "status": "COMPLETED",
                "utc": datetime.now(timezone.utc).isoformat(),
                "counts": counts, "checkpoints": checkpoints, "parameter_changes": changed,
                "all_factors_changed": all(row["changed_entries"] > 0 for row in changed),
                "all_factors_received_nonzero_gradient": all(row["nonzero_gradient_steps"] > 0 for row in changed),
                "first_step_loss": losses[0]["loss"], "last_step_loss": losses[-1]["loss"],
                "base_unchanged": True, "resources": {"wall_seconds": time.perf_counter()-wall,
                    "cpu_seconds": time.process_time()-cpu, "peak_rss_bytes": peak_rss_bytes()}}
            session.detach()
            session = None
        verify_sources(protocol["source_sha256"])
        outcome["resources"] = {"wall_seconds": time.perf_counter()-wall,
            "cpu_seconds": time.process_time()-cpu, "peak_rss_bytes": peak_rss_bytes()}
        auth.check_usage(steps=counts["optimizer_steps"], answer_tokens=counts["answer_tokens"],
            input_tokens=counts["input_tokens"], wall_seconds=outcome["resources"]["wall_seconds"],
            cpu_seconds=outcome["resources"]["cpu_seconds"], rss_bytes=outcome["resources"]["peak_rss_bytes"])
        json_dump_atomic(out / "completed.json", outcome)
        return outcome
    except BaseException as exc:
        failed = True
        json_dump_atomic(out / "failed.json", {**identity, "status": "FAILED", "error": type(exc).__name__,
            "message": str(exc), "counts": counts, "resources": {"wall_seconds": time.perf_counter()-wall,
                "cpu_seconds": time.process_time()-cpu, "peak_rss_bytes": peak_rss_bytes()}})
        raise
    finally:
        if session is not None:
            try:
                session.detach()
            except BaseException as cleanup:
                json_dump_atomic(out / "cleanup-failed.json", {"error": type(cleanup).__name__,
                                 "message": str(cleanup), "prior_run_failed": failed})
                if not failed:
                    raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "protocol", "authorization", "bundle", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-gate", type=Path)
    args = parser.parse_args()
    run(args.base, args.protocol, args.authorization, args.bundle, args.out,
        run_index=args.run_index, smoke=args.smoke, smoke_gate=args.smoke_gate)
