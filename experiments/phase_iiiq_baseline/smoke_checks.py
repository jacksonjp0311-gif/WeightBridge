"""Small supervised pipeline checks; no result here authorizes main-study learning.

Only the dedicated thirty-two-presentation smoke roster is accepted by model
workers. Pure gate checks retain token replay, semantic outcomes and matched
teacher-forced losses as separate requirements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from weightbridge.adaptation.behavior_endpoint import _observation, endpoint_runner
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.provenance import peak_rss_bytes
from .data import PopulationSpec, validate
from .integrity import PARENT, ROOT, read, verify_sources
from .measurement import endpoint, measure, question, summarize
from .train import runtime


DEFAULT_GATES = {"max_final_to_initial_loss_ratio": .5, "min_correct_rate": .9,
    "min_strict_gain": 1, "min_complete_family_gain": 1, "all_adapter_factors_changed": True,
    "zero_init_parent_identical": True, "fresh_reload_predictions_identical": True,
    "disabled_parent_identical": True}
_IDENTITY = ("id", "item", "prompt", "answer", "wording", "group", "family", "operands",
             "pair_id", "comparison_id", "order", "counterfactual", "family_id", "task_id")


def _checked(measurement):
    contract = endpoint(measurement["endpoint"]["decode"])
    if measurement["endpoint"] != contract.manifest():
        raise ValueError("Smoke endpoint identity differs")
    rows = measurement["rows"]
    if len(rows) != 32 or len({row["id"] for row in rows}) != 32:
        raise ValueError("Smoke requires exactly thirty-two distinct presentations")
    for row in rows:
        if not row["id"].startswith("baseline-decimal/smoke/"):
            raise ValueError("Smoke checks cannot read another study partition")
        observed = _observation(row, question(row), contract)
        if any(row[key] != value for key, value in observed.items()):
            raise ValueError("Recorded smoke semantic outcome differs from its actual response")
    return rows


def exact_predictions(first, second):
    """Require identical prompts, native generated tokens and complete decoding."""
    before, after = _checked(first), _checked(second)
    if first["endpoint"] != second["endpoint"] or any(
            [row.get(key) for key in _IDENTITY] != [other.get(key) for key in _IDENTITY]
            for row, other in zip(before, after, strict=True)):
        raise ValueError("Smoke prediction comparison changed semantic or prompt identities")
    return all((row["generated_token_ids"], row["text"], row["stop_reason"])
               == (other["generated_token_ids"], other["text"], other["stop_reason"])
               for row, other in zip(before, after, strict=True))


def validate_smoke(parent, trained, reloaded, disabled, training_receipt, *, gates=None):
    """Classify the frozen overfit smoke; never a fresh-generalization result."""
    settings = dict(DEFAULT_GATES)
    if gates is not None:
        if set(gates) != set(settings):
            raise ValueError("Smoke gate keys differ from the frozen pipeline requirements")
        settings.update(gates)
    for measured in (parent, trained, reloaded, disabled):
        _checked(measured)
    # This also verifies all underlying identities before comparing summary sets.
    reload_exact = exact_predictions(trained, reloaded)
    disabled_exact = exact_predictions(parent, disabled)
    zero = training_receipt.get("zero_init_measurement")
    zero_exact = exact_predictions(parent, zero) if zero is not None else False
    old, post = summarize(parent), summarize(trained)
    if [tuple(row.get(key) if not isinstance(row.get(key), list) else tuple(row[key]) for key in _IDENTITY)
            for row in parent["rows"]] != [tuple(row.get(key) if not isinstance(row.get(key), list) else tuple(row[key]) for key in _IDENTITY)
            for row in trained["rows"]]:
        raise ValueError("Smoke training and parent observations refer to different questions")
    before, after = training_receipt.get("matched_loss_before"), training_receipt.get("matched_loss_after")
    valid_loss = bool(before and after)
    ratio = None
    if valid_loss:
        for field in ("row_ids", "tokenization_sha256", "supervised_tokens", "prompt_representation"):
            if before.get(field) != after.get(field):
                raise ValueError("Smoke loss reduction must use the same complete tokenized population")
        initial, final = before.get("mean_token_nll"), after.get("mean_token_nll")
        valid_loss = (isinstance(initial, (int, float)) and isinstance(final, (int, float))
            and math.isfinite(initial) and math.isfinite(final) and initial > 0 and final >= 0
            and before.get("row_ids") == [row["id"] for row in parent["rows"]]
            and before.get("prompt_representation") == "bare"
            and before.get("forward_calls") == after.get("forward_calls") == 8)
        if valid_loss:
            ratio = final/initial
    strict_gain = post["strict_comparisons"]-old["strict_comparisons"]
    family_gain = post["complete_families"]-old["complete_families"]
    correct_completed = sum(row["correct"] and row["valid"] and row["completed"] for row in trained["rows"])
    changes = training_receipt.get("parameter_changes", ())
    changed = bool(changes) and all(type(row.get("changed_entries")) is int and row["changed_entries"] > 0
        and isinstance(row.get("norm"), (int, float)) and math.isfinite(row["norm"]) and row["norm"] > 0 for row in changes)
    checks = {
        "training_completed": training_receipt.get("status") == "COMPLETED",
        "base_unchanged": training_receipt.get("base_unchanged") is True,
        "matched_loss_finite": bool(valid_loss),
        "matched_loss_reduced": bool(ratio is not None and ratio <= settings["max_final_to_initial_loss_ratio"]),
        "all_adapter_factors_changed": training_receipt.get("all_factors_changed") is True and changed,
        "strict_comparison_gain": strict_gain >= settings["min_strict_gain"],
        "complete_family_gain": family_gain >= settings["min_complete_family_gain"],
        "completed_correct_rate": correct_completed/32 >= settings["min_correct_rate"],
        "zero_init_parent_identical": zero_exact, "fresh_reload_predictions_identical": reload_exact,
        "disabled_parent_identical": disabled_exact,
    }
    return {"schema": "weightbridge.supervised_baseline.smoke_gate.v1", "passed": all(checks.values()),
        "disposition": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "failed_requirements": [key for key, value in checks.items() if not value], "gates": settings,
        "matched_loss_ratio": ratio, "strict_comparison_gain": strict_gain,
        "complete_family_gain": family_gain, "correct_completed_presentations": correct_completed,
        "total_presentations": 32,
        "scope": "Disposable training-set pipeline smoke only; no independent improvement, lineage or deployment authority."}


def matched_completion_loss(runner, rows, settings):
    """Token-weighted causal loss on the same complete smoke population, eight F."""
    import torch
    from weightbridge.adaptation.completion_training import prepare_completion, collate_completions, completion_loss

    validate(rows, "smoke", PopulationSpec(target_families=4))
    tokenizer = endpoint_runner(runner, endpoint(settings)).tokenizer
    prepared = [prepare_completion(tokenizer, row["prompt"], row["answer"], leading_space=True,
        max_input_tokens=settings["max_input_tokens"], max_target_tokens=settings["max_new_tokens"]) for row in rows]
    identities = [{"id": row["id"], "prefix_ids": list(value.prefix_ids), "target_ids": list(value.target_ids)}
                  for row, value in zip(rows, prepared, strict=True)]
    numerator = 0.
    tokens = forwards = 0
    wall, cpu = time.perf_counter(), time.process_time()
    runner.model.eval()
    with torch.inference_mode():
        for first in range(0, 32, 4):
            batch = collate_completions(prepared[first:first+4], pad_token_id=tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, device=runner.device)
            loss, work = completion_loss(runner.model, batch, logits_mode="tail")
            count = work["supervised_tokens"]
            numerator += float(loss)*count
            tokens += count
            forwards += 1
    return {"mean_token_nll": numerator/tokens, "supervised_tokens": tokens, "forward_calls": forwards,
        "backward_calls": 0, "row_ids": [row["id"] for row in rows], "prompt_representation": "bare",
        "tokenization_sha256": hashlib.sha256(json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "resources": {"wall_seconds": time.perf_counter()-wall, "cpu_seconds": time.process_time()-cpu,
                      "peak_rss_bytes": peak_rss_bytes()}}


def _inputs(protocol_path, rows_path, out, *, base):
    config, rows = read(protocol_path), read(rows_path)
    if not config.get("source_sha256"):
        raise ValueError("Smoke execution requires a frozen source manifest")
    verify_sources(config["source_sha256"])
    validate(rows, "smoke", PopulationSpec(target_families=4))
    if len(rows) != 32 or config.get("checkpoint_sha256") != PARENT:
        raise ValueError("Smoke source or population differs")
    out = Path(out)
    if out.resolve().is_relative_to(Path(base).resolve()):
        raise ValueError("Smoke output must never be inside the original checkpoint")
    if out.resolve().is_relative_to(ROOT):
        raise ValueError("Smoke model execution requires an isolated host output directory")
    out.mkdir(parents=True, exist_ok=False)
    return config, rows, out


def zero_init_check(base, protocol_path, rows_path, out):
    from weightbridge.adaptation.supervised_lora import SupervisedLoRA, discover_lora_targets

    config, rows, out = _inputs(protocol_path, rows_path, out, base=base)
    wall, cpu = time.perf_counter(), time.process_time()
    session, result, error = None, None, None
    try:
        with runtime(base, config) as (adapter, runner):
            parent = measure(runner, rows, config["inference"])
            json_dump_atomic(out/"parent-measurement.json", parent)
            matched = matched_completion_loss(runner, rows, config["inference"])
            json_dump_atomic(out/"matched-loss-before.json", matched)
            targets, _ = discover_lora_targets(adapter, runner)
            settings = config["adapter"]
            session = SupervisedLoRA(runner.model, targets, checkpoint_sha256=PARENT,
                rank=settings["rank"], alpha=settings["alpha"], dropout=settings["dropout"],
                seed=config["smoke_training"]["seed"]).attach()
            zero = measure(runner, rows, config["inference"])
            json_dump_atomic(out/"zero-init-measurement.json", zero)
            same = exact_predictions(parent, zero)
            session.assert_base_unchanged(hash_check=True)
            session.detach()
            session = None
            verify_sources(config["source_sha256"])
            result = {"schema": "weightbridge.supervised_baseline.zero_init.v1", "passed": same,
                "parent_measurement": parent, "zero_init_measurement": zero, "matched_loss_before": matched,
                "checkpoint_sha256": PARENT, "protocol_sha256": file_hash(protocol_path),
                "roster_sha256": file_hash(rows_path), "zero_init_parent_identical": same,
                "explicit_forward_calls": matched["forward_calls"], "backward_calls": 0,
                "batched_decode_steps": parent["resources"]["batched_decode_steps"]+zero["resources"]["batched_decode_steps"],
                "generated_response_records": 64, "base_unchanged": True,
                "resources": {"wall_seconds": time.perf_counter()-wall, "cpu_seconds": time.process_time()-cpu,
                              "peak_rss_bytes": peak_rss_bytes()}}
            json_dump_atomic(out/"zero-init-check.json", result)
            return result
    except BaseException as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
        raise
    finally:
        if session:
            session.detach()
        json_dump_atomic(out/"execution.json", {"status": "COMPLETED" if result else "FAILED", "error": error,
            "resources": {"wall_seconds": time.perf_counter()-wall, "cpu_seconds": time.process_time()-cpu,
                          "peak_rss_bytes": peak_rss_bytes()}})


def fresh_adapter_check(base, protocol_path, rows_path, adapter_path, expected_sha, out, *, parent_measurement=None):
    from weightbridge.adaptation.supervised_lora import SupervisedLoRA, discover_lora_targets

    if Path(out).resolve().is_relative_to(Path(adapter_path).resolve().parent):
        raise ValueError("Smoke output must not modify the saved adapter directory")
    config, rows, out = _inputs(protocol_path, rows_path, out, base=base)
    if file_hash(adapter_path) != expected_sha:
        raise ValueError("Fresh smoke adapter differs from its saved hash")
    from safetensors import safe_open
    with safe_open(str(adapter_path), framework="pt", device="cpu") as archive:
        metadata = json.loads(archive.metadata()["host_metadata"])
    if (metadata.get("protocol_sha256") != file_hash(protocol_path) or metadata.get("smoke") is not True
            or metadata.get("parent_sha256") != PARENT):
        raise ValueError("Fresh smoke adapter belongs to another study or training stage")
    wall, cpu = time.perf_counter(), time.process_time()
    session, result, error = None, None, None
    try:
        with runtime(base, config) as (source, runner):
            targets, _ = discover_lora_targets(source, runner)
            session = SupervisedLoRA.load(runner.model, targets, adapter_path, checkpoint_sha256=PARENT,
                                         expected_metadata=metadata).attach()
            trained = measure(runner, rows, config["inference"])
            json_dump_atomic(out/"trained-measurement.json", trained)
            matched = matched_completion_loss(runner, rows, config["inference"])
            json_dump_atomic(out/"matched-loss-after.json", matched)
            with session.disabled():
                disabled = measure(runner, rows, config["inference"])
            json_dump_atomic(out/"disabled-measurement.json", disabled)
            prior = read(parent_measurement) if isinstance(parent_measurement, (Path, str)) else parent_measurement
            restored = exact_predictions(prior, disabled) if prior is not None else None
            session.assert_base_unchanged(hash_check=True)
            session.detach()
            session = None
            verify_sources(config["source_sha256"])
            result = {"schema": "weightbridge.supervised_baseline.fresh_smoke.v1", "checkpoint_sha256": PARENT,
                "adapter_sha256": expected_sha, "protocol_sha256": file_hash(protocol_path),
                "roster_sha256": file_hash(rows_path), "trained_measurement": trained,
                "disabled_measurement": disabled, "matched_loss_after": matched,
                "disabled_parent_identical": restored, "base_unchanged": True,
                "explicit_forward_calls": matched["forward_calls"], "backward_calls": 0,
                "batched_decode_steps": trained["resources"]["batched_decode_steps"]+disabled["resources"]["batched_decode_steps"],
                "generated_response_records": 64,
                "resources": {"wall_seconds": time.perf_counter()-wall, "cpu_seconds": time.process_time()-cpu,
                              "peak_rss_bytes": peak_rss_bytes()}}
            json_dump_atomic(out/"fresh-adapter-check.json", result)
            return result
    except BaseException as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
        raise
    finally:
        if session:
            session.detach()
        json_dump_atomic(out/"execution.json", {"status": "COMPLETED" if result else "FAILED", "error": error,
            "resources": {"wall_seconds": time.perf_counter()-wall, "cpu_seconds": time.process_time()-cpu,
                          "peak_rss_bytes": peak_rss_bytes()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("zero", "fresh"))
    for name in ("base", "protocol", "rows", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--expected-sha")
    parser.add_argument("--parent-measurement", type=Path)
    args = parser.parse_args()
    if args.command == "zero":
        print(zero_init_check(args.base, args.protocol, args.rows, args.out)["zero_init_parent_identical"], flush=True)
    else:
        if not args.adapter or not args.expected_sha:
            parser.error("fresh requires the immutable saved adapter and hash")
        print(fresh_adapter_check(args.base, args.protocol, args.rows, args.adapter, args.expected_sha, args.out,
            parent_measurement=args.parent_measurement)["disabled_parent_identical"], flush=True)
