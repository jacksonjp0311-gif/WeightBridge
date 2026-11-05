"""Fresh-process Phase S free generation with explicit original or sequential parent."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from experiments.phase_iiiq_baseline.integrity import read, verify_sources
from experiments.phase_iiiq_baseline import measurement as decimal
from weightbridge.adaptation.model_adapter import select_hf_adapter
from weightbridge.adaptation.supervised_lora import SupervisedLoRA, discover_lora_targets
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.provenance import peak_rss_bytes
from .measurement import measure_tools, summarize_tools
from .verification import SCHEMA, endpoint_settings, validate_authorization


def run(base, protocol_path, rows_path, out, *, expected_checkpoint, expected_rows, task,
        adapter_path=None, expected_adapter=None, stage, authorization):
    if Path(out).exists():
        raise FileExistsError("S evaluation evidence is write-once")
    if Path(out).resolve().is_relative_to(Path(base).resolve()):
        raise ValueError("Evaluation evidence must stay outside checkpoint storage")
    protocol, _ = validate_authorization(protocol_path, authorization, stage=stage, task=task,
        expected_rows=expected_rows, expected_checkpoint=expected_checkpoint, expected_adapter=expected_adapter)
    if file_hash(rows_path) != expected_rows:
        raise ValueError("S roster changed after host authorization")
    # Only after authorization and hash checks may a worker see its scoped rows.
    rows = read(rows_path)
    import torch
    settings = endpoint_settings(protocol, task)
    wall, cpu = time.perf_counter(), time.process_time()
    torch.set_num_threads(settings["cpu_threads"])
    source = select_hf_adapter(base)
    if source.checkpoint.checkpoint_sha256() != expected_checkpoint:
        raise ValueError("S evaluator was given another immutable checkpoint")
    runner = source.load_runner(device="cpu", dtype="auto", max_rss_bytes=protocol["resources"]["max_memory_bytes"])
    runner.model.eval()
    session = None
    try:
        if adapter_path is not None:
            if (expected_checkpoint != protocol["checkpoint_sha256"] or not expected_adapter
                    or file_hash(adapter_path) != expected_adapter):
                raise ValueError("Adapter evaluation requires its exact explicit training parent and artifact")
            targets, _ = discover_lora_targets(source, runner)
            session = SupervisedLoRA.load(runner.model, targets, adapter_path,
                checkpoint_sha256=expected_checkpoint).attach()
        elif expected_adapter is not None:
            raise ValueError("An adapter identity requires the exact adapter artifact")
        if task == "tool":
            measured = measure_tools(runner, rows, settings, protocol["tool_completion"])
            summary = summarize_tools(measured)
        else:
            measured = decimal.measure(runner, rows, settings)
            summary = decimal.summarize(measured)
        if session:
            session.assert_base_unchanged(hash_check=True)
        if source.checkpoint.checkpoint_sha256() != expected_checkpoint:
            raise RuntimeError("Evaluation changed checkpoint storage")
        verify_sources(protocol["source_sha256"])
        result = {"schema": SCHEMA, "task": task, "stage": stage, "checkpoint_sha256": expected_checkpoint,
            "parent_sha256": protocol["checkpoint_sha256"], "adapter_sha256": expected_adapter,
            "representation": "fp32_adapter_on_bf16_parent" if session else "stored_checkpoint",
            "protocol_sha256": file_hash(protocol_path), "roster_sha256": expected_rows,
            "authorization_sha256": file_hash(authorization), "task_manifest_sha256": protocol["task_manifest_sha256"],
            "source_sha256": protocol["source_sha256"], "measurement": measured, "summary": summary,
            "resources": {"wall_seconds": time.perf_counter()-wall, "cpu_seconds": time.process_time()-cpu,
                          "peak_rss_bytes": peak_rss_bytes()}}
        json_dump_atomic(out, result)
        return result
    finally:
        if session:
            session.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "protocol", "rows", "out", "authorization"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("expected-checkpoint", "expected-rows", "stage", "task"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--expected-adapter")
    args = parser.parse_args()
    run(args.base, args.protocol, args.rows, args.out, expected_checkpoint=args.expected_checkpoint,
        expected_rows=args.expected_rows, task=args.task, adapter_path=args.adapter,
        expected_adapter=args.expected_adapter, stage=args.stage, authorization=args.authorization)
