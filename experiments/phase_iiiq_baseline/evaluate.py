"""Fresh-process free-generation evaluator. No fitting code reads these rosters."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from weightbridge.adaptation.model_adapter import select_hf_adapter
from weightbridge.adaptation.supervised_lora import SupervisedLoRA, discover_lora_targets
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.provenance import peak_rss_bytes
from .integrity import PARENT, read, verify_sources
from .measurement import measure, summarize


def run(base, protocol_path, rows_path, out, *, expected_checkpoint=PARENT, expected_rows,
        adapter_path=None, expected_adapter=None, stage="development", authorization=None):
    import torch
    protocol = read(protocol_path)
    if Path(out).resolve().is_relative_to(Path(base).resolve()):
        raise ValueError("Evaluation evidence must stay outside checkpoint storage")
    verify_sources(protocol["source_sha256"])
    if file_hash(rows_path) != expected_rows:
        raise ValueError("Evaluator roster differs from the frozen host identity")
    if stage in ("selection", "hidden", "preservation-validation"):
        if authorization is None:
            raise ValueError("Independent populations require a host stage authorization")
        auth = read(authorization)
        if (auth.get("stage") != stage or auth.get("protocol_sha256") != file_hash(protocol_path)
                or auth.get("roster_sha256") != expected_rows or adapter_path is not None
                or (auth.get("representation_sha256") != expected_checkpoint
                    and not (expected_checkpoint == PARENT and auth.get("parent_sha256") == PARENT))):
            raise ValueError("Host authorization is bound to another stage/model/protocol")
    wall, cpu = time.perf_counter(), time.process_time()
    torch.set_num_threads(protocol["inference"]["cpu_threads"])
    source = select_hf_adapter(base)
    if source.checkpoint.checkpoint_sha256() != expected_checkpoint:
        raise ValueError("Independent evaluator checkpoint SHA-256 differs")
    runner = source.load_runner(device="cpu", dtype="auto", max_rss_bytes=protocol["resources"]["max_memory_bytes"])
    runner.model.eval()
    session = None
    try:
        if adapter_path:
            if expected_checkpoint != PARENT or not expected_adapter or file_hash(adapter_path) != expected_adapter:
                raise ValueError("Adapter evaluation requires the original base and exact artifact hash")
            targets, _ = discover_lora_targets(source, runner)
            session = SupervisedLoRA.load(runner.model, targets, adapter_path, checkpoint_sha256=PARENT).attach()
        measurement = measure(runner, read(rows_path), protocol["inference"])
        result = {"schema": "weightbridge.supervised_baseline.evaluation.v1", "stage": stage,
            "checkpoint_sha256": expected_checkpoint, "adapter_sha256": expected_adapter,
            "representation": "fp32_adapter_on_bf16_parent" if session else "stored_checkpoint",
            "protocol_sha256": file_hash(protocol_path), "roster_sha256": expected_rows,
            "measurement": measurement, "summary": summarize(measurement),
            "resources": {"wall_seconds": time.perf_counter()-wall, "cpu_seconds": time.process_time()-cpu,
                          "peak_rss_bytes": peak_rss_bytes()}, "source_sha256": protocol["source_sha256"]}
        if session:
            session.assert_base_unchanged(hash_check=True)
        if source.checkpoint.checkpoint_sha256() != expected_checkpoint:
            raise RuntimeError("Evaluation changed checkpoint storage")
        json_dump_atomic(out, result)
        return result
    finally:
        if session:
            session.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "protocol", "rows", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-checkpoint", default=PARENT)
    parser.add_argument("--expected-rows", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--expected-adapter")
    parser.add_argument("--stage", default="development")
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    run(args.base, args.protocol, args.rows, args.out, expected_checkpoint=args.expected_checkpoint,
        expected_rows=args.expected_rows, adapter_path=args.adapter, expected_adapter=args.expected_adapter,
        stage=args.stage, authorization=args.authorization)
