"""Separately authorized conventional BF16 merge, never a production promotion."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

from weightbridge.adaptation.supervised_lora import SupervisedLoRA, discover_lora_targets
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.provenance import peak_rss_bytes
from .integrity import PARENT, read, verify_sources
from .train import runtime


def run(base, protocol_path, adapter_path, expected_adapter, authorization, out, receipt):
    import torch
    protocol = read(protocol_path)
    verify_sources(protocol["source_sha256"])
    if file_hash(adapter_path) != expected_adapter:
        raise ValueError("Merge factors differ from the locked adapter")
    wall, cpu = time.perf_counter(), time.process_time()
    with runtime(base, protocol) as (source, runner):
        targets, _ = discover_lora_targets(source, runner)
        session = SupervisedLoRA.load(runner.model, targets, adapter_path,
                                      checkpoint_sha256=PARENT, expected_sha256=expected_adapter)
        # A streaming denominator includes every original parameter, not only edited matrices.
        original_square = 0.
        with torch.no_grad():
            for parameter in runner.model.parameters():
                flat = parameter.detach().reshape(-1)
                for first in range(0, flat.numel(), 1048576):
                    original_square += float(torch.sum(flat[first:first+1048576].double().square()))
        result = session.merge_candidate(base, out, authorization, receipt)
        parent_norm = math.sqrt(original_square)
        extended = {"schema": "weightbridge.supervised_baseline.merge_verification.v1",
            "merge_receipt_sha256": file_hash(receipt), "adapter_sha256": expected_adapter,
            "protocol_sha256": file_hash(protocol_path), "parent_parameter_norm": parent_norm,
            "relative_stored_update_norm_full_model": result["stored_update_norm"] / parent_norm,
            "resources": {"wall_seconds": time.perf_counter()-wall, "cpu_seconds": time.process_time()-cpu,
                          "peak_rss_bytes": peak_rss_bytes()}, "parent_checkpoint_sha256": PARENT,
            "candidate_sha256": result["candidate_sha256"], "lineage_authority": False,
            "production_authority": False}
        verify_sources(protocol["source_sha256"])
        json_dump_atomic(Path(receipt).with_suffix(".verification.json"), extended)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "protocol", "adapter", "authorization", "out", "receipt"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-adapter", required=True)
    args = parser.parse_args()
    run(args.base, args.protocol, args.adapter, args.expected_adapter, args.authorization, args.out, args.receipt)
