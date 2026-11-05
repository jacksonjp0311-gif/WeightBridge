"""Read-only identity audit; historical private evaluation answers are not opened."""
import argparse
import time
from pathlib import Path

from experiments.phase_iiiq_baseline.integrity import PARENT, ROOT, read, verify_sources
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.provenance import peak_rss_bytes

REFERENCE = "ddf9d6c72eab4cc6cbd7448a5d9c73a429881a26c721ef5726709df7ddf08191"


def audit(base, reference, out):
    started, cpu = time.perf_counter(), time.process_time()
    protocol_path = ROOT / "experiments/phase_iiiq_baseline/protocol.json"
    final_path = ROOT / "results/phase-iii-q-baseline/baseline-v1/final-verification/final-verification.json"
    protocol, final = read(protocol_path), read(final_path)
    verify_sources(protocol["source_sha256"])
    original = load_checkpoint(base).checkpoint_sha256()
    merged = load_checkpoint(reference).checkpoint_sha256()
    checks = {
        "original_parent_identity": original == PARENT,
        "historical_merged_identity": merged == REFERENCE,
        "historical_final_receipt_identity": file_hash(final_path) == "735807c9f56c560660004fa60bd5c110fb84afb8354544c88c646b54e56c6cdf",
        "historical_protocol_identity": file_hash(protocol_path) == final["protocol_sha256"],
        "historical_result": final["passed"] and final["disposition"] == "VERIFIED_BASELINE_PASS",
        "historical_no_promotion": not final["lineage_accepted"] and not final["production_authority"],
        "historical_reconstruction": final["exact_reconstruction"]["passed"] and final["exact_reconstruction"]["checkpoint_sha256"] == merged,
        "historical_replay": all(row["passed"] and row["reconstructed"]["exact_native_generation"] and row["rollback"]["exact_native_generation"] for row in final["replays"].values()),
    }
    record = {"schema": "weightbridge.phase_iiir.reference_audit.v1", "passed": all(checks.values()),
        "checks": checks, "checkpoint_sha256": original, "historical_merged_sha256": merged,
        "protocol_sha256": file_hash(protocol_path), "final_receipt_sha256": file_hash(final_path),
        "frozen_source_count": len(protocol["source_sha256"]), "model_inference_calls": 0,
        "scope": "Read-only checkpoint bytes, historical receipt consistency and frozen source replay. No fresh historical behavioral scoring; prior independent outcomes remain historical evidence.",
        "resources": {"wall_seconds": time.perf_counter()-started, "cpu_seconds": time.process_time()-cpu, "peak_rss_bytes": peak_rss_bytes()}}
    json_dump_atomic(out, record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "reference", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    print(audit(args.base, args.reference, args.out))
