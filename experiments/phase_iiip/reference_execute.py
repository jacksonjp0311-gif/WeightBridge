"""Whole host-process receipt for an isolated, already authorized reference comparison."""
import argparse
from pathlib import Path
import time
from experiments.phase_iiip.reference_study import assess
from experiments.phase_iiih.run_decimal_endpoint import _resources
from weightbridge.core import file_hash, json_dump_atomic


def execute(base, candidate, roster, reconstruction, out, receipt):
    if receipt.exists() or out.exists():
        raise FileExistsError("A reference attempt requires new write-once paths")
    wall, cpu, result, error = time.perf_counter(), time.process_time(), None, None
    try:
        result = assess(base, candidate, roster, reconstruction, out)
        return result
    except BaseException as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
        raise
    finally:
        json_dump_atomic(receipt, {"schema": "weightbridge.phase_iiip.reference_execution.v1", "track": "B",
            "status": "COMPLETED" if result is not None else "INTERRUPTED", "error": error,
            "source_sha256": file_hash(__file__), "resources": _resources(wall, cpu),
            "comparison_sha256": file_hash(out / "comparison.json") if result is not None else None,
            "scope": "Host process CPU only; successful worker CPU must be added once. Host wall includes workers. Failed worker/partial work remains separately recorded, never fabricated as zero."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "candidate", "roster", "reconstruction", "out", "receipt"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(execute(args.base, args.candidate, args.roster, args.reconstruction, args.out, args.receipt)["paired"], flush=True)
