"""Capture completed and interrupted Track A costs without replacing evidence."""
import argparse
from pathlib import Path
import time
from experiments.phase_iiih.run_decimal_endpoint import _resources
from experiments.phase_iiip.run_right import run
from weightbridge.core import file_hash, json_dump_atomic


def execute(base, material, out, receipt):
    if out.exists() or receipt.exists():
        raise FileExistsError("An attempt or restart requires fresh paths")
    wall, cpu, result, error = time.perf_counter(), time.process_time(), None, None
    try:
        result = run(base, material, out)
        return result
    except BaseException as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
        raise
    finally:
        json_dump_atomic(receipt, {"schema": "weightbridge.phase_iiip.execution.v1",
            "status": "COMPLETED" if result is not None else "INTERRUPTED", "error": error,
            "source_sha256": file_hash(__file__), "runner_source_sha256": file_hash(Path(__file__).with_name("run_right.py")),
            "fit_report_sha256": file_hash(out / "fit-report.json") if result is not None else None,
            "resources": _resources(wall, cpu), "restart_policy": "Fresh paths; retain prior failures and original sealed boundary."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "material", "out", "receipt"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(execute(args.base, args.material, args.out, args.receipt)["detail"], flush=True)
