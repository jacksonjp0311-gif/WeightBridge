"""Write a process resource receipt even when paired fitting is interrupted."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from experiments.phase_iiih.run_decimal_endpoint import _resources
from experiments.phase_iiin.run_paired import run
from weightbridge.core import file_hash, json_dump_atomic


def execute(base, material, out, receipt):
    if receipt.exists() or out.exists():
        raise FileExistsError("A restart requires fresh evidence paths; old attempts stay immutable")
    wall, cpu = time.perf_counter(), time.process_time()
    result, error = None, None
    try:
        result = run(base, material, out)
        return result
    except BaseException as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
        raise
    finally:
        json_dump_atomic(receipt, {"schema": "weightbridge.phase_iiin.execution.v1",
            "status": "COMPLETED" if result is not None else "INTERRUPTED", "error": error,
            "source_sha256": file_hash(__file__), "runner_source_sha256": file_hash(Path(__file__).with_name("run_paired.py")),
            "fit_report_sha256": file_hash(out / "fit-report.json") if result is not None else None,
            "resources": _resources(wall, cpu),
            "restart_policy": "Fresh trial path; never overwrite partial evidence or reopen evaluation data."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("base", "material", "out", "receipt"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.base, args.material, args.out, args.receipt)
    print(result["detail"], flush=True)
