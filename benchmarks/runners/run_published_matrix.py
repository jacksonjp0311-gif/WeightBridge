"""Run base and p50c models on the published Exp8 MATH/GSM8K/IFEval matrix.

Defaults reproduce the documented generation settings. Dataset preparation is
separate so each frozen input can be audited and hashed before inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


SAMPLES = {"math500": 4, "gsm8k": 2, "ifeval": 2}


def run_matrix(base: Path, candidate: Path, dataset_dir: Path, output: Path, *,
               ifeval_source: Path, device: str = "auto", dtype: str = "auto",
               cpu_threads: int = 0, timeout_seconds: float = 3600.0,
               max_rss_bytes: int = 8 * 1024**3) -> dict:
    if type(timeout_seconds) not in {int, float} or not 0 < timeout_seconds <= 24 * 3600:
        raise ValueError("timeout_seconds must be in (0, 86400]")
    if type(max_rss_bytes) is not int or not 256 * 1024**2 <= max_rss_bytes <= 128 * 1024**3:
        raise ValueError("max_rss_bytes must be in [256 MiB, 128 GiB]")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    required = [dataset_dir / f"{name}.json" for name in SAMPLES]
    if any(not path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError("prepared dataset file(s) missing: " + ", ".join(missing))
    if not ifeval_source.is_dir():
        raise FileNotFoundError(f"Google instruction_following_eval source path is required: {ifeval_source}")
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    runner = root / "benchmarks" / "runners" / "run_hf_benchmark.py"
    compare = root / "benchmarks" / "runners" / "compare_hf_runs.py"
    run_paths = {}
    try:
        for dataset_name, sample_count in SAMPLES.items():
            dataset_path = dataset_dir / f"{dataset_name}.json"
            grader = "exact" if dataset_name == "ifeval" else "math-verify"
            paths = {}
            for variant, model_path in (("baseline", base), ("published-p50c", candidate)):
                run_dir = output / variant / dataset_name
                command = [sys.executable, str(runner), "--model", str(model_path.resolve()),
                           "--dataset", str(dataset_path.resolve()), "--out", str(run_dir),
                           "--variant", variant, "--split", "holdout", "--n", str(sample_count),
                           "--seed", "0", "--max-new-tokens", "32768", "--max-input-tokens", "8192",
                           "--temperature", "0.6", "--top-p", "0.95", "--top-k", "20",
                           "--presence-penalty", "1.5", "--timeout-seconds", str(timeout_seconds),
                           "--max-rss-bytes", str(max_rss_bytes),
                           "--device", device, "--dtype", dtype, "--cpu-threads", str(cpu_threads),
                           "--grader", grader]
                if dataset_name == "ifeval":
                    command.extend(["--ifeval-source", str(ifeval_source.resolve())])
                completed = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                           timeout=timeout_seconds)
                if completed.returncode != 0:
                    raise RuntimeError(f"{variant}/{dataset_name} failed ({completed.returncode}): "
                                       f"{completed.stderr[-3000:]}")
                paths[variant] = str(run_dir / "run.json")
            comparison_path = output / "comparisons" / f"{dataset_name}.json"
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run([sys.executable, str(compare), "--baseline", paths["baseline"],
                                        "--candidate", paths["published-p50c"], "--out", str(comparison_path)],
                                       cwd=root, capture_output=True, text=True,
                                       timeout=timeout_seconds)
            if completed.returncode != 0:
                raise RuntimeError(f"paired comparison for {dataset_name} failed: {completed.stderr[-3000:]}")
            run_paths[dataset_name] = {**paths, "comparison": str(comparison_path)}
    except Exception as exc:
        failure = {"schema": "weightbridge.published_matrix_failure.v1",
                   "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "exception_type": type(exc).__name__, "error": str(exc)[:4000],
                   "completed_runs": run_paths,
                   "note": "Completed child artifacts remain intact; this failure record does not overwrite them."}
        with (output / "matrix_failure.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(failure, stream, indent=2, sort_keys=True)
            stream.write("\n")
        raise
    result = {"schema": "weightbridge.published_matrix.v1", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "base_model_path": str(base.resolve()), "candidate_model_path": str(candidate.resolve()),
              "dataset_manifest_path": str((dataset_dir / "manifest.json").resolve()),
              "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                           "presence_penalty": 1.5, "thinking_enabled": True,
                           "max_new_tokens": 32768, "seed_base": 0},
              "samples_per_problem": SAMPLES, "runs": run_paths,
              "resource_budgets": {"timeout_seconds_per_subrun": timeout_seconds,
                                   "max_rss_bytes": max_rss_bytes},
              "status": "COMPLETED_FULL_CONFIG"}
    with (output / "matrix_summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path, help="Pinned original HF model directory")
    parser.add_argument("--candidate", required=True, type=Path, help="Reconstructed p50c HF model directory")
    parser.add_argument("--datasets", required=True, type=Path, help="Directory created by prepare_exp8_datasets.py")
    parser.add_argument("--ifeval-source", required=True, type=Path,
                        help="Google google-research/instruction_following_eval source directory")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--max-rss-bytes", type=int, default=8 * 1024**3)
    args = parser.parse_args()
    result = run_matrix(args.base, args.candidate, args.datasets, args.out,
                        ifeval_source=args.ifeval_source, device=args.device, dtype=args.dtype,
                        cpu_threads=args.cpu_threads, timeout_seconds=args.timeout_seconds,
                        max_rss_bytes=args.max_rss_bytes)
    print(json.dumps({"status": result["status"], "runs": result["runs"]}, indent=2))


if __name__ == "__main__":
    main()
