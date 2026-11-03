"""Paired comparison for two frozen outputs from run_hf_benchmark.py."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weightbridge.evaluation.metrics import paired_bootstrap_interval


def compare(baseline_path: Path, candidate_path: Path, output: Path, *, seed: int = 17) -> dict:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    cand = json.loads(candidate_path.read_text(encoding="utf-8"))
    for key in ("dataset_sha256", "split", "samples_per_problem", "sampling"):
        if base.get(key) != cand.get(key):
            raise ValueError(f"baseline and candidate differ in {key}")
    if base.get("model_sha256") == cand.get("model_sha256"):
        raise ValueError("baseline and candidate checkpoint hashes are identical")
    key = lambda item: (item.get("problem_id"), item.get("sample_index"), item.get("seed"))
    base_items = {key(item): item for item in base["generations"]}
    cand_items = {key(item): item for item in cand["generations"]}
    if set(base_items) != set(cand_items):
        raise ValueError("baseline and candidate have different generation identities")
    pairs = [(base_items[item_id], cand_items[item_id]) for item_id in sorted(base_items)]
    completed = [(before, after) for before, after in pairs
                 if before.get("status") == "completed" and after.get("status") == "completed"]
    if not completed:
        raise ValueError("No paired completed generations to compare")
    base_correct = [float(bool(before["correct"])) for before, _ in completed]
    cand_correct = [float(bool(after["correct"])) for _, after in completed]
    base_completion = [float(bool(before.get("answer_completed", before.get("boxed_answer_completed"))))
                       for before, _ in completed]
    cand_completion = [float(bool(after.get("answer_completed", after.get("boxed_answer_completed"))))
                       for _, after in completed]
    base_token_limit = [float(bool(before["token_limit"])) for before, _ in completed]
    cand_token_limit = [float(bool(after["token_limit"])) for _, after in completed]
    regressions = sum(before["correct"] and not after["correct"] for before, after in completed)
    rescues = sum(not before["correct"] and after["correct"] for before, after in completed)
    accuracy_ci = paired_bootstrap_interval(base_correct, cand_correct, seed=seed)
    completion_ci = paired_bootstrap_interval(base_completion, cand_completion, seed=seed)
    limit_ci = paired_bootstrap_interval(base_token_limit, cand_token_limit, seed=seed)
    disposition = "PASS" if accuracy_ci["ci_low"] > 0 else ("FAIL" if accuracy_ci["ci_high"] < 0 else "INCONCLUSIVE")
    result = {
        "schema": "weightbridge.hf_benchmark_comparison.v1",
        "comparison_id": f"{base['run_id']}__vs__{cand['run_id']}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_run": str(baseline_path), "candidate_run": str(candidate_path),
        "dataset_sha256": base["dataset_sha256"], "split": base["split"],
        "paired_generations": len(completed), "attempted_generations": len(pairs),
        "paired_accuracy": {"baseline": sum(base_correct) / len(base_correct),
                            "candidate": sum(cand_correct) / len(cand_correct),
                            "difference_ci": accuracy_ci,
                            "rescues": int(rescues), "regressions": int(regressions),
                            "regression_rate": regressions / len(completed)},
        "paired_answer_completion": {"baseline": sum(base_completion) / len(base_completion),
                                      "candidate": sum(cand_completion) / len(cand_completion),
                                      "difference_ci": completion_ci},
        "paired_token_limit_rate": {"baseline": sum(base_token_limit) / len(base_token_limit),
                                    "candidate": sum(cand_token_limit) / len(cand_token_limit),
                                    "difference_ci": limit_ci},
        "disposition": disposition,
        "interpretation_limit": "A single frozen suite comparison supports only this task/sampling setup; it is not a general capability claim.",
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = compare(Path(args.baseline), Path(args.candidate), Path(args.out), seed=args.seed)
    print(json.dumps({"comparison_id": result["comparison_id"], "disposition": result["disposition"],
                      "paired_accuracy": result["paired_accuracy"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
