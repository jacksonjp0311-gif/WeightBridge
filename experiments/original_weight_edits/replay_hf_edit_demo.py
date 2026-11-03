"""Rebuild and score an archived Qwen repair with the product evaluator."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.core import json_dump_atomic
from weightbridge.evaluation.hf_edit import evaluate_hf_edit
from weightbridge.patches import apply_safetensors_patch

EXAMPLES = {
    "wording": ("qwen-wording-ablation-v1", "case-12", "multi", 10),
    "ai-pilot": ("qwen-ai-edit-pilot-v1", "case-03", "guided", 3),
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def apply_archived(source, out, folder, name, policy):
    with gzip.open(folder / f"{name}_patch.json.gz", "rt", encoding="utf-8") as stream:
        patch = json.load(stream)
    expected = read(folder / f"{name}_receipt.json")
    receipt = apply_safetensors_patch(source, patch, policy, out)
    for field in ("base_sha256", "candidate_sha256", "patch_sha256", "policy_sha256", "changed_entries"):
        if receipt[field] != expected[field]:
            raise ValueError(f"Rebuilt {name} receipt differs in {field}")


def replay_session(damaged, out, case, policy):
    """Exercise the public CLI with two already disclosed pilot choices."""
    out, damaged = out.resolve(), damaged.resolve()
    session = out / "session"
    practice = out / "practice.json"
    policy_path = out / "policy.json"
    json_dump_atomic(practice, {
        "schema": "weightbridge.hf_edit_suite.v1", "choices": ["A", "B"],
        **read(case / "public/practice.json"),
    })
    json_dump_atomic(policy_path, policy)

    def command(*args, request=None):
        start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "-m", "weightbridge", "hf-edit-session", *map(str, args)],
            input=json.dumps(request) if request is not None else "",
            stdout=subprocess.PIPE, text=True, check=True, timeout=300, cwd=ROOT,
        )
        return json.loads(result.stdout), time.perf_counter() - start

    def call(tool, **arguments):
        return command("call", "--session", session, request={"tool": tool, "arguments": arguments})

    _, init_seconds = command(
        "init", "--session", session, "--base", damaged, "--policy", policy_path,
        "--practice", practice, "--max-candidates", 2, "--dtype", "bfloat16", "--max-input-tokens", 96,
    )
    _, describe_seconds = call("describe")
    attempts = []
    for layer, strength in ((21, 0.15), (23, 0.03)):
        tensor = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
        result, seconds = call("try_direction", tensor=tensor, strength=strength)
        attempts.append({"tensor": tensor, "strength": strength, "elapsed_seconds": seconds, **result})
    json_dump_atomic(out / "session-replay.json", {
        "init_seconds": init_seconds, "describe_seconds": describe_seconds, "attempts": attempts,
    })
    if [attempt["disposition"] for attempt in attempts] != ["FAIL", "PASS"]:
        raise ValueError("Session replay should reject the locality regression and accept the repair")
    final = attempts[-1]
    if final["candidate_sha256"] != read(case / "search/guided_receipt.json")["candidate_sha256"]:
        raise ValueError("Session candidate checkpoint hash differs from the archived repair")
    decision, seconds = call("finalize", candidate_id=final["candidate_id"])
    json_dump_atomic(out / "session-finalization.json", {"elapsed_seconds": seconds, **decision})
    return Path(final["candidate_path"])


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pristine", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--example", choices=EXAMPLES, default="wording")
    parser.add_argument("--session-tools", action="store_true", help="Replay two disclosed AI pilot choices through the local tool API")
    args = parser.parse_args()
    if args.session_tools and args.example != "ai-pilot":
        parser.error("--session-tools requires --example ai-pilot")
    args.out.mkdir()
    evidence_name, case_name, arm, min_gain = EXAMPLES[args.example]
    evidence = ROOT / "results" / evidence_name
    case = evidence / case_name
    policy = read(evidence / "manifest.json")["policy"]
    damaged, repaired = args.out / "damaged", args.out / "repaired"
    apply_archived(args.pristine, damaged, case / "evaluator", "fault", policy)
    if args.session_tools:
        repaired = replay_session(damaged, args.out, case, policy)
    else:
        apply_archived(damaged, repaired, case / "search", arm, policy)
    suite = args.out / "holdout.json"
    holdout = read(case / "evaluator/holdout.json")
    json_dump_atomic(suite, {"schema": "weightbridge.hf_edit_suite.v1", "choices": ["A", "B"], **holdout})
    evaluation_started = time.perf_counter()
    report = evaluate_hf_edit(damaged, repaired, suite, args.out / "report.json",
                              min_target_gain=min_gain, dtype="bfloat16")
    if args.session_tools:
        json_dump_atomic(args.out / "replay-timing.json", {
            "holdout_evaluation_seconds": time.perf_counter() - evaluation_started,
            "total_seconds": time.perf_counter() - started,
        })
    expected = read(case / "evaluation/scores.json")
    for group in ("target", "sentinel"):
        observed = [(row["base_prediction"], row["candidate_prediction"]) for row in report[group]]
        archived = [(a["prediction"], b["prediction"])
                    for a, b in zip(expected["damaged"][group], expected[arm][group], strict=True)]
        if observed != archived:
            raise ValueError(f"Reloaded {group} predictions differ from archived scores")
    print(json.dumps({key: report[key] for key in (
        "disposition", "base_target_correct", "candidate_target_correct", "target_count",
        "sentinel_count", "sentinel_prediction_flips",
    )}, indent=2))
    if report["disposition"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
