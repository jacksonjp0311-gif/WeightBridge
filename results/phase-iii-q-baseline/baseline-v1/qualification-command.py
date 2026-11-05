"""Actual host invocation for the locked Phase III-Q finalist.

Run from the repository root with its pinned Python runtime. No training,
promotion, or deployment is performed. Raw protected outputs remain private.
"""
from pathlib import Path
import sys

from experiments.phase_iiiq_baseline import study
from experiments.phase_iiiq_baseline.integrity import PARENT, read, verify_sources
from experiments.phase_iiiq_baseline.run_training import _temporary_sleep_request, launch_owned_tree
from weightbridge.core import file_hash, json_dump_atomic


def run(base, root, hidden):
    root, base, hidden = map(Path, (root, base, hidden))
    protocol = root / "protocol.json"
    verify_sources(read(protocol)["source_sha256"])
    assessment = root / "assessment"
    result = study._replay_screening(assessment, protocol)
    reference = read(assessment / "finalist-reference.json")
    adapter = Path(reference["adapter_directory"]) / "adapter.safetensors"
    digest = reference["adapter_file_sha256"]
    if reference["candidate_id"] != result["winner"] or file_hash(adapter) != digest:
        raise ValueError("Adapter differs from the locked finalist")
    if not read(root / "main-authorization.json")["allow_merge"]:
        raise ValueError("The host did not authorize a merged checkpoint")

    def worker(label, module, args, *, wall, cpu, memory):
        observed = launch_owned_tree([sys.executable, "-m", module, *map(str, args)], root / f"{label}.log",
                                     wall_limit=wall, cpu_limit=cpu, memory_limit=memory)
        json_dump_atomic(root / f"{label}-process.json", {"schema": "weightbridge.q_baseline.host_stage.v1",
            "stage": label, "protocol_sha256": file_hash(protocol), "host_invocation_sha256": file_hash(__file__),
            "limits": {"wall_seconds": wall, "cpu_seconds": cpu, "sum_lifetime_peak_bytes": memory},
            "execution": observed})
        if observed["exit_code"] != 0 or observed["budget_failure"] is not None:
            raise RuntimeError(f"Bounded verification worker failed: {label}")

    def signatures(measurement):
        return [(row["id"], row["generated_token_ids"], row["text"], row["stop_reason"])
                for row in measurement["rows"]]

    gib = 1024**3
    with _temporary_sleep_request() as awake:
        if awake["requested"] and not awake["succeeded"]:
            raise RuntimeError("Windows refused a temporary execution request")
        practice = assessment / "rosters/practice.json"
        replay = root / "adapter-practice-replay.json"
        worker("adapter-replay", "experiments.phase_iiiq_baseline.evaluate",
               ("--base", base, "--protocol", protocol, "--rows", practice, "--out", replay,
                "--expected-checkpoint", PARENT, "--expected-rows", file_hash(practice), "--adapter", adapter,
                "--expected-adapter", digest, "--stage", "practice"), wall=1200, cpu=9600, memory=24*gib)
        repeated = study._validated_worker(replay, practice, protocol, PARENT, digest, "practice")
        prior = read(assessment / "evaluation" / f"practice-{result['winner']}.json")
        exact = signatures(repeated["measurement"]) == signatures(prior["measurement"])
        json_dump_atomic(root / "adapter-replay-check.json", {"exact_predictions": exact, "adapter_sha256": digest,
            "parent_checkpoint_sha256": PARENT, "rows": len(repeated["measurement"]["rows"]),
            "replay_sha256": file_hash(replay), "frozen_prior_sha256": file_hash(
                assessment / "evaluation" / f"practice-{result['winner']}.json")})
        if not exact:
            raise ValueError("The saved adapter did not exactly replay its practice predictions")
        merged, receipt = root / "merged-checkpoint", root / "merge-receipt.json"
        policy = root / "merge-authorization.json"
        worker("merge", "experiments.phase_iiiq_baseline.merge", ("--base", base, "--protocol", protocol,
            "--adapter", adapter, "--expected-adapter", digest, "--authorization", policy,
            "--out", merged, "--receipt", receipt), wall=1200, cpu=9600, memory=24*gib)
        # This sum covers separate sequential workers' lifetime peaks. Each
        # frozen worker retains its own 24-GiB runtime envelope and deadline.
        worker("independent", "experiments.phase_iiiq_baseline.study", ("finish", "--base", base,
            "--protocol", protocol, "--material", root / "material", "--out", assessment,
            "--merged", merged, "--merge-receipt", receipt, "--merge-authorization", policy,
            "--hidden", hidden), wall=6000, cpu=36000, memory=(8*24+1)*gib)
    verify_sources(read(protocol)["source_sha256"])
    json_dump_atomic(root / "qualification-execution.json", {"schema": "weightbridge.q_baseline.qualification_execution.v1",
        "temporary_sleep_request": awake, "independent_result_sha256": file_hash(assessment / "independent.json"),
        "lineage_authority": False, "production_authority": False})
    print(read(assessment / "independent.json")["reason"], flush=True)


if __name__ == "__main__":
    run(*sys.argv[1:])
