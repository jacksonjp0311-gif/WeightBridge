"""Read-only public archive replay; no model, training or private data access.

Run from a clean checkout with the existing Python runtime:
python results/phase-iii-r-replication/verify_public.py --public-export PATH --out NEW_RECEIPT

This verifies public screening scores and identities. Protected confirmation
can only be checked as published aggregate host receipts, not independently
rescored from withheld examples. Archive validity is not a model-learning PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.phase_iiiq_baseline import study as qstudy
from experiments.phase_iiiq_baseline.integrity import PARENT, verify_sources
from experiments.phase_iiir_replication import study as rstudy

ARMS = ("rank8", "rank4")
STAGES = ("preservation-validation", "selection", "hidden")
ROW_KEYS = {"rows", "measurement", "prompt", "operands", "generated_token_ids", "text",
    "strict_comparison_ids", "complete_family_ids", "newly_complete_families",
    "repeatable_comparisons_corrected", "prohibited_regressions", "lost_valid_presentations",
    "lost_required_controls", "correctness_disagreements", "validity_disagreements",
    "complete_continuation_disagreements"}


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def inside(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Public commitment escaped the export")
    return path


def no_rows(value):
    if isinstance(value, dict):
        if ROW_KEYS & value.keys():
            raise ValueError("Protected projection contains per-row evidence")
        for child in value.values():
            no_rows(child)
    elif isinstance(value, list):
        for child in value:
            no_rows(child)


def verify_files(root, manifest):
    if manifest.get("schema") != "weightbridge.phase_iiir.public_export.v1":
        raise ValueError("Not a versioned public R evidence export")
    committed = {**{name: row["public_sha256"] for name, row in manifest["artifacts"].items()},
                 **manifest["derived_artifact_sha256"]}
    if set(manifest["derived_artifact_sha256"]) != {"compute-accounting.json", "arm-summary.json"}:
        raise ValueError("Derived outcome/cost summaries require their own commitments")
    files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if files != set(committed) | {"export-manifest.json"}:
        raise ValueError("Uncommitted or missing export file")
    protected_digests = set()
    global_manifest = read(root/"material/manifest.json")
    for name, row in global_manifest["splits"].items():
        if name.startswith("bank-"):
            protected_digests.add(row["sha256"])
    for name, digest in committed.items():
        path = inside(root, name)
        if sha(path) != digest:
            raise ValueError(f"Changed public artifact: {name}")
        if (path.suffix.lower() in {".safetensors", ".pt", ".pth", ".bin", ".ckpt", ".gguf"}
                or any(part in {"private-evaluation", "private-independent-rosters", "merged-checkpoint"} for part in path.parts)
                or path.name in {"selection.json", "hidden.json", "preservation-validation.json"}
                or digest in protected_digests):
            raise ValueError("Protected roster or model file entered public evidence")
        record = manifest["artifacts"].get(name)
        if record and record["mode"] == "protected_aggregate_projection":
            projected = read(path)
            if projected["private_receipt_sha256"] != record["source_sha256"]:
                raise ValueError("Protected aggregate lost its host evidence commitment")
            no_rows(projected)
        elif record and record["mode"] == "exact_public_scope" and record["source_sha256"] != digest:
            raise ValueError("Exact public copy differs from its declared source")
    return len(committed)


def replay_screening(root, arm):
    folder, protocol_path = root/arm/"assessment", root/arm/"protocol.json"
    if not (folder/"assessment.json").is_file():
        return {"status": "UNMEASURED", "winner": None, "candidates": None, "worker_count": 0}
    protocol, recorded = read(protocol_path), read(folder/"assessment.json")
    if recorded["protocol_sha256"] != sha(protocol_path):
        raise ValueError("Public screening bound to a different frozen arm")
    parents, workers = {}, 0
    for stage in ("practice", "development"):
        path = folder/"evaluation"/f"{stage}-parent.json"
        if path.is_file():
            parents[stage] = qstudy._validated_worker(path, folder/"rosters"/f"{stage}.json",
                protocol_path, PARENT, None, stage)
            workers += 1
    candidates = {}
    for name, record in recorded["candidates"].items():
        if record.get("duplicate_of"):
            owner = recorded["candidates"][record["duplicate_of"]]
            if record["adapter_sha256"] != owner["adapter_sha256"] or any(record[k] != owner[k]
                    for k in ("practice_gate", "development_gate", "eligible")):
                raise ValueError("Duplicate adapter treatment changes its outcomes")
        else:
            for stage in ("practice", "development"):
                worker = qstudy._validated_worker(folder/"evaluation"/f"{stage}-{name}.json",
                    folder/"rosters"/f"{stage}.json", protocol_path, PARENT, record["adapter_sha256"], stage)
                gate = qstudy._gate(parents[stage]["measurement"], worker["measurement"], protocol, stage)
                if gate != record[f"{stage}_gate"]:
                    raise ValueError("Screening gate does not replay from complete generated responses")
                workers += 1
            if record["eligible"] != all(record[f"{stage}_gate"]["passed"] for stage in ("practice", "development")):
                raise ValueError("Screening eligibility differs from frozen gates")
        candidates[name] = {"eligible": record["eligible"], "practice_passed": record["practice_gate"]["passed"],
            "development_passed": record["development_gate"]["passed"], "adapter_sha256": record["adapter_sha256"]}
    eligible = sorted((row for row in recorded["candidates"].values() if row["eligible"]), key=qstudy._rank)
    expected = eligible[0]["id"] if eligible and not recorded["errors"] else None
    if expected != recorded["winner"]:
        raise ValueError("Finalist does not match the frozen ordering")
    return {"status": "PUBLIC_SCREENING_REPLAY_PASS", "winner": expected, "candidates": candidates,
            "worker_count": workers, "complete_assessment": not bool(recorded["errors"])}


def replay_development_retention(root):
    """Replay the pre-merge paired gate from the two locked adapter workers.

    The early rank-four failure is public development evidence, not a Bank B
    observation. Its inputs are adapters on the original BF16 parent; the later
    merged comparison, when reached, remains a separate gate.
    """
    path = root/"rank4/adapter-development-retention.json"
    if not path.is_file():
        return None
    rstudy._matched_arms(root/"rank8/protocol.json", root/"rank4/protocol.json")
    workers, identities = [], {}
    rosters = [root/arm/"assessment/rosters/development.json" for arm in ARMS]
    if sha(rosters[0]) != sha(rosters[1]):
        raise ValueError("Public paired development rosters differ")
    for arm, roster in zip(ARMS, rosters, strict=True):
        folder, protocol = root/arm/"assessment", root/arm/"protocol.json"
        screening = read(folder/"assessment.json")
        winner = screening["winner"]
        if winner is None:
            raise ValueError("Paired development comparison lacks a locked finalist")
        selected = screening["candidates"][winner]
        worker_path = folder/"evaluation"/f"development-{winner}.json"
        worker = qstudy._validated_worker(worker_path, roster, protocol, PARENT,
                                           selected["adapter_sha256"], "development")
        workers.append(worker)
        identities[arm] = {"candidate_id": winner, "adapter_sha256": selected["adapter_sha256"],
                           "evaluation_sha256": sha(worker_path)}
    measured = rstudy.retention(workers[0]["measurement"], workers[1]["measurement"])
    if read(path) != measured:
        raise ValueError("Paired adapter development retention does not replay")
    return {"passed": measured["passed"], "receipt_sha256": sha(path), "inputs": identities,
        "representation": "fp32_adapter_on_bf16_parent", "independent_bank_evidence": False,
        "lost_rank8_correct_responses": len(measured["prohibited_regressions"]),
        "lost_rank8_valid_responses": len(measured["lost_valid_presentations"]),
        "lost_controls": len(measured["lost_required_controls"]),
        "strict_comparison_gain": measured["strict_comparison_gain"],
        "complete_family_gain": measured["complete_family_gain"],
        "complete_continuation_disagreement_count": len(measured["complete_continuation_disagreements"])}


def verify(public_export, out):
    wall, cpu = time.perf_counter(), time.process_time()
    root, out = Path(public_export).resolve(), Path(out).resolve()
    if out.is_relative_to(root):
        raise ValueError("Verification receipt must remain outside the immutable export")
    result = {"schema": "weightbridge.phase_iiir.public_replay.v1", "passed": False,
        "disposition": "PUBLIC_ARCHIVE_CHECK_FAILED", "model_execution": False, "private_roster_access": False,
        "independent_confirmation_raw_replay": False, "scope": "Public screening and artifact replay only; no new neural learning claim."}
    try:
        manifest = read(root/"export-manifest.json")
        count = verify_files(root, manifest)
        protocol = read(root/"protocol.json")
        verify_sources(protocol["source_sha256"])
        if sha(ROOT/"results/phase-iii-r-replication/report.py") != manifest["reporter_sha256"]:
            raise ValueError("Public report generator identity differs")
        result.update(artifact_count=count, frozen_source_count=len(protocol["source_sha256"]),
            protocol_sha256=sha(root/"protocol.json"), export_manifest_sha256=sha(root/"export-manifest.json"))
        summaries = read(root/"arm-summary.json")
        outcomes = {}
        for arm in ARMS:
            replay = replay_screening(root, arm)
            if replay["winner"] != summaries[arm]["selected_checkpoint"]:
                raise ValueError("Derived arm summary changed the selected checkpoint")
            stages = {}
            for stage in STAGES:
                path = root/arm/"assessment"/f"{stage}-gate.json"
                gate = read(path)["aggregate"] if path.is_file() else None
                if summaries[arm]["independent_stages"][stage] != gate:
                    raise ValueError("Unmeasured independent evidence was synthesized or altered")
                if replay["winner"] is None and gate is not None:
                    raise ValueError("Independent evidence was opened without a public qualified finalist")
                stages[stage] = None if gate is None else {"passed": gate["passed"],
                    "evidence_type": "Published aggregate; private rows not independently replayed here"}
            outcomes[arm] = {"screening": replay, "independent_stages": stages}
        paired = replay_development_retention(root)
        summary_paired = summaries["rank4"].get("adapter_development_retention")
        if (paired is None) != (summary_paired is None) or (paired is not None and summary_paired["passed"] != paired["passed"]):
            raise ValueError("Paired development summary differs from measured eligibility")
        result.update(passed=True, disposition="PUBLIC_ARCHIVE_REPLAY_PASS", outcomes=outcomes,
                      paired_adapter_development=paired)
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    result["resources"] = {"wall_seconds": time.perf_counter()-wall, "cpu_seconds": time.process_time()-cpu}
    result["verifier_sha256"] = sha(__file__)
    write(out, result)
    return result


def self_test():
    import tempfile
    start, cpu = time.perf_counter(), time.process_time()
    no_rows({"passed": False, "prohibited_regressions_count": 1, "parent": {"correct": 2}})
    for invalid in ({"rows": []}, {"hidden": {"prohibited_regressions": ["secret"]}}, {"text": "secret"}):
        try:
            no_rows(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Protected row field accepted")
    with tempfile.TemporaryDirectory() as path:
        root = Path(path)
        assert replay_screening(root, "rank8")["status"] == "UNMEASURED"
        assert replay_development_retention(root) is None
        try:
            inside(root, "../escape.json")
        except ValueError:
            pass
        else:
            raise AssertionError("Path escape accepted")
        target = root/"write-once.json"
        write(target, {"passed": True})
        try:
            write(target, {})
        except FileExistsError:
            pass
        else:
            raise AssertionError("Verification receipt was overwritten")
    return {"passed": True, "missing_stages_remain_unmeasured": True, "protected_rows_refused": True,
            "write_once": True, "model_execution": False,
            "resources": {"wall_seconds": time.perf_counter()-start, "cpu_seconds": time.process_time()-cpu}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-export", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        value = self_test()
    else:
        if args.public_export is None or args.out is None:
            parser.error("--public-export and --out required")
        value = verify(args.public_export, args.out)
    print(json.dumps({k: v for k, v in value.items() if k not in {"outcomes"}}, indent=2))
    raise SystemExit(0 if value["passed"] else 1)
