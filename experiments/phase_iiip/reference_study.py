"""Isolated frozen decimal evaluation of the published external checkpoint."""
from __future__ import annotations
import argparse
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiih.run_decimal_endpoint import endpoint, _resources
from experiments.phase_iiil import semantic_roster as generator
from experiments.phase_iiil.semantic_diagnostics import summarize as semantic_summary
from experiments.phase_iiim.semantic_metrics import compare
from weightbridge.adaptation.behavior_endpoint import measure_endpoint
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("reference_protocol.json")


def source_hashes():
    from experiments.phase_iiih.run_decimal_endpoint import sources
    return {**sources(), **{p.relative_to(ROOT).as_posix(): file_hash(p) for p in (
        Path(__file__), PROTOCOL, Path(__file__).with_name("reference_authorization.json"),
        Path(__file__).with_name("reference_import.py"), Path(__file__).with_name("reference_execute.py"),
        Path(generator.__file__), Path(generator.renderer.__file__))}}


def freeze(out):
    if out.exists():
        raise FileExistsError("Reference roster and commitment are write-once")
    config = json.loads(PROTOCOL.read_bytes())
    split = config["split"]
    counts = {"target_pairs": config["target_families"], "positive": config["positive_comparisons"], "preserve": config["copy_items"]}
    compatible = {"roster_counts": {split: counts}, "endpoint": config["endpoint"]}
    rows = generator._generate(compatible, split, config["seed"], set(), set())
    offset = Decimal(config["integer_offset"])
    for row in rows:
        values = [format(Decimal(value) + (offset if Decimal(value) >= 0 else -offset), "f") for value in row["operands"]]
        row["operands"] = values
        row["answer"] = generator.renderer._answer(row["family"], values)
        row["prompt"] = generator.renderer._prompt(row["family"], values, row["wording"])
        if row["family"] == "larger-decimal":
            row["truth_side"] = values.index(row["answer"])
    generator._validate(rows, split, counts)
    if (len(rows) != config["budgets"]["presentations_per_model"]
            or any(abs(Decimal(v)) < offset for row in rows for v in row["operands"])):
        raise ValueError("Reference-only population or disjoint magnitude domain differs")
    json_dump_atomic(out, {"schema": "weightbridge.phase_iiip.reference_roster.v1", "track": "B",
        "protocol_sha256": file_hash(PROTOCOL), "sources": source_hashes(), "rows": rows})


def worker(base, sha, roster, out, roster_sha256):
    if out.exists():
        raise FileExistsError("Reference worker never replaces predictions")
    wall, cpu = time.perf_counter(), time.process_time()
    config, frozen, sources = json.loads(PROTOCOL.read_bytes()), json.loads(roster.read_bytes()), source_hashes()
    if (file_hash(roster) != roster_sha256 or frozen["protocol_sha256"] != file_hash(PROTOCOL)
            or frozen["sources"] != sources):
        raise ValueError("Reference endpoint or roster source identity changed")
    try:
        with _open(base, config, sha) as (_, runner, capabilities):
            measurement = measure_endpoint(runner, frozen["rows"], endpoint(config))
        if (load_checkpoint(base).checkpoint_sha256() != sha or source_hashes() != sources
                or file_hash(roster) != roster_sha256):
            raise ValueError("Reference checkpoint or evaluator changed")
    except BaseException as error:
        json_dump_atomic(out.with_name(out.stem + ".failure.json"), {
            "schema": "weightbridge.phase_iiip.reference_worker_failure.v1", "status": "INTERRUPTED",
            "checkpoint_sha256": sha, "roster_sha256": roster_sha256,
            "error": {"type": type(error).__name__, "message": str(error)}, "resources": _resources(wall, cpu),
            "behavioral_measurement": None, "partial_model_call_count": None})
        raise
    record = {"schema": "weightbridge.phase_iiip.reference_worker.v1", "track": "B", "checkpoint_sha256": sha,
        "roster_sha256": file_hash(roster), "sources": sources, "measurement": measurement,
        "summary": semantic_summary(measurement), "resources": _resources(wall, cpu), "lineage_authority": False}
    json_dump_atomic(out, record)
    return record


def assess(base, candidate, roster, reconstruction, out):
    wall, cpu = time.perf_counter(), time.process_time()
    config = json.loads(PROTOCOL.read_bytes())
    receipt = json.loads(reconstruction.read_bytes())
    if out.exists():
        raise FileExistsError("Use a fresh reference study path")
    out.mkdir(parents=True)
    original = load_checkpoint(base).checkpoint_sha256()
    edited = load_checkpoint(candidate).checkpoint_sha256()
    authorization = Path(__file__).with_name("reference_authorization.json")
    importer = Path(__file__).with_name("reference_import.py")
    if (original != config["base_checkpoint_sha256"] or receipt.get("candidate_checkpoint_sha256") != edited
            or receipt.get("schema") != "weightbridge.phase_iiip.reference_reconstruction.v1"
            or receipt.get("status") != "REFERENCE_RECONSTRUCTED"
            or receipt.get("authorization_sha256") != file_hash(authorization)
            or receipt.get("importer_source_sha256") != file_hash(importer)
            or receipt.get("source_checkpoint_preserved") is not True
            or receipt.get("base_checkpoint_sha256_rechecked") != original):
        raise ValueError("Reference study requires the authorized reconstructed checkpoint identity")
    frozen_hash, expected_sources = file_hash(roster), source_hashes()
    observations = {}
    for name, path, digest in (("original", base, original), ("reference", candidate, edited)):
        destination = out / f"{name}.json"
        subprocess.run([sys.executable, "-m", "experiments.phase_iiip.reference_study", "worker",
            "--base", str(path), "--sha256", digest, "--roster", str(roster), "--roster-sha256", frozen_hash, "--out", str(destination)],
            cwd=ROOT, check=True, timeout=config["inference"]["worker_timeout_seconds"])
        observations[name] = json.loads(destination.read_bytes())
        if (observations[name]["roster_sha256"] != frozen_hash or observations[name]["checkpoint_sha256"] != digest
                or file_hash(roster) != frozen_hash or source_hashes() != expected_sources):
            raise ValueError("Reference worker does not bind its checkpoint and roster")
    paired = compare(observations["original"]["measurement"], observations["reference"]["measurement"],
        min_comparison_gain=1, min_repeatable_corrections=1)
    result = {"schema": "weightbridge.phase_iiip.reference_comparison.v1", "track": "B", "status": "BENCHMARK_COMPLETED",
        "protocol_sha256": file_hash(PROTOCOL), "roster_sha256": file_hash(roster),
        "reconstruction_receipt_sha256": file_hash(reconstruction), "source_model": original, "reference_model": edited,
        "paired": paired, "original_summary": observations["original"]["summary"],
        "reference_summary": observations["reference"]["summary"],
        "worker_sha256": {name: file_hash(out / f"{name}.json") for name in observations},
        "resources": {"host": _resources(wall, cpu), "workers": {name: r["resources"] for name, r in observations.items()}},
        "source_parent_restored_exactly": load_checkpoint(base).checkpoint_sha256() == original,
        "lineage_accepted": False, "new_weightbridge_learning_claimed": False, "long_benchmark_reproduction": False,
        "scope": config["result_scope"]}
    json_dump_atomic(out / "comparison.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "worker", "assess"))
    for field in ("base", "candidate", "roster", "out", "reconstruction"):
        parser.add_argument(f"--{field}", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--roster-sha256")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.out)
    elif args.command == "worker":
        print(worker(args.base, args.sha256, args.roster, args.out, args.roster_sha256)["summary"], flush=True)
    else:
        print(assess(args.base, args.candidate, args.roster, args.reconstruction, args.out)["paired"], flush=True)
