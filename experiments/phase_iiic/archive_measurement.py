"""Export exact measured evidence while keeping the adaptation roster sealed."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiic.measurement_roster import PROTOCOL
from weightbridge.core import file_hash, json_dump_atomic


SOURCE_PATHS = ("experiments/phase_iiic/measurement_protocol.json", "experiments/phase_iiic/measurement_roster.py",
                "experiments/phase_iiic/run_measurement.py", "weightbridge/adaptation/content_scoring.py")


def capture_sources(material, output):
    json_dump_atomic(output, {"schema": "weightbridge.phase_iiic.execution_sources.v1",
                              "recorded_utc": datetime.now(timezone.utc).isoformat(),
                              "scope": "Recorded before starting a replay",
                              "manifest_sha256": file_hash(material / "manifest.json"),
                              "sources": {name: file_hash(ROOT / name) for name in SOURCE_PATHS}})


def export(material, run, output):
    manifest_path, report_path = material / "manifest.json", run / "report.json"
    manifest, report = [json.loads(path.read_bytes()) for path in (manifest_path, report_path)]
    if (report["protocol_sha256"] != file_hash(PROTOCOL)
            or report["base_sha256"] != manifest["model_sha256"]
            or report["material_manifest_sha256"] != file_hash(manifest_path)
            or report["practice_decision_sha256"] != file_hash(run / "practice-decision.json")):
        raise ValueError("Measurement report does not bind the frozen inputs and decision")
    sources = json.loads((run / "execution-sources.json").read_bytes())
    if (sources["manifest_sha256"] != file_hash(manifest_path) or set(sources["sources"]) != set(SOURCE_PATHS)
            or sources["sources"]["experiments/phase_iiic/run_measurement.py"] != report["runner_source_sha256"]
            or sources["sources"]["weightbridge/adaptation/content_scoring.py"] != report["scoring_source_sha256"]):
        raise ValueError("Execution source snapshot does not match the measured run")
    for relative, expected in sources["sources"].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError(f"Execution source changed at {relative}")
    archive_files = {"material/manifest.json": manifest_path, "material/measurement_protocol.json": PROTOCOL}
    for split in ("practice", "selection", "qualification"):
        source = material / f"{split}.json"
        if file_hash(source) != manifest["artifacts"][split]["sha256"]:
            raise ValueError(f"Measured {split} bytes differ from the roster manifest")
        archive_files[f"material/{split}.json"] = source
        archive_files[f"run/{split}-predictions.json"] = run / f"{split}-predictions.json"
    for name in ("report.json", "practice-decision.json", "execution-sources.json", "factorial-analysis.json",
                 "practice-invariance.json"):
        archive_files[f"run/{name}"] = run / name
    expected_report = json.loads((run / "factorial-analysis.json").read_bytes())
    if expected_report["report_sha256"] != file_hash(report_path):
        raise ValueError("Factorial analysis does not bind the measured report")
    output.mkdir(parents=True)
    inventory = {}
    for relative, source in archive_files.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if file_hash(source) != file_hash(destination):
            raise ValueError(f"Archive bytes differ at {relative}")
        inventory[relative] = {"sha256": file_hash(destination), "bytes": destination.stat().st_size}
    json_dump_atomic(output / "artifact_manifest.json", {
        "schema": "weightbridge.phase_iiic.measurement_archive.v1", "artifacts": inventory,
        "sealed_adaptation_sha256": manifest["artifacts"]["sealed-adaptation"]["sha256"],
        "sealed_adaptation_bytes_exported": False,
        "runtime_packages": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "transformers", "psutil")},
    })
    print(f"Verified and archived {len(inventory)} exact files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--capture-sources", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.capture_sources:
        capture_sources(args.material, args.out)
    elif args.run is None:
        parser.error("--run is required to export measured evidence")
    else:
        export(args.material, args.run, args.out)
