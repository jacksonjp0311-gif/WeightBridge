"""Archive verified Phase III-D evidence without opening the adaptation reserve."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.core import file_hash, json_dump_atomic


PROTOCOL = Path(__file__).with_name("protocol.json")
SPLITS = ("practice", "selection", "qualification")
SOURCES = (
    "experiments/phase_iiid/analysis.py",
    "experiments/phase_iiid/roster.py",
    "experiments/phase_iiid/run_grounding.py",
    "weightbridge/adaptation/bic.py",
    "weightbridge/adaptation/content_scoring.py",
    "weightbridge/adaptation/grounding.py",
)


def export(material, run, output, *, admission=None, description=None):
    """Verify the run's hash links and copy only measured artifacts to a new archive."""
    artifacts = {}
    inventory = {}

    def read(relative, source, expected=None):
        data = source.read_bytes()
        digest = sha256(data).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f"Hash mismatch at {relative}")
        inventory[relative] = {"sha256": digest, "bytes": len(data)}
        artifacts[relative] = data
        return json.loads(data)

    protocol = read("material/protocol.json", PROTOCOL)
    protocol_hash = inventory["material/protocol.json"]["sha256"]
    report = read("run/report.json", run / "report.json")
    report_hash = inventory["run/report.json"]["sha256"]
    manifest = read("material/manifest.json", material / "manifest.json", report["material_manifest_sha256"])
    sources = read("run/execution-sources.json", run / "execution-sources.json", report["execution_sources_sha256"])
    decision = read("run/practice-decision.json", run / "practice-decision.json", report["practice_decision_sha256"])

    model = protocol["model_sha256"]
    if (report["schema"] != "weightbridge.phase_iiid.grounding_report.v1"
            or manifest["schema"] != "weightbridge.phase_iiid.grounding_manifest.v1"
            or sources["schema"] != "weightbridge.phase_iiid.grounding_execution.v1"
            or not all(doc["protocol_sha256"] == protocol_hash for doc in (report, manifest, sources, decision))
            or any(digest != model for digest in (report["base_sha256"], manifest["model_sha256"], sources["base_sha256"]))
            or sources["material_manifest_sha256"] != report["material_manifest_sha256"]
            or any(sources[key] != protocol[key] for key in ("inference", "generation"))
            or set(sources["source_sha256"]) != set(SOURCES)
            or set(report["splits"]) != set(SPLITS)
            or manifest["generator_sha256"] != sources["source_sha256"]["experiments/phase_iiid/roster.py"]):
        raise ValueError("Run identities do not match the frozen protocol, source record and manifest")
    for name, digest in sources["source_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f"Measured execution source has changed at {name}")

    for split in SPLITS:
        roster = read(f"material/{split}.json", material / f"{split}.json", manifest["artifacts"][split]["sha256"])
        name = f"{split}-predictions.json"
        predictions = read(f"run/{name}", run / name, report["splits"][split]["predictions_sha256"])
        if (roster["split"] != split or predictions["split"] != split
                or predictions["resources"] != report["splits"][split]["resources"]):
            raise ValueError(f"Measurement and roster disagree for {split}")
    if decision["practice_predictions_sha256"] != report["splits"]["practice"]["predictions_sha256"]:
        raise ValueError("Practice decision does not bind the archived practice predictions")

    for name, path in (("admission", admission), ("description", description)):
        if path is not None and read(f"run/{name}.json", path)["report_sha256"] != report_hash:
            raise ValueError(f"Optional {name} does not bind the measured report")

    output.mkdir(parents=True, exist_ok=False)
    for relative, data in artifacts.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        if file_hash(destination) != inventory[relative]["sha256"]:
            raise ValueError(f"Archive bytes differ at {relative}")
    json_dump_atomic(output / "artifact_manifest.json", {
        "schema": "weightbridge.phase_iiid.grounding_archive.v1",
        "report_sha256": report_hash,
        "artifacts": inventory,
        "sealed_adaptation_sha256": manifest["artifacts"]["sealed-adaptation"]["sha256"],
        "sealed_adaptation_bytes_exported": False,
    })
    print(f"Verified and archived {len(inventory)} exact files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("material", "run", "out"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--description", type=Path)
    args = parser.parse_args()
    export(args.material, args.run, args.out, admission=args.admission, description=args.description)
