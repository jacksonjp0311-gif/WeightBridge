"""Export hash-linked Phase III evidence without copying checkpoints or local paths."""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.core import file_hash, json_dump_atomic


def path_redactor(base, materialized, study_run, controls):
    roots = {base: "$BASE_CHECKPOINT", materialized: "$FROZEN_MATERIAL", study_run: "$STUDY_RUN",
             ROOT: "$REPOSITORY", Path(tempfile.gettempdir()): "$TEMP", Path.home(): "$HOST_HOME"}
    if controls:
        roots[controls] = "$CONTROL_RUN"
    replacements = {}
    for root, placeholder in roots.items():
        absolute = root.resolve()
        for variant in (str(absolute), absolute.as_posix()):
            replacements[variant] = placeholder
            replacements[f"\\\\?\\{variant}"] = placeholder
    ordered = sorted(replacements.items(), key=lambda item: -len(item[0]))

    def redact(value):
        if isinstance(value, str):
            for original, placeholder in ordered:
                value = value.replace(original, placeholder)
        elif isinstance(value, list):
            value = [redact(item) for item in value]
        elif isinstance(value, dict):
            value = {redact(key): redact(item) for key, item in value.items()}
        return value

    return redact


def select_files(root):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (path.is_symlink() or not path.is_file() or path.suffix not in (".json", ".txt")
                or any(part.startswith(".weightbridge-stage-") for part in relative.parts)):
            continue
        if "model" in relative.parts and path.name != ".weightbridge_receipt.json":
            continue
        yield path, relative


def export(base, materialized, study_run, controls, calibrations, out, partial):
    completion = study_run / "summary.json"
    if not partial and not completion.is_file():
        raise ValueError("The study has no completed summary. Export with --partial for an in-progress record.")
    protocol_sha = file_hash(materialized / "protocol.json")
    if completion.exists() and json.loads(completion.read_bytes())["protocol_sha256"] != protocol_sha:
        raise ValueError("Study and frozen data have different protocols")
    redact = path_redactor(base, materialized, study_run, controls)
    out.mkdir(parents=True)
    artifacts = {}

    def publish(path, relative):
        raw = path.read_bytes()
        compress = path.name == "patch.json"
        if compress:
            relative = relative.with_suffix(".json.gz")
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            original = json.loads(raw)
            sanitized = redact(original)
            if compress:
                if sanitized != original:
                    raise ValueError(f"Unexpected machine-local path in a numeric patch: {relative}")
                target.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
            elif sanitized != original:
                json_dump_atomic(target, sanitized)
            else:
                target.write_bytes(raw)
        else:
            original = raw.decode("utf-8", errors="replace")
            sanitized = redact(original)
            target.write_bytes(sanitized.encode("utf-8") if sanitized != original else raw)
        source_sha = sha256(raw).hexdigest()
        exported_sha = file_hash(target)
        artifacts[relative.as_posix()] = {"source_sha256": source_sha, "exported_sha256": exported_sha,
                                           "byte_identical": source_sha == exported_sha}
        if compress:
            artifacts[relative.as_posix()]["encoding"] = "gzip"
            artifacts[relative.as_posix()]["decoded_sha256"] = source_sha

    for root, namespace in ((materialized, "material"), (study_run, "run"), (controls, "controls")):
        if root is None:
            continue
        for path, relative in select_files(root):
            if partial and namespace == "material" and relative.parts[0] == "sealed":
                report = study_run / "state" / "evaluation" / path.stem / "report.json"
                if not report.is_file():
                    continue
            publish(path, Path(namespace) / relative)
    for path in calibrations:
        publish(path, Path("calibration") / path.name)
    manifest = {"schema": "weightbridge.phase_iii.evidence_export.v1", "protocol_sha256": protocol_sha,
                "study_complete": completion.is_file(), "partial_export": partial, "artifacts": artifacts}
    json_dump_atomic(out / "artifact_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("base", "materialized", "study-run", "out"):
        parser.add_argument(f"--{path}", type=Path, required=True)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--calibration", type=Path, action="append", default=[])
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    manifest = export(args.base, args.materialized, args.study_run, args.controls, args.calibration, args.out, args.partial)
    print(json.dumps({"files": len(manifest["artifacts"]), "study_complete": manifest["study_complete"]}))
