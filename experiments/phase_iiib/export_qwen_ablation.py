"""Publish a completed Phase III-B study without local paths or model shards."""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iii.export_natural_qwen_evidence import path_redactor, select_files
from weightbridge.core import file_hash, json_dump_atomic


def export(base, material, run, output):
    protocol_sha = file_hash(material / "protocol.json")
    outcome = json.loads((run / "report.json").read_bytes())
    if outcome["protocol_sha256"] != protocol_sha:
        raise ValueError("Completed report does not bind the frozen material")
    redact = path_redactor(base, material, run, None)
    output.mkdir(parents=True)
    artifacts, omitted_inputs = {}, {}

    for root, prefix in ((material, "material"), (run, "run")):
        for source, relative in select_files(root):
            name = Path(prefix) / relative
            raw = source.read_bytes()
            digest = sha256(raw).hexdigest()
            if source.name == "compile-input.json":
                # Each input consists of frozen practice, intent and policy
                # plus an optional identical copy of a preserved A patch.
                omitted_inputs[name.as_posix()] = digest
                continue
            compress = source.name == "patch.json"
            if compress:
                name = name.with_suffix(".json.gz")
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == ".json":
                original = json.loads(raw)
                portable = redact(original)
                if compress:
                    if portable != original:
                        raise ValueError("Numeric patch unexpectedly contains a machine-local path")
                    target.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
                elif portable != original:
                    json_dump_atomic(target, portable)
                else:
                    target.write_bytes(raw)
            else:
                target.write_text(redact(raw.decode("utf-8", errors="replace")), encoding="utf-8", newline="")
            exported = file_hash(target)
            artifacts[name.as_posix()] = {
                "source_sha256": digest, "exported_sha256": exported,
                "byte_identical": digest == exported,
                **({"encoding": "gzip", "decoded_sha256": digest} if compress else {}),
            }
    manifest = {"schema": "weightbridge.phase_iiib.qwen_evidence.v1", "protocol_sha256": protocol_sha,
                "artifacts": artifacts, "omitted_reconstructible_input_sha256": omitted_inputs}
    json_dump_atomic(output / "artifact_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "material", "run", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    manifest = export(args.base.resolve(), args.material.resolve(), args.run.resolve(), args.out.resolve())
    print(json.dumps({"exported_files": len(manifest["artifacts"]), "protocol_sha256": manifest["protocol_sha256"]}))
