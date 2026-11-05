"""Publish redacted prospective commitments; never open a task roster or bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import tempfile


EXPECTED_MANIFEST = "ccc217d9139d214ecb241acb3099006a9cc35a0b3f85a6f49784adc3de9a813a"
DROP = {"path", "format_example_family_id", "private_seeds", "row_ids", "ordered_row_ids",
        "prompt", "answer", "oracle_answer", "completion", "text", "fixture", "gold_action",
        "generated_token_ids", "prefix_token_ids", "input_ids", "target_ids"}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def public(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in DROP:
                continue
            if key == "rows" and isinstance(item, dict) and set(item) == {"path", "rows", "sha256"} and type(item["rows"]) is int:
                result["combined_roster"] = public(item)
                continue
            if key == "rows" and type(item) is not int:
                raise ValueError("Raw rows cannot be preregistered publicly")
            if key.endswith("_ids") and isinstance(item, list):
                result[key+"_count"] = len(item)
            elif key == "preflight_receipt_sha256":
                result[key] = {PureWindowsPath(k).name: v for k, v in item.items()}
                if len(result[key]) != len(item):
                    raise ValueError("Receipt basename collision")
            else:
                result[key] = public(item)
        return result
    if isinstance(value, list):
        return [public(item) for item in value]
    if isinstance(value, str) and (":\\" in value or value.startswith("C:/") or value.startswith("\\\\")):
        raise ValueError("A private absolute path survived projection")
    return value


def check(value):
    if isinstance(value, dict):
        if set(value) & DROP:
            raise ValueError("Unsafe publication field")
        if "rows" in value and type(value["rows"]) is not int:
            raise ValueError("Unsafe publication rows")
        if any(k.endswith("_ids") and isinstance(v, list) for k, v in value.items()):
            raise ValueError("Per-item identifiers cannot be published")
        for item in value.values():
            check(item)
    elif isinstance(value, list):
        for item in value:
            check(item)
    elif isinstance(value, str) and (":\\" in value or value.startswith("C:/") or value.startswith("\\\\")):
        raise ValueError("Private absolute path in public commitment")


def export(study, out):
    study, out = Path(study), Path(out)
    if out.exists():
        raise FileExistsError("Preregistration publication is write-once")
    inputs = {}
    def read(name):
        payload = (study/name).read_bytes()
        inputs[name] = sha(payload)
        return json.loads(payload)
    source = read("manifest.json")
    if inputs["manifest.json"] != EXPECTED_MANIFEST:
        raise ValueError("Study differs from the operator-confirmed prospective manifest")
    selected = {k: v for k, v in source.items() if k != "historical_exclusion"}
    selected["historical_exclusion"] = {k: v for k, v in source["historical_exclusion"].items()
        if not isinstance(v, (dict, list))}
    artifacts = {"manifest.public.json": public(selected),
                 "initialized.public.json": public(read("initialized.json")),
                 "prospective-data-order.json": public(read("prospective-data-order.json")),
                 "source-manifest.json": source["source_sha256"]}
    for arm in ("sequential", "original-joint"):
        protocol = read(f"{arm}/protocol.json")
        if protocol["task_manifest_sha256"] != EXPECTED_MANIFEST or protocol["source_sha256"] != source["source_sha256"]:
            raise ValueError("Arm protocol differs from frozen task/source commitments")
        artifacts[f"{arm}-protocol.public.json"] = public(protocol)
    for value in artifacts.values():
        check(value)
    out.mkdir(parents=True)
    hashes = {}
    for name, value in artifacts.items():
        payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+"\n").encode()
        (out/name).write_bytes(payload)
        hashes[name] = sha(payload)
    receipt = {"schema": "weightbridge.phase_iiis.public_preregistration.v1",
        "frozen_study_manifest_sha256": EXPECTED_MANIFEST, "original_source_bytes_sha256": inputs,
        "public_bytes_sha256": hashes, "exporter_sha256": sha(Path(__file__).read_bytes()),
        "projection": "Public JSON is a redacted projection, not claimed byte-identical to private originals. Both byte hashes are recorded.",
        "seed_scope": "Only operator-declared optimizer and bootstrap seeds are public; dataset seeds are represented only by commitments.",
        "raw_rosters_opened": False, "raw_rosters_published": False, "private_outputs_published": False,
        "weights_published": False, "model_execution": False}
    (out/"export-manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return verify(out)


def verify(out):
    out = Path(out)
    receipt = json.loads((out/"export-manifest.json").read_bytes())
    for name, expected in receipt["public_bytes_sha256"].items():
        if Path(name).name != name or sha((out/name).read_bytes()) != expected:
            raise ValueError("Public preregistration byte hash mismatch")
        check(json.loads((out/name).read_bytes()))
    return {"passed": True, "artifacts": len(receipt["public_bytes_sha256"]),
            "frozen_study_manifest_sha256": receipt["frozen_study_manifest_sha256"], "model_execution": False}


def self_test():
    value = {"files": {"tool": {"path": "C:\\private\\hidden.json", "sha256": "a"*64, "rows": 256}},
             "private_seeds": "NEVER_EXPOSE", "complete_family_ids": ["PRIVATE_ID"],
             "preflight_receipt_sha256": {"C:\\private\\timing.json": "b"*64}}
    projected = public(value)
    check(projected)
    assert "NEVER_EXPOSE" not in json.dumps(projected) and "PRIVATE_ID" not in json.dumps(projected)
    assert projected["files"]["tool"]["rows"] == 256
    assert projected["preflight_receipt_sha256"] == {"timing.json": "b"*64}
    try:
        public({"rows": [{"id": "private"}]})
    except ValueError:
        pass
    else:
        raise AssertionError("Raw roster export was permitted")
    return {"redaction": "PASS", "count_preservation": "PASS", "raw_roster_refusal": "PASS"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    elif args.verify:
        print(json.dumps(verify(args.verify)))
    elif args.study and args.out:
        print(json.dumps(export(args.study, args.out)))
    else:
        parser.error("Choose --self-test, --verify, or --study with --out")
