"""Report-only allowlist export of Phase S preflight aggregates, never model data.

This code is outside scientific decision sources. Public hash checking validates
the exported archive; it is not a substitute for private model execution/replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile


ARMS = ("original-joint", "sequential")
SCOPES = ("calibration-original-tool", "calibration-sequential-tool",
          "calibration-original-decimal", "calibration-sequential-decimal", "timing-sequential")
FORBIDDEN = {"rows", "prompt", "text", "content", "answer", "oracle_answer", "completion",
             "fixture", "gold_action", "generated_token_ids", "prefix_token_ids", "target_ids",
             "input_ids", "private_seeds", "row_ids", "ordered_row_ids", "command", "argv"}


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {key+"_count" if key.endswith("_ids") and isinstance(item, list) else key:
                    len(item) if key.endswith("_ids") and isinstance(item, list) else clean(item)
                for key, item in value.items() if key not in FORBIDDEN}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def assert_public(value):
    if isinstance(value, dict):
        forbidden = {k for k in value if k in FORBIDDEN and not (k == "rows" and type(value[k]) is int)}
        if forbidden or any(k.endswith("_ids") and isinstance(v, list) for k, v in value.items()):
            raise ValueError("Publication would expose per-row or native-token material")
        for item in value.values():
            assert_public(item)
    elif isinstance(value, list):
        for item in value:
            assert_public(item)


def selected(value, names):
    return {name: value[name] for name in names if name in value}


def export(study: Path, out: Path):
    study, out = Path(study), Path(out)
    if out.exists():
        raise FileExistsError("Public preflight evidence is write-once")
    inputs, artifacts = {}, {}
    def read(relative, required=True):
        path = study/relative
        if not path.is_file() and not required:
            return None
        payload = path.read_bytes()
        inputs[relative] = sha(payload)
        return json.loads(payload)

    env = read("environment-preflight-v1.json")
    artifacts["environment.json"] = {
        **selected(env, ("schema", "python", "packages", "files", "generation_config_present",
                         "r_receipt_sha256", "r_verification_sha256", "r_replication_gate_passed")),
        "parents": [selected(p, ("expected", "sha256", "passed", "tensors")) for p in env["parents"]],
        "tokenizer": selected(env["tokenizer"], ("class", "eos_token", "eos_token_id", "pad_token_id",
                "end_token_encoding", "chat_template_sha256", "template_mentions_enable_thinking"))}
    artifacts["tokenizer-equivalence.json"] = read("parent-tokenizer-equivalence.json")
    old = read("historical-r-public-replay.json")
    artifacts["historical-public-replay.json"] = selected(old, ("schema", "passed", "disposition",
        "model_execution", "private_roster_access", "independent_confirmation_raw_replay", "scope",
        "artifact_count", "frozen_source_count", "export_manifest_sha256", "protocol_sha256",
        "verifier_sha256", "resources"))
    history = read("historical-exclusion-v1.json", required=False)
    artifacts["freshness-coverage.json"] = None if history is None else selected(history, (
        "schema", "coverage", "limitation", "coverage_limitation", "protected_rows_exposed",
        "q_all_seven_populations_verified", "q_manifest_sha256", "q_protocol_sha256", "r_manifest_sha256",
        "r_bank_a_raw_access", "r_bank_b_raw_access", "blocked_numeric_tasks", "blocked_negative_family_anchors",
        "excluded_task_identities", "excluded_negative_anchors", "s_calibration_decimal_sha256",
        "s_calibration_new_task_identities"))
    artifacts["calibration-manifest.json"] = read("calibration-v1/manifest.json")
    summary, raw = {}, {}
    for arm in ARMS:
        protocol = read(f"calibration-v1/{arm}-protocol.json")
        artifacts[f"{arm}-calibration-protocol.json"] = selected(protocol, (
            "schema", "stage", "arm", "checkpoint_sha256", "source_sha256", "task_manifest_sha256",
            "endpoints", "tool_completion", "resources", "limits", "calibration_headroom",
            "training_authorized", "production_authority", "lineage_authority"))
        summary[arm] = {}
        for task in ("tool", "decimal"):
            result = read(f"calibration-v1/{arm}-{task}.json", required=False)
            if result is None:
                summary[arm][task] = None
                continue
            raw[arm, task] = result["measurement"]["rows"]
            item = {"status": "MEASURED", "summary": clean(result["summary"]),
                    "worker_resources": result["resources"], "source_sha256": inputs[f"calibration-v1/{arm}-{task}.json"]}
            if task == "tool":
                wrong = [r for r in raw[arm, task] if r["schema_valid"] and not r["semantic_correct"]]
                kinds = sorted({r["gold_action"]["tool"] for r in wrong} & {"search_files", "read_file", "run_tests"})
                n, families = len(raw[arm, task]), result["summary"]["families"]
                checks = {"competent_schema": result["summary"]["schema_valid"]/n >= .5,
                    "ten_point_gain_possible": 1-result["summary"]["strict_accuracy"] >= .1-1e-12,
                    "incomplete_family_opportunities": 1-result["summary"]["complete_families"]/families >= 6/32,
                    "semantic_action_errors": len(wrong)/n >= 8/256,
                    "semantic_error_family_diversity": len({r["family_id"] for r in wrong})/families >= 4/32,
                    "semantic_error_action_diversity": len(kinds) >= 2}
                item["headroom"] = {"plausible": all(checks.values()), "checks": checks,
                    "schema_valid_semantic_errors": len(wrong),
                    "semantic_error_families": len({r["family_id"] for r in wrong}), "real_action_types": kinds}
            summary[arm][task] = item
    paired = None
    if all((arm, "tool") in raw for arm in ARMS):
        before, after = (raw[arm, "tool"] for arm in ARMS)
        if [r["id"] for r in before] != [r["id"] for r in after]:
            raise ValueError("Calibration comparison identities differ")
        paired = {metric: {"gained": sum(not b[metric] and a[metric] for b,a in zip(before,after,strict=True)),
                           "lost": sum(b[metric] and not a[metric] for b,a in zip(before,after,strict=True))}
                  for metric in ("strict_correct", "schema_valid", "semantic_correct", "completed")}
    artifacts["calibration-summary.json"] = {"arms": summary, "original_to_r8": paired,
        "scope": "Disposable unchanged-parent measurement only; not learning or independent qualification"}
    timing = read("timing-v1/measurement/timing.json", required=False)
    artifacts["training-timing.json"] = None if timing is None else clean(selected(timing, (
        "schema", "scope", "passed", "parent_sha256", "protocol_sha256", "authorization_sha256", "bundle_sha256",
        "optimizer_steps", "counts", "batches", "equivalence", "base_unchanged", "factors_unchanged",
        "initial_factor_sha256", "resources")))
    scopes = {}
    for name in SCOPES:
        record = read(f"execution/{name}.receipt.json", required=False)
        scopes[name] = None if record is None else {"passed": record["passed"], "limits": record["limits"],
            "resources": record["execution"]["resources"], "exit_code": record["execution"]["exit_code"],
            "budget_failure": record["execution"]["budget_failure"],
            "temporary_sleep_request": record["temporary_sleep_request"]}
    observed = [v["resources"] for v in scopes.values() if v is not None]
    artifacts["compute-accounting.json"] = {"scopes": scopes,
        "observed_wall_seconds": sum(v["wall_seconds"] for v in observed) if observed else None,
        "observed_cpu_seconds": sum(v["cpu_seconds"] for v in observed) if observed else None,
        "conservative_peak_bytes": max(v["peak_rss_bytes"] for v in observed) if observed else None,
        "scope": "Five enclosing execution receipts only. Nested worker resources are not summed again. Excludes tests, reporting and unlisted preflight work.",
        "missing_scopes": [name for name, value in scopes.items() if value is None]}
    for artifact in artifacts.values():
        assert_public(artifact)
    out.mkdir(parents=True)
    hashes = {}
    for name, value in artifacts.items():
        payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
        (out/name).write_bytes(payload)
        hashes[name] = sha(payload)
    manifest = {"schema": "weightbridge.phase_iiis.public_preflight.v1", "source_artifact_sha256": inputs,
        "public_artifact_sha256": hashes, "exporter_sha256": sha(Path(__file__).read_bytes()),
        "raw_rosters_published": False, "private_predictions_published": False, "weights_published": False,
        "scope": "Metadata and aggregate publication. Hash verification is not private behavioral replay."}
    (out/"export-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    verify(out)
    return manifest


def verify(out):
    out = Path(out)
    manifest = json.loads((out/"export-manifest.json").read_bytes())
    for name, expected in manifest["public_artifact_sha256"].items():
        if Path(name).name != name or sha((out/name).read_bytes()) != expected:
            raise ValueError("Public preflight artifact hash mismatch")
        assert_public(json.loads((out/name).read_bytes()))
    return {"passed": True, "artifacts": len(manifest["public_artifact_sha256"]),
            "scope": "Public archive integrity only; no model execution or private replay"}


def self_test():
    private = {"rows": [{"prompt": "PRIVATE_SENTINEL"}], "complete_family_ids": ["PRIVATE_ID"],
               "summary": {"correct": 2}, "row_ids": ["PRIVATE_ROW"], "measurement": {"text": "PRIVATE_OUTPUT"}}
    projected = clean(private)
    assert "PRIVATE" not in json.dumps(projected)
    assert projected["complete_family_ids_count"] == 1
    assert_public(projected)
    assert_public({"rows": 80})
    try:
        assert_public(private)
    except ValueError:
        pass
    else:
        raise AssertionError("Raw disclosure was not refused")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        payload = b'{"measured":null}'
        (out/"summary.json").write_bytes(payload)
        (out/"export-manifest.json").write_text(json.dumps({"public_artifact_sha256": {"summary.json": sha(payload)}}))
        assert verify(out)["passed"]
        (out/"summary.json").write_text('{"measured":0}')
        try:
            verify(out)
        except ValueError:
            pass
        else:
            raise AssertionError("Tampering was not refused")
    return {"redaction": "PASS", "raw_refusal": "PASS", "hash_tampering": "PASS", "null_preservation": "PASS"}


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
        result = export(args.study, args.out)
        print(json.dumps({"artifacts": len(result["public_artifact_sha256"]), "raw_rosters_published": False}))
    else:
        parser.error("Choose --self-test, --verify, or --study with --out")
