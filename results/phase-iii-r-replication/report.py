"""Read-only publication projection. Never imported by training or decisions.

Usage: python results/phase-iii-r-replication/report.py --study PRIVATE_ROOT
       --out NEW_PUBLIC_VERSION [--scope label=relative-process-receipt.json]
Only explicitly named metadata and practice/development/smoke evidence can be
copied. Independent gates are projected to counts. Protected rows, weights,
optimizer states and private helper scripts are never copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ARMS = ("rank8", "rank4")
STAGES = ("preservation-validation", "selection", "hidden")
COUNTS = {"strict_comparison_ids", "complete_family_ids", "newly_complete_families",
    "repeatable_comparisons_corrected", "prohibited_regressions", "lost_valid_presentations",
    "lost_required_controls", "correctness_disagreements", "validity_disagreements",
    "complete_continuation_disagreements"}
SCALARS = {"schema", "status", "passed", "disposition", "reason", "stage", "method", "scope",
    "lineage_accepted", "production_authority", "lineage_authority", "protected_generations_private",
    "rank4_adaptation_pass", "rank4_retention_pass", "presentations", "correct", "valid", "completed",
    "valid_completed_first_lines", "completed_first_lines", "strict_comparisons", "comparisons",
    "complete_families", "families", "strict_comparison_gain", "complete_family_gain", "correct_response_gain",
    "observations", "exact_native_generation", "exact_rank8_native_generation", "numerical_merge_receipt_exact",
    "stored_update_norm", "unrounded_update_norm", "rounding_residual_norm", "changed_entries"}
CONTAINERS = {"parent", "candidate", "merged", "preservation_validation", "selection", "hidden",
    "gates", "stages", "rank4_ordinary", "rank8_reference", "rank4_retention", "adapter_development_retention",
    "merged_development_retention", "exact_reconstruction", "rollback", "reconstructed", "replays"}


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def numeric_tree(value):
    """Resources/group summaries contain only numerical aggregate leaves."""
    if isinstance(value, dict):
        return {k: numeric_tree(v) for k, v in value.items() if isinstance(v, (dict, int, float, bool)) or v is None}
    if value is None or isinstance(value, (int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite aggregate")
        return value
    raise ValueError("Non-numeric aggregate")


def protected(value):
    """Strict field projection, not a recursive copy with a short denylist."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Protected summary must be an object or null")
    result = {}
    for key, item in value.items():
        if key in COUNTS:
            if not isinstance(item, list):
                raise ValueError("Expected protected identifier list")
            result[key+"_count"] = len(item)
        elif key in SCALARS and (item is None or isinstance(item, (str, int, float, bool))):
            result[key] = item
        elif key.endswith("sha256") and isinstance(item, str) and len(item) == 64 and all(c in "0123456789abcdef" for c in item):
            result[key] = item
        elif key in ("resources", "groups", "decode_termination", "worker_limits"):
            result[key] = numeric_tree(item)
        elif key in CONTAINERS and (item is None or isinstance(item, dict)):
            if key in ("gates", "stages", "replays") and item is not None:
                result[key] = {name: protected(part) for name, part in item.items()
                               if name in (*STAGES, "practice", "development")}
            else:
                result[key] = protected(item)
        elif key == "error" and item is not None:
            result[key] = {"type": item.get("type")} if isinstance(item, dict) else {"present": True}
    return result


def _inside(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Receipt path escaped its study")
    return path


def check_action_receipt(value):
    """Refuse unexpected row evidence in a purported process-only receipt."""
    if isinstance(value, dict):
        if any(key in value for key in ("rows", "measurement", "prompt", "operands", "generated_token_ids", *COUNTS)):
            raise ValueError("Outer action receipt contains protected per-row evidence")
        for item in value.values():
            check_action_receipt(item)
    elif isinstance(value, list):
        for item in value:
            check_action_receipt(item)


def scope(root, name, relative, mode):
    path = _inside(root, relative)
    if not path.is_file():
        return {"name": name, "receipt": relative, "status": "UNMEASURED", "resources": None}
    body = read(path)
    resources = body.get("resources") if mode == "aggregate" else (body.get("execution") or {}).get("resources")
    if not resources:
        return {"name": name, "receipt": relative, "sha256": sha(path), "status": "UNMEASURED", "resources": None}
    values = numeric_tree(resources)
    return {"name": name, "receipt": relative, "sha256": sha(path),
        "status": body.get("status", body.get("disposition", "RECORDED")), "resources": values}


def compute(root, additional=()):
    """One enclosing receipt per sequential action; no nested worker summation."""
    specs = [(f"{arm}-smoke", f"{arm}/smoke/smoke-gate.json", "aggregate") for arm in ARMS]
    specs += [(f"{arm}-training", f"{arm}/training-process.json", "process") for arm in ARMS]
    specs += [(f"{arm}-{action}", f"execution/{arm}-{action}.json", "process")
              for arm in ARMS for action in ("assess", "qualify", "verify")]
    for declaration in additional:
        label, relative = declaration.split("=", 1)
        if any(label == name or relative == source for name, source, _ in specs):
            raise ValueError("Duplicate accounting scope")
        specs.append((label, relative, "process"))
    entries = [scope(root, *spec) for spec in specs]
    measured = [entry for entry in entries if entry["resources"] is not None]
    digests = [entry["sha256"] for entry in measured]
    if len(digests) != len(set(digests)):
        raise ValueError("One process receipt was supplied as multiple accounting scopes")
    totals = {}
    for field in ("wall_seconds", "cpu_seconds"):
        observed = [e["resources"].get(field) for e in measured]
        totals[field+"_observed_scope_sum"] = sum(x for x in observed if x is not None) if observed else None
    peak = [e["resources"].get("peak_rss_bytes") for e in measured]
    totals["maximum_observed_conservative_peak_rss_bytes"] = max((v for v in peak if v is not None), default=None)
    return {"schema": "weightbridge.phase_iiir.public_compute.v1", "scopes": entries, "totals": totals,
        "exact_total_cpu_seconds": None, "unmeasured_scope_count": len(entries)-len(measured),
        "scope": "Each smoke aggregate, main-training tree, and explicitly listed enclosing assessment/qualification/replay tree once. Child receipts excluded from sums. Software tests, reporting and unlisted preflight costs excluded.",
        "limitation": "Observed process-tree CPU is a lower bound; short-lived unobserved processes are not ruled out. Additional scopes must be disjoint enclosing actions, never nested workers."}


def continuation_comparison(before, after):
    """Report scoped replay differences without claiming endpoint equivalence."""
    a, b = before["measurement"], after["measurement"]
    identities = ("id", "prompt", "answer", "operands", "wording", "order", "comparison_id", "pair_id")
    if (a["endpoint"] != b["endpoint"] or len(a["rows"]) != len(b["rows"])
            or any(any(x.get(k) != y.get(k) for k in identities)
                   for x, y in zip(a["rows"], b["rows"], strict=True))):
        raise ValueError("Continuation comparison requires identical endpoint and practice/development rows")
    pairs = list(zip(a["rows"], b["rows"], strict=True))
    return {"presentations": len(pairs),
        "token_sequence_differences": sum(x["generated_token_ids"] != y["generated_token_ids"] for x, y in pairs),
        "full_text_differences": sum(x["text"] != y["text"] for x, y in pairs),
        "raw_first_line_differences": sum(x["text"].partition("\n")[0] != y["text"].partition("\n")[0] for x, y in pairs),
        "stop_reason_differences": sum(x["stop_reason"] != y["stop_reason"] for x, y in pairs),
        "semantic_correctness_differences": sum(x["correct"] != y["correct"] for x, y in pairs),
        "validity_differences": sum(x["valid"] != y["valid"] for x, y in pairs),
        "scope": "Adapter versus independently loaded merged checkpoint on the same authorized practice/development presentations. Identical scored first lines do not imply identical full continuations."}


def arm_summary(root, arm):
    path = root/arm
    cfg = read(path/"protocol.json") if (path/"protocol.json").is_file() else {}
    result = {"rank": cfg.get("adapter", {}).get("rank"), "alpha": cfg.get("adapter", {}).get("alpha"),
        "training_status": "UNMEASURED", "training_counts": None, "trainable_parameters": None,
        "training_body_resources": None, "selected_checkpoint": None, "checkpoints": {}, "merged": None,
        "independent_stages": {stage: None for stage in STAGES},
        "adapter_development_retention": None,
        "adapter_vs_merged_continuations": {stage: None for stage in ("practice", "development")}}
    for name in ("completed.json", "failed.json"):
        source = path/"training"/name
        if source.is_file():
            body = read(source)
            result.update(training_status=body["status"], training_counts=body.get("counts"),
                          training_body_resources=body.get("resources"))
    inventory = path/"training/inventory.json"
    if inventory.is_file():
        result["trainable_parameters"] = read(inventory).get("trainable_parameters")
    assessment = read(path/"assessment/assessment.json") if (path/"assessment/assessment.json").is_file() else None
    result["selected_checkpoint"] = assessment.get("winner") if assessment else None
    for step in (64, 128):
        candidate = path/"training/checkpoints"/f"step-{step:04d}"
        receipt = candidate/"receipt.json"
        record = None
        if receipt.is_file():
            value = read(receipt)
            effective = value["effective_update"]
            adapter = candidate/"adapter.safetensors"
            record = {"step": step, "receipt_sha256": sha(receipt), "adapter_sha256": value["sha256"],
                "adapter_bytes": adapter.stat().st_size if adapter.is_file() else None,
                "trainable_parameters": value["trainable_parameters"], "training_counts": value["counts"],
                "training_loss": value["training_loss"], "effective_unrounded_norm": effective["proposed_update_norm"],
                "BF16_preview_norm": effective["stored_update_norm"], "BF16_preview_changed_entries": effective["changed_entries"],
                "rounding_residual_norm": effective["rounding_residual_norm"], "practice": None, "development": None}
            if assessment:
                scored = assessment["candidates"].get(f"run-0-step-{step:04d}")
                if scored is not None:
                    record["practice"] = protected(scored.get("practice_gate"))
                    record["development"] = protected(scored.get("development_gate"))
                    record["eligible"] = scored["eligible"]
        result["checkpoints"][str(step)] = record
    merged = path/"merge-receipt.json"
    if merged.is_file():
        value = read(merged)
        result["merged"] = {key: value.get(key) for key in ("candidate_sha256", "base_sha256", "factor_sha256",
            "proposed_update_norm", "stored_update_norm", "rounding_residual_norm", "changed_entries")}
        result["merged"]["receipt_sha256"] = sha(merged)
    for stage in STAGES:
        gate = path/"assessment"/f"{stage}-gate.json"
        if gate.is_file():
            result["independent_stages"][stage] = protected(read(gate))
    if result["selected_checkpoint"]:
        for stage in ("practice", "development"):
            before = path/"assessment/evaluation"/f"{stage}-{result['selected_checkpoint']}.json"
            after = path/"assessment/evaluation"/f"merged-{stage}.json"
            if before.is_file() and after.is_file():
                result["adapter_vs_merged_continuations"][stage] = {
                    **continuation_comparison(read(before), read(after)),
                    "adapter_measurement_sha256": sha(before), "merged_measurement_sha256": sha(after)}
    paired = path/"adapter-development-retention.json"
    if paired.is_file():
        result["adapter_development_retention"] = protected(read(paired))
    return result


def export(study, out, *, additional_scopes=()):
    study, out = Path(study).resolve(), Path(out).resolve()
    if out.exists() or out.is_relative_to(study) or study.is_relative_to(out):
        raise ValueError("Export requires a distinct new write-once version directory")
    out.mkdir(parents=True)
    exported = {}
    def copy(relative):
        source = _inside(study, relative)
        if source.is_file():
            target = out/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(source.read_bytes())
            exported[relative] = {"source_sha256": sha(source), "public_sha256": sha(target), "mode": "exact_public_scope"}
    def project(relative):
        source = _inside(study, relative)
        if source.is_file():
            digest = sha(source)
            target = out/relative
            write(target, {"private_receipt_sha256": digest, "aggregate": protected(read(source)),
                           "projection": "Protected rows, identifiers and private paths omitted; original host receipt retained privately."})
            exported[relative] = {"source_sha256": digest, "public_sha256": sha(target), "mode": "protected_aggregate_projection"}
    for name in ("protocol.json", "architecture-inventory.json", "material/manifest.json"):
        copy(name)
    action_paths = [f"execution/{arm}-{action}.json" for arm in ARMS for action in ("assess", "qualify", "verify")]
    action_paths += [declaration.split("=", 1)[1] for declaration in additional_scopes]
    for relative in action_paths:
        source = _inside(study, relative)
        if source.is_file():
            check_action_receipt(read(source))
            copy(relative)
    project("replication-verification.json")
    for arm in ARMS:
        copy(f"{arm}/adapter-development-retention.json")  # shared public development only, never a bank
        for name in ("protocol.json", "material/manifest.json", "prospective-data-order.json", "main-data-order.json",
                     "main-authorization-plan.json", "main-authorization.json", "smoke-authorization.json", "merge-authorization.json",
                     "main-launch-claim.json", "smoke-launch-claim.json", "training-process.json", "main-execution.json",
                     "merge-receipt.json", "merge-receipt.verification.json"):
            copy(f"{arm}/{name}")
        for name in ("started.json", "completed.json", "failed.json", "inventory.json"):
            copy(f"{arm}/training/{name}")
        for path in sorted((study/arm/"training/steps").glob("*.json")):
            copy(path.relative_to(study).as_posix())
        for step in (64, 128):
            copy(f"{arm}/training/checkpoints/step-{step:04d}/receipt.json")
        for stage in ("practice", "development"):
            copy(f"{arm}/assessment/rosters/{stage}.json")
            for suffix in ("parent", "run-0-step-0064", "run-0-step-0128"):
                copy(f"{arm}/assessment/evaluation/{stage}-{suffix}.json")
            copy(f"{arm}/assessment/evaluation/merged-{stage}.json")
        for name in ("assessment.json", "finalist-lock.json", "finalist-reference.json", "merged-development-gate.json"):
            copy(f"{arm}/assessment/{name}")
        for name in ("independent.json", "bank-b.json", *(f"{s}-gate.json" for s in STAGES),
                     *(f"bank-b-{s}-gate.json" for s in STAGES)):
            project(f"{arm}/assessment/{name}")
        project(f"{arm}/final-verification/final-verification.json")
        project(f"{arm}/comparison-verification/comparison-verification.json")
        copy(f"{arm}/smoke/smoke-gate.json")  # exclusively disposable training-only material
        for name in ("started.json", "zero/zero-init-check.json", "zero/parent-measurement.json",
                     "training/started.json", "training/completed.json", "training/failed.json", "training/inventory.json",
                     "training/checkpoints/step-0032/receipt.json", "fresh-1/fresh-adapter-check.json",
                     "fresh-2/fresh-adapter-check.json", "zero-process.json", "training-process.json",
                     "fresh-1-process.json", "fresh-2-process.json"):
            copy(f"{arm}/smoke/{name}")
        for path in sorted((study/arm/"smoke/training/steps").glob("*.json")):
            copy(path.relative_to(study).as_posix())
    accounting = compute(study, additional_scopes)
    write(out/"compute-accounting.json", accounting)
    write(out/"arm-summary.json", {arm: arm_summary(study, arm) for arm in ARMS})
    result = {"schema": "weightbridge.phase_iiir.public_export.v1", "reporter_sha256": sha(__file__),
        "derived_artifact_sha256": {name: sha(out/name) for name in ("compute-accounting.json", "arm-summary.json")},
        "artifacts": exported, "weight_files_published": False, "protected_rows_published": False,
        "source_rosters_or_candidates_modified": False}
    write(out/"export-manifest.json", result)
    return result


def self_test():
    import tempfile
    hostile = {"passed": False, "prohibited_regressions": ["secret-id"], "rows": [{"text": "secret"}],
        "parent": {"correct": 2, "strict_comparison_ids": ["hidden-family"], "prompt": "secret"},
        "unknown": "secret", "error": {"type": "ValueError", "message": "secret answer"},
        "stages": {"hidden": {"rank4_retention": {"correctness_disagreements": ["secret-id"], "passed": False}}}}
    result = protected(hostile)
    assert "secret" not in json.dumps(result) and "hidden-family" not in json.dumps(result)
    assert result["prohibited_regressions_count"] == 1 and result["parent"]["strict_comparison_ids_count"] == 1
    row = {"id": "fixture", "prompt": "synthetic fixture", "answer": "1", "operands": ["1"], "wording": 0,
        "order": None, "comparison_id": None, "pair_id": None, "generated_token_ids": [1, 2, 3],
        "text": "1\ncontinuation A", "stop_reason": "token_budget", "correct": True, "valid": True}
    before = {"measurement": {"endpoint": {"id": "fixture"}, "rows": [row]}}
    after = {"measurement": {"endpoint": {"id": "fixture"}, "rows": [{**row,
        "generated_token_ids": [1, 2, 4], "text": "1\ncontinuation B"}]}}
    differences = continuation_comparison(before, after)
    assert differences["token_sequence_differences"] == differences["full_text_differences"] == 1
    assert differences["raw_first_line_differences"] == differences["semantic_correctness_differences"] == 0
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        costs = compute(root)
        assert costs["totals"]["cpu_seconds_observed_scope_sum"] is None
        empty = arm_summary(root, "rank4")
        assert empty["training_status"] == "UNMEASURED" and empty["training_counts"] is None
        assert all(value is None for value in empty["independent_stages"].values())
        write(root/"rank8/training-process.json", {"execution": {"resources": {"wall_seconds": 4, "cpu_seconds": 8, "peak_rss_bytes": 20}}})
        write(root/"rank8/training/completed.json", {"resources": {"wall_seconds": 3, "cpu_seconds": 7, "peak_rss_bytes": 15}})
        costs = compute(root)
        assert costs["totals"]["cpu_seconds_observed_scope_sum"] == 8
        assert costs["unmeasured_scope_count"] == 9
        try:
            compute(root, ["duplicate=rank8/training-process.json"])
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate resource scope accepted")
        artifact = root/"artifact-study"
        allowed = ("rank8/smoke/zero/parent-measurement.json", "rank8/smoke/training/steps/0001.json",
                   "rank8/smoke/fresh-1/fresh-adapter-check.json", "rank8/smoke/zero-process.json",
                   "rank8/main-launch-claim.json", "execution/rank8-assess.json", "rank4/adapter-development-retention.json")
        for relative in allowed:
            write(artifact/relative, {"schema": "self-test", "passed": True})
        write(artifact/"rank8/assessment/private-independent-rosters/hidden.json", [{"prompt": "secret"}])
        weights = artifact/"rank8/smoke/training/checkpoints/step-0032/adapter.safetensors"
        weights.parent.mkdir(parents=True); weights.write_bytes(b"private-weight-placeholder")
        result = export(artifact, root/"public")
        assert all(sha(root/"public"/name) == digest for name, digest in result["derived_artifact_sha256"].items())
        assert all(relative in result["artifacts"] for relative in allowed)
        assert not list((root/"public").rglob("*.safetensors"))
        assert not (root/"public/rank8/assessment/private-independent-rosters").exists()
        try:
            check_action_receipt({"outcome": {"measurement": {"rows": []}}})
        except ValueError:
            pass
        else:
            raise AssertionError("Raw action output accepted")
    return {"sanitization": "PASS", "partial_nulls": "PASS", "nonoverlap_accounting": "PASS",
        "explicit_export_boundary": "PASS", "continuation_scope": "PASS"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--scope", action="append", default=[])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    else:
        if args.study is None or args.out is None:
            parser.error("--study and --out are required")
        result = export(args.study, args.out, additional_scopes=args.scope)
        print(json.dumps({"exported_artifacts": len(result["artifacts"]), "reporter_sha256": result["reporter_sha256"]}))
