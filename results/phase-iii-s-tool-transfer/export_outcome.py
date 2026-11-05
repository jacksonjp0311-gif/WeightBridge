"""Report-only Phase S projections and public hash validation; no model execution.

Examples (a new output directory is required for every snapshot)::

    python export_outcome.py --study PRIVATE/study-v1 --ledger PRIVATE/execution --out outcome-v1
    python export_outcome.py --verify outcome-v1
    python export_outcome.py --self-test

Later host gate receipts may be explicitly added with --gate LABEL=RELATIVE.json.
Only whitelisted metadata schemas are accepted. Protected workers and rosters
are never discovered or read. Public validation checks export integrity, not
the scientific truth of private model evidence or authority to admit a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import tempfile


MANIFEST_SHA = "ccc217d9139d214ecb241acb3099006a9cc35a0b3f85a6f49784adc3de9a813a"
ARMS = ("sequential", "original-joint")
TOOLS = {"search_files", "read_file", "run_tests", "clarify", "no_action"}
GATE_SCHEMAS = {
    "weightbridge.phase_iiis.behavior_gate.v1",
    "weightbridge.phase_iiis.arm_lock.v1",
    "weightbridge.phase_iiis.all_arms_locked.v1",
    "weightbridge.phase_iiis.final_verification.v1",
    "weightbridge.supervised_lora_merge_receipt.v1",
    "weightbridge.phase_iiis.reconstruction.v1",
}
FORBIDDEN = {"rows", "records", "outputs", "prompt", "answer", "oracle_answer", "completion",
    "text", "fixture", "scenario", "gold_action", "parsed_action", "operands", "request",
    "messages", "measurement", "parent_measurement", "zero_measurement", "disabled_measurement",
    "generated_token_ids", "prefix_token_ids", "input_ids", "target_ids", "tokenization",
    "command", "message", "workers", "ledger", "path", "protocol_path", "checkpoints",
    "tensor_values", "weights", "state_dict", "optimizer_state", "adapter_factors"}
STRING_FIELDS = {"schema", "status", "disposition", "stage", "mode", "arm", "representation",
    "error", "type", "name", "task", "tensor_name", "module_name", "role", "dtype", "factor_dtype",
    "base_dtype", "merge_math", "method", "failed_stage", "terminal_kind", "budget_failure"}
SAFE_DICT_LISTS = {"parameter_changes", "trainable_tensors", "modules", "tensor_receipts",
    "reference_gates", "required_retention", "required_comparator_retention", "scopes"}
IDENTITY = ("schema", "status", "protocol_sha256", "authorization_sha256", "bundle_sha256",
    "manifest_sha256", "task_manifest_sha256", "parent_sha256", "checkpoint_sha256", "adapter_sha256",
    "representation_sha256", "initial_factor_sha256", "factor_sha256", "run_index", "mode", "smoke")


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def dumps(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+"\n").encode()


def private_path(value):
    return bool(re.match(r"^[A-Za-z]:[/\\]", value) or value.startswith(("\\\\", "/")))


def relative(value):
    path = PurePosixPath(str(value).replace("\\", "/"))
    if not path.parts or path.is_absolute() or ":" in str(path) or ".." in path.parts:
        raise ValueError("Only bounded relative artifact names are allowed")
    return str(path)


def project(value):
    """Project known aggregate metadata; all string lists become counts by default.

    This is never applied to a roster or measurement. In particular an empty
    losses list becomes an observed zero count, while an absent stage stays null.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Aggregate projection requires an object")
    result = {}
    for key, item in value.items():
        if private_path(key):
            continue
        if key in FORBIDDEN or "seed" in key.lower() or key.endswith("_path"):
            continue
        if key.endswith("_id"):
            continue
        if isinstance(item, dict):
            result[key] = project(item)
        elif isinstance(item, list):
            if key in ("real_actions", "corrected_real_actions"):
                if not set(item) <= TOOLS:
                    raise ValueError("Unknown tool name in aggregate metadata")
                result[key] = item
            elif key in SAFE_DICT_LISTS and all(isinstance(row, dict) for row in item):
                result[key] = [project(row) for row in item]
            elif key in ("shape", "strict_accuracy_gain_interval", "complete_family_rate_gain_interval"):
                if not all(type(x) in (int, float) and math.isfinite(x) for x in item):
                    raise ValueError("Expected a numerical aggregate vector")
                result[key] = item
            else:
                result[key+"_count"] = len(item)
        elif item is None or type(item) in (bool, int, float):
            result[key] = item
        elif isinstance(item, str) and (key in STRING_FIELDS or key.endswith("sha256")):
            if private_path(item) or re.search(r"[A-Za-z]:[/\\]", item):
                raise ValueError("A private path survived aggregate selection")
            if key.endswith("sha256") and not re.fullmatch(r"[0-9a-f]{64}", item):
                raise ValueError("Malformed artifact hash")
            result[key] = item
        elif isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item):
            # Source-manifest and already-relative input-commitment dictionaries.
            result[key] = item
    return result


def assert_public(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if (key in FORBIDDEN or "seed" in key.lower() or key.endswith(("_id", "_ids", "_path"))
                    or private_path(key)):
                raise ValueError("Forbidden public field")
            assert_public(item)
    elif isinstance(value, list):
        for item in value:
            assert_public(item)
    elif isinstance(value, str) and re.search(r"[A-Za-z]:[/\\]|^\\\\|^/", value):
        raise ValueError("Private absolute path in publication")
    elif type(value) is float and not math.isfinite(value):
        raise ValueError("Nonfinite public measurement")


def selected(record, names):
    return project({name: record[name] for name in names if name in record}) if record is not None else None


class Reader:
    def __init__(self, study):
        self.study, self.inputs, self.missing = Path(study).resolve(), {}, []

    def read(self, name, *, root=None, label=None):
        name = relative(name)
        root = self.study if root is None else Path(root).resolve()
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Input symlink or traversal refused")
        identity = relative(label or name)
        if not path.is_file():
            self.missing.append(identity)
            return None
        payload = path.read_bytes()
        self.inputs[identity] = sha(payload)
        return json.loads(payload)


def evaluation(reader, name):
    record = reader.read(name)
    if record is None:
        return None
    if record.get("task") not in ("tool", "decimal") or record.get("stage") != "development":
        raise ValueError("Automatic output summaries are limited to development")
    summary = selected(record, (*IDENTITY, "task", "representation", "summary", "resources"))
    measurement = record.get("measurement") or {}
    # Work counters only. Neither raw outputs nor native prompt/token records enter export.
    summary["decoding_work"] = selected(measurement, ("counts", "work", "resources"))
    return summary


def training(reader, prefix):
    completed = reader.read(prefix+"/completed.json")
    started = reader.read(prefix+"/started.json")
    failed = reader.read(prefix+"/failed.json")
    inventory = reader.read(prefix+"/inventory.json")
    step_dir = reader.study/prefix/"steps"
    steps = []
    for path in sorted(step_dir.glob("*.json")):
        if not re.fullmatch(r"\d{4}\.json", path.name):
            raise ValueError("Unknown training step artifact")
        record = reader.read(prefix+"/steps/"+path.name)
        steps.append(selected(record, ("step", "loss", "learning_rate", "gradient_norm_before_clip", "counts",
            "resources", "by_category", "task_examples", "unrounded_effective_update_norm")))
    if steps and [r["step"] for r in steps] != list(range(1, len(steps)+1)):
        raise ValueError("Training step evidence is not contiguous")
    checkpoint_dir = reader.study/prefix/"checkpoints"
    checkpoints = []
    for path in sorted(checkpoint_dir.glob("step-*/receipt.json")):
        if not re.fullmatch(r"step-\d{4}", path.parent.name):
            raise ValueError("Unknown checkpoint receipt")
        receipt = reader.read(prefix+"/checkpoints/"+path.parent.name+"/receipt.json")
        checkpoint = project(receipt)
        artifact = path.parent/"adapter.safetensors"
        checkpoint["artifact_bytes"] = artifact.stat().st_size if artifact.is_file() else None
        checkpoints.append(checkpoint)
    return {"state": "COMPLETED" if completed is not None else "FAILED" if failed is not None else
            "INCOMPLETE" if started is not None else "UNMEASURED",
        "started": selected(started, IDENTITY), "completed": selected(completed, (*IDENTITY, "counts",
            "parameter_changes", "all_factors_changed", "all_factors_received_nonzero_gradient",
            "first_step_loss", "last_step_loss", "base_unchanged", "resources")),
        "failure": selected(failed, (*IDENTITY, "error", "counts", "resources", "retry_authorized")),
        "inventory": project(inventory) if inventory is not None else None,
        "steps": steps if steps else None, "saved_checkpoints": checkpoints if checkpoints else None}


def ledger_summary(reader, ledger):
    ledger = Path(ledger)
    scopes, unfinished = [], []
    for path in sorted(ledger.glob("*.receipt.json")):
        record = reader.read(path.name, root=ledger, label="execution/"+path.name)
        if record.get("schema") != "weightbridge.phase_iiis.execution.v1" or record.get("name") != path.name[:-13]:
            raise ValueError("Unknown or mismatched outer execution receipt")
        execution = record.get("execution") or {}
        scopes.append({"name": record["name"], "passed": record.get("passed"),
            "receipt_sha256": reader.inputs["execution/"+path.name], "limits": project(record.get("limits", {})),
            "resources": project(execution["resources"]) if execution.get("resources") is not None else None,
            "exit_code": execution.get("exit_code"), "budget_failure": execution.get("budget_failure"),
            "log_sha256": record.get("log_sha256"), "error": selected(record.get("error"), ("type",))})
    for path in sorted(ledger.glob("*.started.json")):
        name = path.name[:-13]
        if not (ledger/(name+".receipt.json")).exists():
            reader.read(path.name, root=ledger, label="execution/"+path.name)
            unfinished.append(name)
    complete = not unfinished and all(s["resources"] is not None for s in scopes)
    totals = None
    if scopes and all(s["resources"] is not None for s in scopes):
        totals = {"wall_seconds": sum(s["resources"]["wall_seconds"] for s in scopes),
            "cpu_seconds": sum(s["resources"]["cpu_seconds"] for s in scopes),
            "peak_rss_bytes": max(s["resources"]["peak_rss_bytes"] for s in scopes)}
    return {"scopes": scopes, "unfinished_scope_names": unfinished, "complete_accounting": complete,
        "completed_scope_totals": totals, "all_recorded_scope_totals": totals if complete else None,
        "study_execution_complete": None,
        "accounting": "Disjoint owned worker scopes, including monitor cost; inner training and evaluation costs are not added again. Peak is the maximum conservative process-tree measurement, not a sum across stages."}


def export(study, ledger, out, gates=(), *, expected_manifest=MANIFEST_SHA):
    out = Path(out)
    if out.exists():
        raise FileExistsError("Outcome snapshots are write-once")
    reader = Reader(study)
    frozen = reader.read("manifest.json")
    if frozen is None or reader.inputs["manifest.json"] != expected_manifest:
        raise ValueError("Frozen study manifest does not match this experiment")
    artifacts = {"source-manifest.json": frozen["source_sha256"], "baselines.json": {
        parent: {task: evaluation(reader, f"baselines/{parent}/{task}.json") for task in ("tool", "decimal")}
        for parent in ("original", "r8")}}
    artifacts["parent-development-audit.json"] = project(reader.read("parent-development-audit-v1.json",
        root=reader.study.parent, label="preflight/parent-development-audit-v1.json"))
    artifacts["setup-refusal.json"] = {"model_attempt": False,
        "receipt": project(reader.read("freeze-preflight-refusal-v1.json", root=reader.study.parent,
            label="preflight/freeze-preflight-refusal-v1.json")),
        "scope": "Setup-only validation refusal before formal populations or model work; not counted as a model candidate, training run or behavioral failure."}
    for arm in ARMS:
        protocol = reader.read(f"{arm}/protocol.json")
        if protocol is None or protocol.get("task_manifest_sha256") != expected_manifest or protocol.get("source_sha256") != frozen["source_sha256"]:
            raise ValueError("Arm protocol/source identity differs from formal freeze")
        artifacts[f"{arm}-training.json"] = training(reader, f"{arm}/training")
        artifacts[f"{arm}-smoke-training.json"] = training(reader, f"{arm}/smoke/training")
        artifacts[f"{arm}-metadata.json"] = {
            "protocol_sha256": reader.inputs[f"{arm}/protocol.json"],
            "parent_sha256": protocol.get("checkpoint_sha256"),
            "recipe": selected(protocol, ("adapter", "training", "resources")),
            "data_order": project(reader.read(f"{arm}/prospective-data-order.json")),
            "main_authorization": project(reader.read(f"{arm}/main-authorization.json")),
            "main_admission": project(reader.read(f"{arm}/main-admission.json")),
            "main_claim": project(reader.read(f"{arm}/main-launch-claim.json")),
            "smoke_gate": project(reader.read(f"{arm}/smoke-gate.json")),
            "main_training_validation": project(reader.read(f"{arm}/main-training-validation.json")),
            "execution": selected(reader.read(f"{arm}/run-arm.json"), (*IDENTITY, "arm", "completed", "status",
                "failed_stage", "protected_evaluation_opened", "behavioral_qualification", "production_authority", "lineage_authority"))}
        artifacts[f"{arm}-development.json"] = {str(step): {"diagnostic_only": step == 64,
            "tool": evaluation(reader, f"{arm}/development/step-{step:04d}/tool.json"),
            "decimal": evaluation(reader, f"{arm}/development/step-{step:04d}/decimal.json"),
            "host_gate": project(reader.read(f"{arm}/assessment/step-{step:04d}.json"))}
            for step in (64, 128)}
    availability = {"independent_qualification": None, "sealed_confirmation": None, "final_replay": None}
    for label, name in gates:
        if not re.fullmatch(r"[a-z0-9_-]+", label) or f"gate-{label}.json" in artifacts:
            raise ValueError("Invalid or duplicate gate label")
        name = relative(name)
        if name.endswith(("/tool.json", "/decimal.json", "/rows.json")) or "material" in PurePosixPath(name).parts:
            raise ValueError("An explicitly supplied gate cannot be a roster or worker measurement")
        value = reader.read(name)
        if value is not None and value.get("schema") not in GATE_SCHEMAS:
            raise ValueError("Explicit publication input is not a recognized host metadata receipt")
        if value is not None and value.get("task_manifest_sha256", expected_manifest) != expected_manifest:
            raise ValueError("Explicit gate belongs to another study")
        artifacts[f"gate-{label}.json"] = project(value)
        if value is not None:
            stage = {"qualification": "independent_qualification", "confirmation": "sealed_confirmation"}.get(value.get("stage"))
            if value.get("schema") == "weightbridge.phase_iiis.final_verification.v1":
                stage = "final_replay"
            if stage:
                availability[stage] = availability[stage] or {}
                availability[stage][value["arm"]] = {"passed": value.get("passed"),
                    "disposition": value.get("disposition"), "public_gate_artifact": f"gate-{label}.json"}
    artifacts["execution.json"] = ledger_summary(reader, ledger)
    artifacts["stage-availability.json"] = {**availability, "lineage_admission": False,
        "interpretation": "No result is inferred from a missing stage. Explicit host gate artifacts, when supplied, are authoritative for measured later stages. This exporter does not grant evaluation or lineage authority."}
    for value in artifacts.values():
        assert_public(value)
    out.mkdir(parents=True)
    hashes = {}
    for name, value in artifacts.items():
        payload = dumps(value)
        (out/name).write_bytes(payload)
        hashes[name] = sha(payload)
    receipt = {"schema": "weightbridge.phase_iiis.public_outcome.v1", "study_manifest_sha256": expected_manifest,
        "input_sha256": reader.inputs, "missing_inputs": reader.missing, "artifacts": hashes,
        "exporter_sha256": sha(Path(__file__).read_bytes()), "raw_rosters_published": False,
        "private_outputs_published": False, "model_weights_published": False, "model_execution": False,
        "scope": "Redacted aggregate projections with original private receipt byte hashes. Public hash checks do not replace host scientific verification or reproduce private predictions."}
    assert_public(receipt)
    (out/"export-manifest.json").write_bytes(dumps(receipt))
    return verify(out)


def verify(out):
    out = Path(out)
    receipt = json.loads((out/"export-manifest.json").read_bytes())
    if receipt.get("schema") != "weightbridge.phase_iiis.public_outcome.v1":
        raise ValueError("Unknown public export schema")
    assert_public(receipt)
    if set(p.name for p in out.iterdir()) != set(receipt["artifacts"]) | {"export-manifest.json"}:
        raise ValueError("Uncommitted or missing public files")
    for name, digest in receipt["artifacts"].items():
        if relative(name) != name or PurePosixPath(name).name != name or (out/name).is_symlink():
            raise ValueError("Unsafe public artifact path")
        payload = (out/name).read_bytes()
        if sha(payload) != digest:
            raise ValueError("Public artifact byte hash mismatch")
        assert_public(json.loads(payload))
    return {"passed": True, "artifacts": len(receipt["artifacts"]),
        "study_manifest_sha256": receipt["study_manifest_sha256"], "model_execution": False,
        "scientific_outcomes_reverified": False}


def self_test():
    raw = {"lost_schema_valid": ["PRIVATE_ROW"], "complete_family_ids": ["PRIVATE_FAMILY"],
        "private_seeds": {"seed": 123}, "records": [{"text": "PRIVATE_TEXT"}],
        "prompt": "PRIVATE_PROMPT", "path": "C:/private/roster.json", "never_measured": None,
        "corrected_real_actions": ["search_files"], "strict_accuracy_gain_interval": [.1, .2]}
    clean = project(raw)
    assert_public(clean)
    assert clean["lost_schema_valid_count"] == clean["complete_family_ids_count"] == 1
    assert clean["never_measured"] is None and "PRIVATE" not in json.dumps(clean)
    for bad in ({"records": []}, {"rows": []}, {"seed": 1}, {"where": "C:/private/model"}):
        try:
            assert_public(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Public sensitive material was accepted")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        study, ledger, out = root/"study-v1", root/"execution", root/"public"
        study.mkdir(); ledger.mkdir()
        manifest = {"source_sha256": {"module.py": "a"*64}}
        payload = dumps(manifest); expected = sha(payload)
        (study/"manifest.json").write_bytes(payload)
        for arm in ARMS:
            (study/arm).mkdir()
            (study/arm/"protocol.json").write_bytes(dumps({"task_manifest_sha256": expected,
                "source_sha256": manifest["source_sha256"], "checkpoint_sha256": "b"*64}))
        train = study/"sequential/training"
        (train/"steps").mkdir(parents=True)
        (train/"started.json").write_bytes(dumps({"schema": "training.v1", "parent_sha256": "b"*64}))
        (train/"steps/0001.json").write_bytes(dumps({"step": 1, "loss": 2.5, "learning_rate": .0003,
            "by_category": {"tool": {"supervised_tokens": 8, "negative_log_likelihood_sum": 20.}},
            "resources": {"wall_seconds": 2., "cpu_seconds": 5., "peak_rss_bytes": 100}}))
        (ledger/"test.receipt.json").write_bytes(dumps({"schema": "weightbridge.phase_iiis.execution.v1",
            "name": "test", "passed": True, "execution": {"exit_code": 0, "budget_failure": None,
                "resources": {"wall_seconds": 3., "cpu_seconds": 6., "peak_rss_bytes": 200}},
            "command": ["C:/PRIVATE/python.exe", "PRIVATE_PROMPT"]}))
        (ledger/"active.started.json").write_bytes(dumps({"command": ["PRIVATE_COMMAND"]}))
        result = export(study, ledger, out, expected_manifest=expected)
        assert result["passed"]
        measured = json.loads((out/"sequential-training.json").read_bytes())
        assert measured["completed"] is None and measured["state"] == "INCOMPLETE"
        assert measured["steps"][0]["by_category"]["tool"]["supervised_tokens"] == 8
        cost = json.loads((out/"execution.json").read_bytes())
        assert cost["all_recorded_scope_totals"] is None and cost["unfinished_scope_names"] == ["active"]
        assert cost["completed_scope_totals"]["cpu_seconds"] == 6.  # Not 6+inner 5.
        assert all("PRIVATE" not in p.read_text() for p in out.glob("*.json"))
        try:
            export(study, ledger, out, expected_manifest=expected)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Write-once evidence overwritten")
        (out/"baselines.json").write_bytes(b"{}")
        try:
            verify(out)
        except ValueError:
            pass
        else:
            raise AssertionError("Artifact tampering accepted")
    return {"redaction": "PASS", "sensitive_field_refusal": "PASS", "missing_stages_null": "PASS",
        "partial_snapshot": "PASS", "category_token_preservation": "PASS", "non_nested_accounting": "PASS",
        "interrupted_scope_unknown": "PASS", "write_once": "PASS", "hash_tamper_refusal": "PASS", "model_execution": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gate", action="append", default=[], metavar="LABEL=RELATIVE.json")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    elif args.verify:
        print(json.dumps(verify(args.verify)))
    elif args.study and args.ledger and args.out:
        gates = [value.split("=", 1) for value in args.gate]
        if any(len(pair) != 2 for pair in gates):
            parser.error("Each --gate must be LABEL=RELATIVE.json")
        print(json.dumps(export(args.study, args.ledger, args.out, gates)))
    else:
        parser.error("Choose --self-test, --verify, or --study/--ledger/--out")
