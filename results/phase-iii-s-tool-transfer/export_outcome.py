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
    "weightbridge.phase_iiis.merge_verification.v1",
    "weightbridge.phase_iiis.host_exact_reconstruction.v1",
    "weightbridge.phase_iiis.public_worker_counters.v1",
    "weightbridge.phase_iiis.host_historical_retention.v1",
    "weightbridge.phase_iiis.private_stage_replay.v1",
    "weightbridge.phase_iiis.private_protected_execution.v3",
    "weightbridge.phase_iiis.formal_stage_closure.v3",
    "weightbridge.phase_iiis.closed_historical_metadata_review.v1",
}
FORBIDDEN = {"rows", "records", "outputs", "prompt", "answer", "oracle_answer", "completion",
    "text", "fixture", "scenario", "gold_action", "parsed_action", "operands", "request",
    "messages", "measurement", "parent_measurement", "zero_measurement", "disabled_measurement",
    "generated_token_ids", "prefix_token_ids", "input_ids", "target_ids", "tokenization",
    "command", "message", "workers", "ledger", "path", "protocol_path", "checkpoints",
    "tensor_values", "weights", "state_dict", "optimizer_state", "adapter_factors"}
STRING_FIELDS = {"schema", "status", "disposition", "stage", "action", "mode", "arm", "representation",
    "error", "type", "name", "label", "task", "tensor", "tensor_name", "module_name", "role", "dtype", "factor_dtype",
    "base_dtype", "merge_math", "relative_norm_scope", "method", "failed_stage", "terminal_kind", "budget_failure"}
SAFE_DICT_LISTS = {"parameter_changes", "trainable_tensors", "modules", "tensor_receipts",
    "reference_gates", "required_retention", "required_comparator_retention", "scopes", "selected_workers", "changed_tensors"}
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
            if key in ("real_actions", "corrected_real_actions", "semantic_error_real_actions"):
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


def adapter_merged_diagnostic(record, *, manifest_sha256, source_sha256, protocol_sha256, arm):
    """Keep endpoint agreement distinct from exact full native output identity."""
    if record is None:
        return None
    expected = {"schema": "weightbridge.phase_iiis.adapter_merged_development_diagnostic.v1",
        "arm": arm, "step": 128, "task_manifest_sha256": manifest_sha256,
        "source_sha256": source_sha256, "protocol_sha256": protocol_sha256,
        "diagnostic_only": True, "additional_acceptance_gate": False,
        "evidence_validation_passed": True, "protected_banks_opened": False,
        "model_execution_by_reviewer": False}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Adapter/merged diagnostic differs from its frozen development binding")
    if set(record.get("suites", {})) != {"tool", "decimal"}:
        raise ValueError("Adapter/merged diagnostic needs both declared endpoint suites")
    result = selected(record, (*expected, "adapter_parent_checkpoint_sha256", "adapter_sha256",
        "merged_checkpoint_sha256", "scientific_sources_verified", "input_sha256", "resources",
        "production_authority", "lineage_authority"))
    names = {"generated_token_ids": "generated_tokens_equal", "native_end_token_id": "native_terminator_equal",
        "prefix_token_ids": "native_prefix_tokens_equal", "prefix_tokens": "prefix_length_equal",
        "stop_reason": "stop_reason_equal", "text": "complete_text_equal",
        "unexpected_special_token_ids": "unexpected_special_tokens_equal"}
    result["suites"] = {}
    for task, suite in record["suites"].items():
        n, equal = suite.get("presentations"), suite.get("complete_native_rows_equal")
        fields = suite.get("native_field_equal_counts") or {}
        if (type(n) is not int or n <= 0 or type(equal) is not int or not 0 <= equal <= n
                or set(fields) != set(names) or any(type(v) is not int or not 0 <= v <= n for v in fields.values())
                or suite.get("complete_native_replay_equal") != (equal == n)
                or equal > min(fields.values())):
            raise ValueError("Malformed adapter/merged native equality counters")
        retained = selected(suite, ("presentations", "endpoint_contract_identical", "retention_passed",
            "behavioral_observation_flags_identical", "behavioral_flag_changes", "adapter_summary", "merged_summary",
            "strict_metric_gain_over_adapter", "complete_family_gain_over_adapter", "complete_native_replay_equal",
            "complete_native_rows_equal"))
        retained["native_field_equal_counts"] = {names[k]: v for k, v in fields.items()}
        retained["complete_native_streams_different"] = n-equal
        result["suites"][task] = retained
    result["interpretation"] = "Endpoint agreement does not imply identical unrestricted continuation. This diagnostic creates no additional acceptance rule; exact replay must compare each representation with its own recorded outputs."
    assert_public(result)
    return result


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


def category_exposure(steps):
    """Token shares are measured per-step sums, never assumed from row sampling."""
    if not steps:
        return None
    totals = {}
    for step in steps:
        if not isinstance(step.get("by_category"), dict):
            return None
        for category, values in step["by_category"].items():
            target = totals.setdefault(category, {})
            for key, value in values.items():
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("Nonnumerical training category aggregate")
                target[key] = target.get(key, 0) + value
    denominator = {key: sum(v[key] for v in totals.values()) if all(key in v for v in totals.values()) else None for key in
        ("examples", "supervised_tokens", "weighted_supervised_tokens", "weighted_token_nll_sum")}
    for values in totals.values():
        for metric, base in (("row_fraction", "examples"), ("supervised_token_fraction", "supervised_tokens"),
                             ("weighted_token_fraction", "weighted_supervised_tokens")):
            values[metric] = values[base]/denominator[base] if base in values and denominator[base] else None
        tokens = values.get("weighted_supervised_tokens")
        values["measured_mean_token_nll"] = values["weighted_token_nll_sum"]/tokens if tokens and "weighted_token_nll_sum" in values else None
    for category, values in totals.items():
        per_step = []
        for step in steps:
            observed = step["by_category"]
            if not all("weighted_supervised_tokens" in v for v in observed.values()):
                per_step = []
                break
            denominator_step = sum(v["weighted_supervised_tokens"] for v in observed.values())
            if denominator_step <= 0:
                per_step = []
                break
            per_step.append(observed.get(category, {}).get("weighted_supervised_tokens", 0)/denominator_step)
        values["mean_per_step_objective_weight"] = sum(per_step)/len(per_step) if per_step else None
    return {"measured_steps": len(steps), "categories": totals, "totals": denominator,
        "scope": "Exposure fractions sum tokens across completed steps. Mean objective weights average each category's token fraction within each separately normalized effective batch; they need not equal global token fractions. Neither quantity measures its share of parameter-gradient influence."}


def legacy_sections(summary):
    if summary is None:
        return {"negative_decimal": None, "general": None, "combined_decode_termination": None}
    groups = summary.get("groups", {})
    target = {k: v for k, v in groups.items() if k in ("negative-close", "negative-far")}
    general = {k: v for k, v in groups.items() if k not in target}
    def counts(rows):
        return {key: sum(v[key] for v in rows.values()) for key in ("presentations", "correct", "valid")} if rows else None
    negative = counts(target)
    if negative is not None:
        negative.update({key: summary.get(key) for key in ("strict_comparisons", "comparisons", "complete_families", "families")})
    return {"negative_decimal": negative, "general": counts(general),
        "combined_decode_termination": summary.get("decode_termination")}


def results_tables(artifacts):
    """Derive a compact matrix only from already-redacted public aggregates."""
    def bundle(tool, decimal, *, representation, identity, source):
        if tool is None and decimal is None:
            return None
        return {"representation": representation, "representation_sha256": identity,
            "tool": tool, **legacy_sections(decimal), "source_artifact": source}
    stored, adapters, exposures = {}, {}, {}
    for parent in ("original", "r8"):
        item = artifacts["baselines.json"][parent]
        tool, decimal = item["tool"], item["decimal"]
        stored[parent] = bundle(tool.get("summary") if tool else None, decimal.get("summary") if decimal else None,
            representation="stored_checkpoint", identity=(tool or decimal or {}).get("checkpoint_sha256"), source="baselines.json")
    for arm in ARMS:
        stored[arm+"-merged"] = None
        adapters[arm] = {}
        for step, item in artifacts[f"{arm}-development.json"].items():
            tool, decimal, gate = item["tool"], item["decimal"], item["host_gate"]
            measured = bundle(tool.get("summary") if tool else None, decimal.get("summary") if decimal else None,
                representation="fp32_adapter_on_bf16_parent", identity=(tool or decimal or {}).get("adapter_sha256"),
                source=f"{arm}-development.json")
            adapters[arm][step] = {"endpoint_summary": measured, "diagnostic_only": step == "64", "host_gate": gate}
        exposures[arm] = {"main": category_exposure(artifacts[f"{arm}-training.json"]["steps"]),
            "smoke": category_exposure(artifacts[f"{arm}-smoke-training.json"]["steps"]),
            "inherited_r_main_cost": artifacts[f"{arm}-metadata.json"]["inherited_r_main_cost"]}
    for name, gate in artifacts.items():
        if not name.startswith("gate-") or not isinstance(gate, dict) or gate.get("stage") != "merged-development":
            continue
        arm = gate["arm"]
        if stored[arm+"-merged"] is not None:
            raise ValueError("Multiple merged-development states cannot share one matrix row")
        decimal = gate["decimal"]["reference_gates"][0]["candidate"]
        general = gate["general"]["reference_gates"][0]["candidate"]
        stored[arm+"-merged"] = {"representation": "stored_checkpoint", "representation_sha256": gate["representation_sha256"],
            "tool": gate["tool"]["own_parent_retention"]["candidate"], "negative_decimal": decimal, "general": general,
            "host_gate_passed": gate["passed"], "disposition": gate["disposition"], "source_artifact": name}
    evidence = {}
    for name, value in artifacts.items():
        candidates = [(name, value)] if name.startswith("gate-") else []
        if name.endswith("-development.json"):
            candidates.extend((f"{name}:{step}", row["host_gate"]) for step, row in value.items())
        for label, gate in candidates:
            if not isinstance(gate, dict) or gate.get("schema") != "weightbridge.phase_iiis.behavior_gate.v1":
                continue
            evidence[label] = {key: gate.get(key) for key in ("arm", "stage", "step", "passed", "disposition",
                "tool_recovery", "inherited_decimal", "tool_uncertainty")}
            evidence[label]["tool_gain_and_retention"] = gate.get("tool")
            evidence[label]["decimal_retention"] = gate.get("decimal")
            evidence[label]["general_retention"] = gate.get("general")
    return {"schema": "weightbridge.phase_iiis.public_results_matrix.v1", "development_stored_models": stored,
        "development_adapter_checkpoints": adapters, "token_exposure": exposures, "host_gain_evidence": evidence,
        "adapter_merged_development_diagnostics": {arm: artifacts[f"{arm}-metadata.json"].get("adapter_merged_diagnostic") for arm in ARMS},
        "interpretation": "Merged-model rows remain null until a merged-development receipt exists. Adapter observations cannot fill stored-model rows. Recovery, new semantic corrections, family uncertainty and retention use the host's separate counters; repeated comparator losses are not summed into unique harms."}


def ledger_summary(reader, ledger, *, defer_postdevelopment=()):
    ledger = Path(ledger)
    scopes, unfinished, omitted = [], [], set()
    def deferred(name):
        return any(name.startswith(arm + '-post-') for arm in defer_postdevelopment)
    for path in sorted(ledger.glob("*.receipt.json")):
        if deferred(path.name[:-13]):
            omitted.add(path.name[:-13])
            continue
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
        if deferred(name):
            omitted.add(name)
            continue
        if not (ledger/(name+".receipt.json")).exists():
            reader.read(path.name, root=ledger, label="execution/"+path.name)
            unfinished.append(name)
    complete = not unfinished and all(s["resources"] is not None for s in scopes)
    totals = None
    if scopes and all(s["resources"] is not None for s in scopes):
        totals = {"wall_seconds": sum(s["resources"]["wall_seconds"] for s in scopes),
            "cpu_seconds": sum(s["resources"]["cpu_seconds"] for s in scopes),
            "peak_rss_bytes": max(s["resources"]["peak_rss_bytes"] for s in scopes)}
    return {"scopes": scopes, "unfinished_scope_names": unfinished, "complete_accounting": complete and not defer_postdevelopment,
        "completed_scope_totals": totals, "all_recorded_scope_totals": totals if complete and not defer_postdevelopment else None,
        "deferred_postdevelopment_arms": list(defer_postdevelopment), "omitted_scope_names": sorted(omitted),
        "deferred_scope_cost": None,
        "study_execution_complete": None,
        "accounting": "Disjoint owned worker scopes, including monitor cost; inner training and evaluation costs are not added again. Peak is the maximum conservative process-tree measurement, not a sum across stages."}


def aggregate_worker_counters(specs, out):
    """Explicit completed WorkerSpecs only; no roster reads or path discovery.

    Each entry requires label, worker_spec, worker_sha256 and execution_sha256.
    worker_spec is the existing host object with path, rows, protocol,
    authorization, stage and execution. Its rows path is only compared to the
    recorded command; the roster is never opened. Root must supply these inputs
    only after the associated host stage was authorized and completed.
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError("Counter summaries are write-once")
    counters = ("generate_batches", "generated_examples", "batched_decode_steps", "padded_prompt_token_positions")
    records, used_results, used_executions, labels = [], set(), set(), set()
    def read_checked(path, expected):
        payload = Path(path).read_bytes()
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or sha(payload) != expected:
            raise ValueError("Explicit worker input hash mismatch")
        return json.loads(payload)
    def argument(command, flag):
        if command.count(flag) != 1 or command.index(flag)+1 >= len(command):
            raise ValueError("Ambiguous worker command binding")
        return command[command.index(flag)+1]
    for entry in specs:
        if set(entry) != {"label", "worker_spec", "worker_sha256", "execution_sha256"}:
            raise ValueError("Unknown explicit counter input fields")
        label, spec = entry["label"], entry["worker_spec"]
        if not re.fullmatch(r"[a-z0-9_-]+", label) or label in labels:
            raise ValueError("Invalid or duplicate counter label")
        if set(spec) != {"path", "rows", "protocol", "authorization", "stage", "execution"}:
            raise ValueError("Counter collection needs the complete explicit WorkerSpec")
        result_path, execution_path = Path(spec["path"]).resolve(), Path(spec["execution"]).resolve()
        if result_path in used_results or execution_path in used_executions:
            raise ValueError("Repeated worker scopes would duplicate counts")
        labels.add(label); used_results.add(result_path); used_executions.add(execution_path)
        result = read_checked(result_path, entry["worker_sha256"])
        execution = read_checked(execution_path, entry["execution_sha256"])
        process, limits = execution.get("execution") or {}, execution.get("limits") or {}
        resource, sleep = process.get("resources") or {}, execution.get("temporary_sleep_request") or {}
        command = execution.get("command") or []
        checks = {"worker_schema": result.get("schema") == "weightbridge.phase_iiis.evaluation.v1",
            "worker_stage": result.get("stage") == spec["stage"],
            "frozen_manifest": result.get("task_manifest_sha256") == MANIFEST_SHA,
            "execution_schema": execution.get("schema") == "weightbridge.phase_iiis.execution.v1",
            "successful_execution": execution.get("passed") is True and execution.get("error") is None
                and process.get("exit_code") == 0 and process.get("budget_failure") is None,
            "module": argument(command, "-m") == "experiments.phase_iiis_tool_transfer.evaluate",
            "protocol_hash": sha(Path(spec["protocol"]).read_bytes()) == result.get("protocol_sha256"),
            "authorization_hash": sha(Path(spec["authorization"]).read_bytes()) == result.get("authorization_sha256"),
            "task_binding": argument(command, "--task") == result.get("task"),
            "stage_binding": argument(command, "--stage") == spec["stage"],
            "checkpoint_binding": argument(command, "--expected-checkpoint") == result.get("checkpoint_sha256"),
            "roster_commitment_binding": argument(command, "--expected-rows") == result.get("roster_sha256"),
            "cpu_accounting": resource.get("cpu_accounting_complete_for_observed_processes") is True,
            "sleep_release": sleep.get("requested") is False or sleep.get("requested") is True
                and sleep.get("succeeded") is True and sleep.get("released") is True}
        for flag, key in (("--out", "path"), ("--rows", "rows"), ("--protocol", "protocol"), ("--authorization", "authorization")):
            checks[key+"_binding"] = Path(argument(command, flag)).resolve() == Path(spec[key]).resolve()
        adapter = result.get("adapter_sha256")
        checks["adapter_binding"] = argument(command, "--expected-adapter") == adapter if adapter else "--adapter" not in command
        processes = process.get("processes") or []
        checks["resource_arithmetic"] = bool(processes) and (
            math.isclose(sum(p["cpu_seconds"] for p in processes), resource.get("observed_tree_cpu_seconds", -1), abs_tol=1e-7)
            and math.isclose(resource.get("host_cpu_seconds", -1)+resource.get("observed_tree_cpu_seconds", -1), resource.get("cpu_seconds", -3), abs_tol=1e-7)
            and sum(p["peak_rss_bytes"] for p in processes) == resource.get("peak_rss_bytes"))
        for key, limit in (("wall_seconds", "wall_limit"), ("cpu_seconds", "cpu_limit"), ("peak_rss_bytes", "memory_limit")):
            value, cap = resource.get(key), limits.get(limit)
            checks[key] = type(value) in (int, float) and type(cap) in (int, float) and math.isfinite(value) and 0 <= value <= cap
        if not all(checks.values()):
            raise ValueError("Counter input failed completed-worker bindings: "+", ".join(k for k, v in checks.items() if not v))
        measured = (result.get("measurement") or {}).get("resources") or {}
        counts = {key: measured.get(key) for key in counters}
        if any(value is not None and (type(value) is not int or value < 0) for value in counts.values()):
            raise ValueError("Decode counter is not a nonnegative integer")
        records.append({"label": label, "task": result["task"], "stage": spec["stage"],
            "worker_sha256": entry["worker_sha256"], "execution_sha256": entry["execution_sha256"],
            "protocol_sha256": result["protocol_sha256"], "authorization_sha256": result["authorization_sha256"],
            "task_manifest_sha256": result["task_manifest_sha256"],
            "roster_sha256": result["roster_sha256"], "checkpoint_sha256": result["checkpoint_sha256"],
            "adapter_sha256": adapter, "checks": checks, "counts": counts})
    totals = {key: sum(r["counts"][key] for r in records) if records and all(r["counts"][key] is not None for r in records) else None for key in counters}
    output = {"schema": "weightbridge.phase_iiis.public_worker_counters.v1", "passed": True if records else None,
        "status": "AGGREGATED_COMPLETED_WORKERS" if records else "NOT_EXECUTED",
        "selected_worker_count": len(records), "selected_workers": records, "selected_worker_totals": totals,
        "raw_roster_access": False, "model_execution": False, "scientific_outcomes_reverified": False,
        "scope": "Only explicitly supplied completed workers. Resource counters and byte/command bindings were checked; no raw roster or prediction was scored. These counters are not extra wall/CPU costs and must not be added to enclosing scopes."}
    assert_public(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        stream.write(dumps(output))
    return {"passed": True, "sha256": sha(out.read_bytes()), "selected_worker_count": len(records), "model_execution": False}


def publish_snapshot(artifacts, reader, out, expected_manifest, scope):
    out = Path(out)
    if out.exists():
        raise FileExistsError("Outcome snapshots are write-once")
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
        "scope": scope}
    assert_public(receipt)
    (out/"export-manifest.json").write_bytes(dumps(receipt))
    return verify(out)


def development_supplement(study, out, arm):
    """Fixed completed development metadata only; no banks or live training reads."""
    if arm not in ARMS:
        raise ValueError("Unknown development arm")
    reader = Reader(study)
    manifest = reader.read("manifest.json")
    protocol = reader.read(f"{arm}/protocol.json")
    if reader.inputs.get("manifest.json") != MANIFEST_SHA or protocol.get("task_manifest_sha256") != MANIFEST_SHA:
        raise ValueError("Development supplement differs from frozen study")
    sources = manifest["source_sha256"]
    if protocol.get("source_sha256") != sources:
        raise ValueError("Development supplement protocol source mismatch")
    specs = {"merge-receipt": ("merge-receipt.json", "weightbridge.supervised_lora_merge_receipt.v1"),
        "merge-verification": ("merge-receipt.verification.json", "weightbridge.phase_iiis.merge_verification.v1"),
        "reconstruction-verification": ("reconstruction-merge.verification.json", "weightbridge.phase_iiis.merge_verification.v1"),
        "exact-reconstruction-gate": ("exact-reconstruction-gate.json", "weightbridge.phase_iiis.host_exact_reconstruction.v1"),
        "merged-development-gate": ("merged-development-gate.json", "weightbridge.phase_iiis.behavior_gate.v1"),
        "postdevelopment-completed": ("completed.json", "weightbridge.phase_iiis.private_postdevelopment.v1")}
    artifacts = {"source-manifest.json": sources}
    for label, (name, schema) in specs.items():
        record = reader.read(f"{arm}/post-development-v1/{name}")
        if (record is None or record.get("schema") != schema
                or record.get("task_manifest_sha256", MANIFEST_SHA) != MANIFEST_SHA
                or record.get("protocol_sha256", reader.inputs[f"{arm}/protocol.json"]) != reader.inputs[f"{arm}/protocol.json"]
                or record.get("source_sha256", sources) != sources):
            raise ValueError("Missing or mismatched completed development receipt")
        artifacts[label+".json"] = project(record)
    review = reader.read(f"{arm}-postdevelopment-integrity-audit-v1.json", root=reader.study.parent,
        label=f"diagnostics/{arm}-postdevelopment-integrity-audit-v1.json")
    if (review is None or review.get("schema") != "weightbridge.phase_iiis.closed_postdevelopment_review.v1"
            or review.get("task_manifest_sha256") != MANIFEST_SHA or review.get("source_sha256") != sources
            or review.get("protected_data_access_by_reviewer") is not False):
        raise ValueError("Missing or mismatched closed development integrity review")
    artifacts["postdevelopment-integrity.json"] = project(review)
    artifacts["adapter-merged-diagnostic.json"] = adapter_merged_diagnostic(reader.read(
        f"{arm}-adapter-merged-development-diagnostic-v1.json", root=reader.study.parent,
        label=f"diagnostics/{arm}-adapter-merged-development-diagnostic-v1.json"),
        manifest_sha256=MANIFEST_SHA, source_sha256=sources, protocol_sha256=reader.inputs[f"{arm}/protocol.json"], arm=arm)
    artifacts["stage-availability.json"] = {"independent_qualification": None, "sealed_confirmation": None,
        "historical_retention": None, "independent_replay": None, "lineage_admission": False,
        "interpretation": "This supplement reports completed development only. Exact development replay is not independent qualification. No protected stage was opened by this exporter."}
    return publish_snapshot(artifacts, reader, out, MANIFEST_SHA,
        "Fixed closed development metadata only; no raw rosters, observations, model tensors or changing training records. Endpoint and exact native equality remain separate. All projections bind original private receipt bytes; public hash verification is not private scientific replay.")


def historical_supplement(study, out, arm, *, expected_manifest=MANIFEST_SHA):
    """Only an explicitly selected closed historical transaction; no bank reads."""
    if arm not in ARMS or Path(out).exists():
        raise ValueError("Historical supplement needs a supported arm and new output directory")
    reader = Reader(study)
    manifest, protocol = reader.read("manifest.json"), reader.read(f"{arm}/protocol.json")
    prefix = f"{arm}/protected-v3/historical-retention"
    gate, transaction = reader.read(prefix+"/gate.json"), reader.read(prefix+"/completed.json")
    review = reader.read(f"{arm}-historical-metadata-audit-v1.json", root=reader.study.parent,
        label=f"diagnostics/{arm}-historical-metadata-audit-v1.json")
    if (reader.inputs.get("manifest.json") != expected_manifest
            or protocol.get("task_manifest_sha256") != expected_manifest
            or protocol.get("source_sha256") != manifest["source_sha256"]
            or not gate or gate.get("schema") != "weightbridge.phase_iiis.host_historical_retention.v1"
            or gate.get("task_manifest_sha256") != expected_manifest
            or not transaction or transaction.get("schema") != "weightbridge.phase_iiis.private_protected_execution.v3"
            or transaction.get("completed") is not True or transaction.get("action") != "historical-retention"
            or transaction.get("arm") != arm or transaction.get("task_manifest_sha256") != expected_manifest
            or transaction.get("gate_sha256") != reader.inputs[prefix+"/gate.json"]
            or not review or review.get("schema") != "weightbridge.phase_iiis.closed_historical_metadata_review.v1"
            or review.get("arm") != arm or review.get("task_manifest_sha256") != expected_manifest
            or review.get("gate_sha256") != reader.inputs[prefix+"/gate.json"]
            or review.get("completed_sha256") != reader.inputs[prefix+"/completed.json"]
            or review.get("raw_outputs_read_by_reviewer") is not False
            or review.get("raw_rosters_read_by_reviewer") is not False):
        raise ValueError("Historical metadata does not bind its closed reviewed transaction")
    artifacts = {"source-manifest.json": manifest["source_sha256"],
        "protocol-identity.json": {"protocol_sha256": reader.inputs[f"{arm}/protocol.json"],
            **selected(protocol, ("schema", "arm", "checkpoint_sha256", "task_manifest_sha256"))},
        "historical-gate.json": project(gate), "historical-transaction.json": project(transaction),
        "closed-metadata-review.json": project(review),
        "stage-availability.json": {"historical_retention": {arm: {"reported_transaction_passed": transaction.get("passed"),
            "completed": True, "exact_replays_passed": transaction.get("exact_replays_passed")}},
            "independent_qualification": None, "sealed_confirmation": None, "lineage_admission": False}}
    return publish_snapshot(artifacts, reader, out, expected_manifest,
        "Explicit closed historical metadata only. The previously disclosed R battery measures retention, not fresh transfer. This publication neither reads raw outputs/rosters nor recomputes behavioral or replay claims; the bound host workers and audit supply that evidence.")


def export(study, ledger, out, gates=(), *, expected_manifest=MANIFEST_SHA, defer_postdevelopment=()):
    out = Path(out)
    if out.exists():
        raise FileExistsError("Outcome snapshots are write-once")
    if len(set(defer_postdevelopment)) != len(defer_postdevelopment) or any(arm not in ARMS for arm in defer_postdevelopment):
        raise ValueError("Unknown or repeated deferred postdevelopment arm")
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
            "inherited_r_main_cost": project(protocol.get("inherited_r_main_cost")) if arm == "sequential" else None,
            "data_order": project(reader.read(f"{arm}/prospective-data-order.json")),
            "main_authorization": project(reader.read(f"{arm}/main-authorization.json")),
            "main_admission": project(reader.read(f"{arm}/main-admission.json")),
            "main_claim": project(reader.read(f"{arm}/main-launch-claim.json")),
            "smoke_gate": project(reader.read(f"{arm}/smoke-gate.json")),
            "main_training_validation": project(reader.read(f"{arm}/main-training-validation.json")),
            "assessment": project(reader.read(f"{arm}/assessment/assessment.json")),
            "adapter_merged_diagnostic": None if arm in defer_postdevelopment else adapter_merged_diagnostic(reader.read(
                f"{arm}-adapter-merged-development-diagnostic-v1.json", root=reader.study.parent,
                label=f"diagnostics/{arm}-adapter-merged-development-diagnostic-v1.json"),
                manifest_sha256=expected_manifest, source_sha256=frozen["source_sha256"],
                protocol_sha256=reader.inputs[f"{arm}/protocol.json"], arm=arm),
            "execution": selected(reader.read(f"{arm}/run-arm.json"), (*IDENTITY, "arm", "completed", "status",
                "failed_stage", "protected_evaluation_opened", "behavioral_qualification", "production_authority", "lineage_authority"))}
        artifacts[f"{arm}-development.json"] = {str(step): {"diagnostic_only": step == 64,
            "tool": evaluation(reader, f"{arm}/development/step-{step:04d}/tool.json"),
            "decimal": evaluation(reader, f"{arm}/development/step-{step:04d}/decimal.json"),
            "host_gate": project(reader.read(f"{arm}/assessment/step-{step:04d}.json"))}
            for step in (64, 128)}
    availability = {"independent_qualification": None, "sealed_confirmation": None, "final_replay": None,
        "protected_transactions": None}
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
                availability[stage][value["arm"]] = {"scored_gate_passed": value.get("passed"),
                    "disposition": value.get("disposition"), "public_gate_artifact": f"gate-{label}.json"}
            if value.get("schema") == "weightbridge.phase_iiis.private_protected_execution.v3":
                action, arm = value.get("action"), value.get("arm")
                if action not in ("historical-retention", "qualification", "confirmation", "finalize") or arm not in ARMS:
                    raise ValueError("Unknown protected transaction action or arm")
                availability["protected_transactions"] = availability["protected_transactions"] or {}
                transactions = availability["protected_transactions"].setdefault(action, {})
                if arm in transactions:
                    raise ValueError("Multiple transaction receipts cannot silently replace one another")
                transactions[arm] = {**selected(value, ("completed", "passed", "exact_replays_passed", "disposition", "error")),
                    "public_transaction_artifact": f"gate-{label}.json"}
    artifacts["execution.json"] = ledger_summary(reader, ledger, defer_postdevelopment=defer_postdevelopment)
    artifacts["stage-availability.json"] = {**availability, "lineage_admission": False,
        "interpretation": "No result is inferred from a missing stage. scored_gate_passed describes only the supplied scientific gate, not completed enclosing execution. Protected transaction completion, failure and exact-replay flags are reported separately from explicitly supplied host receipts; an incomplete transaction cannot be credited as a verified pass. This exporter does not rerun private verification or grant evaluation/lineage authority."}
    artifacts["results-matrix.json"] = results_tables(artifacts)
    return publish_snapshot(artifacts, reader, out, expected_manifest,
        "Redacted aggregate projections with original private receipt byte hashes. Public hash checks do not replace host scientific verification or reproduce private predictions.")


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
    gate = project({"tool_recovery": {"recovered_original_successes": ["PRIVATE1"],
        "corrected_beyond_original": ["PRIVATE2", "PRIVATE3"]}, "tool_uncertainty": {
        "families": 16, "seed": 123, "confidence": .95, "strict_accuracy_gain_interval": [.1, .2]},
        "tool": {"corrected_real_actions": ["search_files", "run_tests"], "candidate": {
            "wording_familiarity": {"unseen": {"presentations": 128, "strict_correct": 100}}}}})
    assert gate["tool_recovery"]["recovered_original_successes_count"] == 1
    assert gate["tool_recovery"]["corrected_beyond_original_count"] == 2
    assert gate["tool_uncertainty"]["strict_accuracy_gain_interval"] == [.1, .2]
    assert "seed" not in gate["tool_uncertainty"]
    assert gate["tool"]["candidate"]["wording_familiarity"]["unseen"]["strict_correct"] == 100
    exposure = category_exposure([{"by_category": {
        "tool": {"examples": 1, "supervised_tokens": 9, "weighted_supervised_tokens": 9., "weighted_token_nll_sum": 18.},
        "general": {"examples": 1, "supervised_tokens": 1, "weighted_supervised_tokens": 1., "weighted_token_nll_sum": 1.}}}])
    assert exposure["categories"]["tool"]["row_fraction"] == .5
    assert exposure["categories"]["tool"]["supervised_token_fraction"] == .9
    varying = category_exposure([{"by_category": {"tool": {"weighted_supervised_tokens": 9},
        "general": {"weighted_supervised_tokens": 1}}}, {"by_category": {
        "tool": {"weighted_supervised_tokens": 1}, "general": {"weighted_supervised_tokens": 1}}}])
    assert math.isclose(varying["categories"]["tool"]["weighted_token_fraction"], 10/12)
    assert math.isclose(varying["categories"]["tool"]["mean_per_step_objective_weight"], .7)
    diagnostic = {"schema": "weightbridge.phase_iiis.adapter_merged_development_diagnostic.v1",
        "arm": "sequential", "step": 128, "task_manifest_sha256": "a"*64, "source_sha256": {},
        "protocol_sha256": "b"*64, "diagnostic_only": True, "additional_acceptance_gate": False,
        "evidence_validation_passed": True, "protected_banks_opened": False, "model_execution_by_reviewer": False,
        "suites": {task: {"presentations": 4, "complete_native_rows_equal": 3,
            "complete_native_replay_equal": False, "behavioral_observation_flags_identical": True,
            "native_field_equal_counts": {"generated_token_ids": 3, "text": 3,
                "native_end_token_id": 4, "prefix_token_ids": 4, "prefix_tokens": 4,
                "stop_reason": 4, "unexpected_special_token_ids": 4}} for task in ("tool", "decimal")}}
    identity = dict(manifest_sha256="a"*64, source_sha256={}, protocol_sha256="b"*64, arm="sequential")
    preserved = adapter_merged_diagnostic(diagnostic, **identity)
    assert preserved["suites"]["decimal"]["behavioral_observation_flags_identical"] is True
    assert preserved["suites"]["decimal"]["complete_native_replay_equal"] is False
    assert preserved["suites"]["decimal"]["complete_native_streams_different"] == 1
    assert preserved["suites"]["decimal"]["native_field_equal_counts"]["complete_text_equal"] == 3
    diagnostic["suites"]["tool"]["complete_native_replay_equal"] = True
    try:
        adapter_merged_diagnostic(diagnostic, **identity)
    except ValueError:
        pass
    else:
        raise AssertionError("Endpoint agreement was allowed to masquerade as native identity")
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
        matrix = json.loads((out/"results-matrix.json").read_bytes())
        assert all(v is None for v in matrix["development_stored_models"].values())
        exposure = matrix["token_exposure"]["sequential"]["main"]
        assert exposure["categories"]["tool"]["supervised_token_fraction"] == 1.
        # An active deferred branch is not read, even if its files are incomplete.
        (ledger/"original-joint-post-merge.receipt.json").write_bytes(b"INCOMPLETE")
        (ledger/"original-joint-post-merge.started.json").write_bytes(b"INCOMPLETE")
        (root/"original-joint-adapter-merged-development-diagnostic-v1.json").write_bytes(b"INCOMPLETE")
        deferred_out = root/"deferred-public"
        assert export(study, ledger, deferred_out, expected_manifest=expected,
            defer_postdevelopment=("original-joint",))["passed"]
        deferred = json.loads((deferred_out/"execution.json").read_bytes())
        assert deferred["omitted_scope_names"] == ["original-joint-post-merge"]
        assert deferred["all_recorded_scope_totals"] is None and deferred["deferred_scope_cost"] is None
        assert json.loads((deferred_out/"original-joint-metadata.json").read_bytes())["adapter_merged_diagnostic"] is None
        # A scored PASS followed by an interrupted host transaction remains distinct.
        for name, record in {
            "scored.json": {"schema": "weightbridge.phase_iiis.behavior_gate.v1", "stage": "qualification",
                "arm": "sequential", "passed": True, "disposition": "SCORED_PASS"},
            "failed-transaction.json": {"schema": "weightbridge.phase_iiis.private_protected_execution.v3",
                "action": "qualification", "arm": "sequential", "completed": False, "passed": False,
                "error": {"type": "InterruptedError", "message": "PRIVATE_OUTPUT"},
                "model_workers": ["C:/PRIVATE/worker.json"]},
            "history.json": {"schema": "weightbridge.phase_iiis.host_historical_retention.v1", "passed": False,
                "presentations": 832, "gates": {"hidden": {"lost_correct": ["PRIVATE_ID"]}},
                "input_sha256": {"hidden": {"path": "C:/PRIVATE/bank.json", "sha256": "c"*64}}},
            "replays.json": {"schema": "weightbridge.phase_iiis.private_stage_replay.v1", "passed": False,
                "exact_native": {"qualification/candidate/tool": False},
                "replays": {"qualification/candidate/tool": {"recorded": {"path": "C:/PRIVATE/worker.json", "rows": "PRIVATE_ROWS"}}}},
            "closure.json": {"schema": "weightbridge.phase_iiis.formal_stage_closure.v3", "passed": False,
                "ready_for_final_pass": False, "stages": {"confirmation": {"completed_valid": False,
                    "passed": False, "disposition": "UNOPENED", "gate_path": None}}, "seed": "PRIVATE_SEED"},
        }.items():
            (study/name).write_bytes(dumps({**record, "task_manifest_sha256": expected}))
        protected_out = root/"protected-public"
        assert export(study, ledger, protected_out,
            gates=[(n.removesuffix('.json'), n) for n in ("scored.json", "failed-transaction.json", "history.json", "replays.json", "closure.json")],
            expected_manifest=expected, defer_postdevelopment=("original-joint",))["passed"]
        available = json.loads((protected_out/"stage-availability.json").read_bytes())
        assert available["independent_qualification"]["sequential"]["scored_gate_passed"] is True
        assert available["protected_transactions"]["qualification"]["sequential"]["completed"] is False
        assert available["protected_transactions"]["qualification"]["sequential"]["passed"] is False
        assert available["sealed_confirmation"] is None and available["final_replay"] is None
        assert all("PRIVATE" not in p.read_text() for p in protected_out.glob("*.json"))
        history_dir = study/"sequential/protected-v3/historical-retention"
        history_dir.mkdir(parents=True)
        historical_gate = {"schema": "weightbridge.phase_iiis.host_historical_retention.v1", "passed": True,
            "task_manifest_sha256": expected}
        (history_dir/"gate.json").write_bytes(dumps(historical_gate))
        transaction = {"schema": "weightbridge.phase_iiis.private_protected_execution.v3", "completed": True,
            "passed": True, "action": "historical-retention", "arm": "sequential", "task_manifest_sha256": expected,
            "gate_sha256": sha(dumps(historical_gate)), "exact_replays_passed": True}
        (history_dir/"completed.json").write_bytes(dumps(transaction))
        audit_path = root/"sequential-historical-metadata-audit-v1.json"
        review = {"schema": "weightbridge.phase_iiis.closed_historical_metadata_review.v1", "passed": True,
            "arm": "sequential", "task_manifest_sha256": expected, "gate_sha256": sha(dumps(historical_gate)),
            "completed_sha256": sha(dumps(transaction)), "raw_outputs_read_by_reviewer": False, "raw_rosters_read_by_reviewer": False}
        audit_path.write_bytes(dumps(review))
        assert historical_supplement(study, root/"historical-public", "sequential", expected_manifest=expected)["passed"]
        audit_path.write_bytes(dumps({**review, "gate_sha256": "f"*64}))
        try:
            historical_supplement(study, root/"historical-refused", "sequential", expected_manifest=expected)
        except ValueError:
            assert not (root/"historical-refused").exists()
        else:
            raise AssertionError("Unbound historical review accepted")
        # Counter collection deliberately references a nonexistent roster.
        # Its absence proves the reporter does not open a bank to count work.
        protocol_path, authorization_path = root/"counter-protocol.json", root/"counter-auth.json"
        protocol_path.write_bytes(b"{}"); authorization_path.write_bytes(b"{}")
        worker_path, process_path = root/"counter-worker.json", root/"counter-process.json"
        roster_path = root/"NEVER_OPEN_THIS_ROSTER.json"
        worker = {"schema": "weightbridge.phase_iiis.evaluation.v1", "task": "tool", "stage": "qualification",
            "task_manifest_sha256": MANIFEST_SHA,
            "protocol_sha256": sha(b"{}"), "authorization_sha256": sha(b"{}"),
            "roster_sha256": "c"*64, "checkpoint_sha256": "d"*64, "adapter_sha256": None,
            "measurement": {"rows": [{"prompt": "PRIVATE_ROW", "text": "PRIVATE_RESPONSE"}],
                "resources": {"generate_batches": 2, "generated_examples": 8, "batched_decode_steps": 20,
                    "padded_prompt_token_positions": 100}}}
        command = ["python", "-m", "experiments.phase_iiis_tool_transfer.evaluate", "--out", str(worker_path),
            "--rows", str(roster_path), "--protocol", str(protocol_path), "--authorization", str(authorization_path),
            "--stage", "qualification", "--task", "tool", "--expected-rows", "c"*64, "--expected-checkpoint", "d"*64]
        process = {"schema": "weightbridge.phase_iiis.execution.v1", "passed": True, "error": None,
            "command": command, "limits": {"wall_limit": 10, "cpu_limit": 20, "memory_limit": 1000},
            "temporary_sleep_request": {"requested": False}, "execution": {"exit_code": 0, "budget_failure": None,
                "processes": [{"cpu_seconds": 5., "peak_rss_bytes": 200}], "resources": {"wall_seconds": 3.,
                    "cpu_seconds": 6., "host_cpu_seconds": 1., "observed_tree_cpu_seconds": 5.,
                    "peak_rss_bytes": 200, "cpu_accounting_complete_for_observed_processes": True}}}
        worker_path.write_bytes(dumps(worker)); process_path.write_bytes(dumps(process))
        spec = {"label": "one", "worker_spec": {"path": str(worker_path), "rows": str(roster_path),
            "protocol": str(protocol_path), "authorization": str(authorization_path), "stage": "qualification",
            "execution": str(process_path)}, "worker_sha256": sha(worker_path.read_bytes()),
            "execution_sha256": sha(process_path.read_bytes())}
        counter_out = root/"counters.json"
        assert aggregate_worker_counters([spec], counter_out)["passed"]
        published = json.loads(counter_out.read_bytes())
        assert published["selected_worker_totals"]["generated_examples"] == 8 and not roster_path.exists()
        assert "PRIVATE" not in counter_out.read_text()
        for inputs in ([spec, {**spec, "label": "two"}], [{**spec, "worker_sha256": "e"*64}]):
            try:
                aggregate_worker_counters(inputs, root/"refused.json")
            except ValueError:
                assert not (root/"refused.json").exists()
            else:
                raise AssertionError("Duplicate scope or wrong source hash admitted")
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
        "interrupted_scope_unknown": "PASS", "write_once": "PASS", "hash_tamper_refusal": "PASS",
        "matrix_unknowns": "PASS", "token_share_derivation": "PASS", "explicit_worker_binding": "PASS",
        "recovery_vs_new_gain": "PASS", "family_uncertainty_and_unseen_wording": "PASS",
        "no_roster_open_for_counters": "PASS", "duplicate_counter_scope_refusal": "PASS",
        "adapter_merge_endpoint_vs_native_identity": "PASS", "diagnostic_counter_tamper_refusal": "PASS",
        "active_postdevelopment_exclusion": "PASS", "protected_metadata_redaction_and_incomplete_gate": "PASS",
        "closed_historical_metadata_binding": "PASS",
        "model_execution": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gate", action="append", default=[], metavar="LABEL=RELATIVE.json")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--worker-specs", type=Path, help="Private JSON list of explicitly authorized completed WorkerSpecs and expected hashes")
    parser.add_argument("--counter-out", type=Path, help="New aggregate-only receipt, with no paths or raw observations")
    parser.add_argument("--development-supplement", choices=ARMS, help="Publish only the selected arm's fixed closed stored-development metadata")
    parser.add_argument("--historical-supplement", choices=ARMS, help="Publish only an explicitly selected closed historical gate, transaction and metadata audit")
    parser.add_argument("--defer-postdevelopment", action="append", choices=ARMS, default=[], help="Leave an active arm's postdevelopment metadata and costs unmeasured in this immutable snapshot")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    elif args.worker_specs and args.counter_out:
        print(json.dumps(aggregate_worker_counters(json.loads(args.worker_specs.read_bytes()), args.counter_out)))
    elif args.verify:
        print(json.dumps(verify(args.verify)))
    elif args.development_supplement and args.study and args.out:
        print(json.dumps(development_supplement(args.study, args.out, args.development_supplement)))
    elif args.historical_supplement and args.study and args.out:
        print(json.dumps(historical_supplement(args.study, args.out, args.historical_supplement)))
    elif args.study and args.ledger and args.out:
        gates = [value.split("=", 1) for value in args.gate]
        if any(len(pair) != 2 for pair in gates):
            parser.error("Each --gate must be LABEL=RELATIVE.json")
        print(json.dumps(export(args.study, args.ledger, args.out, gates, defer_postdevelopment=args.defer_postdevelopment)))
    else:
        parser.error("Choose --self-test, --verify, or --study/--ledger/--out")
