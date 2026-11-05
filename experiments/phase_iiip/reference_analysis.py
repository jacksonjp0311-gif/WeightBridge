"""Model-free replay of the isolated external-reference decimal benchmark.

Presentation gains, strict comparisons and eight-presentation families are
different estimands. No result here admits a WeightBridge learning candidate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np

from experiments.phase_iiih.run_decimal_endpoint import endpoint
from experiments.phase_iiil.analyze import _checked_measurement
from experiments.phase_iiil.semantic_diagnostics import summarize as semantic_summary
from experiments.phase_iiim.semantic_metrics import compare
from experiments.phase_iiip import reference_import as importer
from experiments.phase_iiip.reference_study import PROTOCOL, source_hashes
from experiments.phase_iiil import semantic_roster as generator
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.patches.huggingface import _canonical_sha256, _from_float64, _to_float64


def _read(root, name):
    return json.loads((root / name).read_bytes())


def clustered_presentation_interval(parent, candidate, *, seed=1996, resamples=4000):
    """Paired presentation-accuracy change, resampling whole target families."""
    changes = defaultdict(list)
    for before, after in zip(parent["rows"], candidate["rows"], strict=True):
        if before["id"] != after["id"]:
            raise ValueError("Clustered uncertainty requires identical presentation order")
        if before["group"] == "target":
            changes[before["pair_id"]].append(int(after["correct"]) - int(before["correct"]))
    if any(len(values) != 8 for values in changes.values()):
        raise ValueError("Presentation uncertainty requires complete eight-response target families")
    means = [sum(values) / len(values) for _, values in sorted(changes.items())]
    if len(means) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(means, k=len(means))) / len(means) for _ in range(resamples))
    return {"estimand": "Paired target presentation-accuracy change, not strict-comparison or complete-family accuracy",
        "point_estimate": sum(means) / len(means), "low": draws[int(.025 * resamples) - 1],
        "high": draws[int(.975 * resamples) - 1], "confidence_level": .95,
        "clusters": len(means), "resamples": resamples, "seed": seed,
        "postfit_reporting_only": True, "used_for_candidate_selection": False,
        "method": "Exploratory percentile bootstrap over underlying target families; all eight correlated renderings stay together",
        "family_mean_changes": dict(zip(sorted(changes), means, strict=True)),
        "scope": "Six designed families do not establish population-level improvement. This interval does not qualify a candidate or remove observed regressions."}


def _counts(parent, candidate, ids):
    before, after = ({row["id"]: row for row in measured["rows"]} for measured in (parent, candidate))
    corrected = [name for name in ids if not before[name]["correct"] and after[name]["correct"]]
    regressed = [name for name in ids if before[name]["correct"] and not after[name]["correct"]]
    return {"presentations": len(ids), "parent_correct": sum(before[i]["correct"] for i in ids),
        "reference_correct": sum(after[i]["correct"] for i in ids), "gain": len(corrected) - len(regressed),
        "corrected_presentations": corrected, "regressed_presentations": regressed,
        "parent_valid": sum(before[i]["valid"] for i in ids), "reference_valid": sum(after[i]["valid"] for i in ids),
        "parent_completed": sum(before[i]["completed"] for i in ids),
        "reference_completed": sum(after[i]["completed"] for i in ids)}


def _strata(parent, candidate):
    groups = defaultdict(list)
    for row in parent["rows"]:
        target = row["group"] == "target"
        group = f"{row['tier']}/{row['truth_precision']}" if target else row.get("tier") or "copy"
        groups[f"group/{group}"].append(row["id"])
        if target:
            groups[f"wording/{row['wording']}"].append(row["id"])
            groups[f"order/{row['order']}"].append(row["id"])
            groups[f"correct_length/{row['truth_precision']}"].append(row["id"])
    return {name: _counts(parent, candidate, ids) for name, ids in sorted(groups.items())}


def _families(parent, candidate):
    after = {row["id"]: row for row in candidate["rows"]}
    families = defaultdict(list)
    for row in parent["rows"]:
        if row["group"] == "target":
            families[row["pair_id"]].append(row)
    result = {}
    for family, forms in sorted(families.items()):
        indexed = {(row["counterfactual"], row["order"], row["wording"]): row for row in forms}
        if len(forms) != 8 or len(indexed) != 8:
            raise ValueError("Reference family matrix must retain all eight required responses")
        rows = []
        for key, before in sorted(indexed.items()):
            edited = after[before["id"]]
            original_tokens, edited_tokens = before["generated_token_ids"], edited["generated_token_ids"]
            first = next((i for i, (left, right) in enumerate(zip(original_tokens, edited_tokens)) if left != right),
                         min(len(original_tokens), len(edited_tokens)) if len(original_tokens) != len(edited_tokens) else None)
            rows.append({"id": before["id"], "comparison_id": before["comparison_id"], "counterfactual": key[0],
                "order": key[1], "wording": key[2], "operands": before["operands"], "oracle_answer": before["answer"],
                "parent_correct": before["correct"], "reference_correct": edited["correct"],
                "parent_valid": before["valid"], "reference_valid": edited["valid"],
                "parent_semantic_answer": before["exact"], "reference_semantic_answer": edited["exact"],
                "parent_text": before["text"], "reference_text": edited["text"],
                "first_divergent_native_token_position": first,
                "parent_divergent_token_id": original_tokens[first] if first is not None and first < len(original_tokens) else None,
                "reference_divergent_token_id": edited_tokens[first] if first is not None and first < len(edited_tokens) else None})
        result[family] = {"rows": rows, "parent_complete": all(row["parent_correct"] for row in rows),
            "reference_complete": all(row["reference_correct"] for row in rows),
            **_counts(parent, candidate, [row["id"] for row in rows])}
    return result


def _work(measured, contract):
    rows, counts = measured["rows"], measured["resources"]
    lengths = []
    for row in rows:
        tokens = row.get("generated_token_ids")
        if (not isinstance(tokens, list) or not 1 <= len(tokens) <= contract.max_new_tokens
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError("Reference native token evidence exceeds the frozen endpoint budget")
        lengths.append(len(tokens))
    batches = (len(rows) + contract.batch_size - 1) // contract.batch_size
    steps = sum(max(lengths[start:start + contract.batch_size]) for start in range(0, len(rows), contract.batch_size))
    if (counts["generated_examples"] != len(rows) or counts["generate_batches"] != batches
            or counts["batched_decode_steps"] != steps):
        raise ValueError("Reference generation work does not follow its native token records")
    return {"responses": len(rows), "generated_native_tokens": sum(lengths),
        "batched_decode_steps": steps, "forward_calls": steps, "backward_calls": 0}


def _verify_patch_receipt(host, nested, manifest, arrays, authorization):
    """Check every exported value against pinned NPZ and replay actual delta norm.

    Original values are host-recorded BF16 measurements. Full source tensor
    hashes remain reconstruction evidence, not independently recovered weights
    from this model-free report.
    """
    changes = host["changes"]
    if (len(changes) != authorization["expected_changed_entries"]
            or host["touched_entries"] != len(changes)
            or host["policy_sha256"] != _canonical_sha256(authorization["patch_policy"])):
        raise ValueError("Reference receipt does not bind exact support or authorized patch policy")
    square, maximum = 0., 0.
    for i, item in enumerate(manifest["tensors"]):
        records = [row for row in changes if row["tensor"] == item["name"]]
        if (len(records) != item["changed_entries"] or any(row["op"] != "set" or len(row["index"]) != 2 for row in records)
                or any(type(value) is not int or not 0 <= value < item["shape"][axis]
                       for row in records for axis, value in enumerate(row["index"]))):
            raise ValueError("Reference sparse receipt differs from published tensor support")
        indices = np.array([row["index"][0] * item["shape"][1] + row["index"][1] for row in records])
        order, expected_order = np.argsort(indices), np.argsort(arrays[f"t{i}_idx"])
        before, after = (np.array([row[key] for row in records], dtype=float) for key in ("before", "after"))
        expected = importer._to_float64(arrays[f"t{i}_val"].astype('<u2').tobytes(), 'BF16')
        if (len(np.unique(indices)) != len(indices) or not np.array_equal(indices[order], arrays[f"t{i}_idx"][expected_order])
                or not np.isfinite(before).all() or not np.isfinite(after).all()
                or not np.array_equal(after[order], expected[expected_order])
                or any(not np.array_equal(values, _to_float64(_from_float64(values, 'BF16'), 'BF16')) for values in (before, after))):
            raise ValueError("Reference receipt values are not the pinned native BF16 payload")
        delta = after - before
        square += float(delta @ delta)
        maximum = max(maximum, float(np.max(np.abs(delta))))
    if (not np.isclose(np.sqrt(square), nested["absolute_update_norm"], rtol=1e-12, atol=1e-12)
            or maximum != nested["max_abs_delta"] or maximum != host["max_abs_delta"]):
        raise ValueError("Reference stored update norm does not replay from host before/after entries")
    patch = {"schema": "weightbridge.patch.v1", "base_sha256": host["base_sha256"],
        "reason": "Isolated external p50c reference reconstruction; not a Track A adaptation candidate",
        "evidence_ids": [f"{importer.SOURCE_REPOSITORY}@{importer.SOURCE_COMMIT}/exp8_math/runs/p50c"],
        "tensor_sha256": {item["name"]: item["base_sha256"] for item in manifest["tensors"]},
        "changes": [{"tensor": row["tensor"], "index": row["index"], "op": "set", "value": row["after"]} for row in changes],
        "projections": []}
    if host["patch_sha256"] != _canonical_sha256(patch):
        raise ValueError("Reference declarative patch does not replay from its immutable host receipt")
    return {"every_exported_value_matches_pinned_npz": True, "declarative_patch_hash_exact": True,
        "host_before_after_delta_norm_replayed": float(np.sqrt(square)),
        "scope": "Before values and source tensor identity remain hash-bound host measurements; this analysis does not load the original model."}


def summarize(root):
    root = Path(root)
    config, sources = json.loads(PROTOCOL.read_bytes()), source_hashes()
    roster, comparison, execution = (_read(root, name) for name in ("roster.json", "evaluation/comparison.json", "execution.json"))
    reconstruction, reconstruction_execution, compatibility = (_read(root, name) for name in
        ("reconstruction.json", "reconstruction-execution.json", "compatibility.json"))
    if (roster.get("schema") != "weightbridge.phase_iiip.reference_roster.v1" or roster.get("track") != "B"
            or roster["protocol_sha256"] != file_hash(PROTOCOL) or roster["sources"] != sources
            or comparison["protocol_sha256"] != file_hash(PROTOCOL) or comparison["roster_sha256"] != file_hash(root / "roster.json")):
        raise ValueError("Reference roster or comparison does not bind the frozen endpoint sources")
    counts = {"target_pairs": config["target_families"], "positive": config["positive_comparisons"], "preserve": config["copy_items"]}
    generator._validate(roster["rows"], config["split"], counts)
    if len(roster["rows"]) != config["budgets"]["presentations_per_model"]:
        raise ValueError("Reference roster exceeds the frozen presentation budget")
    authorization = PROTOCOL.with_name("reference_authorization.json")
    auth = json.loads(authorization.read_bytes())
    if (reconstruction.get("schema") != "weightbridge.phase_iiip.reference_reconstruction.v1"
            or reconstruction.get("status") != "REFERENCE_RECONSTRUCTED" or reconstruction.get("candidate_created") is not True
            or reconstruction["authorization_sha256"] != file_hash(authorization)
            or reconstruction["importer_source_sha256"] != file_hash(importer.__file__)
            or reconstruction["base_checkpoint_sha256_rechecked"] != config["base_checkpoint_sha256"]
            or reconstruction.get("source_checkpoint_preserved") is not True or reconstruction.get("lineage_accepted") is not False
            or reconstruction.get("production_deployment") is not False or reconstruction.get("generation_evaluations") != 0
            or reconstruction_execution.get("status") != "COMPLETED" or reconstruction_execution.get("error") is not None
            or reconstruction_execution["reconstruction_receipt_sha256"] != file_hash(root / "reconstruction.json")):
        raise ValueError("External reference reconstruction identity or authorization does not replay")
    importer._authorize(authorization)
    manifest, arrays, source_hashes_recorded = importer._inputs(root / "source")
    nested, host = reconstruction["compatibility"], reconstruction["host_patch_receipt"]
    expected_candidate = reconstruction["candidate_checkpoint_sha256"]
    if (nested.get("status") != "REFERENCE_COMPATIBLE" or nested["source_sha256"] != source_hashes_recorded
            or compatibility.get("status") != "REFERENCE_COMPATIBLE" or compatibility["source_sha256"] != source_hashes_recorded
            or nested["base_checkpoint_sha256"] != config["base_checkpoint_sha256"]
            or host["base_sha256"] != config["base_checkpoint_sha256"] or host["candidate_sha256"] != expected_candidate
            or host["changed_entries"] != auth["expected_changed_entries"]
            or set(host["tensor_receipts"]) != set(importer.TENSOR_NAMES)
            or any(host["tensor_receipts"][item["name"]]["base_sha256"] != item["base_sha256"]
                   or host["tensor_receipts"][item["name"]]["candidate_sha256"] != item["edited_sha256"] for item in manifest["tensors"])):
        raise ValueError("Reference artifact or published source/edited tensor identity differs")
    for key in ("tensor_details", "absolute_update_norm", "relative_affected_full_matrices_update_norm", "max_abs_delta", "changed_entries"):
        if nested[key] != compatibility[key]:
            raise ValueError("Compatibility and reconstruction numerical records disagree")
    asset_identity = nested["compatibility"]
    if (asset_identity != compatibility["compatibility"]
            or asset_identity["hf_metadata_sha256"] != file_hash(root / "source/qwen-pinned-api.json")
            or _read(root, "source/qwen-pinned-api.json").get("sha") != importer.MODEL_REVISION):
        raise ValueError("Pinned model revision metadata or compatibility asset identity differs")
    patch_replay = _verify_patch_receipt(host, nested, manifest, arrays, auth)
    if (comparison.get("schema") != "weightbridge.phase_iiip.reference_comparison.v1" or comparison.get("track") != "B"
            or comparison.get("status") != "BENCHMARK_COMPLETED"
            or comparison["reconstruction_receipt_sha256"] != file_hash(root / "reconstruction.json")
            or comparison["source_model"] != config["base_checkpoint_sha256"] or comparison["reference_model"] != expected_candidate
            or comparison.get("source_parent_restored_exactly") is not True
            or any(comparison.get(key) is not False for key in ("lineage_accepted", "new_weightbridge_learning_claimed", "long_benchmark_reproduction"))
            or execution.get("schema") != "weightbridge.phase_iiip.reference_execution.v1" or execution.get("status") != "COMPLETED"
            or execution.get("error") is not None or execution["comparison_sha256"] != file_hash(root / "evaluation/comparison.json")
            or execution["source_sha256"] != file_hash(PROTOCOL.with_name("reference_execute.py"))):
        raise ValueError("Reference comparison or completed execution identity differs")
    measured, workers, work = {}, {}, {}
    contract = endpoint(config)
    for name, digest in (("original", config["base_checkpoint_sha256"]), ("reference", expected_candidate)):
        path = root / f"evaluation/{name}.json"
        worker = _read(root, f"evaluation/{name}.json")
        if (comparison["worker_sha256"][name] != file_hash(path)
                or worker.get("schema") != "weightbridge.phase_iiip.reference_worker.v1" or worker.get("track") != "B"
                or worker["checkpoint_sha256"] != digest or worker["roster_sha256"] != file_hash(root / "roster.json")
                or worker["sources"] != sources or worker.get("lineage_authority") is not False
                or worker["resources"] != comparison["resources"]["workers"][name]):
            raise ValueError("Reference worker does not bind its model, roster, sources and recorded resources")
        measured[name] = _checked_measurement(worker["measurement"], roster["rows"], contract)
        if worker["summary"] != semantic_summary(measured[name]) or comparison[f"{name}_summary"] != worker["summary"]:
            raise ValueError("Reference worker semantic summary does not independently replay")
        workers[name], work[name] = worker, _work(measured[name], contract)
    parent, candidate = measured["original"], measured["reference"]
    gate = compare(parent, candidate, min_comparison_gain=1, min_repeatable_corrections=1)
    if gate != comparison["paired"]:
        raise ValueError("Reference comparison hides a correction, regression or semantic gate change")
    all_ids = [row["id"] for row in parent["rows"]]
    target_ids = [row["id"] for row in parent["rows"] if row["group"] == "target"]
    controls = [row["id"] for row in parent["rows"] if row["group"] != "target"]
    components = {"compatibility_inspection": compatibility["resources"],
        "reconstruction_execution": reconstruction_execution["resources"], "reference_execution_host": execution["resources"],
        **{f"worker_{name}": worker["resources"] for name, worker in workers.items()}}
    compute = {"components": components, "recorded_cpu_seconds_lower_bound": sum(v["cpu_seconds"] for v in components.values()),
        "recorded_serial_wall_seconds": compatibility["resources"]["wall_seconds"] + reconstruction_execution["resources"]["wall_seconds"] + execution["resources"]["wall_seconds"],
        "largest_recorded_peak_rss_bytes": max(v["peak_rss_bytes"] for v in components.values()),
        "model_work": work, "generated_responses": sum(v["responses"] for v in work.values()),
        "generated_native_tokens": sum(v["generated_native_tokens"] for v in work.values()),
        "model_forward_calls": sum(v["forward_calls"] for v in work.values()), "model_backward_calls": 0,
        "preliminary_source_inspection_cpu_seconds": None, "ai_token_tool_cost": None,
        "all_phase_costs_fully_accounted": False,
        "scope": "Count compatibility once, reconstruction wrapper once (not nested function again), evaluation host wrapper once plus two worker CPUs once. Host wall already includes worker wall. Preliminary fetching/manual asset/hash/norm inspections lack exact CPU totals; tests, analyst replay and assistant costs are separate, not zero."}
    return {"schema": "weightbridge.phase_iiip.reference_analysis.v1", "track": "B", "status": comparison["status"],
        "analysis_source_sha256": file_hash(__file__), "protocol_sha256": file_hash(PROTOCOL),
        "roster_sha256": file_hash(root / "roster.json"), "reconstruction_receipt_sha256": file_hash(root / "reconstruction.json"),
        "comparison_sha256": file_hash(root / "evaluation/comparison.json"), "source_model": comparison["source_model"],
        "reference_model": comparison["reference_model"], "source_commit": importer.SOURCE_COMMIT,
        "source_artifacts": source_hashes_recorded, "update_norms": {key: nested[key] for key in
            ("absolute_update_norm", "source_affected_full_matrices_norm", "relative_affected_full_matrices_update_norm", "relative_norm_scope", "max_abs_delta", "changed_entries", "tensor_details")},
        "patch_receipt_replay": patch_replay,
        "preliminary_importer_source_sha256": compatibility["importer_source_sha256"],
        "final_importer_source_sha256": reconstruction["importer_source_sha256"],
        "preliminary_source_version_note": "Preflight inspection predates final authorization/restoration guards; preserve its original receipt. Final reconstruction and evaluation bind the final importer source.",
        "all_presentations": _counts(parent, candidate, all_ids), "target_presentations": _counts(parent, candidate, target_ids),
        "controls": _counts(parent, candidate, controls), "strict_and_family_gate": gate,
        "presentation_gain_interval": clustered_presentation_interval(parent, candidate),
        "strict_comparison_gain_interval": {**gate["family_gain_interval"], "estimand": "Paired strict-comparison accuracy change, clustered by underlying family",
            "scope": "Degenerate observed zero changes do not establish zero population uncertainty."},
        "strata": _strata(parent, candidate), "families": _families(parent, candidate), "compute": compute,
        "lineage_accepted": False, "new_weightbridge_learning_claimed": False, "long_benchmark_reproduction": False,
        "interpretation": "External repetition edit changes some completed decimal answers, including a prohibited loss. Presentation gain is not strict-comparison or complete-family transfer and cannot qualify zero-regression adaptation.",
        "scope": comparison["scope"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    json_dump_atomic(args.out, summarize(args.root))
