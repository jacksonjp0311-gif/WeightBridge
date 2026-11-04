"""Replay actual paired-supervision outcomes and matched family-level contrasts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.phase_iiih.run_decimal_endpoint import endpoint, qualify
from experiments.phase_iiil.analyze import _checked_measurement, _subgroup_outcomes
from experiments.phase_iiil.semantic_diagnostics import summarize as semantic_summary
from experiments.phase_iiim.analyze import (
    _family_changes, _practice, _stored_record, _numerical_rejections, _work_counts, _termination_diagnostics,
)
from experiments.phase_iiin.archive_trial import public_records
from experiments.phase_iiin.semantic_roster import PROTOCOL, load_split, read_manifest
from experiments.phase_iiin.source_manifest import ROOT
from weightbridge.adaptation.factorial_geometry import compare_realized_norms, stored_update_cosines
from weightbridge.adaptation.paired_supervision import mean_direction, pair_gradient_diagnostics
from weightbridge.adaptation.preservation_geometry import DecisionAnchor
from weightbridge.core import file_hash, json_dump_atomic


def _cluster_difference(before, first, second):
    """Paired family bootstrap; all eight related presentations stay together."""
    a, b = _family_changes(before, first), _family_changes(before, second)
    families = sorted(a)
    values = np.array([b[name] - a[name] for name in families], dtype=float)
    rng = np.random.default_rng(73019)
    draws = values[rng.integers(0, len(values), size=(4096, len(values)))].mean(axis=1) / 2
    return {"unit": "underlying truth-flip family", "family_count": len(families),
            "strict_comparison_accuracy_difference": float(values.mean() / 2),
            "descriptive_95pct_family_bootstrap": np.quantile(draws, [.025, .975]).tolist(),
            "per_family_strict_count_difference": dict(zip(families, values.astype(int).tolist())),
            "scope": "Frozen development can rank finalists; this is not independent selection."}


def _distribution(measured):
    summary = semantic_summary(measured)
    target = [row for row in measured["rows"] if row["group"] == "target"]
    return {"families": summary["families"], "comparisons": summary["comparisons"],
            "truth_flip_matrix": summary["truth_flip_matrix"], "subgroups": summary["subgroups"],
            "wording": summary["by_wording"], "order": summary["by_order"],
            "valid_completion_rate": sum(row["valid"] for row in measured["rows"]) / len(measured["rows"]),
            "target_presentations": len(target)}


def _verify_geometry(config, records):
    reference = config["fixed_geometry"]
    for name in ("basis", "fit_report"):
        if file_hash(ROOT / reference[f"{name}_path"]) != reference[f"{name}_sha256"]:
            raise ValueError("Audited Phase III-M geometry source differs")
    if "basis.json" not in records:
        return
    old = json.loads((ROOT / reference["basis_path"]).read_bytes())
    basis, receipt = records["basis.json"], records["reference-geometry.json"]
    if (basis["columns"] != old["columns"] or basis["right_factor"] != old["right_factor"]
            or receipt["columns"] != old["columns"] or receipt["right_factor"] != old["right_factor"]
            or any(receipt[key] != value for key, value in reference.items())
            or receipt["checkpoint_sha256"] != config["checkpoint_sha256"]):
        raise ValueError("Candidate geometry does not exactly reuse audited Phase III-M columns and factor")


def _verify_directions(records):
    if "directions.json" not in records:
        return
    basis, record = records["basis.json"], records["directions.json"]
    ids = basis["item_ids"]
    matrix = np.asarray(basis["projected_gradients"], dtype=float)
    indexes = {name: index for index, name in enumerate(ids)}
    for arm in ("A", "B"):
        selected = record["arm_items"][arm]
        expected = mean_direction(selected, matrix[[indexes[item] for item in selected]])
        if not np.array_equal(expected, record["arms"][arm]["direction"]):
            raise ValueError("Mean direction does not replay from equal-weight comparison gradients")
    if record["arm_items"]["B"] != record["arm_items"]["C"]:
        raise ValueError("Pair-aware arm differs in supervised comparisons")
    families = [next(fork["metadata"]["pair_id"] for fork in basis["forks"] if fork["comparison_id"] == item)
                for item in ids]
    expected = pair_gradient_diagnostics(ids, families, basis["item_groups"], matrix,
                                       margins=np.asarray(basis["comparison_parent_margins"]))
    if expected != record["pairs"]:
        raise ValueError("Paired gradient cosine/norm compatibility does not replay")


def summarize(trial, material):
    raw = public_records(trial, material)
    records = {name: json.loads(value) for name, value in raw.items()}
    report, assessment = records["fit-report.json"], records["assessment.json"]
    config, manifest = json.loads(PROTOCOL.read_bytes()), read_manifest(material)
    _verify_geometry(config, records)
    _verify_directions(records)
    contract = endpoint(config)
    frozen = {name: load_split(material, name, manifest) for name in report["optimization_splits"]}
    parents = {name: _checked_measurement(records[f"parent-{name}.json"]["measurement"], rows, contract)
               for name, rows in frozen.items()}
    for name, measured in parents.items():
        if semantic_summary(measured) != records[f"parent-{name}.json"]["summary"]:
            raise ValueError("Parent semantic summary differs from completed generation")
    if qualify(parents["adaptation"], config) != report["qualification"]:
        raise ValueError("Parent qualification differs from frozen endpoint evidence")
    parent, protected = parents["adaptation"], parents["preservation-fit"]
    basis = records.get("basis.json", {})
    sampling = records.get("sampling.json", {})
    anchors = tuple(DecisionAnchor(**row) for row in sampling.get("anchors", ()))
    arms, deltas, generations, owners, work = {}, {}, {}, {}, {}
    for name, record in records.items():
        if name in report["artifacts"] and not name.startswith("arm-"):
            scopes = _work_counts(record)
            if scopes:
                work[name] = scopes
    for dose in range(1, len(config["probe_design"]["target_norms"]) + 1):
        for label in "ABC":
            name = f"{label}{dose}"
            record = records.get(f"arm-{name}.json")
            if record is None:
                continue
            selected, probe = record["norm_match"].get("selected"), record["probe"]
            digest = selected["tensor_sha256"] if selected else None
            delta = _stored_record(record, config)
            if delta is not None:
                if record["columns"] != basis["columns"] or record["right_factor"] != basis["right_factor"]:
                    raise ValueError("Arms do not share fixed inherited columns/right factor")
                deltas[name] = delta
            alias = record.get("duplicate_of")
            row = {"arm": name, "allocation": config["probe_design"]["arms"][label],
                   "intended_norm": record["intended_norm"], "realized_norm": selected["realized_norm"] if selected else None,
                   "changed_entries": selected["changed_entries"] if selected else None,
                   "tensor_sha256": digest, "norm_match": record["norm_match"]["status"],
                   "reason": probe["reason"], "duplicate_of": alias,
                   "direction_specific_allowance": record["direction_specific_allowance"],
                   "calibration_sufficient": record["calibration_sufficient"], "floor": record["floor"],
                   "numerical_rejections": _numerical_rejections(record, config),
                   "numeric_resources": record["resources"], "probe_resources": probe.get("resources")}
            if alias:
                if alias not in arms or owners.get(digest) != alias:
                    raise ValueError("Alias does not bind an earlier identical stored state")
                for key in ("practice_gate", "subgroups", "truth_flip_matrix", "family_comparison_changes",
                            "fit_regressions", "fit_validity_losses", "minimum_measured_margin", "fork", "fork_token_classes",
                            "distribution", "termination_diagnostics"):
                    row[key] = arms[alias][key]
                generations[name] = generations[alias]
            elif "endpoint" in probe:
                if digest in owners or record["norm_match"]["status"] != "MATCHED" or not record["calibration_sufficient"]:
                    raise ValueError("Decoded state duplicated, unmatched or uncalibrated")
                owners[digest] = name
                # The active preservation reference is common across arms. Arm
                # fork objectives differ only in their authorized fit population.
                reference = records.get(f"probe-reference-{label}.json")
                if reference is None and label == "C":
                    reference = records.get("probe-reference-B.json")
                if reference is None:
                    raise ValueError("Decoded arm lacks its native fork and preservation-prefix reference")
                row.update(_practice(record, parent, protected, frozen, contract, config, anchors, reference["prompt_ids"]))
                parent_classes = records["token-class-parent.json"]["arms"][label]
                if "shared_with" in parent_classes:
                    parent_classes = records["token-class-parent.json"]["arms"][parent_classes["shared_with"]]
                selected_forks = [basis["forks"][i] for i in record["fork_indices"]]
                if [fork["prompt_id"] for fork in selected_forks] != record["fork_prompt_ids"]:
                    raise ValueError("Decoded fork sequence differs from its selected comparison gradients")
                row["termination_diagnostics"] = _termination_diagnostics(record, {"forks": selected_forks},
                    parent_classes, records["fork-diagnosis.json"])
                measured = probe["endpoint"]["measurement"]
                generations[name] = measured
                row["distribution"] = _distribution(measured)
            arms[name] = row
            scope = {"numeric": {"resources": record["resources"]}}
            if probe.get("resources"):
                scope["probe"] = {"resources": probe["resources"]}
            if "endpoint" in probe:
                scope["generation"] = {key: {"resources": probe["endpoint"][key]["resources"]}
                    for key in ("measurement", "preservation_fit_measurement", "fork_token_classes")
                    if key in probe["endpoint"]}
            work[f"arm-{name}.json"] = _work_counts(scope)
    unique = sum("practice_gate" in row and not row["duplicate_of"] for row in arms.values())
    if unique != report["detail"].get("unique_complete_evaluations", 0) or unique > 6:
        raise ValueError("Actual decoding count differs from frozen six-candidate budget")
    matching = {}
    for dose, norm in enumerate(config["probe_design"]["target_norms"], 1):
        selected = {name: {"realized_norm": row["realized_norm"], "tensor_sha256": row["tensor_sha256"]}
                    for name, row in arms.items() if name.endswith(str(dose)) and row["realized_norm"] is not None}
        matching[str(dose)] = compare_realized_norms(selected, intended_norm=norm,
            relative_tolerance=config["probe_design"]["norm_relative_tolerance"]) if selected else None
    development, development_generations = {}, {}
    before_dev = None
    parent_file = "verification/development-transfer-base.json"
    if parent_file in records:
        before_dev = records[parent_file]["measurement"]
        for name, gate in assessment["development"].items():
            path = f"verification/development-transfer-{name}.json"
            if path in records:
                measured = records[path]["measurement"]
                development_generations[name] = measured
                development[name] = {"gate": gate, "distribution": _distribution(measured),
                    "family_comparison_changes": _family_changes(before_dev, measured),
                    "subgroups": _subgroup_outcomes(before_dev, measured, gate)}
    contrasts = {}
    for dose, norm in enumerate(config["probe_design"]["target_norms"], 1):
        for first, second in ((f"A{dose}", f"B{dose}"), (f"B{dose}", f"C{dose}")):
            key = f"{first}:{second}"
            selected = {name: {"realized_norm": arms[name]["realized_norm"], "tensor_sha256": arms[name]["tensor_sha256"]}
                        for name in (first, second) if name in arms and arms[name]["realized_norm"] is not None}
            norms = compare_realized_norms(selected, intended_norm=norm,
                relative_tolerance=config["probe_design"]["norm_relative_tolerance"]) if len(selected) == 2 else None
            matched = bool(norms and norms["cohort_status"] == "MATCHED")
            distinct = len({row["tensor_sha256"] for row in selected.values()}) == 2
            contrast = {"matched_norms": norms, "distinct_stored_states": distinct,
                "causal_scope": "A:B supervision allocation confounds pairing with family diversity; B:C isolates solver on identical paired data."}
            for population, measurements, baseline in (("practice", generations, parent),
                    ("development", development_generations, before_dev)):
                if matched and distinct and first in measurements and second in measurements and baseline:
                    contrast[population] = _cluster_difference(baseline, measurements[first], measurements[second])
                else:
                    contrast[population] = {"informative": False, "reason": "requires distinct norm-matched decoded states"}
            contrasts[key] = contrast
    totals = Counter()
    for scopes in work.values():
        for counts in scopes.values():
            totals.update(counts)
    independent = records.get("independent-outcome.json")
    return {"schema": "weightbridge.phase_iiin.analysis.v1",
        "fit_report_sha256": hashlib.sha256(raw["fit-report.json"]).hexdigest(),
        "assessment_sha256": hashlib.sha256(raw["assessment.json"]).hexdigest(),
        "disposition": independent["disposition"] if independent else assessment["disposition"],
        "reason": independent["reason"] if independent else assessment["reason"], "lineage_accepted": False,
        "hidden_opened": independent.get("hidden_opened") if independent else False,
        "parent_qualification": report["qualification"], "parent_diagnostics": _distribution(parent),
        "fit_detail": report["detail"], "basis": basis, "directions": records.get("directions.json"),
        "selection": records.get("supervision-selection.json"),
        "gradient_compatibility": records.get("directions.json", {}).get("pairs"),
        "sampling": sampling, "arms": arms, "unique_decoded_probes": unique,
        "practice_passers": [name for name, row in arms.items() if row.get("practice_gate", {}).get("passed") and not row["duplicate_of"]],
        "stored_norm_matching": matching, "stored_update_comparison": stored_update_cosines(deltas) if deltas else {},
        "matched_allocation_contrasts": contrasts, "development": development,
        "preservation": assessment["preservation"], "independent_selection": assessment["selection"], "independent": independent,
        "resources": {"fitting": report["resources"], "fitting_stage_counts": work,
                      "fitting_work_totals": dict(sorted(totals.items())),
                      "verification": assessment["worker_resources"], "verification_host": assessment["host_resources"]}}


def plot(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    measured = [(name, row) for name, row in summary["arms"].items() if row.get("practice_gate")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    names = [name for name, _ in measured]
    x = np.arange(len(names))
    for axis, metric, title in ((axes[0], "target_gain", "Strict four-form comparison gains"),
                                (axes[1], "complete_family_gain", "Complete eight-form family gains")):
        axis.bar(x - .18, [row["practice_gate"][metric] for _, row in measured], width=.35, label="Practice")
        axis.bar(x + .18, [summary["development"].get(name, {}).get("gate", {}).get(metric, float("nan")) for name in names],
                 width=.35, label="Development (if authorized)")
        axis.set_xticks(x, names)
        axis.set_title(title)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.axhline(0, color="gray", linewidth=.6)
        axis.legend(frameon=False)
        absent = [name for name in names if name not in summary["development"]]
        if absent:
            axis.text(.02, .97, "Development not measured: " + ", ".join(absent),
                      transform=axis.transAxes, va="top", fontsize=8)
    if not measured:
        axes[0].text(.5, .5, "No eligible complete-generation probes", ha="center")
    fig.savefig(out, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    result = summarize(args.trial, args.material)
    if args.out:
        json_dump_atomic(args.out, result)
    else:
        print(json.dumps(result, indent=2))
    if args.figure:
        plot(result, args.figure)
