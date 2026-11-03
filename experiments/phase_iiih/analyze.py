"""Replay the public Phase III-H practice gates and plot the measured doses."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from math import isclose
from pathlib import Path

from experiments.phase_iiih.run_decimal_endpoint import PROTOCOL, endpoint, read_json, sources
from weightbridge.adaptation.behavior_endpoint import compare_endpoint
from weightbridge.core import file_hash


def replay(run: Path) -> list[dict]:
    report = read_json(run / "report.json")
    config = read_json(PROTOCOL)
    contract = endpoint(config)
    if report["sources"] != sources():
        raise ValueError("Archived endpoint source commitments differ")
    run_root = run.resolve()
    for name, digest in report["artifacts"].items():
        target = (run / name).resolve()
        if not target.is_relative_to(run_root) or file_hash(target) != digest:
            raise ValueError(f"Archived endpoint evidence differs: {name}")
    practice = read_json(run / "practice.json")
    rows = []
    for branch in ("legacy", "corrected"):
        path = run / f"probes-{branch}.json"
        if not path.exists():
            continue
        probes = read_json(path)
        eligible, legacy_eligible = [], []
        for probe in probes["probes"]:
            row = {"branch": branch, "id": probe["id"], "dose": probe["dose"],
                   "changed_entries": probe["changed_entries"], "norm": probe["realized_update_norm"],
                   "reason": probe["reason"]}
            if "endpoint" in probe:
                guard = compare_endpoint(practice["measurement"], probe["endpoint"]["measurement"], contract,
                                         min_repeatable_corrections=1)
                corrected = sorted(set(guard["corrected_repeatable_target_items"])
                                   & practice["qualification"]["repeatable_errors"].keys())
                guard["corrected_repeatable_negative_items"] = corrected
                guard["passed"] = bool(guard["passed"] and corrected)
                if guard != probe["endpoint"]["guard"] or probe.get("restored_exactly") is not True:
                    raise ValueError(f"Endpoint gate or restoration differs for {branch} {probe['id']}")
                row.update({"gate": guard["passed"], "item_gain": guard["target_item_gain"],
                            "presentation_gain": guard["target_presentation_gain"],
                            "regressions": len(guard["prohibited_regressions"]),
                            "lost_validity": len(guard["lost_valid_presentations"]),
                            "surrogate_gain": probe["margin"]["surrogate_gain"],
                            "legacy_margin_gate": probe["margin"]["passed"],
                            "surrogate_corrections": probe["margin"]["corrections"],
                            "mean_kl": probe.get("distribution", {}).get("mean_kl"),
                            "divergences": dict(Counter(event.get("parent_phase", "unclassified")
                                                        for event in probe["first_divergences"]))})
                if row["mean_kl"] is not None:
                    penalties = config["probe_design"]
                    objective = (probe["margin"]["loss"] + penalties["kl_weight"] * row["mean_kl"]
                                 + penalties["norm_weight"] * row["norm"]**2)
                    if not isclose(objective, probe["distribution"]["objective"], rel_tol=1e-12):
                        raise ValueError("Archived regularized objective differs")
                if guard["passed"]:
                    eligible.append(probe)
                if row["legacy_margin_gate"]:
                    legacy_eligible.append(probe)
            rows.append(row)
        selections = {
            "margin": max(legacy_eligible, key=lambda p: (p["margin"]["corrections"], -p["margin"]["loss"]),
                          default={}).get("id"),
            "endpoint": max(eligible, key=lambda p: (p["endpoint"]["guard"]["target_item_gain"],
                                                     -p["realized_update_norm"]), default={}).get("id"),
            "distribution": min(eligible, key=lambda p: p["distribution"]["objective"], default={}).get("id"),
        }
        if selections != probes["selected"]:
            raise ValueError(f"Archived probe selections differ for {branch}")
        expected = {"A": selections["margin"]} if branch == "legacy" else {
            "B": selections["margin"], "C": selections["endpoint"], "D": selections["distribution"]}
        if any(report["arms"][key] != value for key, value in expected.items()):
            raise ValueError("Archived ablation does not match measured selections")
    return rows


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), layout="constrained")
    for branch, color, marker in (("legacy", "#94441F", "o"), ("corrected", "#155A86", "s")):
        probes = [r for r in rows if r["branch"] == branch and "item_gain" in r]
        if not probes:
            continue
        x = [r["norm"] for r in probes]
        for axis, field in zip(axes, ("surrogate_gain", "item_gain", "regressions"), strict=True):
            axis.plot(x, [r[field] for r in probes], label=branch, color=color, marker=marker)
    for axis, title in zip(axes, ("Conditional loss reduction", "Both-wording target gain", "Regressed responses"),
                           strict=True):
        axis.axhline(0, color="#999999", linewidth=.8, zorder=0)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Realized BF16 update norm")
    axes[0].legend(frameon=False)
    fig.savefig(out, dpi=170, metadata={"Description": "Measured Qwen Phase III-H practice probes"})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    result = replay(args.run)
    if args.plot and any("item_gain" in row for row in result):
        plot(result, args.plot)
    print(json.dumps(result, indent=2))
