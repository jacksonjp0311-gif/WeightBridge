"""Build an evidence summary and figures from immutable WeightBridge records.

The script reads the append-only ledger and saved run artifacts. It does not
create demonstration or interpolated metrics. Plots keep different task suites
visually grouped because their accuracy values are not directly comparable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "results" / "experiment_ledger.jsonl"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_ledger() -> list[dict[str, Any]]:
    if not LEDGER.is_file():
        raise FileNotFoundError(f"Experiment ledger not found: {LEDGER}")
    return [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_experiments(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Exclude explicitly superseded records and duplicate measurement keys."""
    superseded: set[str] = set()
    addenda: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("schema") != "weightbridge.ledger_correction.v1":
            continue
        if record.get("disposition") == "METADATA_BACKFILLED":
            target = record.get("references_experiment_id")
            data = record.get("record_addendum")
            if isinstance(target, str) and isinstance(data, dict):
                addenda.setdefault(target, []).append(data)
            continue
        if record.get("disposition") == "IDENTITY_METADATA_BACKFILLED":
            target = record.get("canonical_experiment_id")
            measurement_key = record.get("measurement_key")
            if isinstance(target, str) and isinstance(measurement_key, str):
                addenda.setdefault(target, []).append({"measurement_key": measurement_key})
            continue
        value = record.get("supersedes")
        if isinstance(value, str):
            superseded.add(value)
        elif isinstance(value, list):
            superseded.update(str(item) for item in value)
    selected: list[dict[str, Any]] = []
    keys: set[str] = set()
    for record in records:
        if record.get("schema") != "weightbridge.experiment.v1":
            continue
        if record.get("experiment_id") in superseded:
            continue
        resolved = dict(record)
        for addendum in addenda.get(record.get("experiment_id"), []):
            for key, value in addendum.items():
                if isinstance(resolved.get(key), dict) and isinstance(value, dict):
                    resolved[key] = {**resolved[key], **value}
                else:
                    resolved[key] = value
        key = resolved.get("measurement_key")
        if key and key in keys:
            continue
        if key:
            keys.add(key)
        selected.append(resolved)
    return selected


def paired_cluster_bootstrap(
    baseline_by_seed: list[list[bool]], candidate_by_seed: list[list[bool]], *,
    seed: int = 77124, resamples: int = 10_000,
) -> dict[str, Any]:
    """Bootstrap paired correctness differences by independently generated task seed."""
    if len(baseline_by_seed) != len(candidate_by_seed) or not baseline_by_seed:
        raise ValueError("paired seed groups must be non-empty and have equal length")
    if any(len(a) != len(b) for a, b in zip(baseline_by_seed, candidate_by_seed)):
        raise ValueError("paired correctness arrays must have equal lengths within each seed")
    differences = np.asarray([
        np.mean(np.asarray(c, dtype=float) - np.asarray(b, dtype=float))
        for b, c in zip(baseline_by_seed, candidate_by_seed)
    ], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    for i in range(resamples):
        draws[i] = differences[rng.integers(0, len(differences), len(differences))].mean()
    return {
        "mean_difference": float(differences.mean()),
        "ci95_clustered_by_seed": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "number_of_independent_seeds": int(len(differences)),
        "examples_per_seed": [int(len(x)) for x in baseline_by_seed],
        "bootstrap_seed": seed,
        "resamples": resamples,
    }


def aggregate_mwc(roots: list[Path]) -> dict[str, Any]:
    names = ("no_memory_no_edit", "external_memory_only", "weight_edit_only", "memory_plus_weight_edit")
    examples: dict[str, list[list[bool]]] = {name: [] for name in names}
    raw_paths: list[str] = []
    cpu = wall = 0.0
    evaluator_examples: dict[str, int] = {}
    for root in roots:
        summary = read_json(root / "mwc_ablation.json")
        raw_paths.append(relative(root / "mwc_ablation.json"))
        experiment = next((x for x in load_ledger() if x.get("experiment_id") == summary["experiment_id"]), {})
        resources = experiment.get("resources", {})
        cpu += float(resources.get("cpu_seconds") or 0)
        wall += float(resources.get("wall_seconds") or 0)
        for name, value in resources.get("practice_forward_examples", {}).items():
            evaluator_examples[name] = evaluator_examples.get(name, 0) + int(value)
        for index, _ in enumerate(summary["replicates"]):
            folder = sorted(root.glob(f"replicate-{index:02d}-seed-*/evaluation.json"))
            if len(folder) != 1:
                raise FileNotFoundError(f"Expected one evaluation for replicate {index} under {root}")
            evaluation_path = folder[0]
            raw_paths.append(relative(evaluation_path))
            evaluation = read_json(evaluation_path)
            for name in names:
                examples[name].append([bool(v) for v in evaluation[name]["per_example_correct"]])
    baseline = examples["no_memory_no_edit"]
    conditions: dict[str, Any] = {}
    for name in names:
        scores = np.asarray([np.mean(seed_values) for seed_values in examples[name]], dtype=float)
        record: dict[str, Any] = {
            "accuracy": float(np.mean([np.mean(x) for x in examples[name]])),
            "correct": int(sum(sum(x) for x in examples[name])),
            "n": int(sum(len(x) for x in examples[name])),
            "seed_level_accuracy": [float(x) for x in scores],
            "regression_rate_vs_no_memory_no_edit": None,
            "raw_per_example_path_count": len(roots) * 5,
        }
        if name != "no_memory_no_edit":
            diffs = paired_cluster_bootstrap(baseline, examples[name])
            record["paired_difference_vs_no_memory_no_edit"] = diffs
            regressions = sum(sum(bool(b) and not bool(c) for b, c in zip(bseed, cseed))
                              for bseed, cseed in zip(baseline, examples[name]))
            record["regressions_vs_no_memory_no_edit"] = regressions
            record["regression_rate_vs_no_memory_no_edit"] = regressions / sum(map(len, baseline))
        conditions[name] = record
    return {
        "conditions": conditions,
        "seed_count": len(baseline),
        "example_count_per_condition": int(sum(map(len, baseline))),
        "seed_clustered_uncertainty": True,
        "practice_forward_examples_recorded": evaluator_examples,
        "recorded_cpu_seconds": cpu,
        "recorded_wall_seconds_sum": wall,
        "raw_evidence": raw_paths,
        "interpretation_limit": "Synthetic injected-bias task family only; external memory is a retrieval-time logit bias and does not modify checkpoint parameters.",
    }


def _accuracy_figures(summary: dict[str, Any], outdir: Path) -> None:
    import matplotlib.pyplot as plt

    # Candidate accuracy: deliberately separate task suites with whitespace.
    groups = [
        ("v0.1 demo", [("baseline", summary["v01_demo"]["baseline_accuracy"]),
                       ("pre-specified edit", summary["v01_demo"]["candidate_accuracy"])]),
        ("synthetic discovery", [("baseline", summary["toy_discovery"]["baseline_accuracy"]),
                                  ("diagnostic agent", summary["toy_discovery"]["candidate_accuracy"]),
                                  ("random control", summary["toy_discovery"]["random_control_accuracy"])]),
        ("MWC, pooled 10 seeds", []),
        ("synthetic SGD", [("baseline", summary["gradient_baseline"]["baseline_accuracy"]),
                           ("SGD", summary["gradient_baseline"]["candidate_accuracy"])]),
        ("Qwen CPU smoke, cap 64", []),
    ]
    # Use actual MWC result and avoid hard-coded expected results.
    mwc = summary["mwc_ablation"]["conditions"]
    groups[2] = (groups[2][0], [("no memory / no edit", mwc["no_memory_no_edit"]["accuracy"]),
                               ("external memory", mwc["external_memory_only"]["accuracy"]),
                               ("weight edit", mwc["weight_edit_only"]["accuracy"]),
                               ("memory + edit", mwc["memory_plus_weight_edit"]["accuracy"])])
    groups[4] = (groups[4][0], [("baseline", summary["qwen_smoke_64"]["baseline_accuracy"]),
                               ("published p50", summary["qwen_smoke_64"]["candidate_accuracy"])])
    fig, axes = plt.subplots(len(groups), 1, figsize=(11, 12), constrained_layout=True)
    for ax, (title, items) in zip(axes, groups):
        names, values = zip(*items)
        bars = ax.bar(names, values, color=["#657b8b" if "baseline" in n or "no edit" in n else "#357a66" for n in names])
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Accuracy")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", alpha=.2)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + .015, f"{value:.1%}", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", labelrotation=12)
    fig.suptitle("Measured accuracy by candidate (task suites shown separately)", fontsize=14)
    fig.savefig(outdir / "accuracy_by_candidate.png", dpi=160)
    plt.close(fig)

    # Candidate-minus-baseline differences, not cross-task pooled inference.
    rows = [
        ("v0.1 pre-specified edit", summary["v01_demo"]["candidate_accuracy"] - summary["v01_demo"]["baseline_accuracy"]),
        ("toy diagnostic agent", summary["toy_discovery"]["candidate_accuracy"] - summary["toy_discovery"]["baseline_accuracy"]),
        ("toy random control", summary["toy_discovery"]["random_control_accuracy"] - summary["toy_discovery"]["baseline_accuracy"]),
        ("MWC external memory", mwc["external_memory_only"]["accuracy"] - mwc["no_memory_no_edit"]["accuracy"]),
        ("MWC weight edit", mwc["weight_edit_only"]["accuracy"] - mwc["no_memory_no_edit"]["accuracy"]),
        ("MWC memory + edit", mwc["memory_plus_weight_edit"]["accuracy"] - mwc["no_memory_no_edit"]["accuracy"]),
        ("toy conventional SGD", summary["gradient_baseline"]["candidate_accuracy"] - summary["gradient_baseline"]["baseline_accuracy"]),
        ("Qwen published p50 smoke", summary["qwen_smoke_64"]["candidate_accuracy"] - summary["qwen_smoke_64"]["baseline_accuracy"]),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    labels, values = zip(*rows)
    colors = ["#357a66" if value > 0 else "#657b8b" if value == 0 else "#ba554a" for value in values]
    bars = ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_xlabel("Candidate accuracy minus paired baseline")
    ax.set_title("Measured candidate differences (within task only)")
    ax.grid(axis="x", alpha=.2)
    for bar, value in zip(bars, values):
        ax.text(value + (.006 if value >= 0 else -.006), bar.get_y() + bar.get_height()/2,
                f"{value:+.1%}", ha="left" if value >= 0 else "right", va="center", fontsize=9)
    fig.savefig(outdir / "candidate_differences.png", dpi=160)
    plt.close(fig)


def _qwen_figures(summary: dict[str, Any], outdir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["base 64", "p50 64", "base 256", "p50 256"]
    limits = summary["qwen_smoke_64"]["token_limit_rates"] + summary["qwen_smoke_256"]["token_limit_rates"]
    repeat = summary["qwen_smoke_64"]["tail_repetition_8gram_rates"] + summary["qwen_smoke_256"]["tail_repetition_8gram_rates"]
    loops = summary["qwen_smoke_64"]["reasoning_loop_rates"] + [None, None]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), constrained_layout=True)
    for ax, vals, title in [
        (axes[0], limits, "Token-limit rate"),
        (axes[1], repeat, "8-gram tail repetition proxy"),
        (axes[2], loops, "Repeated-block detector (64-token run only)"),
    ]:
        values = [np.nan if val is None else val for val in vals]
        bars = ax.bar(x, values, color=["#657b8b", "#357a66", "#657b8b", "#357a66"])
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Rate")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", alpha=.2)
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x()+bar.get_width()/2, val+.015, f"{val:.1%}", ha="center", fontsize=9)
    fig.suptitle("Qwen CPU smoke diagnostics; neither cap is a paper reproduction", fontsize=13)
    fig.savefig(outdir / "qwen_smoke_token_and_repetition_rates.png", dpi=160)
    plt.close(fig)


def _patch_figures(summary: dict[str, Any], outdir: Path) -> None:
    import matplotlib.pyplot as plt

    layers = ["layer 15", "layer 19"]
    p50 = summary["published_patch"]["changed_by_layer"]
    random = summary["random_control"]["changed_by_layer"]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.bar(x-.18, [p50[str(i)] for i in (15, 19)], .36, label="published p50c", color="#357a66")
    ax.bar(x+.18, [random[str(i)] for i in (15, 19)], .36, label="matched random control", color="#ba8b42")
    ax.set_xticks(x, layers)
    ax.set_ylabel("Changed BF16 values")
    ax.set_title("Rebuilt checkpoint changes by layer (behavior not evaluated for random control)")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.savefig(outdir / "parameter_changes_by_layer.png", dpi=160)
    plt.close(fig)


def _mcp_agent_figures(summary: dict[str, Any], outdir: Path) -> None:
    import matplotlib.pyplot as plt

    runs = summary["mcp_agent_discovery_pilots"]["runs"]
    if not runs:
        return
    labels = [run["experiment_id"][-8:] for run in runs]
    x = np.arange(len(runs))
    width = .25
    fig, ax = plt.subplots(figsize=(max(7, len(runs) * 1.5), 5), constrained_layout=True)
    for offset, key, label, color in (
        (-width, "baseline_accuracy", "Original", "#657b8b"),
        (0, "candidate_accuracy", "AI MCP candidate", "#357a66"),
        (width, "random_control_accuracy", "Matched random edit", "#ba8b42"),
    ):
        values = [run[key] if run[key] is not None else np.nan for run in runs]
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width()/2, value + .015, f"{value:.1%}",
                        ha="center", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Holdout accuracy")
    ax.set_title("LLM-directed MCP toy candidates (one independent task per run)")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.savefig(outdir / "mcp_agent_discovery_accuracy.png", dpi=160)
    plt.close(fig)


def _resource_figures(summary: dict[str, Any], outdir: Path) -> None:
    import matplotlib.pyplot as plt

    points = summary["cost_improvement_points"]
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for point in points:
        ax.scatter(point["cpu_seconds"], point["accuracy_difference"], s=55)
        ax.annotate(point["label"], (point["cpu_seconds"], point["accuracy_difference"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Recorded process CPU seconds (log scale; task runs are not comparable)")
    ax.set_ylabel("Within-task accuracy difference")
    ax.set_title("Recorded compute cost versus measured accuracy change")
    ax.grid(alpha=.25)
    fig.savefig(outdir / "compute_cost_vs_improvement.png", dpi=160)
    plt.close(fig)

    regressions = summary["regression_rates"]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    labels = [item["label"] for item in regressions]
    values = [item["rate"] for item in regressions]
    bars = ax.barh(labels, values, color="#ba554a")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Previously correct items made incorrect")
    ax.set_title("Observed regression rates within each paired suite")
    ax.grid(axis="x", alpha=.2)
    for bar, val in zip(bars, values):
        ax.text(val + .008, bar.get_y()+bar.get_height()/2, f"{val:.1%}", va="center", fontsize=9)
    fig.savefig(outdir / "regression_rates.png", dpi=160)
    plt.close(fig)

    events = summary["outcomes_over_time"]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    dates = [datetime.fromisoformat(row["timestamp_utc"]) for row in events]
    vals = [row["accuracy_difference"] for row in events]
    ax.plot(dates, vals, marker="o", linewidth=1, color="#426c84")
    for date, row, val in zip(dates, events, vals):
        ax.annotate(row["label"], (date, val), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="black", linewidth=.7)
    ax.set_ylabel("Within-task accuracy difference")
    ax.set_title("Recorded experiment outcomes over time (task-specific differences)")
    ax.grid(alpha=.2)
    fig.autofmt_xdate()
    fig.savefig(outdir / "experiment_outcomes_over_time.png", dpi=160)
    plt.close(fig)


def _blind_qwen_figure(runs: list[dict[str, Any]], outdir: Path) -> None:
    if not runs:
        return
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 4),
                             sharey=True, constrained_layout=True, squeeze=False)
    axes[0, 0].invert_yaxis()
    for ax, run in zip(axes[0], runs):
        scores = run["target_correct"]
        names = ["pristine", "damaged", "repaired", *run["matched_random_controls"]]
        labels = ["Pristine", "Damaged", "Reloaded repair", "Random 1", "Random 2", "Random 3"]
        bars = ax.barh(labels, [scores[name] for name in names],
                       color=["#64748b", "#a85542", "#197a70", "#a5aeba", "#a5aeba", "#a5aeba"])
        ax.bar_label(bars, padding=3)
        ax.set_xlim(0, run["target_n"] * 1.15)
        ax.set_xlabel(f"Correct out of {run['target_n']} target prompts")
        flips = run["sentinel_prediction_flips_vs_damaged"]
        ax.set_title(f"{run['experiment_id'].rsplit('-', 1)[-1].upper()} · {run['disposition']}\n"
                     f"{flips}/{run['sentinel_n']} unrelated predictions changed")
    fig.suptitle("Blind Qwen checkpoint repair on held-out prompts")
    fig.savefig(outdir / "qwen_blind_repair.png", dpi=160)
    plt.close(fig)


def build() -> dict[str, Any]:
    raw_records = load_ledger()
    experiments = canonical_experiments(raw_records)
    by_id = {item["experiment_id"]: item for item in experiments}
    blind_qwen = []
    for item in experiments:
        if item["experiment_id"].startswith("qwen-blind-repair-") and "summary" in item["evaluation"]:
            evidence = ROOT / item["artifact_root"] / item["evaluation"]["summary"]
            blind_qwen.append({"experiment_id": item["experiment_id"], "raw_evidence": relative(evidence),
                               **read_json(evidence)})
    # The seed-731 early records were duplicate replays; the ledger correction
    # excludes the first and one isolated replay is retained as historical only.
    toy = [item for item in experiments if item["experiment_id"] in {
        "toy-discovery-20261103T023455Z-8a4c7497",
        "toy-discovery-20261103T024130Z-ffe29d5c",
        "toy-discovery-20261103T025155Z-de88d976"}]
    if len(toy) != 3:
        raise ValueError(f"Expected three isolated toy-discovery runs, found {len(toy)}")
    toy_baseline = float(np.mean([x["evaluation"]["metrics"]["baseline"]["accuracy"] for x in toy]))
    toy_candidate = float(np.mean([x["evaluation"]["metrics"]["ai_candidate"]["accuracy"] for x in toy]))
    toy_random = float(np.mean([x["evaluation"]["metrics"]["random_control"]["accuracy"] for x in toy]))

    mwc_records = [x for x in experiments if x["experiment_id"].startswith("mwc-ablation-")]
    mwc_roots = [ROOT / x["artifact_root"] for x in mwc_records]
    mwc = aggregate_mwc(mwc_roots)

    v01 = by_id["v01-demo-reproduction-20261103"]
    p50 = read_json(ROOT / "results/published_reproduction/p50c_rebuild_summary.json")
    random_patch = read_json(ROOT / "results/published_reproduction/qwen35_random_control_rebuild_summary.json")
    qwen_pair = read_json(ROOT / "results/published_reproduction/qwen_runner_paired_comparison_v02.json")
    qwen_base_run = read_json(ROOT / "results/published_reproduction/qwen_runner_baseline_v02/run.json")
    qwen_p50_run = read_json(ROOT / "results/published_reproduction/qwen_runner_p50c_v02/run.json")
    qwen64 = {
        "baseline_accuracy": qwen_base_run["summary"]["accuracy"],
        "candidate_accuracy": qwen_p50_run["summary"]["accuracy"],
        "baseline_answer_completion_rate": qwen_base_run["summary"]["answer_completion_rate"],
        "candidate_answer_completion_rate": qwen_p50_run["summary"]["answer_completion_rate"],
        "token_limit_rates": [qwen_base_run["summary"]["token_limit_rate"], qwen_p50_run["summary"]["token_limit_rate"]],
        "tail_repetition_8gram_rates": [qwen_base_run["summary"]["mean_tail_repetition_8gram_rate"], qwen_p50_run["summary"]["mean_tail_repetition_8gram_rate"]],
        "reasoning_loop_rates": [qwen_base_run["summary"]["reasoning_loop_rate"], qwen_p50_run["summary"]["reasoning_loop_rate"]],
        "raw_evidence": ["results/published_reproduction/qwen_runner_baseline_v02/run.json", "results/published_reproduction/qwen_runner_p50c_v02/run.json", "results/published_reproduction/qwen_runner_paired_comparison_v02.json"],
        "disposition": "INCONCLUSIVE",
    }
    old_base = read_json(ROOT / "results/published_reproduction/qwen_cpu_smoke_baseline.json")
    old_p50 = read_json(ROOT / "results/published_reproduction/qwen_cpu_smoke_p50c.json")
    qwen256 = {
        "token_limit_rates": [float(np.mean([row["finish_reason"] == "length" for row in old_base["rows"]])), float(np.mean([row["finish_reason"] == "length" for row in old_p50["rows"]]))],
        "tail_repetition_8gram_rates": [float(np.mean([row["tail_duplicate_8_rate"] for row in old_base["rows"]])), float(np.mean([row["tail_duplicate_8_rate"] for row in old_p50["rows"]]))],
        "raw_evidence": ["results/published_reproduction/qwen_cpu_smoke_baseline.json", "results/published_reproduction/qwen_cpu_smoke_p50c.json"],
        "disposition": "INCONCLUSIVE; exploratory 256-token cap smoke",
    }
    gradient_record = next(x for x in experiments if x["experiment_id"].startswith("toy-gradient-baseline-"))
    toy_derived = read_json(ROOT / "results/original_weight_edits/derived_paired_correctness_v2.json")
    gradient = {"baseline_accuracy": gradient_record["evaluation"]["baseline_accuracy"],
                "candidate_accuracy": gradient_record["evaluation"]["candidate_accuracy"],
                "paired_difference_ci95": gradient_record["evaluation"]["paired_difference_ci95"],
                "changed_parameters_by_seed": gradient_record["affected_parameters"]["parameter_counts_by_seed"],
                "raw_evidence": ["results/original_weight_edits/gradient-20261103T030417Z/experiment_record.json"]}

    random_control_record = next(x for x in experiments if x["experiment_id"].startswith("qwen-p50c-random-control-"))
    p50_record = next(x for x in experiments if x["experiment_id"].startswith("published-p50c-patch-rebuild-"))
    qwen_record = next(x for x in experiments if x["experiment_id"].startswith("qwen-p50c-paired-smoke-"))
    mcp_records = [x for x in experiments
                   if x.get("agent", {}).get("type") == "LLM-directed-MCP-tool-caller"]
    mcp_runs = []
    for item in mcp_records:
        metrics = item["evaluation"]["metrics"]
        candidate = metrics.get("ai_candidate")
        random_candidate = metrics.get("random_control")
        mcp_runs.append({
            "experiment_id": item["experiment_id"],
            "mechanism": item.get("task_oracle", {}).get("mechanism", "not recorded"),
            "baseline_accuracy": metrics["baseline"]["accuracy"],
            "candidate_accuracy": None if candidate is None else candidate["accuracy"],
            "random_control_accuracy": None if random_candidate is None else random_candidate["accuracy"],
            "candidate_gain": None if candidate is None else candidate["accuracy"] - metrics["baseline"]["accuracy"],
            "candidate_vs_baseline_paired_ci95": item["evaluation"].get("ai_vs_baseline_paired_accuracy_ci"),
            "affected_parameters": item["affected_parameters"],
            "disposition": item["disposition"],
            "raw_evidence": relative(ROOT / item["artifact_root"] / "evaluation.json"),
        })
    toy_intervals = [x["evaluation"]["paired_accuracy_difference_ci"] for x in toy]

    outcomes: list[dict[str, Any]] = []
    def outcome(label: str, item: dict[str, Any], difference: float, evidence: str) -> None:
        outcomes.append({"label": label, "timestamp_utc": item["timestamp_utc"],
                         "accuracy_difference": difference, "experiment_id": item["experiment_id"],
                         "evidence": evidence})
    for item in toy:
        task_seed = item["evaluation"]["paired_accuracy_difference_ci"]["seed"]
        outcome(f"toy agent seed {task_seed}", item, item["evaluation"]["metrics"]["ai_candidate"]["accuracy"] - item["evaluation"]["metrics"]["baseline"]["accuracy"], relative(ROOT / item["artifact_root"] / "evaluation.json"))
    for item, root, label in zip(mwc_records, mwc_roots, ("MWC seed batch 1001-1005", "MWC seed batch 1101-1105")):
        outcome(label, item, mwc["conditions"]["memory_plus_weight_edit"]["accuracy"] - mwc["conditions"]["no_memory_no_edit"]["accuracy"], relative(root / "mwc_ablation.json"))
    outcome("toy conventional SGD", gradient_record, gradient["candidate_accuracy"] - gradient["baseline_accuracy"], gradient["raw_evidence"][0])
    outcome("Qwen p50 paired smoke", qwen_record, qwen64["candidate_accuracy"] - qwen64["baseline_accuracy"], qwen64["raw_evidence"][2])
    for item, run in zip(mcp_records, mcp_runs):
        if run["candidate_gain"] is not None:
            outcome("LLM MCP pilot " + item["experiment_id"][-8:], item, run["candidate_gain"], run["raw_evidence"])
    outcomes.sort(key=lambda row: row["timestamp_utc"])

    regressions = []
    toy_baseline_correct = sum(run["metrics"]["baseline"]["correct"] for run in toy_derived["runs"])
    for label, key in (("toy diagnostic agent", "ai_candidate"), ("toy random control", "random_control")):
        count = sum(run["metrics"][key]["regressions_vs_baseline"] for run in toy_derived["runs"])
        regressions.append({"label": label, "rate": count / toy_baseline_correct,
                            "regressions": count, "baseline_correct_items": toy_baseline_correct,
                            "basis": "derived by reloading frozen checkpoints against already scored holdout; analysis only"})
    for name in ("external_memory_only", "weight_edit_only", "memory_plus_weight_edit"):
        item = mwc["conditions"][name]
        regressions.append({"label": "MWC " + name.replace("_", " "), "rate": item["regression_rate_vs_no_memory_no_edit"], "basis": f"paired raw correctness arrays; {mwc['example_count_per_condition']} examples"})
    regressions.append({"label": "Qwen p50 64-token smoke", "rate": qwen_pair["paired_accuracy"]["regression_rate"], "basis": "paired hand-authored prompts, n=2"})
    gradient_regressions = 0
    for replicate in gradient_record["evaluation"]["replicates"]:
        folder = ROOT / gradient_record["artifact_root"] / f"replicate-{len([r for r in gradient_record['evaluation']['replicates'] if r['seed'] < replicate['seed']]):02d}-seed-{replicate['seed']}" / "evaluation.json"
        if folder.is_file():
            row = read_json(folder)
            a = row["baseline"]["per_example_correct"]
            b = row["gradient_candidate"]["per_example_correct"]
            gradient_regressions += sum(bool(x) and not bool(y) for x, y in zip(a, b))
    regressions.append({"label": "toy conventional SGD", "rate": gradient_regressions / 320, "basis": "paired raw correctness arrays, n=320"})
    for item in mcp_records:
        metrics = item["evaluation"]["metrics"]
        candidate = metrics.get("ai_candidate")
        if candidate is not None:
            base_correct = metrics["baseline"]["per_example_correct"]
            candidate_correct = candidate["per_example_correct"]
            denominator = sum(bool(value) for value in base_correct)
            regressions.append({
                "label": "LLM MCP pilot " + item["experiment_id"][-8:],
                "rate": (sum(bool(before) and not bool(after)
                             for before, after in zip(base_correct, candidate_correct)) / denominator)
                        if denominator else 0.0,
                "basis": "paired raw holdout correctness arrays; one task seed",
            })

    cost_points = []
    for item in toy:
        task_seed = item["evaluation"]["paired_accuracy_difference_ci"]["seed"]
        cost_points.append({"label": f"toy agent seed {task_seed}",
                            "cpu_seconds": item["resources"].get("cpu_seconds"),
                            "accuracy_difference": item["evaluation"]["metrics"]["ai_candidate"]["accuracy"] - item["evaluation"]["metrics"]["baseline"]["accuracy"]})
    cost_points.extend([
        {"label": "MWC batch 1001-1005", "cpu_seconds": mwc_records[0]["resources"].get("cpu_seconds"), "accuracy_difference": mwc["conditions"]["memory_plus_weight_edit"]["accuracy"] - mwc["conditions"]["no_memory_no_edit"]["accuracy"]},
        {"label": "MWC batch 1101-1105", "cpu_seconds": mwc_records[1]["resources"].get("cpu_seconds"), "accuracy_difference": mwc["conditions"]["memory_plus_weight_edit"]["accuracy"] - mwc["conditions"]["no_memory_no_edit"]["accuracy"]},
        {"label": "toy SGD", "cpu_seconds": gradient_record["resources"].get("cpu_seconds"), "accuracy_difference": gradient["candidate_accuracy"] - gradient["baseline_accuracy"]},
        {"label": "Qwen paired smoke", "cpu_seconds": qwen_record["resources"].get("cpu_seconds"), "accuracy_difference": qwen64["candidate_accuracy"] - qwen64["baseline_accuracy"]},
    ])
    cost_points.extend({"label": "LLM MCP pilot " + item["experiment_id"][-8:],
                        "cpu_seconds": item["resources"].get("cpu_seconds"),
                        "accuracy_difference": run["candidate_gain"]}
                       for item, run in zip(mcp_records, mcp_runs) if run["candidate_gain"] is not None)
    cost_points = [point for point in cost_points if point["cpu_seconds"] is not None and point["cpu_seconds"] > 0]

    selected = list(experiments)
    cpu_sum = sum(float(x.get("resources", {}).get("cpu_seconds") or 0) for x in selected)
    wall_sum = sum(float(x.get("resources", {}).get("wall_seconds") or 0) for x in selected)
    measured_resources = [{"experiment_id": x["experiment_id"], "wall_seconds": x.get("resources", {}).get("wall_seconds"),
                           "cpu_seconds": x.get("resources", {}).get("cpu_seconds"),
                           "peak_rss_bytes": x.get("resources", {}).get("peak_rss_bytes")}
                          for x in selected]
    return {
        "schema": "weightbridge.benchmark_summary.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "append-only experiment ledger and raw run artifacts; canonicalized by explicit corrections and measurement keys",
        "canonical_ledger_experiment_count": len(selected),
        "published_reference_metrics_included_as_measurements": False,
        "v01_demo": {"baseline_accuracy": v01["evaluation"]["metrics"]["baseline_accuracy"],
                     "candidate_accuracy": v01["evaluation"]["metrics"]["candidate_accuracy"],
                     "paired_gain_ci95": v01["evaluation"]["metrics"]["paired_gain_ci95"],
                     "ai_discovered_edit": False, "raw_evidence": ["results/v01_demo_reproduction/evaluation.json", "results/experiment_ledger.jsonl"]},
        "published_patch": {"status": p50["disposition"], "base_sha256": p50["base_checkpoint_sha256"],
                            "candidate_sha256": p50["candidate_checkpoint_sha256"], "changed_values": p50["host_patch_changed_entries"],
                            "touched_positions": p50["touched_entries"], "changed_by_layer": {"15": p50["tensor_details"][0]["published_changed_entries"], "19": p50["tensor_details"][1]["published_changed_entries"]},
                            "tensor_hashes_match": all(x["match"] for x in p50["tensor_verification"].values()),
                            "behavioral_result": "not established by reconstruction"},
        "random_control": {"status": random_patch["status"], "changed_by_layer": {"15": random_patch["tensor_details"][0]["changed_entries"], "19": random_patch["tensor_details"][1]["changed_entries"]},
                           "changed_values": random_patch["matched_changed_entries"], "behaviorally_evaluated": False,
                           "raw_evidence": ["results/published_reproduction/qwen35_random_control_rebuild_summary.json"]},
        "toy_discovery": {"isolated_runs": len(toy), "holdout_examples": sum(run["holdout_n"] for run in toy_derived["runs"]),
                          "baseline_accuracy": toy_baseline, "candidate_accuracy": toy_candidate,
                          "random_control_accuracy": toy_random, "mean_candidate_gain": toy_candidate - toy_baseline,
                          "per_run_paired_gain_ci95": [[float(row["ci_low"]), float(row["ci_high"])] for row in toy_intervals],
                          "paired_correctness_derivation": relative(ROOT / "results/original_weight_edits/derived_paired_correctness_v2.json"),
                          "agent_type": "deterministic practice-only finite-difference search; no LLM or model-generated tokens",
                          "changed_parameter_count": int(toy[0]["affected_parameters"]["count"]),
                          "parameter_fraction": float(toy[0]["affected_parameters"]["fraction"]),
                          "selected_edit": "output_bias[1] += -4 on three isolated task seeds",
                          "raw_evidence": [relative(ROOT / x["artifact_root"] / "evaluation.json") for x in toy]},
        "mcp_agent_discovery_pilots": {
            "count": len(mcp_runs), "independent_task_count": len(mcp_runs),
            "runs": mcp_runs,
            "scope_note": "LLM-directed MCP sessions are reported separately from the deterministic worker cohort; one task is a pilot, not a transfer estimate.",
        },
        "mwc_ablation": mwc,
        "gradient_baseline": gradient,
        "qwen_smoke_64": qwen64,
        "qwen_smoke_256": qwen256,
        "qwen_blind_repair": blind_qwen,
        "cost_improvement_points": cost_points,
        "regression_rates": regressions,
        "outcomes_over_time": outcomes,
        "recorded_computation_totals": {"cpu_seconds_sum_unique_experiment_records": cpu_sum,
                                         "wall_seconds_sum_unique_experiment_records": wall_sum,
                                         "records": measured_resources,
                                         "scope_note": "Sum of resource fields in canonical ledger experiment records only. It excludes unrecorded coding, development, test, reference-clone and dependency-download time; CPU and wall sums include non-comparable task runs."},
        "reproduction_status": {"v01_tooling_demo": "reproduced", "published_p50_tensor_edit": "reconstructed; hashes matched", "published_MATH_GSM8K_IFEval_scores": "HOLD; not independently measured", "Qwen_CPU_smoke": "INCONCLUSIVE; n=2, cap=64; all generations truncated"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=ROOT / "results" / "benchmark_summary.json")
    parser.add_argument("--figures", type=Path, default=ROOT / "results" / "figures")
    args = parser.parse_args()
    summary = build()
    write_json(args.summary, summary)
    args.figures.mkdir(parents=True, exist_ok=True)
    _accuracy_figures(summary, args.figures)
    _qwen_figures(summary, args.figures)
    _blind_qwen_figure(summary["qwen_blind_repair"], args.figures)
    _patch_figures(summary, args.figures)
    _mcp_agent_figures(summary, args.figures)
    _resource_figures(summary, args.figures)
    print(json.dumps({"summary": relative(args.summary), "figures": relative(args.figures),
                      "canonical_experiments": summary["canonical_ledger_experiment_count"],
                      "mwc_accuracy": {k: v["accuracy"] for k, v in summary["mwc_ablation"]["conditions"].items()}}, indent=2))


if __name__ == "__main__":
    main()
