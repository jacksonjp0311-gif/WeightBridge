"""Separate Track A numerical diagnostics from Track B measured behavior."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.phase_iiil.semantic_diagnostics import summarize
from experiments.phase_iiim.semantic_metrics import compare
from weightbridge.core import file_hash, json_dump_atomic


def generate(track_a_root, track_b_root, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    aroot, broot, outdir = Path(track_a_root), Path(track_b_root), Path(outdir)
    read = lambda path: json.loads(path.read_bytes())
    analysis, feasibility = read(aroot / "analysis.json"), read(aroot / "trial/feasibility.json")
    paired = read(broot / "evaluation/comparison.json")
    workers = {name: read(broot / f"evaluation/{name}.json") for name in ("original", "reference")}
    if analysis["fit_report_sha256"] != file_hash(aroot / "trial/fit-report.json"):
        raise ValueError("Track A analysis does not bind the completed fit")
    for name, worker in workers.items():
        if (file_hash(broot / f"evaluation/{name}.json") != paired["worker_sha256"][name]
                or worker["roster_sha256"] != paired["roster_sha256"]
                or summarize(worker["measurement"]) != paired[f"{name}_summary"]):
            raise ValueError("Track B chart input differs from its worker evidence")
    before, after = (workers[name]["measurement"] for name in ("original", "reference"))
    if compare(before, after, min_comparison_gain=1, min_repeatable_corrections=1) != paired["paired"]:
        raise ValueError("Track B plotted changes do not replay")
    paths = [outdir / name for name in ("track-a-local-norms.png", "track-b-decimal-endpoint.png", "track-b-families.png")]
    if any(path.exists() for path in (*paths, outdir / "figures.json")):
        raise FileExistsError("Use a fresh figure destination; previous artifacts stay unchanged")
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "figure.facecolor": "white"})

    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    x = np.arange(2)
    groups = ("target_only", "with_preservation")
    for offset, color, label, values in (
        (-.25, "#176b87", "Selected-block minimum (approx.)", [feasibility[g]["relaxation"]["minimum_norm"] for g in groups]),
        (0, "#cc8b2a", "Rank-one spectral lower bound", [feasibility[g]["relaxation"]["certificate"]["rank_one_norm_lower_bound"] for g in groups]),
        (.25, "#64569a", "Fixed-right minimum (approx.)", [feasibility[g]["fixed_right"]["minimum_norm"] for g in groups])):
        bars = ax.bar(x + offset, values, width=.23, color=color, label=label)
        ax.bar_label(bars, fmt="%.5f", fontsize=8, padding=4)
    ax.axhline(.025, color="#b53025", linestyle="--", linewidth=1.5, label="Authorized cap: 0.025")
    ax.set_xticks(x, ["Individual target decisions", "Targets + sampled preservation"])
    ax.set(ylabel="Unrounded local update Frobenius norm", title="Track A | Required local intervention exceeds the cap", ylim=(0, .066))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(.5, .02, "Parent-linearized diagnostics. Orientation search stopped at the checked bound; edited answers remain unmeasured.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1)); fig.savefig(paths[0], dpi=180); plt.close(fig)

    rows = [{row["id"]: row for row in measurement["rows"]} for measurement in (before, after)]
    if rows[0].keys() != rows[1].keys():
        raise ValueError("Reference comparison does not contain identical presentation identities")
    targets = [row for row in before["rows"] if row["group"] == "target"]
    denominators = [len(targets), paired["paired"]["target_comparisons"], len({row["pair_id"] for row in targets})]
    counts = [[sum(row["correct"] for row in mapping.values() if row["group"] == "target"),
               paired["paired"][f"{name}_target_correct"], paired["paired"][f"{name}_complete_families"]]
              for name, mapping in zip(("parent", "candidate"), rows)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    for offset, color, label, values in zip((-.18, .18), ("#176b87", "#64569a"), ("Original", "Published edit"), counts):
        bars = axes[0].bar(np.arange(3) + offset, 100 * np.asarray(values) / denominators, width=.33, color=color, label=label)
        axes[0].bar_label(bars, labels=[f"{value}/{total}" for value, total in zip(values, denominators)], fontsize=9, padding=3)
    axes[0].set_xticks(np.arange(3), ["Target\npresentations", "Strict\ncomparisons", "Complete\ntruth-flip families"])
    axes[0].set(ylabel="Correct under each endpoint (%)", ylim=(0, 25)); axes[0].legend(frameon=False)
    corrections = sum(not rows[0][key]["correct"] and rows[1][key]["correct"] for key in rows[0])
    regressions = sum(rows[0][key]["correct"] and not rows[1][key]["correct"] for key in rows[0])
    bars = axes[1].bar(["Corrections", "Regressions"], [corrections, regressions], color=["#27816b", "#c55232"], width=.55)
    axes[1].bar_label(bars, padding=3); axes[1].set(ylabel="Individual complete presentations", ylim=(0, max(corrections, regressions) + 2))
    valid = [sum(row["valid"] for row in mapping.values()) for mapping in rows]
    fig.suptitle("Track B | External edit changes isolated answers, not complete families", fontweight="bold")
    fig.text(.5, .02, f"Valid responses: original {valid[0]}/{len(rows[0])}, published edit {valid[1]}/{len(rows[1])}. Short greedy decimal pilot; not the authors' long benchmark or a new learning result.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .07, 1, .96)); fig.savefig(paths[1], dpi=180); plt.close(fig)

    families = sorted({row["pair_id"] for row in targets})
    ordered = [sorted((row for row in targets if row["pair_id"] == family), key=lambda row: (row["counterfactual"], row["order"], row["wording"])) for family in families]
    if any(len(group) != 8 for group in ordered):
        raise ValueError("Every displayed truth-flip family needs eight presentations")
    matrices = [np.array([[int(mapping[row["id"]]["correct"]) for row in group] for group in ordered]) for mapping in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True)
    for axis, matrix, title in zip(axes, (*matrices, matrices[1]-matrices[0]), ("Original", "Published edit", "Change")):
        colors = ["#f0f2f4", "#176b87"] if title != "Change" else ["#c55232", "#f0f2f4", "#27816b"]
        boundaries = [-.5, .5, 1.5] if title != "Change" else [-1.5, -.5, .5, 1.5]
        axis.imshow(matrix, cmap=ListedColormap(colors), norm=BoundaryNorm(boundaries, len(colors)), aspect="auto")
        for y, row in enumerate(matrix):
            for x, value in enumerate(row): axis.text(x, y, f"{value:+d}" if title == "Change" and value else str(value), ha="center", va="center", fontsize=8, color="white" if value else "#48505b")
        axis.set(title=title, xlabel="Comparison / operand order / wording")
        axis.set_xticks(np.arange(8), [f"{c}/{o}/{w}" for c in (0, 1) for o in (0, 1) for w in (0, 1)], rotation=65, fontsize=8)
    axes[0].set_yticks(np.arange(len(families)), [f"Family {family.rsplit('/', 1)[-1]}" for family in families])
    fig.suptitle("Track B | All eight outcomes remain visible", fontweight="bold")
    fig.text(.5, .015, "Correct = 1; incorrect = 0. Change: +1 correction, -1 regression. No family becomes correct on all eight responses.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94)); fig.savefig(paths[2], dpi=180); plt.close(fig)
    inputs = [aroot / "analysis.json", aroot / "trial/feasibility.json", broot / "evaluation/comparison.json", *(broot / f"evaluation/{name}.json" for name in workers)]
    json_dump_atomic(outdir / "figures.json", {"schema": "weightbridge.phase_iiip.figures.v1",
        "generator_sha256": file_hash(__file__), "input_sha256": {f"{path.parent.name}/{path.name}": file_hash(path) for path in inputs},
        "figure_sha256": {path.name: file_hash(path) for path in paths}, "scope": "Recorded numerical diagnostics and completed external-reference observations only; interventions and norm scales are not pooled."})
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("track-a", "track-b", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print("\n".join(map(str, generate(args.track_a, args.track_b, args.out))))
