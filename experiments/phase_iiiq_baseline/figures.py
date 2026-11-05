"""Static figures from a recorded post-hoc report; missing data stay unplotted."""
from __future__ import annotations

import argparse
from pathlib import Path
from .integrity import read


def generate(report_path, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report, out = read(report_path), Path(outdir)
    if report.get("schema") != "weightbridge.supervised_baseline.report.v1" or not report.get("reporting_only"):
        raise ValueError("Figures require an explicit post-hoc receipt summary")
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    def save(fig, name, note):
        path = out/f"{name}.png"
        if path.exists():
            raise FileExistsError("Research figures are write-once")
        fig.text(.02, .01, note, fontsize=8, wrap=True)
        fig.tight_layout(rect=(0, .08, 1, .94))
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(str(path))
    runs = [("Smoke (training set)", report["smoke"]["training"])] if report.get("smoke") else []
    runs += [(f"Main run {i}", row) for i, row in enumerate(report.get("main_runs", []))]
    measured = [(name, row) for name, row in runs if row.get("steps")]
    if measured:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for name, row in measured:
            points = [step for step in row["steps"] if step.get("loss") is not None]
            axes[0].plot([p["step"] for p in points], [p["loss"] for p in points], label=name)
            points = [step for step in row["steps"] if step.get("unrounded_effective_update_norm") is not None]
            if points:
                axes[1].plot([p["step"] for p in points], [p["unrounded_effective_update_norm"] for p in points], ".-", label=name)
        for axis, title, ylabel in zip(axes, ("Observed training loss", "Unrounded adapter update"), ("Causal token NLL", "Frobenius norm")):
            axis.set(title=title, xlabel="Optimizer step", ylabel=ylabel)
            axis.grid(alpha=.2); axis.legend(fontsize=8)
        save(fig, "training", "Losses are training observations. Adapter update norms are unrounded; neither establishes fresh behavioral improvement.")
    checkpoints = [(name, p) for name, row in runs for p in row.get("checkpoints", []) if p.get("stored_update_norm") is not None]
    if checkpoints:
        fig, axis = plt.subplots(figsize=(8, 4))
        labels = [f"{name}\nstep {p['step']}" for name, p in checkpoints]
        axis.bar(labels, [p["stored_update_norm"] for _, p in checkpoints], color="#4878a8")
        axis.set(title="Stored-BF16 effective update norms", ylabel="Frobenius norm")
        axis.tick_params(axis="x", labelsize=8); axis.grid(axis="y", alpha=.2)
        save(fig, "stored-norms", "Receipt-derived round-once BF16 deltas; unmerged adapter generation can differ from merged checkpoint generation.")
    candidates = (report.get("assessment") or {}).get("candidates", {})
    observed = [(name, row) for name, row in candidates.items() if row.get("development_gate")]
    if observed:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        labels = [name for name, _ in observed]
        positions = list(range(len(observed)))
        axes[0].bar([p-.18 for p in positions], [r["development_gate"]["strict_comparison_gain"] for _, r in observed], .36, label="Strict comparisons")
        axes[0].bar([p+.18 for p in positions], [r["development_gate"]["complete_family_gain"] for _, r in observed], .36, label="Complete families")
        axes[0].set(title="Fresh development gain", ylabel="Candidate minus parent"); axes[0].legend(fontsize=8)
        axes[1].bar(positions, [len(r["development_gate"]["prohibited_regressions"]) for _, r in observed], color="#c45b51")
        axes[1].set(title="Development preservation losses", ylabel="Previously correct presentations lost")
        for axis in axes:
            axis.set_xticks(positions, labels, rotation=25, ha="right"); axis.grid(axis="y", alpha=.2)
        save(fig, "development", "Fresh development is a selection stage, not sealed confirmation. Unopened stages are not plotted as zero.")
    costs = [row for row in report.get("cost_components", []) if row.get("wall_seconds") is not None]
    if costs:
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.barh([r["phase"].replace("_", " ") for r in costs], [r["wall_seconds"]/60 for r in costs], color="#558b6e")
        axis.set(title="Recorded research wall time", xlabel="Minutes")
        save(fig, "cost", "Measured components only. Nested worker/training costs are not double counted; tests, setup and unaccounted independent work are excluded.")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path in generate(args.report, args.out):
        print(path)
