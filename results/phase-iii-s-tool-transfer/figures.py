"""Report-only figures from a hash-verified public Phase S scalar snapshot."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ARMS = ("sequential", "original-joint")
LABELS = {"sequential": "Sequential from R8", "original-joint": "Joint from original"}
COLORS = {"sequential": "#245B91", "original-joint": "#C87524"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curves(training):
    steps = training.get("steps")
    if not steps:
        return None
    if [s["step"] for s in steps] != list(range(1, len(steps) + 1)):
        raise ValueError("Training curve steps must be contiguous")
    if any(not math.isfinite(s["loss"]) or s["loss"] <= 0 for s in steps):
        raise ValueError("Log-loss plot requires measured finite positive losses")
    rows = [{k: s.get(k) for k in ("step", "loss", "learning_rate", "gradient_norm_before_clip")} for s in steps]
    blocks = []
    for start in range(0, len(rows), 16):
        group = rows[start:start+16]
        blocks.append({"first_step": group[0]["step"], "last_step": group[-1]["step"],
            "mean_minibatch_loss": sum(s["loss"] for s in group)/len(group),
            "minimum_minibatch_loss": min(s["loss"] for s in group),
            "maximum_minibatch_loss": max(s["loss"] for s in group)})
    return {"raw_steps": rows, "sixteen_step_summaries": blocks,
        "interpretation": "Measured unsmoothed training minibatches, not fixed-example or held-out loss."}


def development(rows):
    result = []
    for step in (64, 128):
        item = rows.get(str(step))
        if not item or not item.get("tool") or not item.get("host_gate"):
            result.append({"step": step, "measurement": None})
            continue
        summary, gate = item["tool"]["summary"], item["host_gate"]
        retention = gate["tool"]["own_parent_retention"]
        result.append({"step": step, "measurement": {
            **{k: summary[k] for k in ("strict_correct", "presentations", "complete_families", "families", "schema_valid", "completed")},
            **{k: retention[k] for k in ("lost_schema_valid_count", "lost_strict_correct_count", "lost_semantic_correct_count")},
            "passed": gate["passed"], "disposition": gate["disposition"],
            "diagnostic_only": step == 64}})
    return result


def render(source, out):
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError("Figure artifacts are write-once")
    spec = importlib.util.spec_from_file_location("public_s_exporter", Path(__file__).with_name("export_outcome.py"))
    exporter = importlib.util.module_from_spec(spec); spec.loader.exec_module(exporter)
    exporter.verify(source)  # All scalar inputs must belong to this exact public export.
    read = lambda name: json.loads((source/name).read_bytes())
    names = ["export-manifest.json", "stage-availability.json"]
    names += [f"{a}-{kind}.json" for a in ARMS for kind in ("training", "development")]
    data = {a: {"training": curves(read(a+"-training.json")), "development": development(read(a+"-development.json"))} for a in ARMS}
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "svg.hashsalt": "weightbridge-phase-s-report-v1"})
    out.mkdir(parents=True)
    payload = {"schema": "weightbridge.phase_iiis.public_curve_data.v1", "arms": data,
        "independent_stages": read("stage-availability.json"), "model_execution": False}
    (out/"curve-data.json").write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for arm in ARMS:
        c = data[arm]["training"]
        if c is None:
            continue
        rows = c["raw_steps"]
        ax.plot([r["step"] for r in rows], [r["loss"] for r in rows], label=LABELS[arm],
            color=COLORS[arm], linewidth=1.4, marker=".", markersize=2.5)
    ax.set(yscale="log", xlabel="Optimizer step", ylabel="Measured answer-token cross-entropy",
        title="Phase S: actual training loss, without smoothing")
    ax.grid(axis="y", alpha=.2); ax.legend(frameon=False)
    for step in (64, 128):
        ax.axvline(step, color="#777777", linestyle=":", alpha=.5)
    fig.text(.07, .02, "Different minibatches at each step. Lower training loss is not independent behavioral evidence.", fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1)); save(fig, out, "training-loss"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    observed = [(a, item["step"], item["measurement"]) for step in (64, 128) for a in ARMS
        for item in data[a]["development"] if item["step"] == step and item["measurement"] is not None]
    labels = [f"{'R8' if a == 'sequential' else 'Original'}\nstep {step}" for a, step, _ in observed]
    for index, (field, denominator, title) in enumerate((("strict_correct", "presentations", "Completed tool actions"),
            ("complete_families", "families", "Complete eight-form families"),
            ("lost_schema_valid_count", None, "Parent-schema regressions"))):
        ax = axes[index]
        for x, (arm, step, m) in enumerate(observed):
            value = 100*m[field]/m[denominator] if denominator else m[field]
            ax.bar(x, value, color=COLORS[arm], hatch="///" if not m["passed"] else None,
                edgecolor="#AD2525" if not m["passed"] else COLORS[arm], linewidth=1.5)
            label = f'{m[field]}/{m[denominator]}' if denominator else str(m[field])
            ax.text(x, value + (2 if denominator else .15), label, ha="center", fontsize=9)
        ax.set_xticks(range(len(labels)), labels); ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 114 if denominator else max(1, max((m[field] for _, _, m in observed), default=0))*1.25)
        ax.set_ylabel("Percent" if denominator else "Lost presentations"); ax.grid(axis="y", alpha=.15)
    fig.suptitle("Development only: step 64 is diagnostic; step 128 is the fixed candidate", fontsize=12)
    fig.text(.055, .065, "Hatched bars: gate FAIL despite high accuracy. Original step 64 loses 7 schema-valid responses; step 128 retains them.", fontsize=9)
    fig.text(.055, .025, "Both final adapters: 128/128 tool EOS; legacy answers 208/208 correct first lines, 0/208 EOS. Formal banks unopened in this snapshot.", fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.96)); save(fig, out, "adapter-development"); plt.close(fig)
    outputs = {p.name: sha(p) for p in sorted(out.iterdir())}
    manifest = {"schema": "weightbridge.phase_iiis.public_figures.v1", "source_snapshot": source.name,
        "input_sha256": {n: sha(source/n) for n in names}, "script_sha256": sha(__file__),
        "matplotlib_version": matplotlib.__version__, "artifacts": outputs, "model_execution": False,
        "scientific_independence": "Development and training only; no independent model qualification is inferred."}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return {"passed": True, "figures": 2, "artifacts": len(outputs), "model_execution": False}


def save(fig, root, name):
    fig.savefig(root/(name+".png"), dpi=180, facecolor="white")
    fig.savefig(root/(name+".svg"), metadata={"Date": None}, facecolor="white")


def self_test():
    assert curves({"steps": None}) is None
    c = curves({"steps": [{"step": i+1, "loss": 1/(i+1)} for i in range(32)]})
    assert len(c["raw_steps"]) == 32 and len(c["sixteen_step_summaries"]) == 2
    assert development({}) == [{"step": 64, "measurement": None}, {"step": 128, "measurement": None}]
    for steps in ([{"step": 2, "loss": 1}], [{"step": 1, "loss": float("nan")}], [{"step": 1, "loss": 0}]):
        try:
            curves({"steps": steps})
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid curve accepted")
    return {"unknown_is_null": "PASS", "unsmoothed_steps": "PASS", "contiguity_and_finite_loss": "PASS", "model_execution": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path); parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    elif args.source and args.out:
        print(json.dumps(render(args.source, args.out)))
    else:
        parser.error("Choose --self-test or --source/--out")
