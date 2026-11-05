"""At most three research figures from the public, measured Phase III-R export.

This report-only script is never imported by training or admission. It accepts
only an export containing export-manifest.json and arm-summary.json; protected
raw host outputs are neither needed nor read. Missing measurements are omitted
or labelled UNMEASURED. Use an existing Matplotlib environment, never install
packages to make this plotting command work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ARMS = ("rank8", "rank4")
COLORS = {"Parent": "#8994A3", "Rank 8": "#2864A4", "Rank 4": "#D37826"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_export(root):
    root = Path(root).resolve()
    if not (root/"export-manifest.json").is_file() or not (root/"arm-summary.json").is_file():
        raise ValueError("Figures require an explicit public export, not a private study directory")
    inputs = {}
    def read(relative):
        path = root/relative
        if not path.is_file():
            return None
        inputs[relative] = digest(path)
        return json.loads(path.read_bytes())
    manifest, arms = read("export-manifest.json"), read("arm-summary.json")
    losses = {}
    for arm in ARMS:
        points = []
        for path in sorted((root/arm/"training/steps").glob("*.json")):
            row = read(path.relative_to(root).as_posix())
            if (type(row.get("step")) is not int or row["step"] <= 0
                    or not isinstance(row.get("loss"), (int, float)) or not math.isfinite(row["loss"]) or row["loss"] < 0):
                raise ValueError("Invalid measured training point")
            points.append({"step": row["step"], "loss": row["loss"]})
        if points:
            if len({point["step"] for point in points}) != len(points):
                raise ValueError("Duplicate training steps")
            losses[arm] = points
    bank_b = read("rank4/assessment/bank-b.json")
    bank_b = bank_b.get("aggregate") if bank_b else None
    committed = {name: identity["public_sha256"] for name, identity in manifest["artifacts"].items()}
    committed.update(manifest.get("derived_artifact_sha256", {}))
    for relative, observed in inputs.items():
        if relative == "export-manifest.json":
            continue
        if relative not in committed or observed != committed[relative]:
            raise ValueError("Public figure input differs from its export commitment")
    return arms, losses, bank_b, inputs


def gate_facts(gate):
    if not gate:
        return {"gate_passed": None, "lost_correct": None, "lost_valid": None}
    return {"gate_passed": gate.get("passed"),
        "lost_correct": gate.get("prohibited_regressions_count"),
        "lost_valid": gate.get("lost_valid_presentations_count")}


def gate_note(value):
    if value.get("gate_passed") is None:
        return "UNMEASURED"
    state = "PASS" if value["gate_passed"] else "FAIL"
    if not value["gate_passed"]:
        for key, label in (("lost_correct", "lost correct"), ("lost_valid", "lost valid")):
            if value.get(key) is not None:
                state += f" · {label} {value[key]}"
    return state


def behavior_series(arms, bank_b):
    """One record per displayed endpoint condition, with separate decode facts."""
    series = {"Development": [], "Bank A": [], "Bank B": []}
    parent = None
    for arm in ARMS:
        for step in (64, 128):
            saved = arms.get(arm, {}).get("checkpoints", {}).get(str(step))
            gate = saved.get("development") if saved else None
            if gate:
                if parent is None:
                    parent = gate["parent"]
                    series["Development"].append({"label": "Parent", "model": "Parent", "summary": parent})
                elif any(parent.get(key) != gate["parent"].get(key) for key in
                         ("presentations", "correct", "strict_comparisons", "complete_families", "decode_termination")):
                    raise ValueError("Shared development baseline differs across arms")
                model = "Rank 8" if arm == "rank8" else "Rank 4"
                series["Development"].append({"label": f"{model} · step {step}", "model": model,
                    "summary": gate["candidate"], **gate_facts(gate)})
    for stage in ("selection", "hidden"):
        gate = arms.get("rank8", {}).get("independent_stages", {}).get(stage)
        if gate:
            for model, value in (("Parent", gate["parent"]), ("Rank 8", gate["candidate"])):
                series["Bank A"].append({"label": f"{stage.title()} · {model}", "model": model, "summary": value,
                    **(gate_facts(gate) if model != "Parent" else {})})
        record = (bank_b or {}).get("stages", {}).get(stage)
        if record:
            for model, value in (("Parent", record["rank4_ordinary"]["parent"]),
                                  ("Rank 8", record["rank8_reference"]["candidate"]),
                                  ("Rank 4", record["rank4_ordinary"]["candidate"])):
                gate = record["rank8_reference"] if model == "Rank 8" else record["rank4_ordinary"]
                series["Bank B"].append({"label": f"{stage.title()} · {model}", "model": model, "summary": value,
                    **(gate_facts(gate) if model != "Parent" else {}),
                    "paired_retention_passed": record["rank4_retention"]["passed"] if model == "Rank 4" else None})
    return series


def selected_efficiency(arms):
    result = {}
    for arm in ARMS:
        value = arms.get(arm, {})
        name = value.get("selected_checkpoint")
        if not name:
            continue
        step = int(name.rsplit("-", 1)[-1])
        candidate = value.get("checkpoints", {}).get(str(step))
        if candidate is None:
            raise ValueError("Locked checkpoint lacks its numerical receipt")
        merged = value.get("merged") or {}
        result[arm] = {"step": step, "factor_parameters": candidate.get("trainable_parameters"),
            "adapter_bytes": candidate.get("adapter_bytes"), "effective_unrounded_norm": candidate.get("effective_unrounded_norm"),
            "merged_BF16_norm": merged.get("stored_update_norm"),
            "merged_changed_entries": merged.get("changed_entries"),
            "rounding_residual_norm": merged.get("rounding_residual_norm")}
    return result


def render(public_export, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms, losses, bank_b, inputs = read_export(public_export)
    series, efficiency = behavior_series(arms, bank_b), selected_efficiency(arms)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})
    outputs = []
    def save(fig, name):
        fig.savefig(out/name, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append({"file": name, "sha256": digest(out/name)})
    if losses:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        zero_counts = {}
        for arm, points in losses.items():
            model = "Rank 8" if arm == "rank8" else "Rank 4"
            positive = [p for p in points if p["loss"] > 0]
            zero_counts[arm] = len(points)-len(positive)
            # Exact zeros cannot occupy a logarithmic axis; never substitute an
            # invented positive floor. Record and annotate them separately.
            if positive:
                ax.plot([p["step"] for p in positive], [p["loss"] for p in positive], color=COLORS[model],
                        linewidth=1.1, marker=".", markersize=3, label=f"{model} ({arms[arm]['training_status']})")
        ax.set(yscale="log", xlabel="Optimizer step", ylabel="Complete-answer training loss (log scale)",
               title="Observed training losses · unsmoothed logged steps")
        ax.grid(alpha=.2); ax.legend()
        note = "Training loss is diagnostic; it is not a free-generation score."
        if any(zero_counts.values()):
            note += " Exact zero-loss points omitted from log axis: "+str(zero_counts)
        if "rank4" not in losses:
            note += " Rank 4: UNMEASURED."
        fig.text(.08, -.02, note, fontsize=9)
        save(fig, "training-loss.png")
    if any(series.values()):
        fig, axes = plt.subplots(3, 2, figsize=(13, 12), gridspec_kw={"hspace": .65, "wspace": .43})
        for row, (population, records) in enumerate(series.items()):
            preservation = (arms.get("rank8", {}).get("independent_stages", {}).get("preservation-validation")
                if population == "Bank A" else (bank_b or {}).get("stages", {}).get("preservation-validation", {}).get("rank4_ordinary")
                if population == "Bank B" else None)
            preservation_note = f"Preservation gate: {gate_note(gate_facts(preservation))}" if population != "Development" else ""
            bank_b_preservation = (bank_b or {}).get("stages", {}).get("preservation-validation") if population == "Bank B" else None
            if bank_b_preservation:
                preservation_note += ("\nRank 8 reference: "+gate_note(gate_facts(bank_b_preservation["rank8_reference"]))
                    +"; paired retention: "+("PASS" if bank_b_preservation["rank4_retention"]["passed"] else "FAIL"))
            for column, (metric, total, title) in enumerate((("strict_comparisons", "comparisons", "Strict comparisons"),
                                                            ("complete_families", "families", "Complete opposing families"))):
                ax = axes[row, column]
                if not records:
                    ax.text(.5, .5, "UNMEASURED / UNOPENED", ha="center", va="center", transform=ax.transAxes, color="#667080")
                    if preservation_note:
                        ax.text(.5, .35, preservation_note, ha="center", va="center", transform=ax.transAxes,
                                color="#A52F35" if preservation and not preservation["passed"] else "#667080", fontsize=9)
                    ax.set_axis_off(); ax.set_title(f"{population} · {title}")
                    continue
                values = [r["summary"][metric] for r in records]
                denominators = [r["summary"][total] for r in records]
                positions = list(range(len(records)))
                bars = ax.barh(positions, values, color=[COLORS[r["model"]] for r in records], height=.65)
                labels = []
                for bar, record in zip(bars, records, strict=True):
                    label = record["label"]
                    if record["model"] != "Parent":
                        label += "\n"+gate_note(record)
                        if record.get("paired_retention_passed") is False:
                            label += " · paired retention FAIL"
                        if record.get("gate_passed") is False or record.get("paired_retention_passed") is False:
                            bar.set_hatch("//"); bar.set_edgecolor("#A52F35"); bar.set_linewidth(1.8)
                    labels.append(label)
                ax.set_yticks(positions, labels, fontsize=8)
                ax.invert_yaxis(); ax.set_xlim(0, max(denominators)*1.18)
                for y, value, denominator in zip(positions, values, denominators, strict=True):
                    ax.text(value+max(denominators)*.02, y, f"{value}/{denominator}", va="center", fontsize=9)
                ax.set_title(f"{population} · {title}"); ax.set_xlabel("Correct units")
                if preservation_note:
                    ax.text(.02, -.24, preservation_note, transform=ax.transAxes, fontsize=8,
                            color="#A52F35" if preservation and not preservation["passed"] else "#667080")
        fig.suptitle("Complete generated first-line answers · identical endpoint before and after training", y=.98, fontsize=14)
        fig.text(.08, .015, "Strict comparison: all 4 wording/order responses correct. Complete family: all 8 opposing responses correct.\n"
                 "Development selected checkpoints; banks are independent. First-line completion is distinct from full-decode EOS.\n"
                 "Hatched bars flag failed gates even when accuracy is high. Exact valid-line and EOS/token-budget counts are in figure-data.json.", fontsize=9)
        save(fig, "behavioral-outcomes.png")
    if efficiency:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
        fields = (("factor_parameters", 1e6, "Trainable factor parameters", "Millions"),
                  ("adapter_bytes", 1024**2, "Saved adapter artifact", "MiB"),
                  ("merged_BF16_norm", 1, "Realized merged BF16 change", "Frobenius norm"))
        for ax, (field, divisor, title, unit) in zip(axes, fields, strict=True):
            measured = [(arm, row[field]/divisor) for arm, row in efficiency.items() if row[field] is not None]
            ax.set_title(title); ax.set_ylabel(unit)
            if not measured:
                ax.text(.5, .5, "UNMEASURED", transform=ax.transAxes, ha="center"); ax.set_xticks([])
                continue
            labels = ["Rank 8" if arm == "rank8" else "Rank 4" for arm, _ in measured]
            values = [v for _, v in measured]
            ax.bar(labels, values, color=[COLORS[label] for label in labels], width=.55)
            ax.set_ylim(0, max(values)*1.25 if max(values) else 1)
            for i, value in enumerate(values):
                ax.text(i, value+max(values)*.03, f"{value:.4g}", ha="center")
        fig.suptitle("Locked finalists · observed capacity, storage and weight change", fontsize=14)
        note = "; ".join(f"{arm}: step {value['step']}" for arm, value in efficiency.items())
        if "rank4" not in efficiency:
            note += ". Rank 4 finalist: UNMEASURED."
        fig.text(.07, -.015, note+" Fewer factor parameters do not imply a smaller merged weight update.", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .92)); save(fig, "finalist-efficiency.png")
    record = {"schema": "weightbridge.phase_iiir.figures.v1", "plotter_sha256": digest(__file__), "input_sha256": inputs,
        "training_points": losses, "behavioral_series": series, "selected_efficiency": efficiency,
        "figures": outputs, "source": "Public export only; all numbers are recorded measurements.",
        "missing_measurements": "Omitted or UNMEASURED, never fabricated as zero.",
        "decode_note": "Each behavioral series retains valid_completed_first_lines and decode_termination separately. A scored completed first line is not EOS termination."}
    with (out/"figure-data.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    return record


def self_test():
    import tempfile
    empty = {arm: {"checkpoints": {}, "selected_checkpoint": None, "independent_stages": {}} for arm in ARMS}
    assert not any(behavior_series(empty, None).values()) and selected_efficiency(empty) == {}
    summary = {"presentations": 8, "correct": 8, "strict_comparisons": 2, "comparisons": 2,
               "complete_families": 1, "families": 1, "valid_completed_first_lines": 8,
               "decode_termination": {"max_new_tokens": 8}}
    arms = {"rank8": {"checkpoints": {"64": {"development": {"parent": summary, "candidate": summary,
                     "passed": False, "prohibited_regressions_count": 2, "lost_valid_presentations_count": 1}}},
                     "independent_stages": {}}, "rank4": {"checkpoints": {}, "independent_stages": {}}}
    observed = behavior_series(arms, None)
    assert len(observed["Development"]) == 2 and not observed["Bank B"]
    assert observed["Development"][1]["summary"]["decode_termination"] == {"max_new_tokens": 8}
    assert gate_note(observed["Development"][1]) == "FAIL · lost correct 2 · lost valid 1"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root/"arm-summary.json").write_text(json.dumps(empty), encoding="utf-8")
        manifest = {"artifacts": {}, "derived_artifact_sha256": {"arm-summary.json": digest(root/"arm-summary.json")}}
        (root/"export-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        assert read_export(root)[0] == empty
        (root/"arm-summary.json").write_text(json.dumps({**empty, "tampered": True}), encoding="utf-8")
        try:
            read_export(root)
        except ValueError as error:
            assert "commitment" in str(error)
        else:
            raise AssertionError("Changed derived summary accepted")
    return {"missing_data": "PASS", "separate_first_line_and_decode": "PASS",
        "visible_failure_labels": "PASS", "derived_hash_verification": "PASS", "plots_generated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-export", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    else:
        if args.public_export is None or args.out is None:
            parser.error("--public-export and --out are required")
        result = render(args.public_export, args.out)
        print(json.dumps({"figures": result["figures"], "plotter_sha256": result["plotter_sha256"]}))
