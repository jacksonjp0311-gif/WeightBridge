"""Compare one practice wording with three on the same blinded Qwen faults."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

import run_qwen_repair_cohort as prior
from run_blind_qwen_repair import (
    BASE_SHARD, LAYERS, ROOT, Runner, apply_safetensors_patch, correct, digest, flips,
    load_checkpoint, measure, patch_document, peak_rss_bytes, policy, read, timestamp, write,
)
from run_qwen_repair_cohort import (
    FAULT_STRENGTHS, SIZES, STRENGTHS, activation_gate, case_status, practice_metrics,
    random_delta, select_attempt, signed_labels,
)

SOURCE = Path(__file__).resolve()
CONFIG = ROOT / "benchmarks/configs/qwen_wording_ablation_v1.json"
MODES = ("single", "multi")
ARMS = {mode: (mode, f"random_{mode}") for mode in MODES}
SEED = 919426
TEMPLATES = {
    "standard": "Which integer is larger?\nA. {a}\nB. {b}\nAnswer with the letter A or B only.",
    "inline": "Which number is larger, A) {a} or B) {b}? Answer with the letter of the larger number only.",
    "brackets": "Identify the larger of ({a}) and ({b}). Choose A for ({a}) or B for ({b}). Reply with the option letter only.",
    "comparison": "Compare the values below. Which is the greater integer?\nA. {a}\nB. {b}\nAnswer with the letter A or B only.",
    "maximum": "Pick the maximum of these two integers.\nA. {a}\nB. {b}\nAnswer with the letter A or B only.",
}
FORMS = {
    "construction": ("standard",) * 8,
    "single": ("standard",) * 8,
    "multi": ("standard",) * 4 + ("inline",) * 2 + ("brackets",) * 2,
    "holdout": ("standard",) * 12 + ("comparison",) * 6 + ("maximum",) * 6,
}


def protocol():
    original = prior.protocol()
    order = np.random.default_rng(SEED).permutation(len(original["cases"]))
    cases = [{**original["cases"][int(i)], "slot": slot, "seed": SEED + 100 + slot}
             for slot, i in enumerate(order, 1)]
    return {**original, "schema": "weightbridge.qwen_wording_ablation.v1", "seed": SEED, "cases": cases,
            "source_sha256": {**original["source_sha256"], SOURCE.name: digest(SOURCE)},
            "practice_modes": list(MODES), "templates": TEMPLATES,
            "wordings": {k: list(v) for k, v in FORMS.items()},
            "primary_comparison": "paired multi minus single held-out paraphrase accuracy"}


def load_protocol(path):
    frozen = read(path)
    if frozen != protocol():
        raise ValueError("manifest differs from frozen source or protocol")
    return frozen


def build_splits(slot):
    rng = np.random.default_rng(SEED + 1)
    total_targets = sum(n for n, _ in SIZES.values())
    total_sentinels = sum(n for _, n in SIZES.values())
    numbers = iter(rng.choice(70 * 300, 12 * total_targets, replace=False).reshape(12, total_targets)[slot - 1])
    bank = [(kind, good, bad) for kind, goods, bads in prior.SENTINELS for good in goods.split() for bad in bads.split()]
    sentinels = iter(rng.permutation(len(bank))[:12 * total_sentinels].reshape(12, total_sentinels)[slot - 1])
    labels = np.random.default_rng(SEED + 10_000 + slot)
    sets = {}
    for name, (n_target, n_sentinel) in SIZES.items():
        forms = FORMS["multi" if name == "practice" else name]
        pairs = []
        for wording in dict.fromkeys(forms):
            for label in labels.permutation(["A", "B"] * (forms.count(wording) // 2)):
                pair = int(next(numbers))
                low, high = 530 + pair // 300, 630 + pair % 300
                a, b = (high, low) if label == "A" else (low, high)
                pairs.append((a, b, str(label), wording))
        if len(pairs) != n_target:
            raise ValueError("wording schedule changes the target budget")
        unrelated = []
        for i, label in enumerate(labels.permutation(["A", "B"] * (n_sentinel // 2))):
            kind, good, bad = bank[int(next(sentinels))]
            a, b = (good, bad) if label == "A" else (bad, good)
            unrelated.append({"id": f"{slot:02d}-{name}-s{i:02d}", "label": str(label),
                              "text": f"Choose the option that names {kind}.\nA. {a}\nB. {b}\nAnswer with A or B."})
        for mode in (MODES if name == "practice" else (name,)):
            targets = []
            for i, (a, b, label, wording) in enumerate(pairs):
                wording = "standard" if mode == "single" else wording
                targets.append({"id": f"{slot:02d}-{name}-t{i:02d}", "label": label,
                                "form": "same" if wording == "standard" else "paraphrase", "wording": wording,
                                "text": TEMPLATES[wording].format(a=a, b=b)})
            sets[mode] = {"target": targets, "sentinel": unrelated}
    return sets


def prepare(case, pristine, spec, config):
    case.mkdir(parents=True)
    public, sealed = case / "public", case / "evaluator"
    public.mkdir()
    sealed.mkdir()
    if digest(pristine / BASE_SHARD) != config["shard_sha256"]:
        raise ValueError("pristine shard hash differs from manifest")
    sets = build_splits(spec["slot"])
    for mode in MODES:
        write(public / f"{mode}.json", sets[mode])
    for name in ("construction", "holdout"):
        write(sealed / f"{name}.json", sets[name])
    write(sealed / "case.json", {"spec": spec, "pristine_sha256": load_checkpoint(pristine).checkpoint_sha256(),
                                "holdout_sha256": digest(sealed / "holdout.json")})
    runner = Runner(pristine)
    layer, construction = spec["layer"], sets["construction"]
    baseline, activations = measure(runner, construction, capture=(layer,))
    calibration = config["calibration"]
    attempts = []
    accepted = None
    if correct(baseline["target"]) >= calibration["minimum_pristine_correct"]:
        desired = -signed_labels(construction)
        if spec["fault"] == "one_position":
            desired[desired != (-1.0 if spec["replicate"] == 0 else 1.0)] = 0
        columns, vector = activation_gate(activations[layer], desired, sentinel_weight=64.0)
        direction = runner.direction()
        for strength in FAULT_STRENGTHS:
            delta = strength * np.outer(direction, vector)
            with runner.edit(layer, columns, delta) as norm:
                rows, _ = measure(runner, construction)
            loss = correct(baseline["target"]) - correct(rows["target"])
            changes = flips(baseline["sentinel"], rows["sentinel"])
            attempts.append({"strength": strength, "norm": norm, "loss": loss,
                             "unrelated_flips": changes, "rows": rows})
            if loss >= calibration["minimum_lost_targets"] and changes <= calibration["max_unrelated_flips"]:
                accepted = delta
                break
    write(sealed / "calibration.json", {"baseline": baseline, "attempts": attempts, "resources": runner.resources()})
    if accepted is None:
        write(sealed / "hold.json", {"reason": "no bounded fault met frozen construction calibration"})
        return
    patch = patch_document(pristine, layer, columns, accepted, "Evaluator-only injected wording-ablation fault.")
    patch["evidence_ids"] = ["qwen-wording-ablation-v1"]
    write(sealed / "fault_patch.json", patch)
    receipt = apply_safetensors_patch(pristine, patch, policy(), public / "damaged")
    write(sealed / "fault_receipt.json", receipt)
    write(public / "input.json", {"practice_sha256": {mode: digest(public / f"{mode}.json") for mode in MODES},
                                  "damaged_sha256": receipt["candidate_sha256"], "random_seed": spec["seed"] + 1000,
                                  "source_sha256": digest(SOURCE)})


def check_input(public):
    inputs = read(public / "input.json")
    base = public / "damaged"
    if (inputs["source_sha256"] != digest(SOURCE) or
            inputs["practice_sha256"] != {mode: digest(public / f"{mode}.json") for mode in MODES} or
            inputs["damaged_sha256"] != load_checkpoint(base).checkpoint_sha256()):
        raise ValueError("search inputs do not match frozen receipts")
    return inputs


def search(public, out):
    out.mkdir()
    inputs = check_input(public)
    base = public / "damaged"
    runner = Runner(base)
    direction = runner.direction()
    for mode, arms in ARMS.items():
        groups = read(public / f"{mode}.json")
        rng = np.random.default_rng(inputs["random_seed"])
        start = runner.forward_batches
        baseline, activations = measure(runner, groups, capture=LAYERS)
        baseline_forwards = runner.forward_batches - start
        desired = signed_labels(groups)
        logs, proposals = {arm: [] for arm in arms}, {arm: [] for arm in arms}
        for layer in LAYERS:
            columns, vector = activation_gate(activations[layer], desired, sentinel_weight=256.0)
            for strength in STRENGTHS:
                delta = strength * np.outer(direction, vector)
                for arm, proposal in zip(arms, (delta, random_delta(rng, delta))):
                    with runner.edit(layer, columns, proposal) as norm:
                        before = runner.forward_batches
                        rows, _ = measure(runner, groups)
                    logs[arm].append({"layer": layer, "columns": columns, "strength": strength,
                                      "forward_batches": runner.forward_batches - before,
                                      "requested_norm": float(np.linalg.norm(proposal)),
                                      **practice_metrics(baseline, rows, norm)})
                    proposals[arm].append((layer, columns, proposal))
                a, b = (logs[arm][-1]["norm"] for arm in arms)
                if not a or abs(a - b) / a > 0.05:
                    raise ValueError("paired candidates differ by more than 5% in realized norm")
        selected = {arm: select_attempt(logs[arm]) for arm in arms}
        for arm, index in selected.items():
            if index is None:
                continue
            layer, columns, delta = proposals[arm][index]
            patch = patch_document(base, layer, columns, delta, f"Practice-selected wording ablation {arm}.")
            patch["evidence_ids"] = ["qwen-wording-ablation-v1"]
            write(out / f"{arm}_patch.json", patch)
            write(out / f"{arm}_receipt.json", apply_safetensors_patch(base, patch, policy(), out / arm))
        write(out / f"{mode}.json", {"baseline": baseline, "attempts": logs, "selected": selected,
                                    "baseline_forward_batches": baseline_forwards,
                                    "forward_batches": runner.forward_batches - start,
                                    "input_sha256": digest(public / "input.json")})
    write(out / "search_resources.json", runner.resources())


def outcomes(scores, holdout, selected, gate):
    results = {}
    for mode, random in ARMS.values():
        names = {"pristine": "pristine", "damaged": "damaged", "guided": mode, "random": random}
        results[mode] = prior.outcome({key: scores[value] for key, value in names.items()}, holdout,
                                      {"guided": selected[mode], "random": selected[random]}, gate)
    return results


def evaluate(case, pristine, config):
    out = case / "evaluation"
    out.mkdir()
    public, sealed, search_out = (case / name for name in ("public", "evaluator", "search"))
    info = read(sealed / "case.json")
    inputs = check_input(public)
    if (info["holdout_sha256"] != digest(sealed / "holdout.json") or
            info["pristine_sha256"] != load_checkpoint(pristine).checkpoint_sha256()):
        raise ValueError("evaluator inputs differ from frozen receipts")
    holdout = read(sealed / "holdout.json")
    scores, resources, selected = {}, {}, {}
    for name, path in (("pristine", pristine), ("damaged", public / "damaged")):
        runner = Runner(path)
        scores[name], _ = measure(runner, holdout)
        resources[name] = runner.resources()
        del runner
    for mode, arms in ARMS.items():
        log = read(search_out / f"{mode}.json")
        if log["input_sha256"] != digest(public / "input.json"):
            raise ValueError("search evidence uses different inputs")
        practice = read(public / f"{mode}.json")
        for arm in arms:
            index = selected[arm] = log["selected"][arm]
            if index is None:
                scores[arm] = scores["damaged"]
                continue
            path = search_out / arm
            receipt = read(search_out / f"{arm}_receipt.json")
            if (receipt["base_sha256"] != inputs["damaged_sha256"] or
                    receipt["candidate_sha256"] != load_checkpoint(path).checkpoint_sha256()):
                raise ValueError("reloaded checkpoint hash differs from selected receipt")
            runner = Runner(path)
            if measure(runner, practice)[0] != log["attempts"][arm][index]["rows"]:
                raise ValueError("reloaded finalist did not reproduce practice scores")
            scores[arm], _ = measure(runner, holdout)
            resources[arm] = runner.resources()
            del runner
    write(out / "scores.json", scores)
    modes = outcomes(scores, holdout, selected, config["gate"])
    write(out / "summary.json", {"disposition": modes["multi"]["disposition"], "modes": modes,
                                 "resources": resources, "selected": selected,
                                 "scores_sha256": digest(out / "scores.json")})


def report(run, config):
    rows = []
    for spec in config["cases"]:
        case = run / f"case-{spec['slot']:02d}"
        status = case_status(case)
        row = {**spec, "status": status}
        if status in ("PASS", "FAIL", "INCONCLUSIVE"):
            summary = read(case / "evaluation/summary.json")
            scores = read(case / "evaluation/scores.json")
            holdout = read(case / "evaluator/holdout.json")
            selected, batches = {}, {}
            for mode, arms in ARMS.items():
                log = read(case / f"search/{mode}.json")
                for arm in arms:
                    attempts = log["attempts"][arm]
                    selected[arm] = select_attempt(attempts)
                    if (len(attempts) != config["candidate_evaluations_per_arm"] or
                            selected[arm] != log["selected"][arm]):
                        raise ValueError("practice budget or finalist differs from the frozen rule")
                    batches[arm] = sum(attempt["forward_batches"] for attempt in attempts)
            derived = outcomes(scores, holdout, selected, config["gate"])
            if (len(set(batches.values())) != 1 or digest(case / "evaluation/scores.json") != summary["scores_sha256"]
                    or selected != summary["selected"] or derived != summary["modes"]
                    or derived["multi"]["disposition"] != status):
                raise ValueError("case summary or practice budget differs from raw evidence")
            row.update(modes=derived, candidate_forward_batches=batches)
        elif status.startswith("HOLD"):
            path = case / ("evaluator/hold.json" if status == "HOLD_CALIBRATION" else "failure.json")
            row["reason"] = read(path)["reason"]
        resources = [read(path) for path in case.glob("*/resources.json")]
        row["resources"] = {"wall_seconds": sum(r["wall_seconds"] for r in resources),
                            "cpu_seconds": sum(r["cpu_seconds"] for r in resources),
                            "peak_rss_bytes": max((r["peak_rss_bytes"] for r in resources), default=None)}
        rows.append(row)
    statuses = ("PASS", "FAIL", "INCONCLUSIVE", "HOLD_CALIBRATION", "HOLD_EXECUTION", "PARTIAL", "UNRUN")
    counts = {mode: {status: sum(r.get("modes", {}).get(mode, {}).get("disposition", r["status"]) == status
                                 for r in rows) for status in statuses} for mode in MODES}
    dispositions = {mode: ("HOLD" if any(counts[mode][s] for s in ("HOLD_EXECUTION", "PARTIAL", "UNRUN"))
                           else "PASS" if counts[mode]["PASS"] >= config["gate"]["minimum_case_passes"] else "FAIL")
                    for mode in MODES}
    return {"disposition": dispositions["multi"], "dispositions": dispositions, "counts": counts, "cases": rows}


def launch(run, pristine, config, slot):
    case = run / f"case-{slot:02d}"
    if case.exists():
        return slot, case_status(case)
    for stage in ("prepare", "search", "evaluate"):
        cmd = [sys.executable, str(SOURCE), stage]
        if stage == "search":
            cmd += ["--public", str(case / "public"), "--out", str(case / "search")]
        else:
            cmd += ["--case", str(case), "--config", str(config), "--pristine", str(pristine), "--slot", str(slot)]
        with (run / f"case-{slot:02d}-{stage}.log").open("xb") as log:
            try:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=420, check=False)
                failure = f"exit code {result.returncode}" if result.returncode else None
            except subprocess.TimeoutExpired:
                failure = "420-second stage timeout"
        if failure:
            case.mkdir(exist_ok=True)
            write(case / "failure.json", {"stage": stage, "reason": failure})
            break
        if case_status(case) == "HOLD_CALIBRATION":
            break
    return slot, case_status(case)


def export(run, config, out):
    summary = report(run, config)
    out.mkdir(parents=True)
    shutil.copyfile(run / "manifest.json", out / "manifest.json")
    for case in run.glob("case-*"):
        if not case.is_dir():
            continue
        for path in (*case.glob("*.json"), *case.glob("*/*.json")):
            dest = out / path.relative_to(run)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if path.name.endswith("_patch.json"):
                dest.with_suffix(".json.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            else:
                shutil.copyfile(path, dest)
    write(out / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("freeze", "prepare", "evaluate", "run", "export"):
        stage = sub.add_parser(name)
        stage.add_argument("--config", type=Path, default=CONFIG)
        if name in ("prepare", "evaluate"):
            stage.add_argument("--case", type=Path, required=True)
            stage.add_argument("--slot", type=int, choices=range(1, 13), required=True)
        if name in ("prepare", "evaluate", "run"):
            stage.add_argument("--pristine", type=Path, required=True)
        if name in ("run", "export"):
            stage.add_argument("--run", type=Path, required=True)
        if name == "export":
            stage.add_argument("--out", type=Path, required=True)
        if name == "run":
            stage.add_argument("--slots", type=int, nargs="+", choices=range(1, 13), default=list(range(1, 13)))
            stage.add_argument("--jobs", type=int, choices=(1, 2, 3), default=1)
    stage = sub.add_parser("search")
    stage.add_argument("--public", type=Path, required=True)
    stage.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "freeze":
        write(args.config, protocol())
        return
    if args.stage != "search":
        config = load_protocol(args.config)
    if args.stage == "run":
        if len(set(args.slots)) != len(args.slots):
            parser.error("slots must be unique")
        args.run.mkdir(parents=True, exist_ok=True)
        manifest = args.run / "manifest.json"
        if not manifest.exists():
            shutil.copyfile(args.config, manifest)
        elif read(manifest) != config:
            raise ValueError("run directory belongs to a different cohort")
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for slot, status in pool.map(
                lambda slot: launch(args.run.resolve(), args.pristine.resolve(), args.config.resolve(), slot), args.slots
            ):
                print(f"Case {slot:02d}: {status}", flush=True)
        print(report(args.run, config)["counts"], flush=True)
        return
    if args.stage == "export":
        if read(args.run / "manifest.json") != config:
            raise ValueError("run directory belongs to a different cohort")
        export(args.run, config, args.out)
        return
    wall, cpu = time.perf_counter(), time.process_time()
    if args.stage == "prepare":
        prepare(args.case, args.pristine, config["cases"][args.slot - 1], config)
        destination = args.case / "evaluator"
    elif args.stage == "search":
        destination = args.out
        search(args.public, destination)
    else:
        evaluate(args.case, args.pristine, config)
        destination = args.case / "evaluation"
    write(destination / "resources.json", {"wall_seconds": time.perf_counter() - wall,
                                           "cpu_seconds": time.process_time() - cpu,
                                           "peak_rss_bytes": peak_rss_bytes(), "completed_utc": timestamp()})


if __name__ == "__main__":
    main()
