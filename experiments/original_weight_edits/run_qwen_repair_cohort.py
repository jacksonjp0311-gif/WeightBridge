"""Run a frozen, paired Qwen weight repair cohort on CPU."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import itertools
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from run_blind_qwen_repair import (
    BASE_SHARD, LAYERS, MODEL, REVISION, ROOT, WIDTH, Runner, apply_safetensors_patch,
    correct, digest, flips, load_checkpoint, mean_margin, measure, patch_document,
    peak_rss_bytes, policy, read, timestamp, write,
)

SOURCE = Path(__file__).resolve()
BACKEND = SOURCE.with_name("run_blind_qwen_repair.py")
CONFIG = ROOT / "benchmarks/configs/qwen_repair_cohort_v2.json"
ARMS = ("guided", "random")
SEED = 914183
STRENGTHS = (0.03, 0.07, 0.15)
FAULT_STRENGTHS = (0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28)
SIZES = {"construction": (8, 8), "practice": (8, 16), "holdout": (24, 16)}
SHARD_SHA = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
SENTINELS = (
    ("a color", "amber teal green blue purple indigo black brown", "desk shelf floor curtain carpet pillar brick door"),
    ("an animal", "dog wolf horse fox tiger otter lion zebra", "book piano hammer mirror bottle window brush shoe"),
    ("a fruit", "apple pear peach plum mango grape cherry banana", "broom lamp needle wrench rope radio scarf bucket"),
    ("a vehicle", "bus train truck car van bicycle tram tractor", "cup fork plate shirt flower guitar candle pillow"),
    ("an instrument", "violin piano drum flute harp cello trumpet clarinet", "boot table pepper ruler shovel spoon carton sock"),
    ("a tool", "hammer drill saw chisel pliers wrench file vise", "banana apple cheese pear peach cake toast berry"),
    ("a garment", "coat dress shirt scarf vest jacket skirt sweater", "desk stool bench ladder sofa cabinet oven board"),
    ("a beverage", "water juice milk tea coffee soda cider lemonade", "marble rock gravel pebble sand clay tile slate"),
)


def protocol():
    cases = list(itertools.product(LAYERS, ("reversal", "one_position"), (0, 1)))
    order = np.random.default_rng(SEED).permutation(len(cases))
    return {
        "schema": "weightbridge.qwen_repair_cohort.v2",
        "source_sha256": {p.name: digest(p) for p in (SOURCE, BACKEND)},
        "model_id": MODEL, "model_revision": REVISION, "shard_sha256": SHARD_SHA,
        "seed": SEED, "policy": policy(), "strengths": list(STRENGTHS),
        "fault_strengths": list(FAULT_STRENGTHS), "candidate_evaluations_per_arm": len(LAYERS) * len(STRENGTHS),
        "splits": {name: list(size) for name, size in SIZES.items()},
        "calibration": {"minimum_pristine_correct": 6, "minimum_lost_targets": 2, "max_unrelated_flips": 1},
        "gate": {"minimum_case_passes": 8, "minimum_holdout_loss": 2, "minimum_recovery": 0.5,
                 "max_unrelated_flips_vs_pristine": 0, "require_strict_win_against_random": True,
                 "require_no_paraphrase_regression_vs_damaged": True},
        "cases": [
            {"slot": slot, "layer": cases[int(i)][0], "fault": cases[int(i)][1],
             "replicate": cases[int(i)][2], "seed": SEED + 100 + slot}
            for slot, i in enumerate(order, 1)
        ],
    }


def load_protocol(config):
    frozen = read(config)
    if frozen != protocol():
        raise ValueError("cohort manifest does not match the frozen source and protocol")
    return frozen


def build_splits(slot):
    """Partition unique numeric and sentinel pairs before assigning labels."""
    rng = np.random.default_rng(SEED + 1)
    total_target = sum(n for n, _ in SIZES.values())
    total_sentinel = sum(n for _, n in SIZES.values())
    numeric = iter(rng.choice(100 * 300, 12 * total_target, replace=False)
                   .reshape(12, total_target)[slot - 1])
    bank = [(kind, good, bad) for kind, good_words, bad_words in SENTINELS
            for good in good_words.split() for bad in bad_words.split()]
    sentinel = iter(rng.permutation(len(bank))[:12 * total_sentinel]
                    .reshape(12, total_sentinel)[slot - 1])
    labels_rng = np.random.default_rng(SEED + 10_000 + slot)
    sets = {}
    for name, (n_target, n_sentinel) in SIZES.items():
        targets = []
        forms = (("same", n_target),) if name != "holdout" else (("same", n_target // 2), ("paraphrase", n_target // 2))
        for form, size in forms:
            for label in labels_rng.permutation(["A", "B"] * (size // 2)):
                pair = int(next(numeric))
                low, high = 400 + pair // 300, 600 + pair % 300
                a, b = (high, low) if label == "A" else (low, high)
                prompt = (f"Which integer is larger?\nA. {a}\nB. {b}\nAnswer with the letter A or B only."
                          if form == "same" else
                          f"Which number is larger, A) {a} or B) {b}? Answer with the letter of the larger number only.")
                targets.append({"id": f"{slot:02d}-{name}-t{len(targets):02d}", "label": str(label),
                                "form": form, "text": prompt})
        unrelated = []
        for i, label in enumerate(labels_rng.permutation(["A", "B"] * (n_sentinel // 2))):
            kind, good, bad = bank[int(next(sentinel))]
            a, b = (good, bad) if label == "A" else (bad, good)
            unrelated.append({"id": f"{slot:02d}-{name}-s{i:02d}", "label": str(label),
                              "text": f"Pick the option that names {kind}.\nA. {a}\nB. {b}\nAnswer with A or B."})
        sets[name] = {"target": targets, "sentinel": unrelated}
    return sets


def activation_gate(activations, desired, sentinel_weight):
    """Fit a signed practice response, penalizing activation on sentinels."""
    x = activations.astype(np.float64)
    scale = np.maximum(np.sqrt(np.mean(x * x, axis=0)), 1e-6)
    contrast = np.abs(x[:len(desired)].T @ desired) / scale
    indices = np.argsort(-contrast, kind="stable")[:WIDTH].tolist()
    z = x[:, indices] / scale[indices]
    target, unrelated = z[:len(desired)], z[len(desired):]
    gram = target.T @ target + sentinel_weight * (unrelated.T @ unrelated) + np.eye(len(indices))
    return indices, np.linalg.solve(gram, target.T @ desired) / scale[indices]


def signed_labels(groups):
    return np.array([1.0 if row["label"] == "A" else -1.0 for row in groups["target"]])


def prepare(case, pristine, spec, config):
    case.mkdir(parents=True)
    public, sealed = case / "public", case / "evaluator"
    public.mkdir()
    sealed.mkdir()
    if digest(pristine / BASE_SHARD) != config["shard_sha256"]:
        raise ValueError("pristine shard hash differs from manifest")
    groups = build_splits(spec["slot"])
    write(public / "practice.json", groups["practice"])
    for name in ("construction", "holdout"):
        write(sealed / f"{name}.json", groups[name])
    write(sealed / "case.json", {"spec": spec, "pristine_sha256": load_checkpoint(pristine).checkpoint_sha256(),
                                "holdout_sha256": digest(sealed / "holdout.json")})
    runner = Runner(pristine)
    layer = spec["layer"]
    construction = groups["construction"]
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
        reason = "pristine calibration floor failed" if not attempts else "no bounded fault satisfied calibration"
        write(sealed / "hold.json", {"reason": reason})
        return
    patch = patch_document(pristine, layer, columns, accepted, "Evaluator-only injected cohort fault.")
    write(sealed / "fault_patch.json", patch)
    receipt = apply_safetensors_patch(pristine, patch, policy(), public / "damaged")
    write(sealed / "fault_receipt.json", receipt)
    write(public / "input.json", {"practice_sha256": digest(public / "practice.json"),
                                  "damaged_sha256": receipt["candidate_sha256"],
                                  "random_seed": spec["seed"] + 1000,
                                  "source_sha256": digest(SOURCE)})


def random_delta(rng, delta):
    rows, columns = rng.standard_normal(delta.shape[0]), rng.standard_normal(delta.shape[1])
    return np.linalg.norm(delta) * np.outer(rows / np.linalg.norm(rows), columns / np.linalg.norm(columns))


def practice_metrics(baseline, rows, norm):
    drift = sum(abs(a["margin"] - b["margin"]) for a, b in zip(baseline["sentinel"], rows["sentinel"]))
    return {"rows": rows, "target_gain": correct(rows["target"]) - correct(baseline["target"]),
            "margin_gain": mean_margin(rows["target"]) - mean_margin(baseline["target"]),
            "unrelated_flips": flips(baseline["sentinel"], rows["sentinel"]), "drift": drift, "norm": norm}


def select_attempt(attempts):
    eligible = [(-row["target_gain"], row["drift"], row["norm"], -row["margin_gain"], i)
                for i, row in enumerate(attempts) if row["target_gain"] > 0 and row["unrelated_flips"] == 0]
    return min(eligible)[-1] if eligible else None


def search(public, out):
    out.mkdir()
    inputs = read(public / "input.json")
    base = public / "damaged"
    if (inputs["source_sha256"] != digest(SOURCE) or
            inputs["practice_sha256"] != digest(public / "practice.json") or
            inputs["damaged_sha256"] != load_checkpoint(base).checkpoint_sha256()):
        raise ValueError("search inputs do not match frozen receipts")
    groups = read(public / "practice.json")
    rng = np.random.default_rng(inputs["random_seed"])
    runner = Runner(base)
    baseline, activations = measure(runner, groups, capture=LAYERS)
    shared_forwards = runner.forward_batches
    direction, desired = runner.direction(), signed_labels(groups)
    logs, proposals = {arm: [] for arm in ARMS}, {arm: [] for arm in ARMS}
    for layer in LAYERS:
        columns, vector = activation_gate(activations[layer], desired, sentinel_weight=256.0)
        for strength in STRENGTHS:
            delta = strength * np.outer(direction, vector)
            paired = {"guided": delta, "random": random_delta(rng, delta)}
            for arm in ARMS:
                with runner.edit(layer, columns, paired[arm]) as norm:
                    start = runner.forward_batches
                    rows, _ = measure(runner, groups)
                logs[arm].append({"layer": layer, "columns": columns, "strength": strength,
                                  "forward_batches": runner.forward_batches - start,
                                  "requested_norm": float(np.linalg.norm(paired[arm])),
                                  **practice_metrics(baseline, rows, norm)})
                proposals[arm].append((layer, columns, paired[arm]))
            a, b = (logs[arm][-1]["norm"] for arm in ARMS)
            if not a or abs(a - b) / a > 0.05:
                raise ValueError("paired candidates differ by more than 5% in realized norm")
    selected = {arm: select_attempt(logs[arm]) for arm in ARMS}
    for arm, index in selected.items():
        if index is not None:
            layer, columns, delta = proposals[arm][index]
            patch = patch_document(base, layer, columns, delta, f"Practice-selected cohort {arm} intervention.")
            write(out / f"{arm}_patch.json", patch)
            write(out / f"{arm}_receipt.json", apply_safetensors_patch(base, patch, policy(), out / arm))
    write(out / "search.json", {"baseline": baseline, "attempts": logs, "selected": selected,
                               "shared_baseline_forward_batches": shared_forwards, "resources": runner.resources(),
                               "input_sha256": digest(public / "input.json")})


def outcome(scores, holdout, selected, gate):
    target = {name: correct(groups["target"]) for name, groups in scores.items()}
    lost = target["pristine"] - target["damaged"]
    recovery = (target["guided"] - target["damaged"]) / lost if lost > 0 else None
    heldout_forms = {}
    for form in ("same", "paraphrase"):
        indices = [i for i, row in enumerate(holdout["target"]) if row["form"] == form]
        heldout_forms[form] = {name: correct([groups["target"][i] for i in indices])
                               for name, groups in scores.items()}
    pristine = {arm: flips(scores["pristine"]["sentinel"], scores[arm]["sentinel"]) for arm in ARMS}
    damaged = {arm: flips(scores["damaged"]["sentinel"], scores[arm]["sentinel"]) for arm in ARMS}
    passes = (selected["guided"] is not None and recovery is not None and recovery >= gate["minimum_recovery"]
              and target["guided"] > target["random"] and pristine["guided"] <= gate["max_unrelated_flips_vs_pristine"]
              and heldout_forms["paraphrase"]["guided"] >= heldout_forms["paraphrase"]["damaged"])
    return {"disposition": "INCONCLUSIVE" if lost < gate["minimum_holdout_loss"] else "PASS" if passes else "FAIL",
            "target_correct": target, "target_loss": lost, "recovery": recovery, "target_by_form": heldout_forms,
            "sentinel_correct": {name: correct(groups["sentinel"]) for name, groups in scores.items()},
            "sentinel_flips_vs_pristine": pristine, "sentinel_flips_vs_damaged": damaged}


def evaluate(case, pristine, config):
    out = case / "evaluation"
    out.mkdir()
    public, sealed, search_out = (case / name for name in ("public", "evaluator", "search"))
    info, log = read(sealed / "case.json"), read(search_out / "search.json")
    inputs = read(public / "input.json")
    base = public / "damaged"
    if (info["holdout_sha256"] != digest(sealed / "holdout.json") or
            info["pristine_sha256"] != load_checkpoint(pristine).checkpoint_sha256() or
            inputs["damaged_sha256"] != load_checkpoint(base).checkpoint_sha256() or
            inputs["practice_sha256"] != digest(public / "practice.json") or
            log["input_sha256"] != digest(public / "input.json")):
        raise ValueError("evaluator inputs do not match frozen receipts")
    holdout, practice = read(sealed / "holdout.json"), read(public / "practice.json")
    scores, resources = {}, {}
    for name, path in (("pristine", pristine), ("damaged", base), *((arm, search_out / arm) for arm in ARMS)):
        if name in ARMS:
            index = log["selected"][name]
            if index is None:
                scores[name] = scores["damaged"]
                continue
            receipt = read(search_out / f"{name}_receipt.json")
            if (receipt["base_sha256"] != inputs["damaged_sha256"] or
                    receipt["candidate_sha256"] != load_checkpoint(path).checkpoint_sha256()):
                raise ValueError("reloaded checkpoint hash differs from selected receipt")
        runner = Runner(path)
        if name in ARMS and measure(runner, practice)[0] != log["attempts"][name][index]["rows"]:
            raise ValueError("reloaded finalist did not reproduce practice scores")
        scores[name], _ = measure(runner, holdout)
        resources[name] = runner.resources()
        del runner
    write(out / "scores.json", scores)
    summary = outcome(scores, holdout, log["selected"], config["gate"])
    write(out / "summary.json", {**summary, "resources": resources, "selected": log["selected"],
                                 "scores_sha256": digest(out / "scores.json")})


def case_status(case):
    if (case / "evaluation/summary.json").is_file() and (case / "evaluation/resources.json").is_file():
        return read(case / "evaluation/summary.json")["disposition"]
    if (case / "evaluator/hold.json").is_file():
        return "HOLD_CALIBRATION"
    if (case / "failure.json").is_file():
        return "HOLD_EXECUTION"
    return "PARTIAL" if case.exists() else "UNRUN"


def launch(run, pristine, config_path, slot):
    case = run / f"case-{slot:02d}"
    if case.exists():
        return slot, case_status(case)
    for stage in ("prepare", "search", "evaluate"):
        cmd = [sys.executable, str(SOURCE), stage]
        if stage == "search":
            cmd += ["--public", str(case / "public"), "--out", str(case / "search")]
        else:
            cmd += ["--config", str(config_path), "--case", str(case), "--pristine", str(pristine), "--slot", str(slot)]
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


def report(run, config):
    rows = []
    for spec in config["cases"]:
        case = run / f"case-{spec['slot']:02d}"
        status = case_status(case)
        row = {**spec, "status": status}
        if status in ("PASS", "FAIL", "INCONCLUSIVE"):
            summary = read(case / "evaluation/summary.json")
            scores = read(case / "evaluation/scores.json")
            log = read(case / "search/search.json")
            holdout = read(case / "evaluator/holdout.json")
            expected = outcome(scores, holdout, log["selected"], config["gate"])
            if (digest(case / "evaluation/scores.json") != summary["scores_sha256"] or
                    any(summary[key] != value for key, value in expected.items())):
                raise ValueError("cohort summary differs from raw rows")
            for arm in ARMS:
                if (len(log["attempts"][arm]) != config["candidate_evaluations_per_arm"] or
                        select_attempt(log["attempts"][arm]) != log["selected"][arm]):
                    raise ValueError("practice budget or candidate selection differs from manifest")
            counts = {arm: sum(r["forward_batches"] for r in log["attempts"][arm]) for arm in ARMS}
            if counts["guided"] != counts["random"]:
                raise ValueError("practice forward budgets differ between arms")
            row.update(expected, candidate_forward_batches=counts)
        elif status.startswith("HOLD"):
            path = case / ("evaluator/hold.json" if status == "HOLD_CALIBRATION" else "failure.json")
            row["reason"] = read(path)["reason"]
        resources = [read(path) for path in case.glob("*/resources.json")]
        row["resources"] = {"wall_seconds": sum(r["wall_seconds"] for r in resources),
                            "cpu_seconds": sum(r["cpu_seconds"] for r in resources),
                            "peak_rss_bytes": max((r["peak_rss_bytes"] for r in resources), default=None)}
        rows.append(row)
    counts = {status: sum(row["status"] == status for row in rows)
              for status in ("PASS", "FAIL", "INCONCLUSIVE", "HOLD_CALIBRATION", "HOLD_EXECUTION", "PARTIAL", "UNRUN")}
    incomplete = counts["PARTIAL"] + counts["UNRUN"] + counts["HOLD_EXECUTION"]
    disposition = "HOLD" if incomplete else "PASS" if counts["PASS"] >= config["gate"]["minimum_case_passes"] else "FAIL"
    return {"disposition": disposition, "counts": counts, "cases": rows}


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
    sub.add_parser("freeze").add_argument("--config", type=Path, default=CONFIG)
    for name in ("prepare", "evaluate", "run", "export"):
        stage = sub.add_parser(name)
        stage.add_argument("--config", type=Path, default=CONFIG)
        if name in ("prepare", "evaluate"):
            stage.add_argument("--case", type=Path, required=True)
            stage.add_argument("--slot", type=int, choices=range(1, 13), required=True)
        else:
            stage.add_argument("--run", type=Path, required=True)
        if name != "export":
            stage.add_argument("--pristine", type=Path, required=True)
        else:
            stage.add_argument("--out", type=Path, required=True)
        if name == "run":
            stage.add_argument("--slots", type=int, nargs="+", choices=range(1, 13), default=list(range(1, 13)))
            stage.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    stage = sub.add_parser("search")
    stage.add_argument("--public", type=Path, required=True)
    stage.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "freeze":
        write(args.config, protocol())
        return
    wall, cpu = time.perf_counter(), time.process_time()
    if args.stage == "search":
        search(args.public, args.out)
        destination = args.out
    else:
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
            export(args.run, config, args.out)
            return
        if args.stage == "prepare":
            prepare(args.case, args.pristine, config["cases"][args.slot - 1], config)
            destination = args.case / "evaluator"
        else:
            evaluate(args.case, args.pristine, config)
            destination = args.case / "evaluation"
    write(destination / "resources.json", {"wall_seconds": time.perf_counter() - wall,
                                           "cpu_seconds": time.process_time() - cpu,
                                           "peak_rss_bytes": peak_rss_bytes(), "completed_utc": timestamp()})


if __name__ == "__main__":
    main()
