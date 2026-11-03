"""Inject, search, and independently score a bounded Qwen weight repair."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches.huggingface import apply_safetensors_patch
from weightbridge.provenance import append_experiment, peak_rss_bytes


MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
BASE_SHARD = "model.safetensors-00001-of-00001.safetensors"
LAYERS = (19, 21, 23)
WIDTH = 16
BATCH = 8
STRENGTHS = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
FAULT_STRENGTHS = (4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
SPLIT_SEEDS = {"construction": 813005, "practice": 813006, "holdout": 813007}
RANDOM_SEEDS = (813021, 813022, 813023)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, obj):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_name(layer):
    return f"model.language_model.layers.{layer}.mlp.down_proj.weight"


def policy():
    return {
        "schema": "weightbridge.policy.v1", "allow_tensors": [tensor_name(i) for i in LAYERS],
        "max_entries": WIDTH * 1024, "max_fraction": 0.00003, "max_abs_delta": 8.0,
        "max_tensor_bytes": 8 * 1024**2, "max_total_tensor_bytes": 8 * 1024**2,
    }


def build_splits(seeds):
    """Use disjoint numeric ranges and distinct unrelated examples per split."""
    groups = (
        ("a color", ("red", "green", "blue", "yellow", "purple", "white", "black", "orange"),
         ("table", "spoon", "ladder", "chair", "clock", "pencil", "brick", "door")),
        ("an animal", ("dog", "cat", "horse", "fox", "tiger", "rabbit", "lion", "zebra"),
         ("book", "piano", "hammer", "mirror", "bottle", "window", "brush", "shoe")),
        ("a fruit", ("apple", "pear", "peach", "plum", "mango", "grape", "cherry", "banana"),
         ("desk", "lamp", "needle", "broom", "rope", "radio", "scarf", "bucket")),
        ("a vehicle", ("bus", "train", "truck", "car", "van", "bicycle", "scooter", "tractor"),
         ("cup", "fork", "plate", "shirt", "flower", "guitar", "candle", "carpet")),
    )
    suffix = "\nAnswer with the letter A or B only."
    sizes = {"construction": (8, 8), "practice": (12, 12), "holdout": (24, 32)}
    sets = {}
    for split, (n_target, n_other) in sizes.items():
        rng = np.random.default_rng(seeds[split])
        offset = 1000 * list(sizes).index(split)
        lows = rng.choice(np.arange(offset + 100, offset + 400), n_target, replace=False)
        highs = rng.choice(np.arange(offset + 600, offset + 900), n_target, replace=False)
        template = ("Select the greater integer.\nA. {}\nB. {}" if split == "holdout"
                    else "Which integer is larger?\nA. {}\nB. {}")
        target = [{"id": f"{split}-target-{i:02d}", "text": template.format(high, low) + suffix, "label": "A"}
                  for i, (low, high) in enumerate(zip(lows, highs))]
        # Change the actual pairs by split. Balance A and B in every set.
        shifts = {"construction": 0, "practice": 1, "holdout": 3}
        bank = [(kind, good[j], bad[(j + shifts[split]) % len(bad)])
                for kind, good, bad in groups for j in range(len(good))]
        other = []
        for i, j in enumerate(rng.permutation(len(bank))[:n_other]):
            kind, good, bad = bank[j]
            a, b, label = (good, bad, "A") if i % 2 == 0 else (bad, good, "B")
            other.append({"id": f"{split}-sentinel-{i:02d}",
                          "text": f"Which is {kind}?\nA. {a}\nB. {b}" + suffix, "label": label})
        sets[split] = {"target": target, "sentinel": other}
    return sets


class Runner:
    def __init__(self, directory):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        self.torch = torch
        torch.set_num_threads(6)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        self.tokenizer.padding_side = "left"
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            directory, dtype=torch.bfloat16, local_files_only=True,
        ).eval()
        ids = [self.tokenizer.encode(label, add_special_tokens=False) for label in ("A", "B")]
        if any(len(item) != 1 for item in ids) or ids[0] == ids[1]:
            raise ValueError("the two answer labels must be distinct single tokens")
        self.label_ids = [item[0] for item in ids]
        self.modules = {layer: self.model.get_submodule(tensor_name(layer).removesuffix(".weight"))
                        for layer in LAYERS}
        self.forward_batches = self.examples = self.input_tokens = 0

    def direction(self):
        rows = self.model.get_output_embeddings().weight[self.label_ids].detach().float().numpy()
        delta = (rows[0] - rows[1]).astype(np.float64)
        return delta / np.linalg.norm(delta)

    def score(self, rows, *, capture=()):
        captures = {layer: [] for layer in capture}
        handles = []
        for layer in capture:
            def hook(_module, inputs, index=layer):
                captures[index].append(inputs[0][:, -1].detach().float().numpy().copy())
            handles.append(self.modules[layer].register_forward_pre_hook(hook))
        result = []
        try:
            with self.torch.inference_mode():
                for start in range(0, len(rows), BATCH):
                    batch = rows[start:start + BATCH]
                    tokens = self.tokenizer.apply_chat_template(
                        [[{"role": "user", "content": row["text"]}] for row in batch],
                        add_generation_prompt=True, enable_thinking=False, tokenize=True,
                        padding=True, return_dict=True, return_tensors="pt",
                    )
                    if tokens["input_ids"].shape[1] > 96:
                        raise ValueError("frozen prompt exceeds the 96-token cap")
                    logits = self.model(**tokens, use_cache=False, logits_to_keep=1).logits
                    two = logits[:, -1, self.label_ids].float().numpy()
                    if not np.isfinite(two).all():
                        raise ValueError("nonfinite answer logits")
                    for row, (a, b) in zip(batch, two):
                        expected = row["label"]
                        prediction = "A" if a >= b else "B"
                        result.append({"id": row["id"], "prediction": prediction,
                                       "label": expected, "correct": prediction == expected,
                                       "margin": float(a - b if expected == "A" else b - a)})
                    self.forward_batches += 1
                    self.examples += len(batch)
                    self.input_tokens += int(tokens["attention_mask"].sum())
        finally:
            for handle in handles:
                handle.remove()
        return result, {layer: np.concatenate(parts) for layer, parts in captures.items()}

    @contextmanager
    def edit(self, layer, indices, delta):
        weight = self.modules[layer].weight
        with self.torch.no_grad():
            before = weight[:, indices].clone()
            # Match the patcher's float64 addition and BF16 rounding.
            after = (before.double() + self.torch.from_numpy(delta)).float().to(weight.dtype)
            actual = (after.float() - before.float()).numpy()
            if not np.isfinite(actual).all() or np.max(np.abs(actual)) > policy()["max_abs_delta"]:
                raise ValueError("rounded intervention exceeds host scalar cap")
            weight[:, indices] = after
            try:
                yield float(np.linalg.norm(actual.astype(np.float64)))
            finally:
                weight[:, indices] = before

    def resources(self):
        return {"forward_batches": self.forward_batches, "prompt_examples": self.examples,
                "input_tokens": self.input_tokens, "model_output_tokens_generated": 0,
                "peak_rss_bytes": peak_rss_bytes(), "torch": version("torch"),
                "transformers": version("transformers"), "numpy": version("numpy")}


def features(activations, n_target):
    """Fit a small linear gate to target activity and zero sentinel activity."""
    x = activations.astype(np.float64)
    rms = np.maximum(np.sqrt(np.mean(x * x, axis=0)), 1e-6)
    contrast = np.abs(x[:n_target].mean(axis=0) - x[n_target:].mean(axis=0)) / rms
    indices = np.argsort(-contrast, kind="stable")[:WIDTH].tolist()
    z = x[:, indices] / rms[indices]
    desired = np.r_[np.ones(n_target), np.zeros(len(x) - n_target)]
    v = np.linalg.solve(z.T @ z + np.eye(WIDTH), z.T @ desired) / rms[indices]
    return indices, v


def patch_document(base, layer, indices, delta, reason):
    checkpoint = load_checkpoint(base)
    name = tensor_name(layer)
    return {"schema": "weightbridge.patch.v1", "base_sha256": checkpoint.checkpoint_sha256(),
            "tensor_sha256": {name: checkpoint.tensor_sha256(name)}, "reason": reason,
            "evidence_ids": ["qwen-blind-injection-v1"],
            "slices": [{"tensor": name, "axis": "columns", "indices": indices, "op": "add",
                        "values": delta.T.tolist()}]}


def measure(runner, groups, *, capture=()):
    n = len(groups["target"])
    rows, activations = runner.score(groups["target"] + groups["sentinel"], capture=capture)
    return {"target": rows[:n], "sentinel": rows[n:]}, activations


def correct(rows):
    return sum(row["correct"] for row in rows)


def flips(before, after):
    if [row["id"] for row in before] != [row["id"] for row in after]:
        raise ValueError("score rows differ")
    return sum(a["prediction"] != b["prediction"] for a, b in zip(before, after))


def mean_margin(rows):
    return float(np.mean([row["margin"] for row in rows]))


def prepare(run, pristine, fault_seed):
    run.mkdir(parents=True, exist_ok=False)
    public = run / "public"
    sealed = run / "evaluator"
    public.mkdir()
    sealed.mkdir()
    expected = read(ROOT / "results/published_reproduction/p50c_rebuild_summary.json")["base_checkpoint_sha256"]
    if digest(pristine / BASE_SHARD) != expected:
        raise ValueError("pristine weight shard differs from the pinned Qwen checkpoint")
    seeds = dict(SPLIT_SEEDS)
    for name in ("construction", "holdout"):
        seeds[name] = int(np.random.SeedSequence([fault_seed, seeds[name]]).generate_state(1)[0])
    sets = build_splits(seeds)
    write(public / "practice.json", sets["practice"])
    write(sealed / "holdout.json", sets["holdout"])
    write(sealed / "construction.json", sets["construction"])
    config = {
        "schema": "weightbridge.qwen_blind_repair.v1", "timestamp_utc": timestamp(),
        "model_id": MODEL, "model_revision": REVISION, "source_sha256": digest(Path(__file__)),
        "eligible_layers": list(LAYERS), "strengths": list(STRENGTHS),
        "fault_strengths": list(FAULT_STRENGTHS), "policy": policy(),
        "practice_seed": seeds["practice"], "random_control_seeds": list(RANDOM_SEEDS),
        "practice_sha256": digest(public / "practice.json"),
        "holdout_sha256": digest(sealed / "holdout.json"),
        "construction_sha256": digest(sealed / "construction.json"),
        "primary": "fraction of lost forced-choice holdout accuracy recovered",
        "minimum_recovery": 0.5, "max_unrelated_prediction_flips_vs_damaged": 0,
        "minimum_calibration_accuracy_loss_count": 4,
        "inference": "CPU BF16, 6 threads, left padding, no thinking, max 96 input tokens, A/B logits only",
    }
    write(public / "protocol.json", config)
    frozen = load_checkpoint(pristine).checkpoint_sha256()
    runner = Runner(pristine)
    layer = int(np.random.default_rng(fault_seed).choice(LAYERS))
    groups = sets["construction"]
    baseline, activation = measure(runner, groups, capture=(layer,))
    n = len(groups["target"])
    if correct(baseline["target"]) < n - 1:
        raise RuntimeError("HOLD: pristine calibration accuracy is below the frozen floor")
    indices, v = features(activation[layer], n)
    direction = -runner.direction()
    candidates = []
    accepted = None
    for strength in FAULT_STRENGTHS:
        delta = strength * np.outer(direction, v)
        try:
            with runner.edit(layer, indices, delta) as norm:
                scores, _ = measure(runner, groups)
        except ValueError as exc:
            candidates.append({"strength": strength, "failure": str(exc)})
            continue
        loss = correct(baseline["target"]) - correct(scores["target"])
        unrelated_flips = flips(baseline["sentinel"], scores["sentinel"])
        candidates.append({"strength": strength, "rounded_frobenius_norm": norm,
                           "target_loss": loss, "sentinel_flips": unrelated_flips, "rows": scores})
        if loss >= 4 and unrelated_flips <= 1:
            accepted = delta
            break
    write(sealed / "calibration.json", {"pristine": baseline, "attempts": candidates,
                                       "fault_seed": fault_seed, "fault_layer": layer,
                                       "split_seeds": seeds,
                                       "pristine_checkpoint_sha256": frozen, "resources": runner.resources()})
    if accepted is None:
        raise RuntimeError("HOLD: no bounded injected edit met the construction criterion")
    patch = patch_document(pristine, layer, indices, accepted, "Evaluator-only injected directional fault.")
    write(sealed / "fault_patch.json", patch)
    receipt = apply_safetensors_patch(pristine, patch, policy(), public / "damaged")
    write(sealed / "fault_receipt.json", receipt)
    print(f"Damaged checkpoint created with {receipt['changed_entries']} changed values.", flush=True)


def search(public, output):
    """Read the damaged checkpoint and practice evidence through explicit paths."""
    output.mkdir(parents=True, exist_ok=False)
    base = public / "damaged"
    groups = read(public / "practice.json")
    config = read(public / "protocol.json")
    if (config["source_sha256"] != digest(Path(__file__)) or config["policy"] != policy()
            or config["practice_sha256"] != digest(public / "practice.json")):
        raise ValueError("search source or policy differs from the frozen plan")
    runner = Runner(base)
    baseline, activation = measure(runner, groups, capture=LAYERS)
    n = len(groups["target"])
    direction = runner.direction()
    attempts = []
    best = None
    for layer in LAYERS:
        indices, v = features(activation[layer], n)
        for strength in STRENGTHS:
            delta = strength * np.outer(direction, v)
            try:
                with runner.edit(layer, indices, delta) as norm:
                    scores, _ = measure(runner, groups)
            except ValueError as exc:
                attempts.append({"layer": layer, "strength": strength, "failure": str(exc)})
                continue
            flips_other = flips(baseline["sentinel"], scores["sentinel"])
            target_gain = correct(scores["target"]) - correct(baseline["target"])
            margin_gain = mean_margin(scores["target"]) - mean_margin(baseline["target"])
            attempts.append({"layer": layer, "strength": strength, "target_gain": target_gain,
                             "target_margin_gain": margin_gain, "sentinel_flips": flips_other,
                             "rounded_frobenius_norm": norm, "rows": scores})
            key = (-target_gain, norm, -margin_gain)
            if flips_other == 0 and target_gain > 0 and (best is None or key < best[0]):
                best = (key, len(attempts) - 1, layer, indices, delta)
        print(f"Practice search finished layer {layer}.", flush=True)
    write(output / "search.json", {"baseline": baseline, "attempts": attempts,
                                   "selected_attempt": None if best is None else best[1],
                                   "resources": runner.resources(),
                                   "practice_sha256": digest(public / "practice.json"),
                                   "protocol_sha256": digest(public / "protocol.json")})
    if best is None:
        raise RuntimeError("HOLD: no practice improvement preserved all practice sentinels")
    _, _, layer, indices, delta = best
    patch = patch_document(base, layer, indices, delta, "Best practice-only bounded directional intervention.")
    write(output / "repair_patch.json", patch)
    write(output / "repair_receipt.json", apply_safetensors_patch(base, patch, policy(), output / "repaired"))


def evaluate(run, pristine):
    """Score once after the repair has been frozen and saved."""
    public, sealed, search_out = (run / name for name in ("public", "evaluator", "search"))
    output = run / "evaluation"
    output.mkdir(exist_ok=False)
    base = public / "damaged"
    patch = read(search_out / "repair_patch.json")
    receipt = read(search_out / "repair_receipt.json")
    config = read(public / "protocol.json")
    if (config["source_sha256"] != digest(Path(__file__)) or
            config["holdout_sha256"] != digest(sealed / "holdout.json") or
            config["practice_sha256"] != digest(public / "practice.json") or
            load_checkpoint(pristine).checkpoint_sha256() != read(sealed / "calibration.json")["pristine_checkpoint_sha256"] or
            load_checkpoint(search_out / "repaired").checkpoint_sha256() != receipt["candidate_sha256"]):
        raise ValueError("evaluation source or checkpoint differs from its frozen hash")
    groups = read(sealed / "holdout.json")
    practice = read(public / "practice.json")
    selected = read(search_out / "search.json")
    layer = int(patch["slices"][0]["tensor"].split(".layers.")[1].split(".")[0])
    indices = patch["slices"][0]["indices"]
    delta = np.asarray(patch["slices"][0]["values"], dtype=np.float64).T
    scores, resources = {}, {}

    def score(name, path, *, verify_practice=False):
        runner = Runner(path)
        if verify_practice:
            reloaded = measure(runner, practice)[0]
            if reloaded != selected["attempts"][selected["selected_attempt"]]["rows"]:
                raise ValueError("reloaded candidate did not reproduce in-memory practice scores")
        scores[name] = measure(runner, groups)[0]
        resources[name] = runner.resources()
        write(output / f"{name}_scores.json", scores[name])
        print(f"{name}: {correct(scores[name]['target'])}/{len(groups['target'])} target choices", flush=True)

    score("pristine", pristine)
    score("damaged", base)
    score("repaired", search_out / "repaired", verify_practice=True)
    # Same layer, column support, rank and requested Frobenius norm. Record rounded norms.
    norm = float(np.linalg.norm(delta))
    actual_norm = selected["attempts"][selected["selected_attempt"]]["rounded_frobenius_norm"]
    controls = {}
    for seed in RANDOM_SEEDS:
        rng = np.random.default_rng(seed)
        u, v = rng.standard_normal(delta.shape[0]), rng.standard_normal(delta.shape[1])
        random_delta = norm * np.outer(u / np.linalg.norm(u), v / np.linalg.norm(v))
        name = f"random_{seed}"
        doc = patch_document(base, layer, indices, random_delta, "Seeded norm and support matched random control.")
        write(output / f"{name}_patch.json", doc)
        random_path = output / name
        write(output / f"{name}_receipt.json", apply_safetensors_patch(base, doc, policy(), random_path))
        # Calculate the realized norm from saved BF16 bytes.
        from weightbridge.patches.huggingface import _to_float64

        target = tensor_name(layer)
        a = _to_float64(load_checkpoint(base).read_raw(target), "BF16")
        b = _to_float64(load_checkpoint(random_path).read_raw(target), "BF16")
        rounded = float(np.linalg.norm(b - a))
        if abs(rounded - actual_norm) / actual_norm > 0.05:
            raise ValueError("random control realized norm differs by more than 5 percent")
        controls[name] = {"rounded_frobenius_norm": rounded, "requested_frobenius_norm": norm}
        score(name, random_path)
    targets = {name: correct(rows["target"]) for name, rows in scores.items()}
    margins = {name: mean_margin(rows["target"]) for name, rows in scores.items()}
    sentinels = {name: correct(rows["sentinel"]) for name, rows in scores.items()}
    lost = targets["pristine"] - targets["damaged"]
    recovered = (targets["repaired"] - targets["damaged"]) / lost if lost > 0 else None
    unrelated_flips = flips(scores["damaged"]["sentinel"], scores["repaired"]["sentinel"])
    passed = (recovered is not None and recovered >= config["minimum_recovery"]
              and targets["repaired"] > max(targets[name] for name in controls)
              and unrelated_flips <= config["max_unrelated_prediction_flips_vs_damaged"])
    write(output / "summary.json", {
        "disposition": "PASS" if passed else "FAIL" if lost > 0 else "INCONCLUSIVE",
        "target_correct": targets, "target_n": len(groups["target"]), "target_mean_margin": margins,
        "sentinel_correct": sentinels, "sentinel_n": len(groups["sentinel"]),
        "sentinel_prediction_flips_vs_damaged": unrelated_flips, "lost_accuracy_recovered": recovered,
        "matched_random_controls": controls, "resources": resources,
        "repair_checkpoint_sha256": receipt["candidate_sha256"],
        "holdout_sha256": digest(sealed / "holdout.json"), "practice_replay_exact": True,
    })


def export(run, target):
    summary = read(run / "evaluation" / "summary.json")
    target.mkdir(parents=True, exist_ok=False)
    for section in ("public", "evaluator", "search", "evaluation"):
        directory = target / section
        directory.mkdir()
        for source in (run / section).glob("*.json"):
            shutil.copyfile(source, directory / source.name)
    protocol = read(target / "public" / "protocol.json")
    receipt = read(target / "search" / "repair_receipt.json")
    record = {
        "schema": "weightbridge.experiment.v1", "experiment_id": target.name,
        "measurement_key": digest(target / "evaluation" / "summary.json"),
        "timestamp_utc": protocol["timestamp_utc"],
        "hypothesis": "A bounded practice-only search can recover at least half the held-out accuracy lost to an injected Qwen weight defect.",
        "source_model": {"id": MODEL, "revision": REVISION, "damaged_sha256": receipt["base_sha256"],
                         "pristine_sha256": read(target / "evaluator" / "calibration.json")["pristine_checkpoint_sha256"]},
        "proposed_modification": "search/repair_patch.json",
        "affected_parameters": {"changed_entries": receipt["changed_entries"],
                                "parameter_fraction_changed": receipt["parameter_fraction_changed"]},
        "evaluation": {"dataset": "frozen synthetic two-choice prompts", "practice_test_separated": True,
                       "seeds": read(target / "evaluator" / "calibration.json")["split_seeds"],
                       "summary": "evaluation/summary.json"},
        "resources": {"search": read(target / "search" / "search.json")["resources"],
                      "final_evaluation": summary["resources"]},
        "artifact_root": target.resolve().relative_to(ROOT).as_posix(),
        "artifacts": sorted(path.relative_to(target).as_posix() for path in target.rglob("*.json")),
        "disposition": summary["disposition"],
    }
    write(target / "experiment_record.json", record)
    append_experiment(ROOT / "results" / "experiment_ledger.jsonl", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--pristine", type=Path, required=True)
    p.add_argument("--fault-seed", type=int, required=True)
    p = sub.add_parser("search")
    p.add_argument("--public", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--pristine", type=Path, required=True)
    p = sub.add_parser("export")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    wall, cpu = time.perf_counter(), time.process_time()
    if args.stage == "prepare":
        prepare(args.run, args.pristine, args.fault_seed)
        destination = args.run / "evaluator"
    elif args.stage == "search":
        search(args.public, args.out)
        destination = args.out
    elif args.stage == "evaluate":
        evaluate(args.run, args.pristine)
        destination = args.run / "evaluation"
    else:
        export(args.run, args.out)
        return
    write(destination / "stage_resources.json", {"wall_seconds": time.perf_counter() - wall,
                                                 "cpu_seconds": time.process_time() - cpu,
                                                 "peak_rss_bytes": peak_rss_bytes()})


if __name__ == "__main__":
    main()
