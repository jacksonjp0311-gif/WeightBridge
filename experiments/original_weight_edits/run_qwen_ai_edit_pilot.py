"""Run a practice-only AI proposal pilot against injected Qwen faults."""
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

import run_blind_qwen_repair as base
import run_qwen_repair_cohort as prior
import run_qwen_wording_ablation as wording

SOURCE = Path(__file__).resolve()
CONFIG = base.ROOT / "benchmarks/configs/qwen_ai_edit_pilot_v1.json"
SEED = 924132
TRIALS = 6
OPTIONS = [(layer, strength) for layer in base.LAYERS for strength in prior.STRENGTHS]


def protocol():
    return {
        "schema": "weightbridge.qwen_ai_edit_pilot.v1",
        "source_sha256": {**wording.protocol()["source_sha256"], SOURCE.name: base.digest(SOURCE)},
        "model_id": base.MODEL, "model_revision": base.REVISION, "shard_sha256": prior.SHARD_SHA,
        "policy": base.policy(), "gate": prior.protocol()["gate"], "seed": SEED,
        "candidate_evaluations_per_arm": TRIALS, "options": [list(option) for option in OPTIONS],
        "cases": [{"slot": i, "layer": layer} for i, layer in enumerate(base.LAYERS, 1)],
        "controller": "Pinned pristine Qwen, greedy selection over allowed letter tokens, practice feedback only",
        "compiler": "Signed activation ridge fit on practice with sentinel weight 256",
    }


def splits(slot):
    rng = np.random.default_rng(SEED + slot)
    numbers = iter(np.random.default_rng(SEED).choice(3000, 120, replace=False).reshape(3, 40)[slot - 1])
    words = [(kind, good, bad) for kind, goods, bads in prior.SENTINELS
             for good in goods.split() for bad in bads.split()]
    sentinels = iter(np.random.default_rng(SEED + 10).permutation(len(words))[:120].reshape(3, 40)[slot - 1])
    sets = {}
    for name, (_, n_sentinel) in prior.SIZES.items():
        targets, unrelated = [], []
        forms = wording.FORMS["multi" if name == "practice" else name]
        for form in dict.fromkeys(forms):
            for label in rng.permutation(["A", "B"] * (forms.count(form) // 2)):
                index = int(next(numbers))
                low, high = 500 + index // 100, 900 + index % 100
                a, b = (high, low) if label == "A" else (low, high)
                targets.append({"id": f"{slot}-{name}-t{len(targets)}", "label": str(label),
                                "form": "same" if form == "standard" else "paraphrase",
                                "text": wording.TEMPLATES[form].format(a=a, b=b)})
        for i, label in enumerate(rng.permutation(["A", "B"] * (n_sentinel // 2))):
            kind, good, bad = words[int(next(sentinels))]
            a, b = (good, bad) if label == "A" else (bad, good)
            unrelated.append({"id": f"{slot}-{name}-s{i}", "label": str(label),
                              "text": f"Select the option that names {kind}.\nA. {a}\nB. {b}\nAnswer with A or B."})
        sets[name] = {"target": targets, "sentinel": unrelated}
    return sets


def prepare(case, pristine, spec):
    case.mkdir()
    public, evaluator = case / "public", case / "evaluator"
    public.mkdir()
    evaluator.mkdir()
    if base.digest(pristine / base.BASE_SHARD) != prior.SHARD_SHA:
        raise ValueError("pristine shard differs from the pinned Qwen checkpoint")
    sets = splits(spec["slot"])
    base.write(public / "practice.json", sets["practice"])
    for name in ("construction", "holdout"):
        base.write(evaluator / f"{name}.json", sets[name])
    base.write(evaluator / "input.json", {"pristine_sha256": base.load_checkpoint(pristine).checkpoint_sha256(),
                                         "holdout_sha256": base.digest(evaluator / "holdout.json"),
                                         "construction_sha256": base.digest(evaluator / "construction.json")})
    runner = base.Runner(pristine)
    layer = spec["layer"]
    baseline, activations = base.measure(runner, sets["construction"], capture=(layer,))
    indices, vector = prior.activation_gate(activations[layer], -prior.signed_labels(sets["construction"]), 64)
    direction = runner.direction()
    attempts = []
    accepted = None
    if base.correct(baseline["target"]) >= 6:
        for strength in prior.FAULT_STRENGTHS:
            delta = strength * np.outer(direction, vector)
            with runner.edit(layer, indices, delta) as norm:
                rows, _ = base.measure(runner, sets["construction"])
            loss = base.correct(baseline["target"]) - base.correct(rows["target"])
            flips = base.flips(baseline["sentinel"], rows["sentinel"])
            attempts.append({"strength": strength, "norm": norm, "loss": loss, "unrelated_flips": flips, "rows": rows})
            if loss >= 2 and flips <= 1:
                accepted = delta
                break
    base.write(evaluator / "calibration.json", {"baseline": baseline, "attempts": attempts, "resources": runner.resources()})
    if accepted is None:
        base.write(evaluator / "hold.json", {"reason": "construction calibration did not meet the frozen floor"})
        return
    patch = base.patch_document(pristine, layer, indices, accepted, "Evaluator-only directional pilot fault.")
    patch["evidence_ids"] = ["qwen-ai-edit-pilot-v1"]
    base.write(evaluator / "fault_patch.json", patch)
    receipt = base.apply_safetensors_patch(pristine, patch, base.policy(), public / "damaged")
    base.write(evaluator / "fault_receipt.json", receipt)
    base.write(public / "input.json", {"practice_sha256": base.digest(public / "practice.json"),
                                      "damaged_sha256": receipt["candidate_sha256"]})


def choose(controller, prompt, letters):
    inputs = controller.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=True, return_tensors="pt",
    )
    tokens = [controller.tokenizer.encode(letter, add_special_tokens=False) for letter in letters]
    if any(len(ids) != 1 for ids in tokens) or len({ids[0] for ids in tokens}) != len(tokens):
        raise ValueError("controller choices must be distinct single tokens")
    with controller.torch.inference_mode():
        logits = controller.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1]
    scores = {letter: float(logits[ids[0]]) for letter, ids in zip(letters, tokens)}
    if not all(np.isfinite(value) for value in scores.values()):
        raise ValueError("nonfinite controller scores")
    return max(scores, key=scores.__getitem__), scores, int(inputs["attention_mask"].sum())


def search(case, controller_path, slot):
    public, output = case / "public", case / "search"
    output.mkdir()
    damaged = public / "damaged"
    inputs = base.read(public / "input.json")
    if (base.digest(public / "practice.json") != inputs["practice_sha256"] or
            base.load_checkpoint(damaged).checkpoint_sha256() != inputs["damaged_sha256"]):
        raise ValueError("damaged checkpoint or practice data changed")
    practice = base.read(public / "practice.json")
    runner, controller = base.Runner(damaged), base.Runner(controller_path)
    baseline, activations = base.measure(runner, practice, capture=base.LAYERS)
    direction = runner.direction()
    proposals = {layer: prior.activation_gate(activations[layer], prior.signed_labels(practice), 256)
                 for layer in base.LAYERS}
    rng = np.random.default_rng(SEED + 100 + slot)
    remaining = [OPTIONS[i] for i in rng.permutation(len(OPTIONS))]
    history, attempts, edits = [], {"guided": [], "random": []}, {"guided": [], "random": []}
    for _ in range(TRIALS):
        letters = "ABCDEFGHI"[:len(remaining)]
        menu = "\n".join(f"{letter}. layer {layer}, strength {strength}" for letter, (layer, strength) in zip(letters, remaining))
        feedback = "\n".join(f"Layer {r['layer']}, strength {r['strength']}: "
                             f"target gain {r['target_gain']}, sentinel flips {r['unrelated_flips']}, "
                             f"sentinel margin drift {r['drift']:.3f}."
                             for r in attempts["guided"]) or "No edits have been tried."
        prompt = (
            "Choose the next bounded rank-one repair for a damaged language model. Improve target accuracy "
            "while keeping unrelated predictions stable. Each layer and strength pair is untested. The host fits "
            "the direction using practice activations, and returns practice results. "
            f"Damaged practice target accuracy is {base.correct(baseline['target'])}/{len(baseline['target'])}.\n"
            f"Prior guided results:\n{feedback}\n\nAvailable edits:\n{menu}\n\nReply with the single option letter."
        )
        picked, logits, token_count = choose(controller, prompt, letters)
        layer, strength = remaining.pop(letters.index(picked))
        history.append({"prompt": prompt, "choice": picked, "logits": logits, "input_tokens": token_count,
                        "layer": layer, "strength": strength})
        indices, vector = proposals[layer]
        delta = strength * np.outer(direction, vector)
        for arm, edit in (("guided", delta), ("random", prior.random_delta(rng, delta))):
            with runner.edit(layer, indices, edit) as norm:
                rows, _ = base.measure(runner, practice)
            attempts[arm].append({"layer": layer, "strength": strength, "indices": indices,
                                  "requested_norm": float(np.linalg.norm(edit)),
                                  **prior.practice_metrics(baseline, rows, norm)})
            edits[arm].append((layer, indices, edit))
        a, b = (attempts[arm][-1]["norm"] for arm in ("guided", "random"))
        if not a or abs(a - b) / a > 0.05:
            raise ValueError("matched control norm differs after BF16 rounding")
    selected = {arm: prior.select_attempt(rows) for arm, rows in attempts.items()}
    for arm, index in selected.items():
        if index is None:
            continue
        layer, indices, edit = edits[arm][index]
        patch = base.patch_document(damaged, layer, indices, edit, f"Practice-selected pilot {arm} edit.")
        patch["evidence_ids"] = ["qwen-ai-edit-pilot-v1"]
        base.write(output / f"{arm}_patch.json", patch)
        base.write(output / f"{arm}_receipt.json",
                   base.apply_safetensors_patch(damaged, patch, base.policy(), output / arm))
    base.write(output / "search.json", {"baseline": baseline, "controller": history, "attempts": attempts,
                                       "selected": selected, "resources": runner.resources(),
                                       "controller_rss_bytes": base.peak_rss_bytes(),
                                       "practice_sha256": inputs["practice_sha256"]})


def evaluate(case, pristine, config):
    output = case / "evaluation"
    output.mkdir()
    public, search_dir = case / "public", case / "search"
    holdout_path = case / "evaluator/holdout.json"
    frozen = base.read(case / "evaluator/input.json")
    inputs = base.read(public / "input.json")
    if (base.digest(holdout_path) != frozen["holdout_sha256"] or
            base.load_checkpoint(pristine).checkpoint_sha256() != frozen["pristine_sha256"] or
            base.load_checkpoint(public / "damaged").checkpoint_sha256() != inputs["damaged_sha256"] or
            base.digest(public / "practice.json") != inputs["practice_sha256"]):
        raise ValueError("evaluator inputs changed after preparation")
    holdout = base.read(holdout_path)
    practice = base.read(public / "practice.json")
    search_log = base.read(search_dir / "search.json")
    scores, resources = {}, {}
    for arm, path in (("pristine", pristine), ("damaged", public / "damaged"),
                      ("guided", search_dir / "guided"), ("random", search_dir / "random")):
        if arm in search_log["selected"]:
            index = search_log["selected"][arm]
            if index is None:
                scores[arm] = scores["damaged"]
                continue
            receipt = base.read(search_dir / f"{arm}_receipt.json")
            if (receipt["base_sha256"] != inputs["damaged_sha256"] or
                    receipt["candidate_sha256"] != base.load_checkpoint(path).checkpoint_sha256()):
                raise ValueError("saved candidate differs from its receipt")
        runner = base.Runner(path)
        if arm in search_log["selected"] and base.measure(runner, practice)[0] != search_log["attempts"][arm][index]["rows"]:
            raise ValueError("reloaded finalist did not reproduce practice scores")
        scores[arm], _ = base.measure(runner, holdout)
        resources[arm] = runner.resources()
        del runner
    base.write(output / "scores.json", scores)
    result = prior.outcome(scores, holdout, search_log["selected"], config["gate"])
    base.write(output / "summary.json", {**result, "resources": resources,
                                         "scores_sha256": base.digest(output / "scores.json")})


def run_case(run, pristine, config_path, slot):
    case = run / f"case-{slot:02d}"
    for stage in ("prepare", "search", "evaluate"):
        with (run / f"case-{slot:02d}-{stage}.log").open("xb") as log:
            result = subprocess.run([sys.executable, str(SOURCE), stage, "--config", str(config_path),
                                     "--pristine", str(pristine), "--run", str(run), "--slot", str(slot)],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=420, check=False)
        if result.returncode:
            return f"{slot}: {stage} failed ({result.returncode})"
        if (case / "evaluator/hold.json").exists():
            return f"{slot}: HOLD_CALIBRATION"
    return f"{slot}: {base.read(case / 'evaluation/summary.json')['disposition']}"


def export(run, out, config):
    out.mkdir()
    base.write(out / "manifest.json", config)
    for case in sorted(run.glob("case-*")):
        if not case.is_dir():
            continue
        for path in case.glob("*/*.json"):
            target = out / path.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.name.endswith("_patch.json"):
                target.with_suffix(".json.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            else:
                shutil.copyfile(path, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "prepare", "search", "evaluate", "export"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--pristine", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--slot", type=int, choices=range(1, 4))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = protocol()
    if args.stage == "freeze":
        base.write(args.config, config)
        return
    if base.read(args.config) != config:
        raise ValueError("pilot manifest differs from frozen source and protocol")
    if args.stage == "export":
        export(args.run, args.out, config)
        return
    if args.stage == "run":
        args.run.mkdir()
        with ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(lambda slot: run_case(args.run, args.pristine, args.config, slot), range(1, 4)):
                print(result, flush=True)
        return
    case = args.run / f"case-{args.slot:02d}"
    wall, cpu = time.perf_counter(), time.process_time()
    if args.stage == "prepare":
        prepare(case, args.pristine, config["cases"][args.slot - 1])
        destination = case / "evaluator"
    elif args.stage == "search":
        search(case, args.pristine, args.slot)
        destination = case / "search"
    else:
        evaluate(case, args.pristine, config)
        destination = case / "evaluation"
    base.write(destination / "resources.json", {"wall_seconds": time.perf_counter() - wall,
                                                "cpu_seconds": time.process_time() - cpu,
                                                "peak_rss_bytes": base.peak_rss_bytes(),
                                                "completed_utc": base.timestamp()})


if __name__ == "__main__":
    main()
