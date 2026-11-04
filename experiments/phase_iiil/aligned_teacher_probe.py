"""Score token-aligned displayed options on the verified, unchanged fit parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiih.run_decimal_endpoint import endpoint
from experiments.phase_iiil.semantic_diagnostics import summarize
from experiments.phase_iiil.semantic_roster import PROTOCOL, load_split, read_manifest
from experiments.phase_iiil.source_manifest import ROOT, source_hashes
from weightbridge.adaptation.behavior_endpoint import _observation, endpoint_runner
from weightbridge.adaptation.content_scoring import prepare_continuations, score_continuations
from weightbridge.adaptation.fork_update import _batches, _check_memory, _resources
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint


def _read(path):
    return json.loads(path.read_bytes())


def _verified_parent(material, fit_dir, config, contract):
    manifest = read_manifest(material)
    fit_path, parent_path = fit_dir / "fit-report.json", fit_dir / "parent-adaptation.json"
    fit, parent = _read(fit_path), _read(parent_path)["measurement"]
    fit_sha, parent_sha = file_hash(fit_path), file_hash(parent_path)
    if (fit.get("schema") != "weightbridge.phase_iiil.fit.v1"
            or fit.get("checkpoint_sha256") != config["checkpoint_sha256"]
            or fit.get("sources") != source_hashes()
            or fit.get("manifest_sha256") != file_hash(material / "manifest.json")
            or fit.get("artifacts", {}).get("parent-adaptation.json") != parent_sha
            or parent.get("endpoint") != contract.manifest()):
        raise ValueError("Supplemental probe requires the source-bound original adaptation receipt")
    frozen = load_split(material, "adaptation", manifest)
    for expected, observed in zip(frozen, parent["rows"], strict=True):
        classification = _observation(observed, contract.question(expected), contract)
        if any(observed.get(key) != value for key, value in {**expected, **classification}.items()):
            raise ValueError("Parent adaptation receipt differs from the frozen roster or parser")
    return parent, {"fit_report_sha256": fit_sha, "parent_adaptation_sha256": parent_sha,
                    "manifest_sha256": fit["manifest_sha256"],
                    "adaptation_roster_sha256": manifest["splits"]["adaptation"]["sha256"]}


def _prepare(runner, contract, row):
    surfaces = [f" {operand}\n" for operand in row["operands"]]
    semantic = tuple(contract.parse(surface, "token_budget") for surface in surfaces)
    answers = tuple(result.answer for result in semantic)
    if (len(set(answers)) != 2 or set(answers) != set(contract.question(row).allowed_answers or ())
            or not all(result.completed for result in semantic)):
        raise ValueError("Probe surfaces must complete as the two distinct frozen displayed options")
    paths = prepare_continuations(runner.tokenizer, [row["prompt"]], [surfaces],
                                  max_input_tokens=contract.max_input_tokens)
    if any(len(path.prefix_ids) != row["prefix_tokens"]
           or runner.tokenizer.decode(path.answer_ids, skip_special_tokens=False) != surface
           for path, surface in zip(paths, surfaces, strict=True)):
        raise ValueError("Continuation tokenization does not preserve the measured prompt and answer surface")
    return surfaces, answers, paths


def run(base, material, fit_dir, out):
    if out.exists():
        raise FileExistsError(out)
    wall, cpu = time.perf_counter(), time.process_time()
    config = _read(PROTOCOL)
    sha, contract = config["checkpoint_sha256"], endpoint(config)
    parent, evidence = _verified_parent(material, fit_dir, config, contract)
    rows = [row for row in parent["rows"] if row["group"] == "target"]
    limits = config["intent"]["limits"]
    records, continuations, padded = [], [], 0
    script = Path(__file__).resolve()
    script_sha = file_hash(script)
    with _open(base, config, sha) as (_, raw_runner, _):
        runner = endpoint_runner(raw_runner, contract)
        for row in rows:
            record = {"id": row["id"], "comparison_id": row["comparison_id"], "pair_id": row["pair_id"],
                      "order": row["order"], "wording": row["wording"], "tier": row["tier"],
                      "truth_precision": row["truth_precision"], "correct": contract.question(row).answer,
                      "greedy": row["exact"], "greedy_valid": row["valid"], "greedy_token_ids": row["generated_token_ids"],
                      "sum_preference": None, "mean_preference": None}
            records.append(record)
            try:
                surfaces, answers, paths = _prepare(runner, contract, row)
            except ValueError as exc:
                record.update(status="unavailable", reason=str(exc))
                continue
            record.update(status="scored", options=[
                {"surface": surface, "answer": answer, "answer_token_ids": list(path.answer_ids)}
                for surface, answer, path in zip(surfaces, answers, paths, strict=True)
            ])
            continuations.extend(paths)
        batches = tuple(_batches(continuations, contract.batch_size))
        if len(batches) > limits["max_forward_calls"]:
            raise ValueError("Supplemental probe exceeds the frozen diagnostic forward allowance")
        scores = []
        _check_memory(limits["max_memory_bytes"])
        for batch in batches:
            values, resources = score_continuations(runner, batch, batch_size=contract.batch_size)
            scores.extend(values)
            padded += resources["padded_input_token_positions"]
            _check_memory(limits["max_memory_bytes"])
        scored_options = [option for row in records for option in row.get("options", ())]
        for option, score in zip(scored_options, scores, strict=True):
            option.update(score)
    summary = {}
    for metric in ("sum", "mean"):
        key = f"{metric}_preference"
        for record in records:
            if record["status"] != "scored":
                continue
            left, right = record["options"]
            margin = left[f"{metric}_log_probability"] - right[f"{metric}_log_probability"]
            record[f"{metric}_first_minus_second"] = margin
            record[key] = (left if margin > 0 else right)["answer"] if margin else None
            record[f"{metric}_agrees_with_greedy"] = (record[key] == record["greedy"]
                                                      if record["greedy_valid"] and record[key] is not None else None)
        measured = [row for row in records if row[key] is not None]
        diagnostic = summarize(parent, teacher_preferences={row["id"]: row[key] for row in measured})
        summary[metric] = {"preferences": len(measured),
                           "correct_preferences": sum(row[key] == row["correct"] for row in measured),
                           "greedy_agreements": sum(row[f"{metric}_agrees_with_greedy"] is True for row in measured),
                           "greedy_disagreements": sum(row[f"{metric}_agrees_with_greedy"] is False for row in measured),
                           "by_subgroup": diagnostic["subgroups"]}
    if (_verified_parent(material, fit_dir, config, contract)[1] != evidence
            or file_hash(script) != script_sha or load_checkpoint(base).checkpoint_sha256() != sha):
        raise ValueError("Probe inputs changed during measurement")
    report = {"schema": "weightbridge.phase_iiil.aligned_teacher_probe.v1", "scope": "post-hoc parent adaptation only",
              "scoring": "exact native continuation with one leading ASCII space and one scored trailing newline; no EOS",
              "source": {"script": script.relative_to(ROOT).as_posix(), "sha256": script_sha, **evidence},
              "checkpoint_sha256": sha, "summary": summary, "rows": records,
              "resources": _resources(wall, cpu, len(batches), 0, padded_input_token_positions=padded)}
    json_dump_atomic(out, report)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "material", "fit-dir", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.base, args.material, args.fit_dir, args.out), indent=2))
