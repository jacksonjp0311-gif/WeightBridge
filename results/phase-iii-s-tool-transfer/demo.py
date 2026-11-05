"""Public illustrative fixtures and a command for the existing frozen S evaluator.

This helper never executes inference, trains, merges, repairs an answer or opens
a protected roster. Preparing a demo requires an already materialized, exactly
reconstructed stored candidate and a host declaration that the study is final.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.phase_iiis_tool_transfer import data
from experiments.phase_iiis_tool_transfer.populations import normalize_tool_rows
from weightbridge.adaptation import tool_task

PUBLIC_SEED = "weightbridge-phase-iii-s-public-demo-v1-2026-11-05"
NAMESPACE = "public-demo-"
LIMITS = {"wall_seconds": 300, "cpu_seconds": 2000, "memory_bytes": 8 * 1024**3}


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def dump(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+"\n").encode()


def fixtures():
    rows = data.generate_tool_families("development", 5, PUBLIC_SEED, wording_group="confirmation")
    family_order = list(dict.fromkeys(row["family_id"] for row in rows))
    for index, identity in enumerate(family_order):
        group = [row for row in rows if row["family_id"] == identity]
        slug = group[0]["fixture"]["documents"][0]["path"].split("/")[-1].removesuffix("-legacy.md")
        def rename(value):
            if isinstance(value, str):
                return value.replace(slug, NAMESPACE+slug)
            if isinstance(value, list):
                return [rename(v) for v in value]
            if isinstance(value, dict):
                return {k: rename(v) for k, v in value.items()}
            return value
        pair = [rename(next(row["fixture"] for row in group if row["counterfactual"] == side)) for side in (0, 1)]
        for row in group:
            row["fixture"] = deepcopy(pair[row["counterfactual"]])
            row["family_id"] = data.family_key(pair)
            row["id"] = f"tool/public-demo-v1/{index:04d}/{row['counterfactual']}/{row['wording']}/{row['ordering']}"
            row["gold_action"] = tool_task.gold_action(row["fixture"])
            row["completion"] = tool_task.canonical_action(row["gold_action"])
            row["prompt"] = data.render_prompt(row["fixture"], row["wording_group"], row["wording"], row["ordering"])
    data.validate_tool_rows(rows, expected_families=5)
    return rows


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_bytes())
    payload = (directory/"rows.json").read_bytes()
    if sha(payload) != manifest["rows_sha256"] or payload != dump(fixtures()):
        raise ValueError("Public fixture bytes differ from their prospective recipe")
    rows = normalize_tool_rows(json.loads(payload))
    summary = data.validate_tool_rows(rows, expected_families=5)
    if summary != manifest["population"] or summary["strata"] != {tool: 8 for tool in tool_task.TOOLS}:
        raise ValueError("Public demo does not cover five balanced actions")
    for name, digest in manifest["reused_source_sha256"].items():
        if sha((ROOT/name).read_bytes()) != digest:
            raise ValueError("Frozen fixture/scoring source changed")
    if any(not row["id"].startswith("tool/public-demo-v1/") or NAMESPACE not in row["prompt"] for row in rows):
        raise ValueError("A demo identifier lacks its public namespace")
    if any("Format example (unrelated disposable fixture): "+data.FORMAT_EXAMPLE not in row["prompt"] for row in rows):
        raise ValueError("Demo changed the frozen format example")
    # Oracle/scorer contract test only, not a model-generated success.
    if not all(tool_task.score_response(row, row["completion"], completed=True, stop_reason="eos")["strict_correct"] for row in rows):
        raise ValueError("Public fixture oracle and strict scorer disagree")
    return {"passed": True, "rows_sha256": manifest["rows_sha256"], "presentations": 40,
        "families": 5, "fixture_oracle_validation_only": True, "model_execution": False}


def build(directory, tokenizer_path=None):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Public demo fixtures are write-once")
    rows = fixtures()
    token_check = None
    if tokenizer_path is not None:
        from transformers import AutoTokenizer
        from weightbridge.adaptation.task_completion import prepare_tool_completion
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
        spec = {"schema": "weightbridge.native_tool_completion.v1", "end_token_id": 248046, "end_token_surface": "<|im_end|>"}
        encoded = [prepare_tool_completion(tokenizer, row["prompt"], row["gold_action"], spec) for row in rows]
        token_check = {"native_template_sha256": sha(tokenizer.chat_template.encode()),
            "minimum_prompt_tokens": min(len(row.prefix_ids) for row in encoded),
            "maximum_prompt_tokens": max(len(row.prefix_ids) for row in encoded),
            "maximum_answer_with_terminator_tokens": max(len(row.target_ids) for row in encoded),
            "prompt_limit": 512, "generation_limit": 96, "native_end_token": 248046,
            "tokenization_only": True, "model_execution": False}
    payload = dump(rows)
    names = ("experiments/phase_iiis_tool_transfer/data.py", "experiments/phase_iiis_tool_transfer/measurement.py",
        "experiments/phase_iiis_tool_transfer/evaluate.py", "experiments/phase_iiis_tool_transfer/verification.py",
        "experiments/phase_iiis_tool_transfer/populations.py", "weightbridge/adaptation/tool_task.py",
        "weightbridge/adaptation/task_completion.py")
    manifest = {"schema": "weightbridge.phase_iiis.public_demo_fixtures.v1", "public_generation_seed": PUBLIC_SEED,
        "public_identifier_namespace": NAMESPACE, "population": data.validate_tool_rows(rows, expected_families=5),
        "rows_sha256": sha(payload), "generator_sha256": sha(Path(__file__).read_bytes()),
        "reused_source_sha256": {name: sha((ROOT/name).read_bytes()) for name in names},
        "format_example_sha256": sha(data.FORMAT_EXAMPLE.encode()), "tokenization": token_check,
        "execution_limits": LIMITS, "inference_status": "NOT_EXECUTED", "model_result": None,
        "scope": "Public illustrative cases only. Existing confirmation phrasing with new explicitly public identifiers; not a new linguistic-transfer test, qualification bank, training population or independent evidence.",
        "private_rosters_accessed": False, "model_execution": False, "automatic_promotion": False}
    directory.mkdir(parents=True)
    (directory/"rows.json").write_bytes(payload)
    (directory/"manifest.json").write_bytes(dump(manifest))
    return verify(directory)


def prepare(directory, *, protocol, lock, checkpoint, ledger, out, study_finalized=False):
    """Write authorization/argv for host execution, without running a model."""
    if not study_finalized:
        raise ValueError("Demo may only be prepared after the host finalizes the study")
    from experiments.phase_iiis_tool_transfer.study import validate_arm_lock
    from experiments.phase_iiis_tool_transfer.verification import authorize_evaluation
    from weightbridge.core import file_hash
    checked = verify(directory)
    protocol, lock, checkpoint, ledger, out = map(lambda p: Path(p).resolve(), (protocol, lock, checkpoint, ledger, out))
    locked = validate_arm_lock(lock)
    identity = locked.get("representation_sha256")
    if not identity or locked.get("protocol_sha256") != file_hash(protocol):
        raise ValueError("No locked stored candidate exists for this protocol; demo stays NOT_EXECUTED")
    for name in ("merge_verification", "reconstruction"):
        receipt = json.loads(Path(locked["input_sha256"][name]["path"]).read_bytes())
        if receipt.get("passed") is not True or receipt.get("candidate_sha256") != identity or (name == "reconstruction" and receipt.get("exact_reconstruction") is not True):
            raise ValueError("Demo requires the exact existing merged and reconstructed artifact")
    if out.exists() or out.is_relative_to(checkpoint) or not checkpoint.is_dir():
        raise ValueError("Demo evidence must be new and separate from the existing checkpoint")
    out.mkdir(parents=True)
    authorization, result = out/"authorization.json", out/"result.json"
    authorize_evaluation(authorization, protocol, stage="demo", task="tool", roster_sha256=checked["rows_sha256"],
        representation_sha256=identity, adapter_sha256=None)
    name = "public-demo-"+locked["arm"]
    worker = ["--base", str(checkpoint), "--protocol", str(protocol), "--rows", str((Path(directory)/"rows.json").resolve()),
        "--out", str(result), "--authorization", str(authorization), "--stage", "demo", "--task", "tool",
        "--expected-checkpoint", identity, "--expected-rows", checked["rows_sha256"]]
    argv = [sys.executable, "-m", "experiments.phase_iiis_tool_transfer.execute", "--ledger", str(ledger),
        "--name", name, "--module", "experiments.phase_iiis_tool_transfer.evaluate", "--wall", str(LIMITS["wall_seconds"]),
        "--cpu", str(LIMITS["cpu_seconds"]), "--memory", str(LIMITS["memory_bytes"]), "--", *worker]
    plan = {"schema": "weightbridge.phase_iiis.public_demo_plan.v1", "argv": argv, "checkpoint_sha256": identity,
        "lock_sha256": file_hash(lock), "locked_behavior_disposition": locked["disposition"],
        "scientific_qualification_claim": False, "illustrative_only": True, "model_execution": False}
    (out/"command.json").write_bytes(dump(plan))
    return {"prepared": True, "checkpoint_sha256": identity, "model_execution": False, "illustrative_only": True}


def self_test():
    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp)/"public"
        assert build(out)["passed"]
        try:
            build(out)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Public fixture overwrite permitted")
        (out/"rows.json").write_bytes(b"[]")
        try:
            verify(out)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed fixture accepted")
    return {"five_actions_balanced": True, "eight_forms_per_family": True, "source_and_fixture_hashes": True,
        "frozen_format_example": True, "write_once_and_tamper_refusal": True, "model_execution": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--self-test", action="store_true")
    for field in ("protocol", "lock", "checkpoint", "ledger", "out"):
        parser.add_argument("--"+field, type=Path)
    parser.add_argument("--study-finalized", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.build:
        result = build(args.build, args.tokenizer)
    elif args.verify:
        result = verify(args.verify)
    elif args.prepare and all(getattr(args, field) for field in ("protocol", "lock", "checkpoint", "ledger", "out")):
        result = prepare(args.prepare, **{key: getattr(args, key) for key in ("protocol", "lock", "checkpoint", "ledger", "out", "study_finalized")})
    else:
        parser.error("Choose self-test, build, verify, or prepare with all host artifact paths")
    print(json.dumps(result))
