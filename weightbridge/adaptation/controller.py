"""Host-owned continual adaptation of local research checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time

from ..agents.hf_session import (
    describe_hf_session, finalize_hf_candidate, init_hf_session, stop_hf_session, try_hf_direction,
)
from ..core import file_hash, json_dump_atomic
from ..evaluation.hf_edit import _load_suite, _score_checkpoint, evaluate_hf_edit
from ..evaluation.metrics import paired_bootstrap_interval
from ..inference import InferenceConfig
from ..inspection.safetensors import load_checkpoint
from ..patches.huggingface import _policy
from .awl import parse_awl
from .lineage import (
    init_lineage, lineage_status, record_lineage_decision, record_lineage_rejection, rollback_lineage, workspace_lock,
)
from .synthesis import plan_candidates


_PROTOCOL = "weightbridge.phase_iii.protocol.v1"
_CYCLE = re.compile(r"cycle-[0-9]{2}")
_SHA256 = re.compile(r"[a-f0-9]{64}")
_INFERENCE = {"device", "dtype", "cpu_threads", "batch_size", "max_input_tokens"}
_GATES = {"min_target_gain", "max_sentinel_flips", "max_target_regressions"}
_PENALTIES = {"preservation_penalty", "edit_cost_penalty"}
_CYCLE_FIELDS = {"id", "family", "intent_sha256", "practice_sha256", "holdout_sha256"}


def _read(path):
    return json.loads(Path(path).read_bytes())


def _input(path, digest):
    with Path(path).open("rb") as stream:
        raw = stream.read(8 * 1024**2 + 1)
    if len(raw) > 8 * 1024**2 or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Input exceeds the size limit or differs from the frozen protocol")
    return raw


def _validate_protocol(protocol):
    keys = {"schema", "base_sha256", "policy", "inference", "max_candidates", "cycles", "final_sha256"}
    if not isinstance(protocol, dict) or protocol.keys() != keys | _GATES | _PENALTIES:
        raise ValueError("Phase III protocol must contain the exact v1 fields")
    if protocol["schema"] != _PROTOCOL:
        raise ValueError("Unsupported Phase III protocol")
    _policy(protocol["policy"])
    inference = protocol["inference"]
    if not isinstance(inference, dict) or inference.keys() != _INFERENCE:
        raise ValueError("Protocol must freeze the inference configuration")
    if (not isinstance(inference["device"], str) or not inference["device"].strip()
            or inference["dtype"] not in ("auto", "float32", "float16", "bfloat16")
            or type(inference["batch_size"]) is not int or not 1 <= inference["batch_size"] <= 16):
        raise ValueError("Invalid protocol inference configuration")
    InferenceConfig(cpu_threads=inference["cpu_threads"], max_input_tokens=inference["max_input_tokens"]).validate(
        max_input_budget=8192,
    )
    for key in _GATES | {"max_candidates"}:
        lo = int(key in {"max_candidates", "min_target_gain"})
        hi = 32 if key == "max_candidates" else 1024
        if type(protocol[key]) is not int or not lo <= protocol[key] <= hi:
            raise ValueError(f"Invalid protocol {key}")
    for key in _PENALTIES:
        if type(protocol[key]) not in (float, int) or not 0 <= protocol[key] <= 1000:
            raise ValueError(f"Invalid protocol {key}")
    cycles = protocol["cycles"]
    if not isinstance(cycles, list) or not 1 <= len(cycles) <= 32:
        raise ValueError("Protocol requires 1 to 32 cycles")
    suites = [protocol["final_sha256"]]
    digests = [protocol["base_sha256"]]
    for index, spec in enumerate(cycles, 1):
        if (not isinstance(spec, dict) or spec.keys() != _CYCLE_FIELDS or spec["id"] != f"cycle-{index:02d}"
                or not isinstance(spec["family"], str) or not 1 <= len(spec["family"].strip()) <= 96):
            raise ValueError("Cycle IDs must be consecutive and each cycle must declare a family and input hashes")
        suites.extend((spec["practice_sha256"], spec["holdout_sha256"]))
        digests.append(spec["intent_sha256"])
    if any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in digests + suites):
        raise ValueError("Protocol identities must be lowercase SHA-256 digests")
    if len(set(suites)) != len(suites):
        raise ValueError("Practice, cycle holdout and final suite hashes must be distinct")
    return protocol


def _protocol(root):
    frozen = _read(root / "research.json")
    return _validate_protocol(json.loads(_input(root / "protocol.json", frozen["protocol_sha256"])))


def init_research(root, base, protocol_path):
    """Freeze protocol and original checkpoint in a new research workspace."""
    root = Path(root).resolve()
    protocol = _validate_protocol(_read(protocol_path))
    checkpoint = load_checkpoint(base)
    if not checkpoint.root.is_dir() or root.is_relative_to(checkpoint.root):
        raise ValueError("Research workspace must be outside a local model directory")
    if checkpoint.checkpoint_sha256() != protocol["base_sha256"]:
        raise ValueError("Original checkpoint differs from the frozen protocol")
    root.mkdir(parents=True)
    (root / "cycles").mkdir()
    (root / "evaluation").mkdir()
    status = init_lineage(root / "lineage", checkpoint.root)
    if status["root_sha256"] != protocol["base_sha256"]:
        raise ValueError("Original checkpoint changed during initialization")
    json_dump_atomic(root / "protocol.json", protocol)
    json_dump_atomic(root / "research.json", {"protocol_sha256": file_hash(root / "protocol.json")})
    return status


def research_status(root):
    root = Path(root).resolve()
    _protocol(root)
    return lineage_status(root / "lineage")


def _cycle(root, cycle_id):
    if not isinstance(cycle_id, str) or not _CYCLE.fullmatch(cycle_id):
        raise ValueError("Invalid cycle ID")
    protocol = _protocol(root)
    sequence = [entry["id"] for entry in protocol["cycles"]]
    if cycle_id not in sequence:
        raise ValueError("Cycle is outside the frozen protocol")
    lineage = lineage_status(root / "lineage")
    decided = {event["cycle_id"] for event in lineage["events"] if event.get("cycle_id") is not None}
    if any(earlier not in decided for earlier in sequence[:sequence.index(cycle_id)]):
        raise ValueError("Earlier cycles must be decided first")
    return protocol, protocol["cycles"][sequence.index(cycle_id)], lineage, cycle_id in decided


def _rank(attempt, protocol):
    return (attempt["target_gain"] - protocol["preservation_penalty"] * attempt["sentinel_prediction_flips"]
            - protocol["edit_cost_penalty"] * attempt["parameter_fraction_changed"])


def _diagnostic(attempt, protocol):
    if attempt["disposition"] not in ("PASS", "FAIL"):
        return attempt.get("failure_code", attempt["disposition"])
    if attempt["sentinel_prediction_flips"] > protocol["max_sentinel_flips"]:
        return "SENTINEL_REGRESSION"
    if attempt["target_regressions"] > protocol["max_target_regressions"]:
        return "TARGET_REGRESSION"
    if not attempt["target_corrections"] and not attempt["target_regressions"] and not attempt["sentinel_prediction_flips"]:
        return "PREDICTIONS_UNCHANGED"
    if attempt["disposition"] == "FAIL":
        return "TARGET_GAIN_BELOW_GATE"
    return "ELIGIBLE"


def search_cycle(root, cycle_id, intent_path, practice_path):
    """Search on authorized practice only. This API takes no sealed evaluation inputs."""
    root = Path(root).resolve()
    with workspace_lock(root):
        started = time.perf_counter()
        protocol, spec, lineage, decided = _cycle(root, cycle_id)
        if decided or (root / "evaluation" / "final.json").exists():
            raise ValueError("Research cycle is already closed")
        intent = parse_awl(json.loads(_input(intent_path, spec["intent_sha256"])))
        practice = _load_suite(_input(practice_path, spec["practice_sha256"]))
        work = root / "cycles" / cycle_id
        work.mkdir(exist_ok=True)
        session = work / "session"
        preferences = []
        specs = {entry["id"]: entry for entry in protocol["cycles"]}
        for ancestor in reversed(lineage["ancestry"][1:]):
            prior_id = ancestor["cycle_id"]
            if specs[prior_id]["family"] == spec["family"]:
                previous = _read(root / "cycles" / prior_id / "selection.json")
                preferences.append(previous["finalist"]["tensor"])
        planned = plan_candidates(intent, protocol["policy"]["allow_tensors"], protocol["max_candidates"],
                                  preferred_tensors=preferences)
        plan = {"cycle_id": cycle_id, "family": spec["family"], "parent_sha256": lineage["head_sha256"],
                "intent_sha256": spec["intent_sha256"], "practice_sha256": spec["practice_sha256"],
                "attempts": [{"tensor": tensor, "strength": strength} for tensor, strength in planned],
                "preferred_tensors": preferences}
        plan_path = work / "plan.json"
        if not plan_path.exists():
            json_dump_atomic(plan_path, plan)
        elif _read(plan_path) != plan:
            raise ValueError("Previously frozen cycle plan differs from the current parent or inputs")
        policy_path, practice_copy = work / "policy.json", work / "practice.json"
        for path, value in ((policy_path, protocol["policy"]), (practice_copy, practice)):
            if not path.exists():
                json_dump_atomic(path, value)
            elif _read(path) != value:
                raise ValueError("Cycle input copy changed")
        gate = {key: protocol[key] for key in ("min_target_gain", "max_sentinel_flips")}
        if session.exists():
            status = describe_hf_session(session)
        else:
            status = init_hf_session(session, lineage["head_path"], policy_path, practice_copy,
                                     max_candidates=len(planned), **gate, **protocol["inference"])
        if status["base_sha256"] != plan["parent_sha256"] or status["policy"] != protocol["policy"]:
            raise ValueError("Practice session differs from the frozen parent or policy")
        if status["decision"] is None:
            for tensor, strength in planned[status["used_candidates"]:]:
                try_hf_direction(session, tensor, strength)
            status = describe_hf_session(session)
            eligible = [attempt for attempt in status["attempts"] if _diagnostic(attempt, protocol) == "ELIGIBLE"]
            if eligible:
                winner = max(eligible, key=lambda attempt: _rank(attempt, protocol))
                decision = finalize_hf_candidate(session, winner["candidate_id"])
            else:
                decision = stop_hf_session(session, "No candidate cleared the frozen practice and preservation gates")
        else:
            decision = status["decision"]
        finalist = None
        if decision["disposition"] == "FINALIZED":
            finalist = {key: decision[key] for key in ("candidate_id", "candidate_sha256", "tensor", "strength")}
        summary = {"cycle_id": cycle_id, "parent_sha256": plan["parent_sha256"], "finalist": finalist,
                   "plan_sha256": file_hash(plan_path), "session_decision_sha256": file_hash(session / "decision.json"),
                   "attempts": [{"candidate_id": attempt["candidate_id"], "disposition": attempt["disposition"],
                                 "diagnostic": _diagnostic(attempt, protocol)} for attempt in status["attempts"]],
                   "elapsed_seconds": time.perf_counter() - started}
        selection_path = work / "selection.json"
        if selection_path.exists():
            existing = _read(selection_path)
            if {key: value for key, value in existing.items() if key != "elapsed_seconds"} != {
                key: value for key, value in summary.items() if key != "elapsed_seconds"
            }:
                raise ValueError("Existing selection differs from the frozen practice decision")
            return existing
        json_dump_atomic(selection_path, summary)
        return summary


def _rows(suite):
    return suite["target"] + suite["sentinel"]


def _text_keys(rows):
    return {" ".join(row["text"].split()).casefold() for row in rows}


def _verify_suite(raw):
    suite = _load_suite(raw)
    if len(_text_keys(_rows(suite))) != len(_rows(suite)):
        raise ValueError("Sealed suites must use distinct prompts")
    return suite


def _retention_suite(holdout, lineage, paths):
    ancestors = lineage["ancestry"][1:]
    if len(paths) != len(ancestors):
        raise ValueError("Verification requires the frozen holdout of every active accepted ancestor")
    combined = {**holdout, "sentinel": list(holdout["sentinel"])}
    for ancestor, path in zip(ancestors, paths, strict=True):
        prior = _verify_suite(_input(path, ancestor["holdout_sha256"]))
        if prior["choices"] != holdout["choices"]:
            raise ValueError("Retention requires compatible answer choice encoding")
        combined["sentinel"].extend(_rows(prior))
    return _verify_suite(json.dumps(combined).encode())


def _check_retention_parent(lineage, rows):
    verified = {event["cycle_id"]: event["verification"]["report"] for event in lineage["events"]
                if event["kind"] == "ACCEPT"}
    expected = {}
    for ancestor in lineage["ancestry"][1:]:
        ref = verified[ancestor["cycle_id"]]
        original = json.loads(_input(ref["path"], ref["sha256"]))
        for row in _rows(original):
            expected[row["id"]] = row["candidate_prediction"]
    if any(row["base_prediction"] != expected.get(row["id"]) for row in rows):
        raise ValueError("Parent predictions did not reproduce previously verified retention rows")


def _check_practice_overlap(root, sealed):
    keys = _text_keys(_rows(sealed))
    for path in (root / "cycles").glob("cycle-*/practice.json"):
        if keys & _text_keys(_rows(_read(path))):
            raise ValueError("Sealed prompts overlap cycle practice data")


def verify_cycle(root, cycle_id, holdout_path, retention_paths=()):
    """Independently reload a frozen finalist and record one lineage decision."""
    root, holdout_path = Path(root).resolve(), Path(holdout_path).resolve()
    with workspace_lock(root):
        protocol, spec, lineage, decided = _cycle(root, cycle_id)
        work = root / "cycles" / cycle_id
        evaluation = root / "evaluation" / cycle_id
        review_path = evaluation / "review.json"
        if decided:
            recorded = next(event for event in lineage["events"] if event.get("cycle_id") == cycle_id)
            review = _read(review_path)
            expected = {"cycle_id": cycle_id, "parent_sha256": recorded["parent_sha256"],
                        "reason": recorded["reason"]}
            if "verification" in recorded:
                expected.update(disposition=recorded["kind"], candidate_sha256=recorded["candidate"]["sha256"],
                                report_sha256=recorded["verification"]["report"]["sha256"])
            else:
                expected.update(disposition="NO_FINALIST")
            if any(review.get(key) != value for key, value in expected.items()):
                raise ValueError("Saved review differs from the lineage event")
            return review
        if not (work / "selection.json").is_file():
            raise ValueError("Complete practice selection before requesting verification")
        started = time.perf_counter()
        selection = _read(work / "selection.json")
        parent = lineage["head_sha256"]
        if (selection["parent_sha256"] != parent or selection["cycle_id"] != cycle_id
                or file_hash(work / "plan.json") != selection["plan_sha256"]
                or file_hash(work / "session" / "decision.json") != selection["session_decision_sha256"]):
            raise ValueError("Practice selection changed or has a stale parent")
        evaluation.mkdir(exist_ok=True)
        # The verifier opens a sealed suite only for a frozen finalist.
        finalist = selection["finalist"]
        decision = _read(work / "session" / "decision.json")
        if finalist is None:
            if decision["disposition"] != "STOPPED":
                raise ValueError("No-finalist selection disagrees with the session decision")
            review = {"cycle_id": cycle_id, "disposition": "NO_FINALIST", "parent_sha256": parent,
                      "reason": "No candidate passed practice", "elapsed_seconds": time.perf_counter() - started}
            if not review_path.exists():
                json_dump_atomic(review_path, review)
            record_lineage_rejection(root / "lineage", cycle_id=cycle_id, reason=review["reason"],
                                     expected_parent_sha256=parent)
            return _read(review_path)
        if decision["disposition"] != "FINALIZED" or any(decision.get(key) != value for key, value in finalist.items()):
            raise ValueError("Finalist differs from the immutable practice decision")
        candidate = Path(decision["candidate_path"]).resolve()
        if (not candidate.is_relative_to(work / "session" / "attempts")
                or load_checkpoint(candidate).checkpoint_sha256() != finalist["candidate_sha256"]):
            raise ValueError("Finalist checkpoint changed before verification")
        holdout = _verify_suite(_input(holdout_path, spec["holdout_sha256"]))
        combined = _retention_suite(holdout, lineage, retention_paths)
        _check_practice_overlap(root, combined)
        suite_path = evaluation / "suite.json"
        if not suite_path.exists():
            json_dump_atomic(suite_path, combined)
        elif _read(suite_path) != combined:
            raise ValueError("Resumed verification suite differs from the original")
        report_path = evaluation / "report.json"
        # Prior prediction changes are always disallowed. The report's cap
        # applies to total flips, and the review checks the prior rows separately.
        gates = {key: protocol[key] for key in ("min_target_gain", "max_sentinel_flips")}
        if report_path.exists():
            report = _read(report_path)
        else:
            report = evaluate_hf_edit(lineage["head_path"], candidate, suite_path, report_path,
                                      **gates, **protocol["inference"])
        if (report.get("configuration") != protocol["inference"]
                or report.get("suite_sha256") != file_hash(suite_path)
                or any(report.get(key) != value for key, value in gates.items())):
            raise ValueError("Verification report differs from the frozen protocol")
        retention_rows = report["sentinel"][len(holdout["sentinel"]):]
        _check_retention_parent(lineage, retention_rows)
        retention_flips = sum(row["base_prediction"] != row["candidate_prediction"] for row in retention_rows)
        regressions = sum(row["base_prediction"] == row["label"] != row["candidate_prediction"]
                          for row in report["target"])
        accept = (report["disposition"] == "PASS" and not retention_flips
                  and regressions <= protocol["max_target_regressions"])
        if retention_flips:
            reason = "Prior verified prediction changed"
        elif regressions > protocol["max_target_regressions"]:
            reason = "Target regressions exceeded the frozen threshold"
        elif accept:
            reason = "Held-out target gain and preservation gates passed"
        else:
            reason = "Held-out target gain or sentinel gate failed"
        review = {"cycle_id": cycle_id, "disposition": "ACCEPT" if accept else "REJECT", "reason": reason,
                  "parent_sha256": parent, "candidate_sha256": finalist["candidate_sha256"],
                  "holdout_sha256": spec["holdout_sha256"], "report_sha256": file_hash(report_path),
                  "target_gain": report["target_gain"], "target_regressions": regressions,
                  "sentinel_prediction_flips": report["sentinel_prediction_flips"] - retention_flips,
                  "retention_count": len(retention_rows), "retention_flips": retention_flips,
                  "elapsed_seconds": time.perf_counter() - started}
        if not review_path.exists():
            json_dump_atomic(review_path, review)
        elif {key: value for key, value in _read(review_path).items() if key != "elapsed_seconds"} != {
            key: value for key, value in review.items() if key != "elapsed_seconds"
        }:
            raise ValueError("Resumed review differs from the original decision")
        record_lineage_decision(root / "lineage", candidate, report_path, suite=suite_path, cycle_id=cycle_id,
                                accept=accept, reason=reason, expected_parent_sha256=parent, source_holdout=holdout_path)
        return _read(review_path)


def score_final(root, final_path):
    """Open the final suite only after all protocol cycles have immutable decisions."""
    root = Path(root).resolve()
    with workspace_lock(root):
        protocol = _protocol(root)
        lineage = lineage_status(root / "lineage")
        decided = {event.get("cycle_id") for event in lineage["events"]}
        if any(spec["id"] not in decided for spec in protocol["cycles"]):
            raise ValueError("Complete every frozen cycle before opening the final suite")
        destination = root / "evaluation" / "final.json"
        if destination.exists():
            raise ValueError("The sealed final suite was already evaluated")
        started = time.perf_counter()
        suite = _verify_suite(_input(final_path, protocol["final_sha256"]))
        _check_practice_overlap(root, suite)
        for path in (root / "evaluation").glob("cycle-*/suite.json"):
            if _text_keys(_rows(suite)) & _text_keys(_rows(_read(path))):
                raise ValueError("Final prompts overlap a cycle verification suite")
        paths = [Path(lineage[key]) for key in ("root_path", "head_path")]
        hashes = [lineage[key] for key in ("root_sha256", "head_sha256")]
        if [load_checkpoint(path).checkpoint_sha256() for path in paths] != hashes:
            raise ValueError("Checkpoint changed before final scoring")
        rows = _rows(suite)
        before = _score_checkpoint(paths[0], suite["choices"], rows, **protocol["inference"])
        after = before if hashes[0] == hashes[1] else _score_checkpoint(paths[1], suite["choices"], rows, **protocol["inference"])
        if [load_checkpoint(path).checkpoint_sha256() for path in paths] != hashes:
            raise ValueError("Checkpoint changed during final scoring")
        paired = [{"id": row["id"], "label": row["label"], "base_prediction": a, "head_prediction": b}
                  for row, a, b in zip(rows, before, after, strict=True)]
        size = len(suite["target"])
        base_correct = [row["base_prediction"] == row["label"] for row in paired[:size]]
        head_correct = [row["head_prediction"] == row["label"] for row in paired[:size]]
        summary = {"schema": "weightbridge.phase_iii.final.v1", "suite_sha256": protocol["final_sha256"],
                   "root_sha256": hashes[0], "head_sha256": hashes[1], "accepted_cycles": len(lineage["ancestry"]) - 1,
                   "configuration": protocol["inference"], "target_count": size, "base_target_correct": sum(base_correct),
                   "head_target_correct": sum(head_correct), "target_gain": sum(head_correct) - sum(base_correct),
                   "target_regressions": sum(a and not b for a, b in zip(base_correct, head_correct, strict=True)),
                   "sentinel_count": len(suite["sentinel"]),
                   "sentinel_prediction_flips": sum(a != b for a, b in zip(before[size:], after[size:], strict=True)),
                   "paired_prompt_bootstrap": paired_bootstrap_interval(base_correct, head_correct),
                   "target": paired[:size], "sentinel": paired[size:], "elapsed_seconds": time.perf_counter() - started}
        json_dump_atomic(destination, summary)
        return summary


def rollback_research(root, ancestor_sha256, *, reason):
    root = Path(root).resolve()
    with workspace_lock(root):
        _protocol(root)
        return rollback_lineage(root / "lineage", ancestor_sha256, reason=reason)
