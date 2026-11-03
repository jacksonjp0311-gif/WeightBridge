"""Rebuild the accepted Phase III Qwen lineage from archived numeric patches.

This audits evidence and replays weight bytes without rerunning inference.
Redacted event files keep their original source hash links in the manifest.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.adaptation.controller import _validate_protocol
from weightbridge.adaptation.lineage import (
    _event_checksum, audit_lineage, init_lineage, record_lineage_decision, rollback_lineage,
)
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches.huggingface import _canonical_sha256, apply_safetensors_patch


STATE = "run/state"
MAX_PATCH_BYTES = 32 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class Archive:
    def __init__(self, root):
        self.root = root
        raw = (root / "artifact_manifest.json").read_bytes()
        self.manifest_sha256 = sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        _require(self.manifest["schema"] == "weightbridge.phase_iii.evidence_export.v1",
                 "Unsupported Phase III evidence archive")
        self.artifacts = self.manifest["artifacts"]

    def source(self, name):
        return self.artifacts[name]["source_sha256"]

    def verify_exports(self):
        for name, entry in self.artifacts.items():
            parts = name.split("/")
            _require(not any(part in ("", ".", "..") or ":" in part or "\\" in part for part in parts),
                     f"Invalid archive artifact path: {name}")
            path = self.root.joinpath(*parts)
            _require(path.resolve().is_relative_to(self.root) and path.is_file()
                     and file_hash(path) == entry["exported_sha256"]
                     and entry["byte_identical"] == (entry["source_sha256"] == entry["exported_sha256"]),
                     f"Exported artifact differs: {name}")

    def document(self, name, *, exact=False, source=None):
        entry = self.artifacts[name]
        _require(source is None or self.source(name) == source, f"Source hash link differs: {name}")
        path = self.root / name
        raw = path.read_bytes()
        _require(sha256(raw).hexdigest() == entry["exported_sha256"], f"Exported artifact changed: {name}")
        if entry.get("encoding") == "gzip":
            with gzip.GzipFile(fileobj=BytesIO(raw)) as stream:
                raw = stream.read(MAX_PATCH_BYTES + 1)
            _require(len(raw) <= MAX_PATCH_BYTES and sha256(raw).hexdigest() == entry["decoded_sha256"]
                     == self.source(name), f"Decoded patch bytes differ: {name}")
        elif exact:
            _require(entry["byte_identical"], f"Archive lacks exact source bytes: {name}")
        return json.loads(raw)


def _accepted_patch(archive, protocol, spec, event):
    cycle = spec["id"]
    parent, candidate = event["parent_sha256"], event["candidate"]["sha256"]
    verification = event["verification"]
    work, evaluation = f"{STATE}/cycles/{cycle}", f"{STATE}/evaluation/{cycle}"
    selection = archive.document(f"{work}/selection.json", exact=True)
    archive.document(f"{work}/plan.json", exact=True, source=selection["plan_sha256"])
    finalist = selection["finalist"]
    _require(selection["cycle_id"] == cycle and selection["parent_sha256"] == parent
             and finalist["candidate_sha256"] == candidate, f"Selection does not bind {cycle}")
    decision = archive.document(f"{work}/session/decision.json", source=selection["session_decision_sha256"])
    candidate_id = finalist["candidate_id"]
    _require(re.fullmatch(r"candidate-[0-9]{4}", candidate_id) and decision["disposition"] == "FINALIZED"
             and decision["practice_disposition"] == "PASS"
             and all(decision.get(key) == value for key, value in finalist.items()),
             f"Practice decision does not bind {cycle}")
    attempt = f"{work}/session/attempts/{candidate_id}"
    patch_name = f"{attempt}/patch.json.gz"
    _require(archive.artifacts[patch_name].get("encoding") == "gzip", f"Expected compressed patch: {patch_name}")
    patch = archive.document(patch_name, exact=True, source=decision["patch_file_sha256"])
    receipt_name = f"{attempt}/model/.weightbridge_receipt.json"
    receipt = archive.document(receipt_name, exact=True, source=verification["receipt"]["sha256"])
    _require(patch["base_sha256"] == parent and receipt["base_sha256"] == parent
             and receipt["candidate_sha256"] == candidate and receipt["schema"] == "weightbridge.receipt.v2"
             and receipt["status"] == "candidate_only_not_promoted"
             and receipt["patch_sha256"] == decision["patch_sha256"] == _canonical_sha256(patch)
             and receipt["policy_sha256"] == _canonical_sha256(protocol["policy"]),
             f"Patch or receipt does not bind {cycle}")

    review = archive.document(f"{evaluation}/review.json", exact=True)
    report_sha256 = verification["report"]["sha256"]
    report = archive.document(f"{evaluation}/report.json", exact=True, source=report_sha256)
    suite_sha256 = verification["suite"]["sha256"]
    suite = archive.document(f"{evaluation}/suite.json", exact=True, source=suite_sha256)
    holdout_sha256 = spec["holdout_sha256"]
    holdout = archive.document(f"material/sealed/{cycle}.json", exact=True, source=holdout_sha256)
    _require(holdout["choices"] == suite["choices"] and holdout["target"] == suite["target"]
             and holdout["sentinel"] == suite["sentinel"][:len(holdout["sentinel"])],
             f"Evaluation suite differs from the frozen holdout: {cycle}")
    _require(verification["source_holdout"]["sha256"] == review["holdout_sha256"] == holdout_sha256
             and review["cycle_id"] == cycle and review["parent_sha256"] == report["base_sha256"] == parent
             and review["candidate_sha256"] == report["candidate_sha256"] == candidate
             and review["reason"] == event["reason"] and review["disposition"] == "ACCEPT"
             and review["report_sha256"] == report_sha256 and report["suite_sha256"] == suite_sha256
             and report["receipt_sha256"] == archive.source(receipt_name),
             f"Independent acceptance evidence differs: {cycle}")
    _verify_gate(protocol, cycle, suite, holdout, report, review, verification)
    return patch, receipt, {"cycle_id": cycle, "parent_sha256": parent, "candidate_sha256": candidate,
                            "patch_source_sha256": archive.source(patch_name), "patch_sha256": receipt["patch_sha256"],
                            "receipt_source_sha256": archive.source(receipt_name),
                            "receipt_sha256": _canonical_sha256(receipt), "report_sha256": report_sha256}


def _verify_gate(protocol, cycle, suite, holdout, report, review, verification):
    for group in ("target", "sentinel"):
        for row, expected in zip(report[group], suite[group], strict=True):
            _require(row["id"] == expected["id"] and row["label"] == expected["label"]
                     and row["base_prediction"] in suite["choices"]
                     and row["candidate_prediction"] in suite["choices"], f"Prediction row differs: {cycle}")
    before = [row["base_prediction"] == row["label"] for row in report["target"]]
    after = [row["candidate_prediction"] == row["label"] for row in report["target"]]
    gain = sum(after) - sum(before)
    regressions = sum(base and not candidate for base, candidate in zip(before, after, strict=True))
    flips = [row["base_prediction"] != row["candidate_prediction"] for row in report["sentinel"]]
    boundary = len(holdout["sentinel"])
    _require(verification["disposition"] == report["disposition"] == "PASS"
             and sum(before) == report["base_target_correct"] and sum(after) == report["candidate_target_correct"]
             and gain == review["target_gain"] == verification["target_gain"] == report["target_gain"]
             >= report["min_target_gain"] == protocol["min_target_gain"]
             and sum(flips) == verification["sentinel_prediction_flips"] == report["sentinel_prediction_flips"]
             <= report["max_sentinel_flips"] == protocol["max_sentinel_flips"]
             and sum(flips[:boundary]) == review["sentinel_prediction_flips"]
             and regressions == review["target_regressions"] == report["target_regressions"]
             <= protocol["max_target_regressions"]
             and len(flips[boundary:]) == review["retention_count"]
             and sum(flips[boundary:]) == review["retention_flips"] == 0
             and report["configuration"] == protocol["inference"], f"Acceptance gate differs: {cycle}")


def _rollback(archive, base, out, steps, protocol, ancestor):
    workspace = out / "lineage"
    init_lineage(workspace, base)
    for step in steps:
        cycle = step["cycle_id"]
        evaluation = archive.root / STATE / "evaluation" / cycle
        record_lineage_decision(
            workspace, out / step["checkpoint_directory"], evaluation / "report.json",
            suite=evaluation / "suite.json", source_holdout=archive.root / "material" / "sealed" / f"{cycle}.json",
            cycle_id=cycle, accept=True, reason="Verified archival replay",
            expected_parent_sha256=step["parent_sha256"],
        )
    former_head = steps[-1]
    rolled = rollback_lineage(workspace, ancestor, reason="Verify selection of a replayed ancestor")
    verified = audit_lineage(workspace)
    _require(rolled["head_sha256"] == verified["head_sha256"] == ancestor
             and verified["root_sha256"] == protocol["base_sha256"]
             and load_checkpoint(out / former_head["checkpoint_directory"]).checkpoint_sha256()
             == former_head["candidate_sha256"], "Rollback did not preserve checkpoint identities")
    return {"disposition": "PASS", "active_sha256": ancestor,
            "preserved_former_head_sha256": former_head["candidate_sha256"],
            "lineage_directory": workspace.name}


def replay(archive_path, base, out, *, expected_protocol_sha256=None, rollback_ancestor_sha256=None):
    archive = Archive(Path(archive_path).resolve(strict=True))
    _require(not Path(out).is_symlink(), "Replay output must be a fresh directory")
    base, out = Path(base).resolve(strict=True), Path(out).resolve()
    _require(base.is_dir() and not out.is_relative_to(base) and not out.is_relative_to(archive.root),
             "Use a local checkpoint directory and a fresh output outside both inputs")
    archive.verify_exports()
    protocol_sha256 = archive.manifest["protocol_sha256"]
    _require(expected_protocol_sha256 is None or protocol_sha256 == expected_protocol_sha256,
             "Archive protocol differs from the supplied SHA-256 pin")
    protocol = _validate_protocol(archive.document("material/protocol.json", exact=True, source=protocol_sha256))
    frozen = archive.document(f"{STATE}/research.json", exact=True)
    _require(archive.document(f"{STATE}/protocol.json", exact=True, source=frozen["protocol_sha256"]) == protocol,
             "Run protocol differs from the archived frozen material")
    root_sha256 = protocol["base_sha256"]
    lineage_name = f"{STATE}/lineage/lineage.json"
    lineage = archive.document(lineage_name)
    _require(lineage["schema"] == "weightbridge.checkpoint_lineage.v1" and lineage["root"]["sha256"] == root_sha256
             and load_checkpoint(base).checkpoint_sha256() == root_sha256,
             "Checkpoint differs from the archived lineage root; rebasing is unsupported")

    event_names = sorted(name for name in archive.artifacts if name.startswith(f"{STATE}/lineage/events/"))
    specs = protocol["cycles"]
    complete = archive.manifest["study_complete"]
    _require(len(event_names) <= len(specs) and (not complete or len(event_names) == len(specs)),
             "Archive event count differs from the frozen protocol")
    previous, head = archive.source(lineage_name), root_sha256
    heads, accepted, rejected = {root_sha256}, [], []
    for index, (name, spec) in enumerate(zip(event_names, specs), 1):
        event = archive.document(name)
        _require(name == f"{STATE}/lineage/events/{index:08d}.json" and event["sequence"] == index
                 and event["schema"] == "weightbridge.checkpoint_lineage_event.v1"
                 and event["previous_sha256"] == previous and event["parent_sha256"] == head
                 and event["cycle_id"] == spec["id"] and event["kind"] in ("ACCEPT", "REJECT"),
                 "Archive event order or parent differs; rollback and composition are unsupported")
        if archive.artifacts[name]["byte_identical"]:
            _require(event["event_sha256"] == _event_checksum(event), f"Event checksum differs: {name}")
        cycle = spec["id"]
        archive.document(f"material/practice/{cycle}.json", exact=True, source=spec["practice_sha256"])
        archive.document(f"material/practice/{cycle}-intent.json", exact=True, source=spec["intent_sha256"])
        previous = archive.source(name)
        if event["kind"] == "ACCEPT":
            item = _accepted_patch(archive, protocol, spec, event)
            head = event["candidate"]["sha256"]
            _require(head not in heads, "Archive repeats an accepted checkpoint")
            heads.add(head)
            accepted.append(item)
        else:
            review = archive.document(f"{STATE}/evaluation/{cycle}/review.json", exact=True)
            _require(review["cycle_id"] == cycle and review["parent_sha256"] == head
                     and review["reason"] == event["reason"]
                     and review["disposition"] == ("REJECT" if "verification" in event else "NO_FINALIST"),
                     f"Rejected cycle review differs: {cycle}")
            rejected.append(cycle)
    if complete:
        summary = archive.document("run/summary.json")
        final = archive.document(f"{STATE}/evaluation/final.json", exact=True)
        archive.document("material/sealed/final.json", exact=True, source=protocol["final_sha256"])
        _require(summary["protocol_sha256"] == protocol_sha256 and summary["original_sha256"] == root_sha256
                 and summary["final_head_sha256"] == final["head_sha256"] == head
                 and final["root_sha256"] == root_sha256 and final["suite_sha256"] == protocol["final_sha256"]
                 and summary["accepted_cycles"] == final["accepted_cycles"] == len(accepted),
                 "Completed study summary differs from the accepted lineage")
    _require(rollback_ancestor_sha256 is None or (rollback_ancestor_sha256 in heads and rollback_ancestor_sha256 != head),
             "Rollback target must be an earlier accepted ancestor")

    out.mkdir(parents=True)
    parent = base
    steps = []
    for patch, expected, step in accepted:
        candidate = out / step["cycle_id"]
        receipt = apply_safetensors_patch(parent, patch, protocol["policy"], candidate)
        _require(receipt == expected and _canonical_sha256(receipt) == step["receipt_sha256"]
                 and file_hash(candidate / ".weightbridge_receipt.json") == step["receipt_source_sha256"]
                 and load_checkpoint(candidate).checkpoint_sha256() == step["candidate_sha256"],
                 f"Reconstructed candidate or receipt differs: {step['cycle_id']}")
        steps.append({**step, "checkpoint_directory": candidate.name, "changed_entries": receipt["changed_entries"]})
        parent = candidate
    result = {"schema": "weightbridge.phase_iii.accepted_lineage_replay.v1", "disposition": "PASS",
              "study_complete": complete, "artifact_manifest_sha256": archive.manifest_sha256,
              "protocol_sha256": protocol_sha256, "root_sha256": root_sha256, "head_sha256": head,
              "accepted": steps, "rejected_cycles_skipped": rejected, "inference_rerun": False}
    if rollback_ancestor_sha256 is not None:
        result["rollback"] = _rollback(archive, base, out, steps, protocol, rollback_ancestor_sha256)
    json_dump_atomic(out / "replay.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("archive", "base", "out"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    parser.add_argument("--protocol-sha256", help="Optional SHA-256 pin obtained independently of the archive")
    parser.add_argument("--rollback-ancestor", help="Verify rollback to this earlier checkpoint SHA-256")
    args = parser.parse_args()
    try:
        result = replay(args.archive, args.base, args.out, expected_protocol_sha256=args.protocol_sha256,
                        rollback_ancestor_sha256=args.rollback_ancestor)
    except (EOFError, KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"Replay failed: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
