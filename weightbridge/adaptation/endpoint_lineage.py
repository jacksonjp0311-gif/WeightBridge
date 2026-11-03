"""Attach independently verified generative behavior to the research hash chain."""
from __future__ import annotations

import json
from pathlib import Path

from ..core import file_hash, json_dump_atomic
from .behavior_endpoint import BehavioralEndpointContract, compare_endpoint
from .lineage import _append, _artifact, _check_cycle, _checkpoint, _describe, workspace_lock


def _read(path):
    return json.loads(Path(path).read_bytes())


def record_endpoint_lineage(workspace, *, base, candidate, contract: BehavioralEndpointContract,
                            hidden_roster, manifest, base_report, candidate_report, replay_report,
                            rollback_report, selection_lock, cycle_id, min_target_gain):
    """Host-only adapter for a qualified independent generative finalist.

    Independent verifier reports stay local with the sealed source. The legacy
    binary-choice verifier remains unchanged and is never used to encode text
    responses. This method does not deploy the selected checkpoint.
    """
    root = Path(workspace).resolve(strict=True)
    with workspace_lock(root):
        state = _describe(root)
        parent = _checkpoint(base, root)
        selected = _checkpoint(candidate, root)
        reason = "Frozen independent generative endpoint and exact reload passed"
        _check_cycle(state, cycle_id, reason, parent["sha256"])
        if (not Path(selected["path"]).is_relative_to(root.parent) or selected["sha256"] == state["root_sha256"]
                or any(event["kind"] == "ACCEPT" and event["candidate"]["sha256"] == selected["sha256"]
                       for event in state["events"])):
            raise ValueError("Candidate is outside the research workspace or already accepted")

        roster_ref, raw = _artifact(hidden_roster)
        roster = json.loads(raw)
        manifest_ref, raw = _artifact(manifest)
        commitment = json.loads(raw)
        lock_ref, raw = _artifact(selection_lock)
        lock = json.loads(raw)
        if (roster.get("split") != "hidden" or commitment["splits"]["hidden"]["sha256"] != roster_ref["sha256"]
                or lock.get("schema") != "weightbridge.phase_iiih.candidate_lock.v1"
                or lock.get("manifest_sha256") != manifest_ref["sha256"]
                or lock.get("base_sha256") != parent["sha256"]
                or lock.get("winner_sha256") != selected["sha256"]
                or lock.get("selection", {}).get(lock.get("winner"), {}).get("passed") is not True):
            raise ValueError("Sealed roster, selection lock and checkpoint identities differ")

        receipt_ref, raw = _artifact(Path(selected["path"]) / ".weightbridge_receipt.json")
        receipt = json.loads(raw)
        if (receipt.get("schema") != "weightbridge.receipt.v2"
                or receipt.get("status") != "candidate_only_not_promoted"
                or receipt.get("base_sha256") != parent["sha256"]
                or receipt.get("candidate_sha256") != selected["sha256"]):
            raise ValueError("Checkpoint receipt differs from the sealed finalist")

        measurements, worker_refs = {}, {}
        for label, path, expected in (
            ("base", base_report, parent), ("candidate", candidate_report, selected),
            ("replay", replay_report, selected), ("rollback", rollback_report, parent),
        ):
            ref, raw = _artifact(path)
            worker = json.loads(raw)
            if (worker.get("schema") != "weightbridge.phase_iiih.verifier.v1"
                    or worker.get("split") != "hidden" or worker.get("roster_sha256") != roster_ref["sha256"]
                    or worker.get("checkpoint_sha256") != expected["sha256"]):
                raise ValueError(f"Invalid independent endpoint report: {label}")
            measurement = worker["measurement"]
            if len(measurement["rows"]) != len(roster["rows"]) or any(
                any(observed.get(key) != value for key, value in frozen.items())
                for observed, frozen in zip(measurement["rows"], roster["rows"], strict=True)
            ):
                raise ValueError(f"Independent report does not match the sealed presentations: {label}")
            measurements[label], worker_refs[label] = measurement, {"sha256": ref["sha256"], "report": worker}
        if (measurements["replay"] != {**measurements["candidate"], "resources": measurements["replay"]["resources"]}
                or measurements["rollback"] != {**measurements["base"], "resources": measurements["rollback"]["resources"]}):
            raise ValueError("Independent candidate replay or parent rollback differs")
        gate = compare_endpoint(measurements["base"], measurements["candidate"], contract,
                                min_target_gain=min_target_gain)
        if not gate["passed"]:
            raise ValueError("Failed independent endpoint gate cannot advance the lineage")

        # Write a local, self-contained verification bundle so normal lineage audit
        # binds the full evidence without changing the historical lineage schema.
        bundle = root / f"{cycle_id}.endpoint-verification.json"
        if bundle.exists():
            raise FileExistsError(bundle)
        json_dump_atomic(bundle, {"schema": "weightbridge.endpoint_lineage_verification.v1",
                                  "manifest_sha256": manifest_ref["sha256"], "lock_sha256": lock_ref["sha256"],
                                  "workers": worker_refs, "gate": gate})
        bundle_ref = {"path": str(bundle), "sha256": file_hash(bundle)}
        return _append(root, state, "ACCEPT", reason, cycle_id=cycle_id, candidate=selected, verification={
            "report": bundle_ref, "receipt": receipt_ref, "suite": roster_ref, "source_holdout": roster_ref,
            "disposition": "PASS", "target_gain": gate["target_item_gain"], "prohibited_regressions": 0,
            "endpoint_schema": contract.manifest()["schema"],
        })
