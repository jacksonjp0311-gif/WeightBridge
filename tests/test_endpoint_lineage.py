import json

import numpy as np
import pytest

from weightbridge.adaptation.behavior_endpoint import decimal_endpoint
from weightbridge.adaptation.endpoint_lineage import record_endpoint_lineage
from weightbridge.adaptation.lineage import audit_lineage, init_lineage, rollback_lineage
from weightbridge.core import file_hash
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches import apply_safetensors_patch


def test_generative_lineage_replays_independent_gate_and_exact_rollback(tmp_path):
    save = pytest.importorskip("safetensors.numpy").save_file

    def write(name, document):
        path = tmp_path / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    base = tmp_path / "base"
    base.mkdir()
    write("base/config.json", {})
    save({"weight": np.zeros(2, dtype=np.float32)}, str(base / "model.safetensors"))
    parent = load_checkpoint(base)
    base_sha = parent.checkpoint_sha256()
    candidate = tmp_path / "candidate"
    receipt = apply_safetensors_patch(base, {
        "schema": "weightbridge.patch.v1", "base_sha256": base_sha,
        "tensor_sha256": {"weight": parent.tensor_sha256("weight")}, "reason": "Endpoint lineage fixture",
        "changes": [{"tensor": "weight", "index": [0], "op": "add", "value": .25}],
    }, {"schema": "weightbridge.policy.v1", "allow_tensors": ["weight"],
        "max_entries": 1, "max_fraction": .5, "max_abs_delta": .5}, candidate)
    child_sha = receipt["candidate_sha256"]
    workspace = tmp_path / "lineage"
    init_lineage(workspace, base)
    contract = decimal_endpoint({"max_input_tokens": 128, "max_new_tokens": 12, "batch_size": 8})
    rows = [{"id": f"target-{i}", "item": "target", "prompt": f"Form {i}\nValues: -2 or -3\nAnswer:",
             "wording": i, "family": "larger-decimal", "group": "target", "operands": ["-2", "-3"],
             "answer": "-2"} for i in range(2)]
    hidden = write("hidden.json", {"split": "hidden", "rows": rows})
    digest = file_hash(hidden)
    manifest = write("manifest.json", {"splits": {"hidden": {"sha256": digest}}})
    lock = write("lock.json", {"schema": "weightbridge.phase_iiih.candidate_lock.v1", "base_sha256": base_sha,
                               "winner": "tested", "winner_sha256": child_sha, "manifest_sha256": file_hash(manifest),
                               "selection": {"tested": {"passed": True}}})

    def report(name, sha, outputs):
        return write(name, {"schema": "weightbridge.phase_iiih.verifier.v1", "split": "hidden",
                            "checkpoint_sha256": sha, "roster_sha256": digest, "measurement": {
                                "endpoint": contract.manifest(), "resources": {"wall": name}, "rows": [
                                    {**row, "text": output, "stop_reason": "eos"}
                                    for row, output in zip(rows, outputs, strict=True)]}})

    args = dict(base=base, candidate=candidate, contract=contract, hidden_roster=hidden, manifest=manifest,
                selection_lock=lock, base_report=report("base.json", base_sha, ["-3", "-3"]),
                candidate_report=report("candidate.json", child_sha, ["-2", "-2"]),
                rollback_report=report("rollback.json", base_sha, ["-3", "-3"]), cycle_id="cycle", min_target_gain=1)
    with pytest.raises(ValueError, match="replay or parent rollback"):
        record_endpoint_lineage(workspace, **args, replay_report=report("replay.json", child_sha, ["-2", "-3"]))
    assert audit_lineage(workspace)["events"] == []
    replay = report("replay.json", child_sha, ["-2", "-2"])
    with pytest.raises(ValueError, match="endpoint gate"):
        record_endpoint_lineage(workspace, **{**args, "min_target_gain": 2}, replay_report=replay)
    accepted = record_endpoint_lineage(workspace, **args, replay_report=replay)
    assert accepted["head_sha256"] == child_sha
    assert audit_lineage(workspace) == accepted
    restored = rollback_lineage(workspace, base_sha, reason="Restore pinned source")
    assert restored["head_sha256"] == base_sha
    assert [event["kind"] for event in restored["events"]] == ["ACCEPT", "ROLLBACK"]
