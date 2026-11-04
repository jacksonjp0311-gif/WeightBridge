import numpy as np
import pytest

from experiments.phase_iiim import reconstruct as study
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches import apply_safetensors_patch


def test_exact_reconstruction_and_frozen_input_refusal(tmp_path, monkeypatch):
    save = pytest.importorskip("safetensors.numpy").save_file
    base = tmp_path / "base"
    base.mkdir()
    json_dump_atomic(base / "config.json", {})
    save({"weight": np.zeros(2, dtype=np.float32)}, str(base / "model.safetensors"))
    checkpoint = load_checkpoint(base)
    original = (base / "model.safetensors").read_bytes()
    patch = {"schema": "weightbridge.patch.v1", "base_sha256": checkpoint.checkpoint_sha256(),
             "tensor_sha256": {"weight": checkpoint.tensor_sha256("weight")}, "reason": "M exact replay fixture",
             "changes": [{"tensor": "weight", "index": [0], "op": "add", "value": .25}]}
    policy = {"schema": "weightbridge.policy.v1", "allow_tensors": ["weight"], "max_entries": 1,
              "max_fraction": .5, "max_abs_delta": .5}
    trial = tmp_path / "trial"
    candidate = trial / "checkpoints" / "D2"
    receipt = apply_safetensors_patch(base, patch, policy, candidate)
    for suffix, value in (("patch", patch), ("policy", policy), ("receipt", receipt)):
        json_dump_atomic(trial / "patches" / f"D2-{suffix}.json", value)
    report = {"checkpoint_sha256": checkpoint.checkpoint_sha256(), "manifest_sha256": "a" * 64,
        "artifacts": {f"patches/D2-{suffix}.json": file_hash(trial / "patches" / f"D2-{suffix}.json")
                      for suffix in ("patch", "policy", "receipt")},
        "candidates": {"D2": {"checkpoint_sha256": receipt["candidate_sha256"],
                               "tensor_sha256": load_checkpoint(candidate).tensor_sha256("weight"),
                               "changed_entries": 1, "realized_update_norm": .25}}}
    json_dump_atomic(trial / "fit-report.json", report)
    monkeypatch.setattr(study, "validate_fit_report", lambda *args: report)
    result = study.reconstruct(base, trial, None, tmp_path / "replayed", tmp_path / "rebuild.json")
    assert result["exact_candidate_reconstruction"] and result["candidate_sha256"] == receipt["candidate_sha256"]
    assert (base / "model.safetensors").read_bytes() == original
    path = trial / "patches/D2-patch.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="immutable fitting evidence"):
        study.reconstruct(base, trial, None, tmp_path / "another-replay", tmp_path / "bad-rebuild.json")
