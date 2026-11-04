"""Resource accounting fixtures are synthetic, never research observations."""
import json

import pytest

from experiments.phase_iiio.compute import summarize
from weightbridge.core import file_hash, json_dump_atomic


@pytest.fixture
def archive(tmp_path):
    def write(name, value):
        json_dump_atomic(tmp_path / name, value)

    def resources(cpu, wall, rss):
        return {"cpu_seconds": cpu, "wall_seconds": wall, "peak_rss_bytes": rss}

    parent = {"rows": [{"generated_token_ids": tokens} for tokens in ([1], [1, 2], [1, 2, 3])],
              "resources": {"batched_decode_steps": 4, "cpu_seconds": 4.}}
    worker = {"rows": [{"generated_token_ids": tokens} for tokens in ([4], [4, 5])],
              "resources": {"batched_decode_steps": 2, "cpu_seconds": 2.}}
    write("trial/parent-adaptation.json", {"measurement": parent})
    write("trial/arm-A1.json", {"probe": {"reason": "individual_linear_margin_refusal"}})
    write("trial/secants.json", {name: {"nominal_radius": .0125, "realized_update_norm": .0124,
        "changed_entries": count, "candidate_tensor_sha256": digest * 64,
        "restored_exactly": True, "resources": resources(5., 1., 50)}
        for name, count, digest in (("A", 5, "a"), ("target_preview", 7, "b"))})
    write("trial/verification/development-transfer-base.json",
          {"resources": resources(3., 3., 80), "measurement": worker})
    write("trial/assessment.json", {"host_resources": resources(2., 4., 20),
        "worker_resources": {"development-transfer-base.json": {"cpu_seconds": 3.}}})
    write("trial/fit-report.json", {"sources": {"experiments/phase_iiio/run_joint.py": "c" * 64},
        "optimization_splits": ["adaptation"], "resources": resources(11., 9., 100)})
    write("execution.json", {"status": "COMPLETED",
        "fit_report_sha256": file_hash(tmp_path / "trial/fit-report.json"),
        "runner_source_sha256": "c" * 64, "resources": resources(12., 10., 100)})
    write("reconstruction-fixture.json", {"resources": resources(1., .5, 30)})
    write("analysis.json", {"resources": {"fitting_work_totals": {
        "forward_calls": 5, "backward_calls": 2, "batched_decode_steps": 4, "generated_examples": 3}},
        "arms": {"A1": {"behavior_measured": False}}, "unique_decoded_probes": 0, "independent": None})
    return tmp_path


def test_outer_cpu_scopes_count_once_and_secant_events_do_not_become_candidate_edits(archive):
    result = summarize(archive)
    # Execution includes the fit; host wall includes worker wall. Nested
    # measurements/secants are not extra CPU components to add a second time.
    assert result["completed_study_cpu_seconds"] == 12 + 2 + 3 + 1
    assert result["completed_fit_and_assessment_wall_seconds"] == 10 + 4
    assert result["largest_recorded_peak_rss_bytes"] == 100
    assert result["all_recorded_model_forward_calls"] == 5 + 4 + 2
    assert result["all_recorded_backward_calls"] == 2
    assert result["all_generated_response_records"] == 5
    assert result["all_generated_native_tokens"] == 9
    assert result["diagnostic_changed_entry_events"] == 12
    assert result["diagnostic_distinct_changed_parameter_positions"] is None
    assert all(row["restored_exactly"] for row in result["temporary_preservation_secants"].values())
    assert result["temporary_preservation_secants"]["A"]["changed_entries"] == 5
    assert result["unique_complete_candidate_practice_probes"] == 0
    assert result["decoded_candidate_changed_entry_events"] == 0
    assert result["decoded_candidate_distinct_changed_parameter_positions"] == 0
    assert result["verified_independent_improvements"] == 0
    assert result["compute_per_verified_improvement"] is None


@pytest.mark.parametrize("field,value", [("status", "INTERRUPTED"),
    ("fit_report_sha256", "d" * 64), ("runner_source_sha256", "e" * 64)])
def test_execution_identity_tampering_is_rejected(archive, field, value):
    path = archive / "execution.json"
    record = json.loads(path.read_bytes())
    record[field] = value
    # Deliberately bypass immutable creation to simulate damaged evidence.
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="source-bound completed fit"):
        summarize(archive)
