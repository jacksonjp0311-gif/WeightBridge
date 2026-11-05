"""Synthetic accounting fixtures, not additional experimental observations."""
import json

import pytest

from experiments.phase_iiip.compute import summarize
from weightbridge.core import file_hash, json_dump_atomic


@pytest.fixture
def archive(tmp_path):
    def write(name, value):
        json_dump_atomic(tmp_path / name, value)
    def resources(forwards=0, backwards=0):
        # Nested CPU belongs to the whole execution and is not another cost.
        return {"forward_calls": forwards, "backward_calls": backwards, "cpu_seconds": 999.}
    for name, count, steps in (("adaptation", 152, 228), ("preservation-fit", 56, 84), ("calibration", 56, 84)):
        write(f"trial/parent-{name}.json", {"measurement": {"rows": [
            {"generated_token_ids": list(range(12))} for _ in range(count)],
            "resources": {"generated_examples": count, "batched_decode_steps": steps, "cpu_seconds": 999.}}})
    for name, forwards, backwards in (("fork-diagnosis", 14, 0), ("target-block-metadata", 3, 14),
        ("preservation-block-metadata", 1, 8), ("calibration-block-metadata", 1, 8),
        ("native-parent-targets", 3, 0), ("preservation-decisions", 10, 0)):
        write(f"trial/{name}.json", {"resources": resources(forwards, backwards)})
    write("trial/secants.json", {name: {"resources": resources(20)} for name in ("A", "target_preview")})
    for name in ("preservation-sensitivities", "calibration-sensitivities"):
        write(f"trial/{name}.json", {"attempts": [{"replay": {"resources": resources(1, 8)}}]})
    write("trial/calibration-decisions.json", {"pool": {"resources": resources(4)}})
    for name in ("A1", "C1", "A2", "C2"):
        write(f"trial/arm-{name}.json", {"numerical": {"resources": resources()}, "probe": {"reason": "not_generated"}})
    write("trial/fit-report.json", {"sources": {"experiments/phase_iiip/run_right.py": "a" * 64},
        "qualification": {"disposition": "PASS"}, "optimization_splits": ["adaptation", "preservation-fit", "calibration"],
        "detail": {"explicit_forward_calls": 78, "backward_calls": 46}, "candidates": {},
        "resources": {"cpu_seconds": 19., "wall_seconds": 9., "peak_rss_bytes": 99}})
    write("execution.json", {"status": "COMPLETED", "fit_report_sha256": file_hash(tmp_path / "trial/fit-report.json"),
        "runner_source_sha256": "a" * 64, "resources": {"cpu_seconds": 20., "wall_seconds": 10., "peak_rss_bytes": 100}})
    write("trial/assessment.json", {"host_resources": {"cpu_seconds": 3., "wall_seconds": 1., "peak_rss_bytes": 20},
        "worker_resources": {}})
    return tmp_path


def test_runtime_discovery_probe_is_counted_once_and_explicitly_source_derived(archive):
    result = summarize(archive)
    assert result["explicit_forward_calls"] == 78 + 1
    assert result["backward_calls"] == 46 + 1
    assert result["greedy_batched_steps"] == 396
    assert result["total_model_forward_calls"] == 79 + 396
    assert result["generated_response_records"] == 264
    assert result["generated_native_tokens"] == 3168
    assert result["runtime_discovery_work"]["evidence_type"] == "source_derived_from_successful_constrained_plan"
    assert result["model_work_by_stage"]["runtime-capability-probe-source-derived"]["forward_calls"] == 1
    assert result["recorded_cpu_seconds"] == 20 + 3
    assert result["fit_and_assessment_wall_seconds"] == 10 + 1
    assert result["maximum_peak_rss_bytes"] == 100
    assert result["saved_candidates"] == result["unique_complete_candidate_practice_probes"] == 0
    assert result["compute_per_verified_improvement"] is None


def test_missing_successful_plan_evidence_does_not_invent_runtime_probe_count(archive):
    path = archive / "trial/fit-report.json"
    fit = json.loads(path.read_bytes())
    fit["detail"].pop("backward_calls")
    # Forge fixture bytes deliberately; historical evidence writers stay immutable.
    path.write_text(json.dumps(fit), encoding="utf-8")
    execution_path = archive / "execution.json"
    execution = json.loads(execution_path.read_bytes())
    execution["fit_report_sha256"] = file_hash(path)
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    result = summarize(archive)
    assert result["explicit_forward_calls"] == 78
    assert result["backward_calls"] == 46
    assert result["runtime_discovery_work"]["forward_calls"] is None
    assert result["runtime_discovery_work"]["backward_calls"] is None
    assert "runtime-capability-probe-source-derived" not in result["model_work_by_stage"]


@pytest.mark.parametrize("field,value", [("status", "INTERRUPTED"),
    ("fit_report_sha256", "b" * 64), ("runner_source_sha256", "c" * 64)])
def test_compute_refuses_changed_execution_or_source_binding(archive, field, value):
    path = archive / "execution.json"
    record = json.loads(path.read_bytes())
    record[field] = value
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="source-bound completed"):
        summarize(archive)
