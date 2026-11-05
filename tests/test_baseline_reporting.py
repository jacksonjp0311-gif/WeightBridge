"""Reporting-only fixtures; no model execution or material roster access."""
import json
from pathlib import Path

import pytest

from experiments.phase_iiiq_baseline import analyze
from experiments.phase_iiiq_baseline.integrity import PARENT
from weightbridge.core import file_hash


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def protocol(tmp_path):
    return put(tmp_path/"protocol.json", {"checkpoint_sha256": PARENT, "source_sha256": {"frozen.py": "a"*64}})


def test_missing_observations_are_unknown_and_not_pass(protocol, tmp_path):
    report = analyze.summarize(protocol, smoke_root=tmp_path/"smoke", run_roots=[tmp_path/"run-0"],
                               assessment_root=tmp_path/"assessment")
    assert report["disposition"] == "HOLD"
    assert report["stages"] == dict.fromkeys(("merged", "preservation_validation", "selection", "hidden", "exact_reconstruction"))
    assert report["measured_cost_totals"]["cpu_seconds"] is None
    assert report["main_runs"][0]["counts"] is None
    assert report["smoke"]["matched_loss_after"] == [None, None]


def test_protocol_binding_and_parent_are_checked(protocol, tmp_path):
    put(tmp_path/"run/completed.json", {"protocol_sha256": "bad"})
    with pytest.raises(ValueError, match="protocol identity"):
        analyze.summarize(protocol, run_roots=[tmp_path/"run"])
    put(protocol, {"checkpoint_sha256": "wrong"})
    with pytest.raises(ValueError, match="parent identity"):
        analyze.summarize(protocol)


def test_cost_ignores_launcher_aggregate_and_counts_worker_bodies_once(protocol, tmp_path):
    put(tmp_path/"smoke/smoke-gate.json", {"identity": {"protocol_sha256": file_hash(protocol)},
        "worker_processes": [{"cpu_seconds": .25}]*4,
        "resources": {"cpu_seconds": .1, "host_cpu_seconds": 1, "wall_seconds": 10, "peak_rss_bytes": 5}})
    put(tmp_path/"smoke/training/completed.json", {"status": "COMPLETED", "counts": {"forward_calls": 32},
        "resources": {"cpu_seconds": 12, "wall_seconds": 5, "peak_rss_bytes": 80}})
    for name, cpu in (("zero", 2), ("fresh-1", 3), ("fresh-2", 2)):
        put(tmp_path/f"smoke/{name}/execution.json", {"status": "COMPLETED", "resources": {
            "cpu_seconds": cpu, "wall_seconds": 1, "peak_rss_bytes": 100}})
    put(tmp_path/"run-0/completed.json", {"status": "COMPLETED", "counts": {"forward_calls": 64},
        "resources": {"cpu_seconds": 30, "wall_seconds": 15, "peak_rss_bytes": 200}})
    report = analyze.summarize(protocol, smoke_root=tmp_path/"smoke", run_roots=[tmp_path/"run-0"])
    assert report["measured_cost_totals"] == {"cpu_seconds": None, "wall_seconds": 25, "peak_rss_bytes": 200,
        "known_cpu_lower_bound_seconds": 51, "conservative_cpu_upper_bound_seconds": None}
    assert report["smoke"]["corrected_resources"]["peak_rss_bytes"] == 100
    assert report["smoke"]["training"]["counts"]["forward_calls"] == 32


def test_independent_smoke_audit_has_priority_and_raw_hash_binding(protocol, tmp_path):
    raw = put(tmp_path/"smoke/smoke-gate.json", {"resources": {"cpu_seconds": .1, "wall_seconds": 10, "peak_rss_bytes": 5}})
    admission = put(tmp_path/"smoke-admission.json", {"protocol_sha256": file_hash(protocol),
        "source_gate_sha256": file_hash(raw), "cost_audit": {"known_cpu_lower_bound_seconds": 99,
            "conservative_cpu_upper_bound_seconds": 123, "total_exact_cpu_seconds": None,
            "wall_seconds": 10, "peak_known_rss_bytes": 500}})
    report = analyze.summarize(protocol, smoke_root=raw.parent)
    assert report["measured_cost_totals"]["cpu_seconds"] is None
    assert report["measured_cost_totals"]["known_cpu_lower_bound_seconds"] == 99
    assert report["measured_cost_totals"]["conservative_cpu_upper_bound_seconds"] == 123
    assert report["measured_cost_totals"]["peak_rss_bytes"] == 500
    changed = json.loads(admission.read_text())
    changed["source_gate_sha256"] = "wrong"
    put(admission, changed)
    with pytest.raises(ValueError, match="another raw gate"):
        analyze.summarize(protocol, smoke_root=raw.parent)


def test_main_attempt_process_tree_supersedes_nested_body_cost(protocol, tmp_path):
    complete = put(tmp_path/"run-1/completed.json", {"status": "COMPLETED", "run_index": 1,
        "resources": {"cpu_seconds": 30, "wall_seconds": 7, "peak_rss_bytes": 500}})
    for name, complete_sha, index, status, cpu, wall, rss in (
        ("run-0", None, 0, "HOLD", 5, 2, 150),
        ("run-1", file_hash(complete), 1, "COMPLETED", 40, 10, 700),
    ):
        log = tmp_path/f"{name}.log"
        log.write_text("fixture", encoding="utf-8")
        put(tmp_path/f"{name}-process.json", {"schema": "weightbridge.supervised_baseline.training_process_tree.v1",
            "status": status, "run_index": index, "protocol_sha256": file_hash(protocol),
            "log_sha256": file_hash(log), "training_completed_receipt_sha256": complete_sha,
            "execution": {"resources": {"cpu_seconds": cpu, "wall_seconds": wall, "peak_rss_bytes": rss,
                "cpu_accounting_complete_for_observed_processes": True}}})
    report = analyze.summarize(protocol, run_roots=[tmp_path/"run-0", tmp_path/"run-1"])
    assert [row["status"] for row in report["main_runs"]] == ["HOLD", "COMPLETED"]
    assert report["measured_cost_totals"]["known_cpu_lower_bound_seconds"] == 45
    assert report["measured_cost_totals"]["wall_seconds"] == 12
    assert report["measured_cost_totals"]["peak_rss_bytes"] == 700
    assert report["measured_cost_totals"]["cpu_seconds"] is None


def test_checkpoint_effective_norms_and_hash_tampering(protocol, tmp_path):
    checkpoint = tmp_path/"run/checkpoints/step-0064"
    checkpoint.mkdir(parents=True)
    artifact = checkpoint/"adapter.safetensors"
    artifact.write_bytes(b"local fixture, never model loaded")
    put(checkpoint/"receipt.json", {"step": 64, "sha256": file_hash(artifact),
        "host_metadata": {"protocol_sha256": file_hash(protocol)},
        "effective_update": {"stored_update_norm": 2.5, "changed_entries": 42}})
    report = analyze.summarize(protocol, run_roots=[tmp_path/"run"])
    assert report["main_runs"][0]["checkpoints"][0]["stored_update_norm"] == 2.5
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash differs"):
        analyze.summarize(protocol, run_roots=[tmp_path/"run"])


def test_report_never_opens_rosters_or_sealed_worker_records(protocol, tmp_path, monkeypatch):
    root = tmp_path/"assessment"
    put(root/"assessment.json", {"protocol_sha256": file_hash(protocol), "candidates": {}, "winner": None})
    put(root/"independent.json", {"protocol_sha256": file_hash(protocol), "hidden": None})
    original = analyze.read
    def guarded(path):
        path = Path(path)
        assert "rosters" not in path.parts and "material" not in path.parts
        assert path.name != "hidden-merged.json"
        return original(path)
    monkeypatch.setattr(analyze, "read", guarded)
    report = analyze.summarize(protocol, assessment_root=root)
    assert report["stages"]["hidden"] is None
    assert report["lineage_accepted"] is False


def test_behavior_and_reconstruction_not_inferred_from_adapter_success(protocol, tmp_path):
    put(tmp_path/"assessment/assessment.json", {"protocol_sha256": file_hash(protocol),
        "candidates": {}, "winner": "run-0-step-64", "disposition": "PASS"})
    put(tmp_path/"assessment/independent.json", {"protocol_sha256": file_hash(protocol),
        "selection": {"passed": True}, "hidden": {"passed": True}, "exact_reconstruction": None})
    report = analyze.summarize(protocol, assessment_root=tmp_path/"assessment")
    assert report["stages"]["selection"]["passed"] is True
    assert report["stages"]["exact_reconstruction"] is None
    assert report["disposition"] == "HOLD"


def test_figures_only_plot_recorded_components_and_are_write_once(protocol, tmp_path):
    pytest.importorskip("matplotlib")
    from experiments.phase_iiiq_baseline.figures import generate
    report = analyze.summarize(protocol)
    report_path = put(tmp_path/"report.json", report)
    assert generate(report_path, tmp_path/"empty-figures") == []
    report["cost_components"] = [{"phase": "fixture", "wall_seconds": 60}]
    put(report_path, report)
    assert len(generate(report_path, tmp_path/"figures")) == 1
    with pytest.raises(FileExistsError, match="write-once"):
        generate(report_path, tmp_path/"figures")
