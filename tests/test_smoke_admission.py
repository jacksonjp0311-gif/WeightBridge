"""Post-freeze host cost correction; never model-efficacy evidence."""
import math

import pytest

from experiments.phase_iiiq_baseline.admit_smoke import _cost_audit, _number
from weightbridge.core import json_dump_atomic


def fixture_cost(tmp_path, *, budget=4500):
    for name, cpu, wall in (("zero", 350., 80.), ("fresh-1", 450., 95.), ("fresh-2", 450., 96.)):
        json_dump_atomic(tmp_path/name/"execution.json", {"status": "COMPLETED", "error": None,
            "resources": {"cpu_seconds": cpu, "wall_seconds": wall, "peak_rss_bytes": 4_000_000_000}})
    completed = {"resources": {"cpu_seconds": 1600., "wall_seconds": 297., "peak_rss_bytes": 7_000_000_000}}
    processes = [{"cpu_seconds": .01} for _ in range(4)]
    raw = {"host_cpu_seconds": 1., "cpu_seconds": 1.04, "wall_seconds": 587., "peak_rss_bytes": 5_000_000}
    limits = {"smoke_max_wall_seconds": 900, "smoke_max_cpu_seconds": budget, "max_memory_bytes": 24*1024**3}
    return completed, processes, raw, limits


def test_launcher_underreporting_is_corrected_without_false_complete_accounting(tmp_path):
    args = fixture_cost(tmp_path)
    audit = _cost_audit(tmp_path, *args)
    assert audit["known_cpu_lower_bound_seconds"] == 2851.04
    assert audit["peak_known_rss_bytes"] == 7_000_000_000
    assert audit["conservative_cpu_upper_bound_seconds"] == 3459.04
    assert audit["known_cpu_within_budget"] and audit["conservative_cpu_upper_bound_within_budget"]
    assert audit["total_exact_cpu_seconds"] is None and not audit["cpu_accounting_complete"]
    assert audit["raw_launcher_accounting_incomplete"]
    assert audit["startup_and_post_receipt_child_cpu"] == "UNMEASURED"


def test_raw_tiny_cpu_cannot_hide_actual_worker_budget_failure(tmp_path):
    args = fixture_cost(tmp_path, budget=2800)
    audit = _cost_audit(tmp_path, *args)
    assert args[2]["cpu_seconds"] < args[3]["smoke_max_cpu_seconds"]
    assert not audit["known_cpu_within_budget"]
    assert not audit["conservative_cpu_upper_bound_within_budget"]


def test_known_lower_bound_alone_cannot_authorize_unmeasured_gap(tmp_path):
    args = fixture_cost(tmp_path, budget=3000)
    audit = _cost_audit(tmp_path, *args)
    assert audit["known_cpu_within_budget"]
    assert not audit["conservative_cpu_upper_bound_within_budget"]


@pytest.mark.parametrize("value", [math.nan, math.inf, -1., True])
def test_invalid_cost_measurements_refused(value):
    with pytest.raises(ValueError):
        _number(value)
