from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from experiments.phase_iiil.archive_trial import _private_fields, _public_bytes
from experiments.phase_iiin import analyze, semantic_roster as roster
from experiments.phase_iiin.archive_trial import FIT_FILES, WORKER_FILES
from experiments.phase_iiih.run_decimal_endpoint import endpoint
from weightbridge.adaptation.behavior_endpoint import _observation


def test_n_allowlist_matches_only_bound_public_fitting_and_scored_workers(tmp_path):
    path = tmp_path / "arm-C2.json"
    path.write_bytes(b"{}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _public_bytes(tmp_path, path.name, FIT_FILES, digest) == b"{}"
    for allowed in ("supervision-selection.json", "reference-geometry.json", "probe-reference-A.json", "probe-reference-B.json"):
        assert FIT_FILES.fullmatch(allowed)
    for forbidden in ("../arm-C2.json", "hidden.json", "selection.json", "preservation-validation.json",
                      "verification/hidden-base.json", "checkpoints/C2/config.json", "arm-D2.json"):
        with pytest.raises(ValueError, match="allowlist"):
            _public_bytes(tmp_path, forbidden, FIT_FILES, digest)
    assert WORKER_FILES.fullmatch("verification/development-transfer-C2.json")
    assert not WORKER_FILES.fullmatch("verification/hidden-base.json")
    assert _private_fields({"payload": {"rows": [{"text": "sealed"}]}}) == {"rows", "text"}


def _measure(rows, config, wrong=()):
    contract, result = endpoint(config), []
    for row in rows:
        answer = row["answer"]
        if row["comparison_id"] in wrong:
            answer = next(value for value in row["operands"] if value != answer)
        generated = {**row, "text": answer + "\n", "stop_reason": "token_budget"}
        result.append({**generated, **_observation(generated, contract.question(row), contract)})
    return {"endpoint": contract.manifest(), "rows": result}


def test_comparison_and_complete_family_difference_cluster_transforms_together(tmp_path):
    material, hidden = tmp_path / "material", tmp_path / "hidden.json"
    manifest = roster.freeze(material, hidden, hidden_seed="n-analysis-test")
    rows = roster.load_split(material, "development-transfer", manifest)
    config = json.loads(roster.PROTOCOL.read_bytes())
    first = rows[0]["comparison_id"]
    second = next(row["comparison_id"] for row in rows if row["pair_id"] == rows[0]["pair_id"] and row["comparison_id"] != first)
    parent = _measure(rows, config, wrong=(first, second))
    partial = _measure(rows, config, wrong=(second,))
    complete = _measure(rows, config)
    from experiments.phase_iiin.semantic_metrics import compare
    assert compare(parent, partial)["complete_family_gain"] == 0
    assert compare(parent, complete)["complete_family_gain"] == 1
    contrast = analyze._cluster_difference(parent, partial, complete)
    assert contrast["family_count"] == 6
    assert sum(contrast["per_family_strict_count_difference"].values()) == 1
    assert contrast["strict_comparison_accuracy_difference"] == pytest.approx(1 / 12)
    distribution = analyze._distribution(complete)
    assert distribution["truth_flip_matrix"]["both_correct"] == 6
    assert distribution["wording"]["0"]["correct"] == 24
    assert distribution["order"]["1"]["correct"] == 24


def test_n_geometry_replay_requires_exact_m_reference_arrays():
    config = json.loads(roster.PROTOCOL.read_bytes())
    from experiments.phase_iiin.source_manifest import ROOT
    prior = json.loads((ROOT / config["fixed_geometry"]["basis_path"]).read_bytes())
    record = {"basis.json": {"columns": prior["columns"], "right_factor": prior["right_factor"]},
        "reference-geometry.json": {**config["fixed_geometry"], "columns": prior["columns"],
            "right_factor": prior["right_factor"], "checkpoint_sha256": config["checkpoint_sha256"]}}
    analyze._verify_geometry(config, record)
    changed = deepcopy(record)
    changed["basis.json"]["right_factor"][0] += .001
    with pytest.raises(ValueError, match="exactly reuse"):
        analyze._verify_geometry(config, changed)
