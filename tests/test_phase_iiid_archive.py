import json

import pytest

from experiments.phase_iiid.archive_grounding import PROTOCOL, ROOT, SOURCES, SPLITS, export
from weightbridge.core import file_hash, json_dump_atomic


@pytest.fixture
def evidence(tmp_path):
    material, run = tmp_path / "material", tmp_path / "run"
    protocol = json.loads(PROTOCOL.read_bytes())
    protocol_hash = file_hash(PROTOCOL)
    source_hashes = {name: file_hash(ROOT / name) for name in SOURCES}
    artifacts, splits = {}, {}
    for split in SPLITS:
        roster, predictions = material / f"{split}.json", run / f"{split}-predictions.json"
        json_dump_atomic(roster, {"split": split})
        json_dump_atomic(predictions, {"split": split, "resources": {}})
        artifacts[split] = {"sha256": file_hash(roster)}
        splits[split] = {"predictions_sha256": file_hash(predictions), "resources": {}}
    artifacts["sealed-adaptation"] = {"sha256": "a" * 64}
    manifest, sources, decision = material / "manifest.json", run / "execution-sources.json", run / "practice-decision.json"
    base = protocol["model_sha256"]
    json_dump_atomic(manifest, {"schema": "weightbridge.phase_iiid.grounding_manifest.v1", "model_sha256": base,
                               "protocol_sha256": protocol_hash, "artifacts": artifacts,
                               "generator_sha256": source_hashes["experiments/phase_iiid/roster.py"]})
    json_dump_atomic(sources, {"schema": "weightbridge.phase_iiid.grounding_execution.v1", "base_sha256": base,
                              "protocol_sha256": protocol_hash, "material_manifest_sha256": file_hash(manifest),
                              "source_sha256": source_hashes, "inference": protocol["inference"],
                              "generation": protocol["generation"], "generation_contexts": ["task", "counterfactual"],
                              "scored_contexts": ["task", "counterfactual", "neutral_absent", "neutral_generic", "irrelevant"]})
    json_dump_atomic(decision, {"protocol_sha256": protocol_hash,
                               "practice_predictions_sha256": splits["practice"]["predictions_sha256"]})
    json_dump_atomic(run / "report.json", {"schema": "weightbridge.phase_iiid.grounding_report.v1", "base_sha256": base,
                                          "protocol_sha256": protocol_hash, "material_manifest_sha256": file_hash(manifest),
                                          "execution_sources_sha256": file_hash(sources),
                                          "practice_decision_sha256": file_hash(decision), "splits": splits})
    return material, run, tmp_path / "archive"


def test_export_verifies_and_copies_measured_bytes_without_reserve(evidence):
    material, run, output = evidence
    # There is no reserve file to open, only its recorded hash.
    export(material, run, output)
    inventory = json.loads((output / "artifact_manifest.json").read_bytes())
    assert len(inventory["artifacts"]) == 11
    assert not inventory["sealed_adaptation_bytes_exported"]
    assert inventory["sealed_adaptation_sha256"] == "a" * 64
    assert not (output / "material" / "sealed-adaptation.json").exists()
    for name, record in inventory["artifacts"].items():
        source = PROTOCOL if name == "material/protocol.json" else material.parent / name
        assert (output / name).read_bytes() == source.read_bytes()
        assert record["sha256"] == file_hash(source)
    with pytest.raises(FileExistsError):
        export(material, run, output)


@pytest.mark.parametrize("changed", ["material/practice.json", "run/selection-predictions.json",
                                    "run/practice-decision.json", "run/execution-sources.json"])
def test_export_refuses_changed_bytes_before_creating_archive(evidence, changed):
    material, run, output = evidence
    path = material.parent / changed
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Hash mismatch"):
        export(material, run, output)
    assert not output.exists()


def test_export_rejects_optional_analysis_from_another_run(evidence):
    material, run, output = evidence
    description = run / "description.json"
    json_dump_atomic(description, {"report_sha256": "different"})
    with pytest.raises(ValueError, match="does not bind"):
        export(material, run, output, description=description)
    assert not output.exists()
