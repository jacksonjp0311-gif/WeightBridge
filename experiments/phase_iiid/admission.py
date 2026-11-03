"""Verify the frozen grounding evidence before authorizing a research compiler."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiid.analysis import analyze_split, ranker_predictions, select_practice
from experiments.phase_iiid.roster import FAMILIES, MANIFEST_SCHEMA, PROTOCOL, SCHEMA
from experiments.phase_iiid.run_grounding import (
    CONTEXTS, DISPLAYS, GENERATION_CONTEXTS, SPLITS, _rows, _source_hashes,
)
from weightbridge.adaptation.grounding import parse_generation
from weightbridge.core import json_dump_atomic


REPORT_SCHEMA = "weightbridge.phase_iiid.grounding_report.v1"
EXECUTION_SCHEMA = "weightbridge.phase_iiid.grounding_execution.v1"
ADMISSION_SCHEMA = "weightbridge.phase_iiid.admission.v1"


def _read(path):
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return value, hashlib.sha256(payload).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _population(rows, *, choose):
    """Classify all question clusters; missing or incomplete responses stay invalid."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["item_id"]].append((index, row))
    result = {family: {} for family in FAMILIES}
    for item_id, group in groups.items():
        family = group[0][1]["family"]
        contexts = {}
        for context, position_key in (("task", "correct_position"), ("counterfactual", "counterfactual_position")):
            decisions = [choose(index, row, context) for index, row in group]
            valid = [type(position) is int and position in (0, 1) for position in decisions]
            correct = sum(ok and position == row[position_key]
                          for (_, row), position, ok in zip(group, decisions, valid, strict=True))
            if not all(valid):
                state = "invalid"
            elif correct == len(group):
                state = "solved"
            elif correct == 0:
                state = "persistent_wrong"
            else:
                state = "unstable"
            canonical = [row["order"][position] for (_, row), position, ok
                         in zip(group, decisions, valid, strict=True) if ok]
            contexts[context] = {"classification": state, "valid": sum(valid), "correct": correct,
                                 "presentations": len(group),
                                 "content_consistent": all(valid) and len(set(canonical)) == 1}
        result[family][item_id] = contexts
    return result


def _generation_position(row, display, parser, context):
    response = row["displays"][display]["generation"][context]
    return response[f"{parser}_position"] if response["stop_reason"] == "eos" else None


def _validate_generation(rows, split):
    for row in rows:
        for display in DISPLAYS:
            for context in GENERATION_CONTEXTS:
                response = row["displays"][display]["generation"][context]
                _require(response["stop_reason"] in ("eos", "token_budget"),
                         f"{split} generation has an unknown stop reason")
                expected = parse_generation(response["text"], display=display, labels=row["labels"],
                                            answers=row["answers"], aliases=row["aliases"],
                                            completed=response["stop_reason"] == "eos")
                _require(all(response[key] == value for key, value in expected.items()),
                         f"{split} generation decision does not replay from the recorded response")


def _headroom(populations, thresholds, ranker_qualified):
    eligibility, scope = {}, {}
    for family in FAMILIES:
        by_split = {}
        for split in ("practice", "selection"):
            bank = populations[split]["ranker"]
            target = sorted(item_id for item_id, states in bank[family].items()
                            if states["task"]["classification"] == "persistent_wrong")
            preserve = sorted(item_id for other in FAMILIES if other != family
                              for item_id, states in bank[other].items()
                              if all(value["classification"] == "solved" for value in states.values()))
            passed = (len(target) >= thresholds["minimum_persistent_errors_per_split"]
                      and len(preserve) >= thresholds["minimum_cross_family_controls_per_split"])
            by_split[split] = {"persistent_wrong_item_ids": target, "preservation_item_ids": preserve,
                               "passed": passed}
        eligibility[family] = by_split
        if ranker_qualified and all(record["passed"] for record in by_split.values()):
            scope[family] = {
                "target_item_ids": by_split["practice"]["persistent_wrong_item_ids"],
                "preservation_item_ids": by_split["practice"]["preservation_item_ids"],
            }
    return eligibility, scope


def assess(material, measurement, *, protocol=PROTOCOL):
    """Read only measured splits, reproduce the practice decision and reapply gates."""
    material, measurement, protocol = Path(material), Path(measurement), Path(protocol)
    config, protocol_hash = _read(protocol)
    manifest, manifest_hash = _read(material / "manifest.json")
    report, report_hash = _read(measurement / "report.json")
    execution, execution_hash = _read(measurement / "execution-sources.json")
    decision, decision_hash = _read(measurement / "practice-decision.json")
    _require(config["schema"] == "weightbridge.phase_iiid.grounding_protocol.v1", "Wrong grounding protocol schema")
    _require(manifest["schema"] == MANIFEST_SCHEMA
             and manifest["model_sha256"] == config["model_sha256"]
             and manifest["protocol_sha256"] == protocol_hash, "Grounding manifest is not bound to the protocol")
    _require(report["schema"] == REPORT_SCHEMA and report["protocol_sha256"] == protocol_hash
             and report["base_sha256"] == config["model_sha256"]
             and report["material_manifest_sha256"] == manifest_hash
             and report["execution_sources_sha256"] == execution_hash
             and report["practice_decision_sha256"] == decision_hash
             and set(report["splits"]) == set(SPLITS), "Grounding report is not bound to frozen evidence")
    sources = _source_hashes()
    _require(execution["schema"] == EXECUTION_SCHEMA and execution["base_sha256"] == config["model_sha256"]
             and execution["protocol_sha256"] == protocol_hash and execution["material_manifest_sha256"] == manifest_hash
             and execution["source_sha256"] == sources
             and manifest["generator_sha256"] == sources["experiments/phase_iiid/roster.py"]
             and execution["inference"] == config["inference"] and execution["generation"] == config["generation"]
             and execution["generation_contexts"] == list(GENERATION_CONTEXTS)
             and execution["scored_contexts"] == list(CONTEXTS), "Grounding execution source or settings differ")

    computed_decision = None
    gates, populations, seen = {}, {}, set()
    for split in SPLITS:
        frozen, frozen_hash = _read(material / f"{split}.json")
        expected = manifest["artifacts"][split]
        keys = {item["semantic_key"] for item in frozen["items"]}
        counts = Counter(item["family"] for item in frozen["items"])
        _require(frozen["schema"] == SCHEMA and frozen["split"] == split and frozen_hash == expected["sha256"]
                 and counts == dict.fromkeys(FAMILIES, config["underlying_items_per_family"][split])
                 and len(keys) == len(frozen["items"]) == expected["underlying_items"] and not (seen & keys),
                 f"Frozen {split} material differs or overlaps")
        seen.update(keys)
        raw, raw_hash = _read(measurement / f"{split}-predictions.json")
        summary = report["splits"][split]
        _require(raw["split"] == split and raw_hash == summary["predictions_sha256"]
                 and raw["resources"] == summary["resources"], f"{split} prediction evidence differs")
        rows = raw["rows"]
        strip_displays = lambda row: {key: value for key, value in row.items() if key != "displays"}
        _require([strip_displays(row) for row in rows] == [strip_displays(row) for row in _rows(frozen["items"])],
                 f"{split} predictions do not correspond to the frozen presentations")
        _validate_generation(rows, split)
        if split == "practice":
            computed_decision = select_practice(rows, config)
            _require(decision == {**computed_decision, "protocol_sha256": protocol_hash,
                                  "practice_predictions_sha256": raw_hash},
                     "Practice decision does not replay from practice evidence")
        analysis = analyze_split(rows, config, computed_decision)
        _require(analysis == summary["analysis"], f"{split} analysis does not replay from bound evidence")
        gates[split] = {"ranker": analysis["ranker_gate"], "strict_generation": analysis["generation_gate"]}
        ranker = ranker_predictions(rows, decision["selected_ranker"]["rule"])
        display = decision["selected_generation"]["display"]
        populations[split] = {"ranker": _population(rows, choose=lambda i, _, context: ranker[i][context])}
        for parser in ("strict", "semantic"):
            populations[split][f"{parser}_generation"] = _population(
                rows, choose=lambda _, row, context: _generation_position(row, display, parser, context))

    ranker_pass = all(gates[split]["ranker"]["passed"] for split in SPLITS)
    generator_pass = all(gates[split]["strict_generation"]["passed"] for split in SPLITS)
    headroom, scope = _headroom(populations, config["headroom_gate"], ranker_pass)
    if ranker_pass:
        reason = None if scope else "NO_REPEATABLE_RANKER_HEADROOM"
    elif generator_pass:
        reason = "QUALIFIED_GENERATION_WITHOUT_QUALIFIED_RANKER_SURROGATE"
    else:
        reason = "NO_MEASUREMENT_QUALIFIED"
    return {
        "schema": ADMISSION_SCHEMA, "disposition": "PASS" if scope else "HOLD", "hold_reason": reason,
        "model_sha256": config["model_sha256"], "protocol_sha256": protocol_hash,
        "report_sha256": report_hash, "material_manifest_sha256": manifest_hash,
        "practice_decision_sha256": decision_hash,
        "qualification": {"ranker": ranker_pass, "strict_generation": generator_pass, "splits": gates},
        "ranker_rule": decision["selected_ranker"]["rule"],
        "strict_generation_display": decision["selected_generation"]["display"],
        "population": populations, "headroom_by_family": headroom,
        "eligible_families": list(scope), "authorized_practice": scope,
        "sealed_adaptation_opened": False, "candidate_attempts": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("material", "measurement", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    result = assess(args.material, args.measurement)
    json_dump_atomic(args.out, result)
    print(json.dumps({key: result[key] for key in ("disposition", "hold_reason", "eligible_families")}, indent=2))
