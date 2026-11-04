from experiments.phase_iiim import semantic_metrics, semantic_roster
from experiments.phase_iiih.run_decimal_endpoint import endpoint
from weightbridge.adaptation.behavior_endpoint import _observation


def test_local_strict_and_eight_form_family_corrections_are_distinct(tmp_path):
    material, hidden = tmp_path / "material", tmp_path / "hidden.json"
    manifest = semantic_roster.freeze(material, hidden, hidden_seed="m-metric-hidden")
    rows = semantic_roster.load_split(material, "development-transfer", manifest)
    contract = endpoint(semantic_roster._read(semantic_roster.PROTOCOL))
    family = rows[0]["pair_id"]
    target = [row["id"] for row in rows if row["pair_id"] == family]

    def measure(corrected):
        measured = []
        for row in rows:
            wrong = row["id"] in target and row["id"] not in corrected
            answer = next(value for value in row["operands"] if value != row["answer"]) if wrong else row["answer"]
            # Trailing zeros preserve the exact Decimal answer's semantic value.
            text = answer + "0\n"
            output = {**row, "text": text, "stop_reason": "token_budget"}
            measured.append({**output, **_observation(output, contract.question(row), contract)})
        return {"endpoint": contract.manifest(), "rows": measured}

    parent = measure(set())
    seven = semantic_metrics.compare(parent, measure(set(target[:7])), min_family_gain=1)
    assert len(seven["local_corrected_target_presentations"]) == 7
    assert seven["target_gain"] == 1 and seven["complete_family_gain"] == 0 and not seven["passed"]
    all_eight = semantic_metrics.compare(parent, measure(set(target)), min_family_gain=1)
    assert len(all_eight["local_corrected_target_presentations"]) == 8
    assert all_eight["target_gain"] == 2 and all_eight["complete_family_gain"] == 1 and all_eight["passed"]
