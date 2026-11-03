from collections import Counter

import pytest

from experiments.phase_iiif.matrix_data import PROTOCOL, freeze, load_split, read_json, render_rows
from experiments.phase_iiif.matrix_scoring import compare, qualify


def _scores(items):
    return [{
        **row, "correct": row["answers"][0].strip(), "ranker": row["answers"][0].strip(),
        "exact": row["answers"][0].strip(),
    } for row in render_rows(items)]


def test_roster_is_hash_bound_and_balanced(tmp_path):
    config = read_json(PROTOCOL)
    path = tmp_path / "matrix"
    manifest = freeze(path)
    practice = load_split(path, "practice", manifest, config["roster_counts_per_split"])
    evaluation = load_split(path, "evaluation", manifest, config["roster_counts_per_split"])
    assert not ({abs(item["value"]) for item in practice} & {abs(item["value"]) for item in evaluation})
    for items in (practice, evaluation):
        assert Counter((row["family"], row["answers"][0]) for row in render_rows(items)) == {
            ("parity", " true"): 8, ("parity", " false"): 8,
            ("sign", " true"): 4, ("sign", " false"): 4,
        }
    with (path / "evaluation.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="hash changed"):
        load_split(path, "evaluation", manifest, config["roster_counts_per_split"])


def test_matrix_requires_persistent_errors_and_preserves_exact_behavior(tmp_path):
    config = read_json(PROTOCOL)
    path = tmp_path / "matrix"
    manifest = freeze(path)
    items = load_split(path, "practice", manifest, config["roster_counts_per_split"])
    candidate = _scores(items)
    baseline = [dict(row) for row in candidate]
    assert qualify(baseline, config["qualification"])["disposition"] == "HOLD"
    targets = [item["id"] for item in items if item["family"] == "parity"][:2]
    for row in baseline:
        if row["item"] in targets:
            row.update(ranker=row["answers"][1].strip(), exact=row["answers"][1].strip())
    assert qualify(baseline, config["qualification"])["disposition"] == "PASS"
    verified = compare(baseline, candidate, min_gain=1, max_regressions=0)
    assert verified["disposition"] == "PASS"
    assert verified["exact_item_gain"] == 2
    sentinel = next(row for row in candidate if row["preserve"])
    sentinel["exact"] = None
    rejected = compare(baseline, candidate, min_gain=1, max_regressions=0)
    assert rejected["disposition"] == "FAIL"
    assert rejected["preservation_changes"] == [sentinel["item"]]
