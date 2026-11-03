import json
from contextlib import nullcontext
from datetime import date
from fractions import Fraction
from types import SimpleNamespace

import pytest

from experiments.phase_iiie import qwen_measure
from experiments.phase_iiie.qwen_analysis import analyze, classify_headroom, lock, project_row
from experiments.phase_iiie.qwen_measure import _admission, _load_rows
from experiments.phase_iiie.qwen_roster import CONTEXTS, FAMILIES, LABELS, ORDERS, PROTOCOL, RESERVE, SPLITS, freeze


CONFIG = json.loads(PROTOCOL.read_bytes())


def _scores(margin, order):
    return [
        {"sum_log_probability": -5 + (margin / 2 if value == 0 else -margin / 2),
         "mean_log_probability": -5 + (margin / 2 if value == 0 else -margin / 2),
         "answer_tokens": 1, "answer_token_ids": [11 + value]}
        for value in order
    ]


def _observations(*, original=2.0, reversed_question=-2.0, neutral=0.0):
    margins = (original, reversed_question, original, reversed_question, neutral, neutral + 0.1)
    rows = []
    for family in FAMILIES:
        for order in ORDERS:
            for labels in LABELS:
                row = {"item_id": family, "family": family, "order": order, "labels": labels,
                       "content": {key: _scores(value, order) for key, value in zip(CONTEXTS, margins, strict=True)},
                       "letter": {key: _scores(value, order) for key, value in
                                  (("task", original), ("reversed", reversed_question))}}
                if labels == LABELS[0]:
                    row["generation"] = {
                        "task": {"strict_position": order.index(0), "stop_reason": "eos"},
                        "reversed": {"strict_position": order.index(1), "stop_reason": "eos"},
                    }
                rows.append(row)
    return rows


def test_paired_correctness_cannot_bypass_standalone_raw_reversal_gate():
    rows = _observations(original=2.0, reversed_question=1.0, neutral=1.0)
    analysis, projections = analyze(rows, CONFIG)
    paired = analysis["rankers"]["paired"]
    assert paired["joint_accuracy"] == 1.0
    assert paired["order_stability"] == 1.0
    assert paired["label_stability"] == 1.0
    assert paired["paraphrase_stability"] == 1.0
    assert analysis["rankers"]["prior"]["joint_accuracy"] == 1.0
    assert analysis["rankers"]["raw"]["joint_accuracy"] == 0.0
    assert not paired["gate"]["raw_standalone_joint_accuracy"]
    assert not paired["passed"]
    assert analysis["generation"]["pass"]
    selected = lock(analysis, CONFIG)
    report = {**analysis, "headroom": classify_headroom(rows, projections, selected)}
    admission = _admission(dict.fromkeys(SPLITS, report), selected, CONFIG)
    assert admission["disposition"] == "HOLD"
    assert admission["hold_reason"] == "MEASUREMENT"


def test_paired_score_aligns_permutations_and_abstains_at_ties():
    outputs = [project_row(row, 1.5) for row in _observations()]
    assert {p["outcomes"]["paired"][0] for p in outputs} == {(0, 1)}
    assert {p["preference"] for p in outputs} == {0.0}
    ties = project_row(_observations(original=1.0, reversed_question=1.0)[0], 1.5)
    assert ties["outcomes"]["paired"][0] == (None, None)
    assert ties["directional_advantage"] < 0


def test_roster_freeze_crosses_labels_and_orders_and_checks_hash(tmp_path):
    root = tmp_path / "frozen"
    manifest = freeze(root)
    assert set(manifest["artifacts"]) == {*SPLITS, RESERVE}
    seen = set()
    for split, count in zip(SPLITS, (9, 9, 12), strict=True):
        rows = _load_rows(root / f"{split}.json", split, CONFIG, manifest, seen)
        assert len(rows) == count * 4
        first = rows[:4]
        assert len({tuple(row["labels"]) for row in first}) == 2
        assert len({tuple(row["order"]) for row in first}) == 2
        assert all(row["prompts"]["task"] != row["prompts"]["reversed"] for row in rows)
        for row in rows:
            first, second = (row["answers"][row["order"].index(i)] for i in (0, 1))
            if row["family"] == "word-order":
                assert first < second
            elif row["family"] == "date-order":
                canonical_dates = [date.fromisoformat(row["aliases"][row["order"].index(i)][-1])
                                   for i in (0, 1)]
                assert canonical_dates[0] < canonical_dates[1]
            else:
                assert Fraction(first) > Fraction(second)
    with pytest.raises(FileExistsError):
        freeze(root)
    material = root / "practice.json"
    material.write_bytes(material.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Roster hash changed"):
        _load_rows(material, "practice", CONFIG, manifest, set())


def test_archive_replays_practice_lock_and_excludes_reserve(tmp_path, monkeypatch):
    material, run, archive = (tmp_path / name for name in ("material", "run", "archive"))
    freeze(material)
    read = qwen_measure._read

    def never_read_reserve(path):
        assert path.name != f"{RESERVE}.json"
        return read(path)

    def measure(_, rows, config, split):
        templates = {(tuple(row["order"]), tuple(row["labels"])): row for row in _observations()}
        records = []
        for row in rows:
            template = templates[tuple(row["order"]), tuple(row["labels"])]
            records.append({**template, **{key: row[key] for key in ("order", "labels", "answers", "item_id", "family")}})
        return records, {"model_fixture": True}

    monkeypatch.setattr(qwen_measure, "_read", never_read_reserve)
    monkeypatch.setattr(qwen_measure, "_runner", lambda *args: nullcontext((None, None)))
    monkeypatch.setattr(qwen_measure, "_measure", measure)
    monkeypatch.setattr(qwen_measure.importlib.metadata, "version", lambda name: "fixture")
    monkeypatch.setattr(qwen_measure, "load_checkpoint", lambda path: SimpleNamespace(
        root=path, checkpoint_sha256=lambda: CONFIG["model_sha256"],
    ))
    report = qwen_measure.run(tmp_path, material, run)
    assert report["selected"] == "raw"
    assert all(report["admission"]["ranker_gates"].values())
    assert report["admission"]["hold_reason"] == "HEADROOM"
    qwen_measure.archive(material, run, archive)
    assert not (archive / "material" / f"{RESERVE}.json").exists()
    assert read(archive / "report.json") == report

    path = run / "practice-lock.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source bindings"):
        qwen_measure.archive(material, run, tmp_path / "mutated")
