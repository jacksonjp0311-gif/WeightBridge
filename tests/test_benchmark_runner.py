import json
from pathlib import Path

import pytest

from benchmarks.runners.run_hf_benchmark import extract_answer, normalize_answer, read_dataset, run
from weightbridge.evaluation.loops import loop_signals


def test_loop_signals_detects_repeated_terminal_block_and_short_text_is_false():
    assert loop_signals("one two three") == {"verbatim_loop_end": False, "repetitive_end": False}
    repeated = " ".join(["alpha beta gamma delta"] * 12 + ["cut"])
    assert loop_signals(repeated) == {"verbatim_loop_end": True, "repetitive_end": True}


def test_math_answer_helpers_use_last_boxed_value():
    assert extract_answer("Earlier \\boxed{1}, final \\boxed{\\frac{2}{2}}") == "\\frac{2}{2}"
    assert normalize_answer(" 1, 234 ") == "1234"


def test_read_dataset_accepts_official_ifeval_row_without_answer(tmp_path: Path):
    dataset = tmp_path / "ifeval.json"
    dataset.write_text(json.dumps({"split": "holdout", "items": [{
        "id": "ifeval-1", "prompt": "Say hello", "kind": "ifeval",
        "instruction_id_list": ["keywords:existence"], "kwargs": [{"keywords": ["hello"]}],
    }]}), encoding="utf-8")
    rows, digest = read_dataset(dataset, "holdout")
    assert len(digest) == 64
    assert rows[0]["kind"] == "ifeval"
    assert rows[0]["instruction_id_list"] == ["keywords:existence"]
    assert "answer" not in rows[0]


def test_read_dataset_rejects_unpaired_ifeval_instruction_metadata(tmp_path: Path):
    dataset = tmp_path / "bad_ifeval.json"
    dataset.write_text(json.dumps([{"id": "x", "prompt": "say hi", "kind": "ifeval",
                                    "instruction_id_list": ["a"], "kwargs": []}]), encoding="utf-8")
    with pytest.raises(ValueError, match="matched"):
        read_dataset(dataset, "practice")


def test_benchmark_runner_preserves_model_load_failure(tmp_path: Path):
    dataset = tmp_path / "items.json"
    dataset.write_text(json.dumps([{"id": "x", "prompt": "Compute 1 + 1", "answer": "2"}]),
                       encoding="utf-8")
    output = tmp_path / "failed-run"
    with pytest.raises(Exception):
        run(tmp_path / "missing-model", dataset, output, variant="test", split="practice",
            n=1, seed=0, max_new_tokens=8, max_input_tokens=64, temperature=0.6, top_p=0.95,
            top_k=20, presence_penalty=1.5, timeout_seconds=30, device="cpu", dtype="float32",
            cpu_threads=1, thinking_enabled=True)
    failure = json.loads((output / "run_failure.json").read_text(encoding="utf-8"))
    assert failure["schema"] == "weightbridge.hf_benchmark_failure.v1"
    assert failure["exception_type"]
