import hashlib
import json
import sys
from types import SimpleNamespace

from benchmarks.runners.prepare_exp8_datasets import prepare
from benchmarks.runners.run_hf_benchmark import ifeval_correct, math_verify_correct


def test_dataset_preparer_hashes_and_freezes_gsm8k_rows(tmp_path, monkeypatch):
    rows = [{"question": "What is 1 + 1?", "answer": "work #### 2"}]
    class FakeDataset(list):
        info = SimpleNamespace(version="1")
        _fingerprint = "fingerprint-1"

    dataset = FakeDataset(rows)

    def fake_load_dataset(dataset_id, *args, **kwargs):
        assert dataset_id == "openai/gsm8k"
        assert args == ("main",)
        assert kwargs["split"] == "test"
        assert kwargs["revision"] == "commit123"
        return dataset

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    out = tmp_path / "prepared"
    manifest = prepare(out, gsm8k_revision="commit123", selected=("gsm8k",))
    raw = (out / "gsm8k.json").read_bytes()
    document = json.loads(raw)
    assert manifest["datasets"]["gsm8k"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert document["items"][0]["answer"] == "2"
    assert document["items"][0]["prompt"] == "What is 1 + 1?"
    assert document["instruction_suffix"].endswith("\\boxed{}.")


def test_math_verify_grader_uses_boxed_parser_and_verifier(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        parse=lambda text, parsing_timeout: calls.append((text, parsing_timeout)) or text,
        verify=lambda gold, prediction, timeout_seconds: gold.split("{")[-1] == prediction.split("{")[-1],
    )
    monkeypatch.setitem(sys.modules, "math_verify", fake)
    assert math_verify_correct("2", "2") is True
    assert calls == [("\\boxed{2}", 5), ("\\boxed{2}", 5)]
    assert math_verify_correct(None, "2") is False


def test_ifeval_grader_uses_google_instruction_registry_and_thinking_answer(monkeypatch):
    class Contains:
        def __init__(self, _identifier):
            pass
        def build_description(self, **_kwargs):
            pass
        def get_instruction_args(self):
            return {}
        def check_following(self, response):
            return "hello" in response

    monkeypatch.setitem(sys.modules, "instructions_registry",
                        SimpleNamespace(INSTRUCTION_DICT={"contains": Contains}))
    row = {"prompt": "Say hello", "instruction_id_list": ["contains"], "kwargs": [{}]}
    assert ifeval_correct(row, "thinking text</think>hello", thinking_enabled=True) == (True, [True])
    assert ifeval_correct(row, "hello", thinking_enabled=True) == (False, [False])
