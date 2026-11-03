"""Regression tests for stored precision and a generative veto on margin gains."""
from hashlib import sha256
from math import log
from types import SimpleNamespace

import pytest

from weightbridge.adaptation.endpoint_update import _distribution_report, _live_probe, _tensor_bytes, fit_endpoint_probes


torch = pytest.importorskip("torch")


class Tokenizer:
    eos_token_id = 0
    pad_token_id = 7
    alphabet = "\na bcdef"

    def encode(self, text, *, add_special_tokens=False):
        return [self.alphabet.index(token) for token in text]

    def decode(self, tokens, *, skip_special_tokens=False):
        return "".join(self.alphabet[token] for token in tokens)


class CausalModel(torch.nn.Module):
    def __init__(self, off_option):
        super().__init__()
        self.affine = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        tokenizer = Tokenizer()
        a, b, c, d, e = tokenizer.encode("abcde")
        with torch.no_grad():
            self.affine.weight.fill_(.5)
            self.affine.weight[c, a] = .625
            self.affine.weight[d, a] = .875
            self.affine.weight[e, a] = 1.5 if off_option else .5
            self.affine.weight[c, b] = .875

    def forward(self, input_ids, attention_mask, use_cache=False, logits_to_keep=0):
        encoded = torch.nn.functional.one_hot(input_ids, num_classes=8).to(self.affine.weight.dtype)
        return SimpleNamespace(logits=self.affine(encoded[:, -logits_to_keep:]))


def fixture(off_option, max_forward_calls=32):
    tokenizer = Tokenizer()
    model = CausalModel(off_option).eval()
    runner = SimpleNamespace(model=model, tokenizer=tokenizer, device="cpu")
    weight = model.affine.weight
    raw = _tensor_bytes(weight, torch)
    digest = sha256(raw).hexdigest()
    reference = SimpleNamespace(shape=tuple(weight.shape), dtype="BF16", nbytes=len(raw))
    checkpoint = SimpleNamespace(tensors={"affine.weight": reference}, read_raw=lambda *_a, **_kw: raw,
                                 checkpoint_sha256=lambda: digest)
    plan = SimpleNamespace(method="constrained", checkpoint_sha256=digest, max_columns=1,
                           region=SimpleNamespace(tensor_name="affine.weight", module_name="affine"),
                           limits=SimpleNamespace(max_candidates=3, max_update_norm=.4, max_memory_bytes=2**40,
                                                  max_update_entries=8, max_rank=1, max_forward_calls=max_forward_calls,
                                                  max_backward_calls=2))
    rows = [{"id": "target", "prompt": "a", "answers": ["c", "d"], "preserve": False},
            {"id": "control", "prompt": "b", "answers": ["c", "d"], "preserve": True}]
    wanted = tokenizer.encode("c")[0]

    def measure():
        readings = []
        for row in rows:
            prefix = torch.tensor([tokenizer.encode(row["prompt"])])
            first = int(model(prefix, torch.ones_like(prefix)).logits[0, -1].argmax())
            continuation = torch.tensor([[first]])
            second = int(model(continuation, torch.ones_like(continuation)).logits[0, -1].argmax())
            readings.append({"id": row["id"], "generated_token_ids": [first, second],
                             "completed": second == tokenizer.eos_token_id,
                             "correct": first == wanted and second == tokenizer.eos_token_id})
        return {"rows": readings}

    with torch.inference_mode():
        baseline = measure()

    def evaluate():
        current = measure()
        gain = int(current["rows"][0]["correct"]) - int(baseline["rows"][0]["correct"])
        return {"guard": {"passed": gain > 0 and current["rows"][1]["correct"], "target_item_gain": gain},
                "measurement": current}

    return SimpleNamespace(checkpoint=checkpoint), runner, plan, rows, baseline, evaluate, raw


def test_realized_dose_margin_pass_cannot_emit_patch_when_generation_fails():
    adapter, runner, plan, rows, baseline, evaluate, raw = fixture(off_option=True)
    patches, report = fit_endpoint_probes(adapter, runner, plan, rows, evaluate_candidate=evaluate,
                                          baseline_endpoint=baseline, doses=(1e-9, .1, .3),
                                          preservation_prefix_rows=(1,), batch_size=4)
    noop, smaller, large = report["probes"]
    assert noop["reason"] == "no_stored_change"
    assert smaller["margin"]["surrogate_gain"] > 0
    assert large["margin"]["passed"] is True
    assert large["endpoint"]["guard"]["passed"] is False
    assert large["distribution"]["positions"] == 2
    assert large["distribution"]["mean_kl"] == 0
    assert large["restored_exactly"] is True
    assert report["selected"] == {"margin": "dose-3", "endpoint": None, "distribution": None}
    assert report["resources"]["total_numerical_forward_calls"] == 6
    assert report["resources"]["backward_calls"] == 2
    assert _tensor_bytes(runner.model.affine.weight, torch) == raw
    assert not patches


def test_endpoint_pass_emits_exact_stored_patch_and_token_divergence():
    adapter, runner, plan, rows, baseline, evaluate, raw = fixture(off_option=False)
    patches, report = fit_endpoint_probes(adapter, runner, plan, rows, evaluate_candidate=evaluate,
                                          baseline_endpoint=baseline, doses=(.3,),
                                          preservation_prefix_rows=(1,), batch_size=4)
    assert report["selected"] == dict.fromkeys(("margin", "endpoint", "distribution"), "dose-1")
    (probe,) = report["probes"]
    assert probe["first_divergences"] == [{"id": "target", "position": 0, "parent_token_id": 5,
                                           "candidate_token_id": 4, "parent_phase": "first_token"}]
    (edit,) = patches["dose-1"]["slices"]
    patched = runner.model.affine.weight.detach().clone()
    patched[:, edit["indices"]] = torch.as_tensor(edit["values"], dtype=patched.dtype).T
    assert sha256(_tensor_bytes(patched, torch)).hexdigest() == probe["candidate_tensor_sha256"]
    assert _tensor_bytes(runner.model.affine.weight, torch) == raw


def test_exact_live_restore_after_decoder_error_including_unedited_columns():
    weight = CausalModel(False).affine.weight
    raw = _tensor_bytes(weight, torch)
    original = weight.detach().clone()
    intended = original.clone()
    intended[:, 1] = .25
    with pytest.raises(RuntimeError, match="decoder failed"):
        with _live_probe(weight, [1], intended[:, [1]].double(), _tensor_bytes(intended, torch), raw, torch):
            assert torch.equal(weight, intended)
            weight[:, 3] = .125
            raise RuntimeError("decoder failed")
    assert _tensor_bytes(weight, torch) == raw


def test_forward_budget_refuses_before_live_evaluation():
    adapter, runner, plan, rows, baseline, _, raw = fixture(off_option=False, max_forward_calls=1)

    def evaluate():
        pytest.fail("budget refusal must precede the endpoint")

    patches, report = fit_endpoint_probes(adapter, runner, plan, rows, evaluate_candidate=evaluate,
                                          baseline_endpoint=baseline, doses=(.1, .3), batch_size=4)
    assert report["reason"] == "insufficient_solver_budget"
    assert report["resources"]["total_numerical_forward_calls"] == 0
    assert _tensor_bytes(runner.model.affine.weight, torch) == raw
    assert not patches


def test_full_distribution_kl_is_parent_to_candidate():
    parent = torch.tensor([[.8, .2]], dtype=torch.float64).log()
    candidate = torch.tensor([[.5, .5]], dtype=torch.float64).log()
    report = _distribution_report((("control", (1,), (0,)),), (parent,), (candidate,), Tokenizer())
    assert report["mean_kl"] == pytest.approx(.8 * log(.8 / .5) + .2 * log(.2 / .5))
    assert report["prefixes"][0]["positions"][0]["parent_top"][0]["token_id"] == 0
