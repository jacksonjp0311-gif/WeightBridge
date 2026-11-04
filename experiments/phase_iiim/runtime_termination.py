"""Measure a runtime-only termination preference on the unchanged fit parent."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import copy
import hashlib
import json
from pathlib import Path
import time

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiih.run_decimal_endpoint import endpoint
from experiments.phase_iiil.semantic_diagnostics import compare_semantic, summarize
from experiments.phase_iiim.semantic_roster import PROTOCOL, load_split, read_manifest
from experiments.phase_iiim.source_manifest import ROOT, source_hashes
from weightbridge.adaptation.behavior_endpoint import _observation, measure_endpoint
from weightbridge.adaptation.endpoint_update import _tensor_bytes
from weightbridge.adaptation.fork_update import _resources
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint


BIASES = (-.125, .125)
AUTHORIZED_DIAGNOSTIC = (
    "A post-fit, unmodified-parent diagnostic may apply fixed +0.125 or -0.125 logit bias to EOS/newline only, "
    "with at most two adaptation endpoint evaluations. It supplies no candidate selection evidence and no persistent changes."
)


def _read(path):
    return json.loads(Path(path).read_bytes())


def _verified_parent(material, fit_dir, config, contract):
    manifest = read_manifest(material)
    fit_path, parent_path = fit_dir / "fit-report.json", fit_dir / "parent-adaptation.json"
    fit, parent = _read(fit_path), _read(parent_path)["measurement"]
    if (fit.get("schema") != "weightbridge.phase_iiim.fit.v1"
            or fit.get("checkpoint_sha256") != config["checkpoint_sha256"]
            or fit.get("sources") != source_hashes()
            or fit.get("manifest_sha256") != file_hash(material / "manifest.json")
            or fit.get("artifacts", {}).get("parent-adaptation.json") != file_hash(parent_path)
            or parent.get("endpoint") != contract.manifest()):
        raise ValueError("Termination probe requires the source-bound original fit parent")
    frozen = load_split(material, "adaptation", manifest)
    for expected, measured in zip(frozen, parent["rows"], strict=True):
        parsed = _observation(measured, contract.question(expected), contract)
        if any(measured.get(key) != value for key, value in {**expected, **parsed}.items()):
            raise ValueError("Fit parent receipt differs from frozen rows or endpoint parsing")
    return parent, {
        "fit_report_sha256": file_hash(fit_path), "parent_adaptation_sha256": file_hash(parent_path),
        "manifest_sha256": file_hash(material / "manifest.json"),
        "adaptation_roster_sha256": manifest["splits"]["adaptation"]["sha256"],
    }


@contextmanager
def termination_bias(model, token_ids, bias):
    """Clone returned logits and remove the forward hook even when decoding fails."""
    if (not token_ids or len(set(token_ids)) != len(token_ids)
            or any(type(token) is not int or token < 0 for token in token_ids)
            or not isinstance(bias, (int, float)) or not -1 <= bias <= 1):
        raise ValueError("Termination probe requires distinct token IDs and a bounded logit bias")
    counts = {"forward_calls": 0, "logit_positions": 0}

    def adjust(_module, _inputs, output):
        logits = getattr(output, "logits", None)
        if logits is None or logits.ndim != 3 or any(token >= logits.shape[-1] for token in token_ids):
            raise ValueError("Runtime cannot expose the native next-token logits")
        modified = copy(output)
        values = logits.float().clone()
        values[..., list(token_ids)] += bias
        modified.logits = values
        counts["forward_calls"] += 1
        counts["logit_positions"] += values.shape[0] * values.shape[1]
        return modified

    handle = model.register_forward_hook(adjust)
    try:
        yield counts
    finally:
        handle.remove()


def _token_set(tokenizer):
    eos = getattr(tokenizer, "eos_token_id", None)
    if type(eos) is not int or eos < 0:
        raise ValueError("Termination diagnostic needs the frozen endpoint's native EOS token")
    newline = tuple(tokenizer.encode("\n", add_special_tokens=False))
    tokens = tuple(dict.fromkeys((eos, *newline))) if len(newline) == 1 else (eos,)
    return tokens, {"eos_token_id": eos, "newline_token_ids": list(newline),
                    "newline_bias_applied": len(newline) == 1,
                    "biased_tokens": [{"id": token, "text": tokenizer.decode([token], skip_special_tokens=False)}
                                      for token in tokens]}


def run(base, material, fit_dir, out):
    import torch

    if out.exists():
        raise FileExistsError(out)
    wall, cpu = time.perf_counter(), time.process_time()
    config = _read(PROTOCOL)
    contract, sha = endpoint(config), config["checkpoint_sha256"]
    parent, evidence = _verified_parent(material, fit_dir, config, contract)
    script = Path(__file__).resolve()
    sources = {"script": script.relative_to(ROOT).as_posix(), "sha256": file_hash(script), **evidence}
    specification = config["probe_design"].get("optional_runtime_diagnostic")
    if specification != AUTHORIZED_DIAGNOSTIC:
        raise ValueError("The runtime-only +/-0.125 comparison must be prospectively frozen")
    arms = {}
    with _open(base, config, sha) as (adapter, runner, capabilities):
        regions = [region for region in capabilities.editable_regions
                   if region.layer_index == config["probe_design"]["fixed_layer"] and region.role == "ffn_output"]
        if len(regions) != 1:
            raise ValueError("Frozen region does not resolve to one native affine tensor")
        region = regions[0]
        module = runner.model.get_submodule(region.module_name)
        original = _tensor_bytes(module.weight, torch)
        tensor_sha = hashlib.sha256(original).hexdigest()
        if tensor_sha != adapter.checkpoint.tensor_sha256(region.tensor_name):
            raise ValueError("Live parent affine tensor differs from the pinned checkpoint")
        tokens, token_record = _token_set(runner.tokenizer)
        for bias in BIASES:
            start, start_cpu = time.perf_counter(), time.process_time()
            work = {"forward_calls": 0, "logit_positions": 0}
            try:
                with termination_bias(runner.model, tokens, bias) as work:
                    measured = measure_endpoint(runner, load_split(material, "adaptation", read_manifest(material)), contract)
                result = {"measurement": measured, "summary": summarize(measured),
                          "comparison": compare_semantic(parent, measured), "runtime_counts": work,
                          "disposition": "DIAGNOSTIC_COMPLETE"}
            except (ValueError, RuntimeError, OSError) as error:
                result = {"disposition": "HOLD", "error_type": type(error).__name__, "reason": str(error)}
            current = _tensor_bytes(module.weight, torch)
            if current != original:
                raise RuntimeError("Termination diagnostic changed live parent parameters")
            result.update(bias=bias, tensor_sha256_before=tensor_sha,
                          tensor_sha256_after=hashlib.sha256(current).hexdigest(),
                          weights_exactly_preserved=True,
                          resources=_resources(start, start_cpu, work["forward_calls"], 0))
            arms[str(bias)] = result
    if (_verified_parent(material, fit_dir, config, contract)[1] != evidence
            or file_hash(script) != sources["sha256"] or load_checkpoint(base).checkpoint_sha256() != sha):
        raise ValueError("Termination probe sources or original checkpoint changed")
    report = {"schema": "weightbridge.phase_iiim.runtime_termination.v1",
              "scope": "post-fit unchanged-parent diagnostic with runtime logits adjustment",
              "checkpoint_sha256": sha, "source": sources, "tokens": token_record,
              "frozen_diagnostic": specification,
              "baseline": summarize(parent), "arms": arms, "promotion_authority": False,
              "persistent_parameter_changes": 0,
              "resources": _resources(wall, cpu, sum(row["resources"]["forward_calls"] for row in arms.values()), 0)}
    json_dump_atomic(out, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("base", "material", "fit-dir", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base, args.material, args.fit_dir, args.out)
    print(json.dumps({name: {"disposition": value["disposition"],
                             "target_gain": value.get("comparison", {}).get("target_gain")}
                      for name, value in result["arms"].items()}), flush=True)
