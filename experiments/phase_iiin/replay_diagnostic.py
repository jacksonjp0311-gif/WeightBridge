"""Bounded parent-only audit of diagnostic versus gradient batch replay.

No gradient is fitted and no candidate is constructed. Only the already
authorized adaptation diagnostic is read. At most 48 forward batches are used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import time

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiih.run_decimal_endpoint import endpoint, read_json, _resources
from experiments.phase_iiii.run_causal_fork import _plan
from experiments.phase_iiin.run_paired import _restore_fork
from experiments.phase_iiin.semantic_roster import PROTOCOL
from weightbridge.adaptation.behavior_endpoint import endpoint_runner
from weightbridge.adaptation.fork_update import _only_train, _region
from weightbridge.adaptation.generative_fork import next_token_logits
from weightbridge.adaptation.semantic_update import _item_batches
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint


def run(base, trial, out):
    import torch

    if out.exists():
        raise FileExistsError(out)
    wall, cpu = time.perf_counter(), time.process_time()
    config = read_json(PROTOCOL)
    diagnosis = read_json(trial / "fork-diagnosis.json")
    selection = read_json(trial / "supervision-selection.json")
    ids = tuple(dict.fromkeys(item for arm in selection["arms"].values() for item in arm["comparison_ids"]))
    parent = read_json(trial / "parent-adaptation.json")["measurement"]
    lookup = {r["id"]: r for r in parent["rows"]}
    semantic = [_restore_fork(r) for r in diagnosis["forks"] if r["status"] == "semantic"]
    wanted = {p for row in selection["population"] if row["comparison_id"] in ids
              for p in row["eligible_error_prompt_ids"]}
    items = defaultdict(list)
    for item in ids:
        items[item] = [f for f in semantic if f.prompt_id in wanted and lookup[f.prompt_id]["comparison_id"] == item]
    batch_size = config["inference"]["batch_size"]
    original = [semantic[i:i + batch_size] for i in range(0, len(semantic), batch_size)
                if any(f.prompt_id in wanted for f in semantic[i:i + batch_size])]
    packed = [[f for _, members in batch for f in members] for batch in _item_batches(items, batch_size)]
    if len(original) * 2 + len(packed) > 48:
        raise ValueError("Replay diagnostic forward budget exceeded before execution")
    observations, calls = {}, 0
    with _open(base, config, config["checkpoint_sha256"], gradient_probe=True) as (adapter, native, capabilities):
        runner = endpoint_runner(native, endpoint(config))
        plan = _plan(config, capabilities, config["probe_design"]["fixed_layer"])
        _, module, raw = _region(adapter, runner, plan, torch)

        def measure(name, batches):
            nonlocal calls
            rows = []
            for batch in batches:
                logits = next_token_logits(runner, [f.prefix_ids for f in batch])
                calls += 1
                for fork, row in zip(batch, logits, strict=True):
                    if fork.prompt_id not in wanted:
                        continue
                    values, tokens = row.float().topk(5)
                    predicted = int(row.argmax())
                    rows.append({"prompt_id": fork.prompt_id, "original_token": fork.actual_token_id,
                        "predicted_token": predicted, "matched": predicted == fork.actual_token_id,
                        "accepted": predicted in fork.accepted_token_ids,
                        "batch_ids": [f.prompt_id for f in batch],
                        "top_tokens": tokens.tolist(), "top_logits": values.tolist()})
            observations[name] = rows

        with torch.inference_mode():
            measure("original_diagnostic_packing", original)
            measure("comparison_packing", packed)
        with _only_train(runner.model, (module.weight,)):
            measure("original_packing_gradient_enabled", original)
        from weightbridge.adaptation.endpoint_update import _tensor_bytes
        unchanged = _tensor_bytes(module.weight, torch) == raw
    if not unchanged or load_checkpoint(base).checkpoint_sha256() != config["checkpoint_sha256"]:
        raise ValueError("Parent-only replay audit changed checkpoint identity")
    result = {"schema": "weightbridge.phase_iiin.replay_diagnostic.v1",
        "source_sha256": file_hash(__file__), "source_diagnosis_sha256": file_hash(trial / "fork-diagnosis.json"),
        "checkpoint_sha256": config["checkpoint_sha256"], "observations": observations,
        "mismatches": {k: sum(not r["matched"] for r in v) for k, v in observations.items()},
        "parent_unchanged": unchanged, "candidate_count": 0, "backward_calls": 0,
        "resources": {**_resources(wall, cpu), "forward_calls": calls, "backward_calls": 0}}
    json_dump_atomic(out, result)
    print(result["mismatches"], flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("base", "trial", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.trial, args.out)
