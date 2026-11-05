"""Tool endpoint measurements, paired preservation and frozen behavioral gates.

No fitting, simulator execution outside the bounded task oracle, or checkpoint
promotion lives here. Decimal/general controls retain the historical Q endpoint.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import time

from experiments.phase_iiiq_baseline import measurement as decimal
from weightbridge.adaptation.behavior_endpoint import BehavioralEndpointContract, ParsedOutput, Question
from weightbridge.adaptation.content_scoring import _left_pad, _pad_id
from weightbridge.provenance import peak_rss_bytes


FORMS = {(c, w, o) for c in (0, 1) for w in (0, 1) for o in (0, 1)}
_OUTPUT = {"text", "stop_reason", "generated_token_ids", "prefix_token_ids", "prefix_tokens",
           "native_end_token_id", "completed", "schema_valid", "schema_valid_without_completion",
           "tool_correct", "arguments_correct", "simulator_success", "strict_correct",
           "semantic_correct", "correct", "valid", "status", "parsed_action", "simulator_reason",
           "unexpected_special_token_ids", "whole_json_valid"}


def tool_endpoint(settings):
    """Shared contract representation; token-aware scoring also checks native termination."""
    from weightbridge.adaptation.tool_task import canonical_action, parse_action, validate_row
    def question(row):
        validate_row(row)
        return Question(canonical_action(row["gold_action"]))
    def parse(text, reason):
        action = parse_action(text)
        return ParsedOutput(reason == "eos", canonical_action(action) if action is not None else None)
    return BehavioralEndpointContract(
        "weightbridge.phase_iiis.whole_tool_json.v1", "native",
        "Exactly one complete JSON action; no duplicate keys, extra fields, prose or special-token text",
        "Whole JSON followed by the frozen native assistant terminator; token-budget stops are incomplete",
        parse, question, settings["max_input_tokens"], settings["max_new_tokens"], settings["batch_size"],
        (0, 1), 1)


def _validate_tool_roster(rows):
    from weightbridge.adaptation.tool_task import validate_row
    families = defaultdict(list)
    identifiers = set()
    for row in rows:
        validate_row(row)
        if row["id"] in identifiers:
            raise ValueError("Duplicate tool presentation identity")
        identifiers.add(row["id"])
        families[row["family_id"]].append(row)
    if not families:
        raise ValueError("Tool endpoint roster cannot be empty")
    for group in families.values():
        if len(group) != 8 or {(r["counterfactual"], r["wording"], r["ordering"]) for r in group} != FORMS:
            raise ValueError("Tool families require exactly eight distinct frozen forms")
    return families


def generate_tools(runner, prompts, settings, completion):
    """Unrestricted greedy decoding, preserving every native output token for replay."""
    import torch
    from weightbridge.adaptation.task_completion import native_tool_prefix, validate_native_terminator
    tokenizer = runner.tokenizer
    end, _ = validate_native_terminator(tokenizer, completion)
    if type(end) is not int:
        raise ValueError("Native completion validator must return the frozen terminator token ID")
    prefixes = [tuple(tokenizer.encode(native_tool_prefix(tokenizer, p), add_special_tokens=False)) for p in prompts]
    if not prefixes or any(not 1 <= len(p) <= settings["max_input_tokens"] for p in prefixes):
        raise ValueError("Tool prompt exceeds its frozen native-token budget")
    pad = _pad_id(tokenizer)
    results = []
    batches = steps = positions = 0
    with torch.inference_mode():
        for start in range(0, len(prefixes), settings["batch_size"]):
            group = prefixes[start:start + settings["batch_size"]]
            inputs, mask = _left_pad(torch, group, pad, runner.device)
            outputs = runner.model.generate(input_ids=inputs, attention_mask=mask, do_sample=False, num_beams=1,
                max_new_tokens=settings["max_new_tokens"], use_cache=True, pad_token_id=pad, eos_token_id=end,
                return_dict_in_generate=False)
            if outputs.ndim != 2 or outputs.shape[0] != len(group) or not torch.equal(outputs[:, :inputs.shape[1]], inputs):
                raise ValueError("Tool decoder changed the native prompt prefix")
            suffix = outputs[:, inputs.shape[1]:]
            if not 1 <= suffix.shape[1] <= settings["max_new_tokens"]:
                raise ValueError("Tool decoder violated the output budget")
            batches += 1
            steps += suffix.shape[1]
            positions += inputs.numel()
            for prefix, tokens in zip(group, suffix.tolist(), strict=True):
                ended = end in tokens
                if not ended and len(tokens) != settings["max_new_tokens"]:
                    raise ValueError("Tool generation stopped before native termination or its budget")
                body = tokens[:tokens.index(end)] if ended else tokens
                # Only generation padding after an observed end token may be discarded.
                if ended and any(token != pad for token in tokens[tokens.index(end)+1:]):
                    raise ValueError("Unexpected nonpadding tokens after native termination")
                results.append({"text": tokenizer.decode(body, skip_special_tokens=False),
                    "generated_token_ids": body + ([end] if ended else []), "prefix_token_ids": list(prefix),
                    "prefix_tokens": len(prefix), "native_end_token_id": end,
                    "unexpected_special_token_ids": [t for t in body if t in tokenizer.all_special_ids],
                    "stop_reason": "eos" if ended else "token_budget"})
    return results, {"generate_batches": batches, "generated_examples": len(results),
        "padded_prompt_token_positions": positions, "batched_decode_steps": steps}


def _score(row, output):
    from weightbridge.adaptation.tool_task import score_response
    if output.get("stop_reason") not in ("eos", "token_budget"):
        raise ValueError("Unknown tool completion condition")
    tokens, end = output.get("generated_token_ids"), output.get("native_end_token_id")
    if (not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens)
            or type(end) is not int or end < 0):
        raise ValueError("Tool measurement must retain native output tokens")
    completed = output["stop_reason"] == "eos"
    if (completed and (tokens[-1] != end or end in tokens[:-1])) or (not completed and end in tokens):
        raise ValueError("Tool native termination evidence disagrees with completion status")
    scored = score_response(row, output["text"], completed, stop_reason=output["stop_reason"])
    unexpected = output.get("unexpected_special_token_ids", [])
    if not isinstance(unexpected, list) or any(type(t) is not int or t not in tokens for t in unexpected):
        raise ValueError("Malformed unexpected-special-token evidence")
    if unexpected:
        for key in ("schema_valid", "schema_valid_without_completion", "tool_correct", "arguments_correct",
                    "simulator_success", "strict_correct", "semantic_correct", "correct", "valid"):
            scored[key] = False
        scored["status"] = "unexpected_special_token"
    return scored


def measure_tools(runner, rows, settings, completion, *, decoder=generate_tools):
    _validate_tool_roster(rows)
    wall, cpu = time.perf_counter(), time.process_time()
    runner.model.eval()
    outputs, resources = decoder(runner, [r["prompt"] for r in rows], settings, completion)
    measured = []
    for row, output in zip(rows, outputs, strict=True):
        if set(row) & (_OUTPUT | set(output)):
            raise ValueError("Tool decoder cannot replace frozen task metadata")
        measured.append({**row, **output, **_score(row, output)})
    return {"endpoint": tool_endpoint(settings).manifest(), "completion": dict(completion), "rows": measured,
        "resources": {**resources, "wall_seconds": time.perf_counter()-wall,
                      "cpu_seconds": time.process_time()-cpu, "peak_rss_bytes": peak_rss_bytes()}}


def checked_tools(measurement, rows, settings, completion):
    """Recompute correctness from frozen tasks and recorded whole outputs, never trust booleans."""
    _validate_tool_roster(rows)
    if measurement.get("endpoint") != tool_endpoint(settings).manifest() or measurement.get("completion") != completion:
        raise ValueError("Tool measurement endpoint changed")
    if len(measurement.get("rows", [])) != len(rows):
        raise ValueError("Tool measurement has missing or additional presentations")
    for original, observed in zip(rows, measurement["rows"], strict=True):
        if any(observed.get(k) != v for k, v in original.items()):
            raise ValueError("Tool measurement differs from its frozen task")
        if observed.get("native_end_token_id") != completion["end_token_id"]:
            raise ValueError("Tool measurement uses another native terminator")
        tokens = observed.get("generated_token_ids", [])
        if len(tokens) > settings["max_new_tokens"] or (observed.get("stop_reason") == "token_budget"
                and len(tokens) != settings["max_new_tokens"]):
            raise ValueError("Tool output length violates the frozen decode budget")
        if (not isinstance(observed.get("prefix_token_ids"), list) or not observed["prefix_token_ids"]
                or len(observed["prefix_token_ids"]) != observed.get("prefix_tokens")
                or observed["prefix_tokens"] > settings["max_input_tokens"]):
            raise ValueError("Tool output lacks its bounded native prompt identity")
        expected = _score(original, observed)
        if any(observed.get(k) != v for k, v in expected.items()):
            raise ValueError("Tool correctness or validity does not replay")
    return measurement


def summarize_tools(measurement):
    rows = measurement["rows"]
    families = _validate_tool_roster(rows)
    complete = sorted(key for key, forms in families.items() if all(r["strict_correct"] for r in forms))
    counts = {key: sum(bool(r[key]) for r in rows) for key in
              ("strict_correct", "semantic_correct", "whole_json_valid", "schema_valid", "schema_valid_without_completion",
               "tool_correct", "arguments_correct", "simulator_success", "completed")}
    strata = {}
    for tool in sorted({r["gold_action"]["tool"] for r in rows}):
        group = [r for r in rows if r["gold_action"]["tool"] == tool]
        strata[tool] = {"presentations": len(group), "strict_correct": sum(r["strict_correct"] for r in group),
            "semantic_correct": sum(r["semantic_correct"] for r in group),
            "schema_valid": sum(r["schema_valid"] for r in group)}
    wordings = {}
    for group_id in sorted({r["wording_group"] for r in rows}):
        group = [r for r in rows if r["wording_group"] == group_id]
        wordings[group_id] = {"presentations": len(group),
            "strict_correct": sum(r["strict_correct"] for r in group),
            "semantic_correct": sum(r["semantic_correct"] for r in group),
            "schema_valid": sum(r["schema_valid"] for r in group)}
    familiarity = {}
    for label in sorted({r.get("wording_familiarity", "UNDECLARED") for r in rows}):
        group = [r for r in rows if r.get("wording_familiarity", "UNDECLARED") == label]
        family_ids = {r["family_id"] for r in group}
        familiarity[label] = {"presentations": len(group), "strict_correct": sum(r["strict_correct"] for r in group),
            "semantic_correct": sum(r["semantic_correct"] for r in group),
            "schema_valid": sum(r["schema_valid"] for r in group), "families": len(family_ids),
            "complete_families": len(family_ids & set(complete))}
    return {"presentations": len(rows), **counts, "strict_accuracy": counts["strict_correct"]/len(rows),
        "families": len(families), "complete_families": len(complete), "complete_family_ids": complete,
        "decode_termination": dict(Counter(r["stop_reason"] for r in rows)), "strata": strata,
        "wording_groups": wordings, "wording_familiarity": familiarity}


def _aligned(reference, candidate):
    if reference.get("endpoint") != candidate.get("endpoint") or reference.get("completion") != candidate.get("completion"):
        raise ValueError("Tool comparisons require the identical frozen endpoint")
    if len(reference["rows"]) != len(candidate["rows"]):
        raise ValueError("Tool comparisons require the identical roster")
    pairs = list(zip(reference["rows"], candidate["rows"], strict=True))
    for a, b in pairs:
        # Gold/fixture/prompt fields are all part of the task, never just row IDs.
        if {k: v for k, v in a.items() if k not in _OUTPUT} != {k: v for k, v in b.items() if k not in _OUTPUT}:
            raise ValueError("Tool comparison task identities differ")
    return pairs


def tool_retention(reference, candidate):
    pairs = _aligned(reference, candidate)
    losses = {key: [a["id"] for a, b in pairs if a[key] and not b[key]]
              for key in ("strict_correct", "semantic_correct", "schema_valid", "schema_valid_without_completion")}
    before, after = summarize_tools(reference), summarize_tools(candidate)
    lost_families = sorted(set(before["complete_family_ids"]) - set(after["complete_family_ids"]))
    return {"passed": not any(losses.values()) and not lost_families,
        "lost_strict_correct": losses["strict_correct"], "lost_semantic_correct": losses["semantic_correct"],
        "lost_schema_valid": losses["schema_valid"],
        "lost_schema_valid_without_completion": losses["schema_valid_without_completion"],
        "lost_complete_families": lost_families, "reference": before, "candidate": after}


def headroom_tools(parent, *, real_actions=("search_files", "read_file", "run_tests")):
    summary = summarize_tools(parent)
    errors = [r for r in parent["rows"] if r["schema_valid"] and not r["semantic_correct"]]
    families = {r["family_id"] for r in errors}
    actions = {r["gold_action"]["tool"] for r in errors} & set(real_actions)
    checks = {"ten_point_headroom": summary["strict_accuracy"] <= .90,
        "six_family_headroom": summary["families"] - summary["complete_families"] >= 6,
        "semantic_errors": len(errors) >= 8, "semantic_error_families": len(families) >= 4,
        "semantic_error_real_actions": len(actions) >= 2}
    return {"passed": all(checks.values()), "checks": checks, "schema_valid_semantic_errors": len(errors),
        "error_families": len(families), "real_actions": sorted(actions),
        "disposition": "SUFFICIENT_HEADROOM" if all(checks.values()) else "INSUFFICIENT_IMPROVEMENT_HEADROOM"}


def compare_tools(parent, candidate, *, formal=False, preservation_only=False, required_references=(),
                  real_actions=("search_files", "read_file", "run_tests")):
    """Formal count thresholds apply only to separately frozen independent populations."""
    pairs = _aligned(parent, candidate)
    own = tool_retention(parent, candidate)
    additional = [tool_retention(ref, candidate) for ref in required_references]
    before, after = own["reference"], own["candidate"]
    corrected = [b for a, b in pairs if a["schema_valid"] and not a["semantic_correct"] and b["strict_correct"]]
    families = sorted({r["family_id"] for r in corrected})
    actions = sorted({r["gold_action"]["tool"] for r in corrected} & set(real_actions))
    strict_gain = after["strict_correct"] - before["strict_correct"]
    family_gain = after["complete_families"] - before["complete_families"]
    gain = strict_gain / before["presentations"]
    checks = {"own_parent_retention": own["passed"], "required_comparator_retention": all(x["passed"] for x in additional)}
    if not preservation_only:
        checks.update(accuracy=after["strict_accuracy"] >= .90, ten_percentage_point_gain=gain >= .10 - 1e-12)
    if formal and not preservation_only:
        checks.update(frozen_formal_population=before["families"] == 32 and before["presentations"] == 256,
            six_new_complete_families=family_gain >= 6, eight_semantic_corrections=len(corrected) >= 8,
            four_corrected_semantic_families=len(families) >= 4, two_corrected_real_actions=len(actions) >= 2)
    headroom = headroom_tools(parent, real_actions=real_actions) if formal and not preservation_only else None
    return {"passed": all(checks.values()), "checks": checks, "formal": formal,
        "disposition": ("INSUFFICIENT_IMPROVEMENT_HEADROOM" if headroom is not None and not headroom["passed"] else
                        "PASS" if all(checks.values()) else "FAIL"),
        "strict_response_gain": strict_gain, "strict_accuracy_gain": gain, "complete_family_gain": family_gain,
        "corrected_schema_valid_semantic_errors": [r["id"] for r in corrected],
        "corrected_semantic_families": families, "corrected_real_actions": actions,
        "headroom": headroom, "own_parent_retention": own, "required_retention": additional}


def legacy_retention(references, candidate):
    """Sequential: R8 decimals, original+R8 general controls. Control: original only."""
    if not references:
        raise ValueError("Legacy preservation requires its declared reference checkpoint(s)")
    gates = [decimal.compare(ref, candidate, min_strict_gain=0, min_family_gain=0) for ref in references]
    return {"passed": all(g["passed"] for g in gates), "reference_gates": gates}

