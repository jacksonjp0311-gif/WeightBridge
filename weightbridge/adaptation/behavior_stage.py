"""Host boundary for a measurement-qualified single-cycle comparison."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from ..core import file_hash, json_dump_atomic
from ..inspection.safetensors import load_checkpoint
from ..patches import apply_safetensors_patch


@dataclass(frozen=True)
class StageOneAdmission:
    base_sha256: str
    measurement_report_sha256: str
    protocol_sha256: str
    scoring: str | None
    families: tuple[str, ...]
    practice_scope: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]
    reasons: tuple[str, ...]

    @property
    def admitted(self) -> bool:
        return not self.reasons


def stage_one_admission(protocol_path, report_path) -> StageOneAdmission:
    """Bind the trusted measurement decision without reading an adaptation holdout."""
    protocol_path, report_path = Path(protocol_path), Path(report_path)
    protocol_hash = file_hash(protocol_path)
    protocol = json.loads(protocol_path.read_bytes())
    report = json.loads(report_path.read_bytes())
    if (protocol.get("schema") != "weightbridge.phase_iiic.measurement_protocol.v1"
            or report.get("schema") != "weightbridge.phase_iiic.measurement_report.v1"
            or report.get("protocol_sha256") != protocol_hash
            or report.get("base_sha256") != protocol.get("model_sha256")
            or not isinstance(report.get("base_sha256"), str)):
        raise ValueError("Measurement report does not bind the frozen protocol and base")
    measurement = report.get("qualification_gate", {})
    headroom = report.get("headroom_gate", {})
    scoring = report.get("selected_rule")
    families = tuple(headroom.get("eligible_families", ()))
    reasons = []
    if measurement.get("passed") is not True or scoring not in ("sum", "mean"):
        reasons.append("MEASUREMENT_NOT_QUALIFIED")
    if headroom.get("passed") is not True or not families:
        reasons.append("NO_AUTHORIZED_HEADROOM")
    scope = []
    if not reasons:
        for family in families:
            profile = headroom["by_family"][family]["practice"]
            scope.append((family, tuple(profile["persistent_error_ids"]), tuple(profile["cross_family_control_ids"])))
    return StageOneAdmission(report["base_sha256"], file_hash(report_path), protocol_hash,
                             scoring, families, tuple(scope), tuple(reasons))


def _practice_items(rows):
    counts = {}
    for row in rows:
        item = row["item_id"]
        counts[item] = counts.get(item, 0) + 1
    if not counts or any(count != 4 for count in counts.values()):
        raise ValueError("Stage 1 requires four authorized practice renderings per item")
    return set(counts)


def compile_stage_one_candidate(
    admission: StageOneAdmission, *, arm, base, output, family, targets, preservation, tensor_names,
    inference, policy, radius=0.25,
):
    """Save one policy-checked candidate from authorized practice rows, or abstain.

    Each target has a content and a letter rendering of the same example.
    The host supplies a count-matched task-aware preservation set for Arm D.
    The model-selection and independent evaluation sets stay with the verifier.
    """
    if not admission.admitted or family not in admission.families:
        raise ValueError("Stage 1 requires qualified measurement and authorized headroom")
    if arm not in ("A", "B", "C", "D"):
        raise ValueError("Stage 1 arm must be A, B, C, or D")
    if load_checkpoint(base).checkpoint_sha256() != admission.base_sha256:
        raise ValueError("Stage 1 base differs from the qualified checkpoint")
    if set(preservation) != {"generic", "task_aware"} or len(preservation["generic"]) != len(preservation["task_aware"]):
        raise ValueError("Stage 1 requires matched generic and task-aware preservation row counts")
    target_ids, control_ids = {entry[0]: entry[1:] for entry in admission.practice_scope}[family]
    if (_practice_items(targets) != set(target_ids)
            or any(not _practice_items(rows).issubset(control_ids) for rows in preservation.values())):
        raise ValueError("Stage 1 rows differ from the authorized practice scope")
    from .content_update import propose_content_update

    output = Path(output)
    output.mkdir(parents=True)
    view = "letter" if arm in ("A", "B") else "content"
    protected = preservation["task_aware" if arm == "D" else "generic"]
    patch, diagnostics = propose_content_update(
        base, [row[view] for row in targets], [row[view] for row in protected], tensor_names,
        inference, policy=policy, scoring="sum" if view == "letter" else admission.scoring,
        guard=arm != "A", radius=radius,
    )
    result = {"arm": arm, "base_sha256": admission.base_sha256,
              "measurement_report_sha256": admission.measurement_report_sha256, "compiler": diagnostics,
              "disposition": "ABSTAINED"}
    if patch is not None:
        json_dump_atomic(output / "patch.json", patch)
        receipt = apply_safetensors_patch(base, patch, policy, output / "model")
        result.update(receipt=receipt, disposition="PRECISION_NOOP")
        if receipt["changed_entries"]:
            if receipt["changed_entries"] != diagnostics["changed_entries"]:
                raise ValueError("Persisted change count differs from the measured rounded probe")
            checked = _remeasure(output / "model", [row[view] for row in [*targets, *protected]],
                                 inference, "sum" if view == "letter" else admission.scoring,
                                 diagnostics["selected_probe"]["observed_margins"])
            if load_checkpoint(output / "model").checkpoint_sha256() != receipt["candidate_sha256"]:
                raise ValueError("Persisted candidate changed during practice remeasurement")
            result.update(practice_reload=checked, disposition="PENDING_SELECTION" if checked["matched"]
                          else "PRACTICE_RELOAD_MISMATCH")
    json_dump_atomic(output / "candidate.json", result)
    return result


def _remeasure(candidate, rows, inference, scoring, measured):
    from .bic import _runner
    from .content_scoring import prepare_continuations, score_continuations

    with _runner(candidate, inference) as (runner, _):
        continuations = prepare_continuations(runner.tokenizer, [row["prompt"] for row in rows],
                                              [row["answers"] for row in rows],
                                              max_input_tokens=inference["max_input_tokens"])
        scores, resources = score_continuations(runner, continuations, batch_size=inference["batch_size"])
    key = "mean_log_probability" if scoring == "mean" else "sum_log_probability"
    margins = [a[key] - b[key] for a, b in zip(scores[::2], scores[1::2], strict=True)]
    maximum = max(abs(expected - actual) for expected, actual in zip(measured, margins, strict=True))
    return {"matched": maximum <= 1e-6, "max_absolute_margin_error": maximum,
            "margins": margins, "resources": resources}


def compare_behavior_rows(before, after):
    """Evaluate paired likelihood and completed generation on fixed item renderings.

    Expected row fields are id, item_id, role, correct, scored and generated.
    Generated is an answer index or None for an incomplete/unparsed response.
    A verified gain requires both measurements on every rendering of an item.
    """
    identities = ("id", "item_id", "role", "correct")
    item_results = {}
    seen, target_regressions, preservation_changes = set(), 0, 0
    preservation_rows = 0
    for parent, candidate in zip(before, after, strict=True):
        if (any(parent[key] != candidate[key] for key in identities) or parent["id"] in seen
                or parent["role"] not in ("target", "preservation")):
            raise ValueError("Behavior rows do not match the frozen evaluation order and roles")
        seen.add(parent["id"])
        if any(type(row[key]) is not int or row[key] not in (0, 1)
               for row in (parent, candidate) for key in ("correct", "scored")):
            raise ValueError("Behavior rows need binary scored choices and labels")
        if any(row["generated"] is not None and (type(row["generated"]) is not int or row["generated"] not in (0, 1))
               for row in (parent, candidate)):
            raise ValueError("Generated choices must be binary or unparsed")
        label = parent["correct"]
        if parent["role"] == "preservation":
            preservation_rows += 1
            preservation_changes += int(parent["generated"] is None or candidate["generated"] is None
                                        or any(parent[key] != candidate[key] for key in ("scored", "generated")))
        else:
            target_regressions += int(any(parent[key] == label and candidate[key] != label
                                          for key in ("scored", "generated")))
            item_results.setdefault(parent["item_id"], []).append(tuple(
                (row["scored"] == label and row["generated"] == label, row["generated"] == label)
                for row in (parent, candidate)
            ))
    if not item_results or not preservation_rows or any(len(renderings) != 4 for renderings in item_results.values()):
        raise ValueError("Verification needs four target renderings per item and preservation rows")
    joint, generated = [[sum(all(pair[index][metric] for pair in renderings) for renderings in item_results.values())
                         for index in (0, 1)] for metric in (0, 1)]
    old, new = joint
    return {"target_items": len(item_results), "parent_correct_items": old, "candidate_correct_items": new,
            "joint_item_gain": new - old, "generated_item_gain": generated[1] - generated[0],
            "target_regression_rows": target_regressions,
            "preservation_rows": preservation_rows, "preservation_changed_rows": preservation_changes,
            "passed": new > old and generated[1] > generated[0] and target_regressions == 0
            and preservation_changes == 0}
