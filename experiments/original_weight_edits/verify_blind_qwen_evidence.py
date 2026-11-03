"""Check the frozen plan, patch chain and scores without model weights."""
import argparse
import hashlib
import json
from pathlib import Path

from run_blind_qwen_repair import correct, digest, flips, mean_margin, read


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_receipt(patch, receipt, protocol):
    def canonical_hash(value):
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    require(canonical_hash(patch) == receipt["patch_sha256"], "patch receipt digest differs")
    require(canonical_hash(protocol["policy"]) == receipt["policy_sha256"], "patch policy digest differs")
    require(patch["base_sha256"] == receipt["base_sha256"], "patch base digest differs")
    require(patch["tensor_sha256"] == {name: row["base_sha256"]
            for name, row in receipt["tensor_receipts"].items()}, "tensor base digest differs")


def verify(root, frozen_source=None):
    public, oracle, search, evaluation = (root / name for name in (
        "public", "evaluator", "search", "evaluation"))
    protocol = read(public / "protocol.json")
    if frozen_source is not None:
        require(digest(frozen_source) == protocol["source_sha256"], "frozen runner source differs")
    log = read(search / "search.json")
    summary = read(evaluation / "summary.json")
    require(log["protocol_sha256"] == digest(public / "protocol.json"), "protocol hash mismatch")
    require(log["practice_sha256"] == protocol["practice_sha256"] == digest(public / "practice.json"),
            "practice hash mismatch")
    holdout = read(oracle / "holdout.json")
    require(summary["holdout_sha256"] == protocol["holdout_sha256"] == digest(oracle / "holdout.json"),
            "holdout hash mismatch")
    require(protocol["construction_sha256"] == digest(oracle / "construction.json"),
            "construction hash mismatch")
    partitions = [{row["text"] for rows in group.values() for row in rows} for group in (
        read(oracle / "construction.json"), read(public / "practice.json"), holdout)]
    require(all(not (a & b) for i, a in enumerate(partitions) for b in partitions[i + 1:]),
            "prompt overlap between frozen partitions")
    attempts = log["attempts"]
    require([(row["layer"], row["strength"]) for row in attempts] ==
            [(layer, strength) for layer in protocol["eligible_layers"] for strength in protocol["strengths"]],
            "practice grid does not match protocol")
    eligible = []
    for i, row in enumerate(attempts):
        if "failure" in row:
            continue
        target = correct(row["rows"]["target"]) - correct(log["baseline"]["target"])
        margin = mean_margin(row["rows"]["target"]) - mean_margin(log["baseline"]["target"])
        sentinel = flips(log["baseline"]["sentinel"], row["rows"]["sentinel"])
        require((target, margin, sentinel) == (row["target_gain"], row["target_margin_gain"], row["sentinel_flips"]),
                "stored practice aggregates differ from raw rows")
        if target > 0 and sentinel == 0:
            eligible.append((-target, row["rounded_frobenius_norm"], -margin, i))
    require(bool(eligible) and min(eligible)[-1] == log["selected_attempt"], "repair did not follow practice selection")
    fault = read(oracle / "fault_receipt.json")
    repair = read(search / "repair_receipt.json")
    require(fault["base_sha256"] == read(oracle / "calibration.json")["pristine_checkpoint_sha256"]
            and fault["candidate_sha256"] == repair["base_sha256"]
            and repair["candidate_sha256"] == summary["repair_checkpoint_sha256"],
            "checkpoint receipt chain is invalid")
    names = ["pristine", "damaged", "repaired"] + [f"random_{seed}" for seed in protocol["random_control_seeds"]]
    scores = {name: read(evaluation / f"{name}_scores.json") for name in names}
    for name, groups in scores.items():
        for group, rows in groups.items():
            require([(row["id"], row["label"]) for row in rows] ==
                    [(row["id"], row["label"]) for row in holdout[group]], f"score rows differ for {name}")
            require(all((row["label"] == row["prediction"]) == row["correct"] for row in rows),
                    f"incorrect classification flag for {name}")
    for key, field in (("target", "target_correct"), ("sentinel", "sentinel_correct")):
        require({name: correct(groups[key]) for name, groups in scores.items()} == summary[field],
                f"score summary differs for {key}")
    target = summary["target_correct"]
    lost = target["pristine"] - target["damaged"]
    recovered = (target["repaired"] - target["damaged"]) / lost if lost > 0 else None
    unrelated = flips(scores["damaged"]["sentinel"], scores["repaired"]["sentinel"])
    require(recovered == summary["lost_accuracy_recovered"]
            and unrelated == summary["sentinel_prediction_flips_vs_damaged"], "recovery summary differs")
    control_names = names[3:]
    repair_norm = attempts[log["selected_attempt"]]["rounded_frobenius_norm"]
    repair_doc = read(search / "repair_patch.json")
    repair_slice = repair_doc["slices"][0]
    check_receipt(repair_doc, repair, protocol)
    check_receipt(read(oracle / "fault_patch.json"), fault, protocol)
    for name in control_names:
        patch = read(evaluation / f"{name}_patch.json")
        receipt = read(evaluation / f"{name}_receipt.json")
        check_receipt(patch, receipt, protocol)
        require(patch["base_sha256"] == receipt["base_sha256"] == repair["base_sha256"],
                f"random control uses a different base for {name}")
        require(all(patch["slices"][0][key] == repair_slice[key] for key in ("tensor", "axis", "indices")),
                f"random support differs for {name}")
        norm = summary["matched_random_controls"][name]["rounded_frobenius_norm"]
        require(abs(norm - repair_norm) / repair_norm <= 0.05, f"random norm differs for {name}")
    passed = (recovered is not None and recovered >= protocol["minimum_recovery"]
              and target["repaired"] > max(target[name] for name in control_names)
              and unrelated <= protocol["max_unrelated_prediction_flips_vs_damaged"])
    expected = "PASS" if passed else "FAIL" if lost > 0 else "INCONCLUSIVE"
    require(expected == summary["disposition"], "decision gate replay differs")
    return {"disposition": expected, "target_correct": target, "sentinel_flips": unrelated,
            "lost_accuracy_recovered": recovered}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--frozen-source", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.evidence, args.frozen_source), sort_keys=True))
