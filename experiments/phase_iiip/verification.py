"""Isolated optimized-right bindings for the established host verification stages.

Only trusted repository functions are rebound. Adaptation documents contain data,
never executable code. Track B is absent from this verifier's data and authority.
"""
from __future__ import annotations

import types
import zipfile
import numpy as np

from experiments.phase_iiio import verification as _parent_verifier


def _inherit(namespace):
    # Local namespace identity stays stable while imported globals are copied.
    # This avoids mutating the old module or accidentally following its own
    # inherited-module alias partway through binding.
    for name, value in tuple(namespace.items()):
        if name.startswith("__"):
            continue
        if isinstance(value, types.FunctionType) and value.__globals__ is namespace:
            copied = types.FunctionType(value.__code__, globals(), name, value.__defaults__, value.__closure__)
            copied.__kwdefaults__ = value.__kwdefaults__
            globals()[name] = copied
        else:
            globals()[name] = value


_inherit(vars(_parent_verifier))

from experiments.phase_iiip.semantic_roster import PROTOCOL, load_split, read_manifest
from experiments.phase_iiip.source_manifest import ROOT, source_hashes
from experiments.phase_iiip.semantic_metrics import compare
from weightbridge.adaptation.right_factor_geometry import canonical_factors

LOCK_SCHEMA = "weightbridge.phase_iiip.stage_lock.v1"


def _blocks(path):
    """Read bounded numerical data only; no pickle or unbounded ZIP expansion."""
    with zipfile.ZipFile(path) as archive:
        if sum(member.file_size for member in archive.infolist()) > 32 * 1024 * 1024:
            raise ValueError("Selected-block derivative archive exceeds its memory bound")
    with np.load(path, allow_pickle=False) as archive:
        if not {"targets", "preservation"} <= set(archive.files) or set(archive.files) - {"targets", "preservation", "calibration"}:
            raise ValueError("Selected-block derivative archive has unknown or missing arrays")
        result = {key: np.asarray(archive[key], dtype=np.float64) for key in archive.files}
    shape = result["targets"].shape
    if (len(shape) != 3 or not 1 <= shape[0] <= 16 or not 1 <= shape[1] <= 1024 or not 1 <= shape[2] <= 16
            or any(value.ndim != 3 or value.shape[1:] != shape[1:] or len(value) > 64
                   or not np.isfinite(value).all() for value in result.values())):
        raise ValueError("Selected-block derivatives exceed the authorized finite dimensions")
    return result


def _native_candidate(record, info, basis, protected, blocks, config, required_floor):
    """Independently project the actual per-arm rank-one factor through every row."""
    probe = record["probe"]
    native = probe.get("native_measurement", {})
    metadata = basis["forks"]
    required = {row["prompt_id"]: row for row in metadata}
    left, right = np.asarray(record["left_factor"], dtype=float), np.asarray(record["right_factor"], dtype=float)
    shape = blocks["targets"].shape
    limit = config["intent"]["limits"]["max_update_norm"]
    if (len(required) != len(metadata) or len(metadata) != shape[0]
            or left.shape != (shape[1],) or right.shape != (shape[2],)
            or not np.isfinite(left).all() or not np.isfinite(right).all()
            or not np.isclose(np.linalg.norm(right), 1., rtol=1e-10, atol=1e-10)
            or right[int(np.argmax(np.abs(right)))] < 0
            or (record["arm"].startswith("A") and not np.array_equal(right, canonical_factors(np.ones(1), basis["right_factor"])[1]))
            or not 0 < np.linalg.norm(left) <= limit):
        raise ValueError("Candidate factors do not bind the bounded canonical rank-one geometry")
    predicted = np.array([row["parent_margin"] for row in metadata]) + np.einsum("ijk,j,k->i", blocks["targets"], left, right)
    targets = native.get("targets", {}).get("rows", ())
    if (native.get("passed") is not True or native.get("restored_exactly") is not True
            or native.get("tensor_sha256") != info["tensor_sha256"]
            or {row["prompt_id"] for row in targets} != set(required) or len(targets) != len(required)
            or not np.isfinite(predicted).all() or np.any(predicted < config["probe_design"]["target_margin"])
            or any(not isfinite(row["measured_margin"])
                   or row["measured_margin"] < config["probe_design"]["target_margin"]
                   or row["accepted_token_ids"] != required[row["prompt_id"]]["accepted_token_ids"]
                   or row["greedy_token_id"] not in row["accepted_token_ids"] for row in targets)):
        raise ValueError("Candidate lacks every individually corrected native stored-precision fork")
    protected_native = native.get("protected", {}).get("rows", ())
    prior = np.asarray([row["parent_margin"] for row in protected])
    floor = native["preservation_floor"]
    allowance = record["numerical"]["direction_specific_allowance"]
    strata = {(row["group"], row["wording"]) for row in protected}
    required_strata = {(group, wording) for group in config["probe_design"]["preservation_groups"]
                       for wording in config["endpoint"]["wordings"]}
    if (len(protected) != len(blocks["preservation"]) or len(protected) != config["probe_design"]["anchors"]
            or strata != required_strata or not isfinite(floor) or floor <= 0 or floor != required_floor
            or not isfinite(allowance) or allowance < 0):
        raise ValueError("Candidate lacks bounded positive preservation requirements")
    protected_predictions = prior + np.einsum("ijk,j,k->i", blocks["preservation"], left, right)
    identities = {(row["prompt_id"], row["position"]): row for row in protected}
    if (not np.isfinite(protected_predictions).all() or np.any(protected_predictions < floor + allowance)
            or len(protected_native) != len(protected) or len(identities) != len(protected)
            or {(row["prompt_id"], row["position"]) for row in protected_native} != set(identities)
            or any(not isfinite(row["margin"]) or row["margin"] <= floor
                   or row["replay_argmax_token_id"] != identities[(row["prompt_id"], row["position"])]["actual_token_id"]
                   or row["parent_token_id"] != identities[(row["prompt_id"], row["position"])]["actual_token_id"]
                   for row in protected_native)):
        raise ValueError("Candidate loses an individually protected decision or positive slack")


def validate_fit_report(out, material):
    report, config = _read(out / "fit-report.json"), _read(PROTOCOL)
    read_manifest(material)
    if (report.get("schema") != "weightbridge.phase_iiip.fit.v1" or report["sources"] != source_hashes()
            or report["manifest_sha256"] != file_hash(material / "manifest.json")
            or report["checkpoint_sha256"] != config["checkpoint_sha256"]
            or report["optimization_splits"] != ["adaptation", "preservation-fit", "calibration"]
            or report["hidden_opened"] is not False):
        raise ValueError("Fit evidence does not bind source, fitting scope, model and roster")
    for name, digest in report["artifacts"].items():
        path = (out / name).resolve()
        if (not name or not path.is_relative_to(out.resolve())
                or Path(name).parts[0] in {"checkpoints", "verification"}
                or not _digest(digest) or file_hash(path) != digest):
            raise ValueError("Fitting artifact does not match its frozen hash")
    states, candidates = set(), report["candidates"]
    if len(candidates) > config["intent"]["limits"]["max_candidates"]:
        raise ValueError("Candidate count exceeds the predeclared budget")
    if candidates and not {"basis.json", "derivative-blocks.npz", "preservation-block-metadata.json", "floor.json"} <= set(report["artifacts"]):
        raise ValueError("Candidate native target identity requires hash-bound basis and derivative blocks")
    blocks = _blocks(out / "derivative-blocks.npz") if candidates else None
    basis = _read(out / "basis.json") if candidates else None
    protected = _read(out / "preservation-block-metadata.json")["rows"] if candidates else None
    floor = _read(out / "floor.json")["selected_floor"] if candidates else None
    if candidates and floor not in config["probe_design"]["positive_floor_grid"]:
        raise ValueError("Candidate preservation floor differs from the frozen positive rule")
    for arm, info in candidates.items():
        if (re.fullmatch(r"[AC][12]", arm) is None
                or not all(_digest(info[key]) for key in ("checkpoint_sha256", "tensor_sha256", "receipt_sha256"))
                or info["checkpoint_sha256"] == config["checkpoint_sha256"]
                or not isfinite(info["realized_update_norm"])
                or not 0 < info["realized_update_norm"] <= config["intent"]["limits"]["max_update_norm"]
                or type(info["changed_entries"]) is not int
                or not 0 < info["changed_entries"] <= config["intent"]["limits"]["max_update_entries"]
                or info["tensor_sha256"] in states or f"arm-{arm}.json" not in report["artifacts"]
                or f"patches/{arm}-receipt.json" not in report["artifacts"]):
            raise ValueError("Practice candidate identity or numerical authorization differs")
        record = _read(out / f"arm-{arm}.json")
        receipt = _read(out / "patches" / f"{arm}-receipt.json")
        encoded = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        probe = record["probe"]
        matched = record.get("norm_match", {}).get("status") == "MATCHED"
        if (hashlib.sha256(encoded).hexdigest() != info["receipt_sha256"]
                or (arm.endswith("2") and not matched)
                or not probe.get("restored_exactly") or probe.get("candidate_tensor_sha256") != info["tensor_sha256"]
                or probe.get("endpoint", {}).get("guard", {}).get("passed") is not True
                or record.get("numerical", {}).get("status") != "NATIVE_AND_CALIBRATION_PASS"):
            raise ValueError("Candidate lacks calibrated, restored, practice-qualified stored checkpoint")
        if arm.endswith("2"):
            target = config["probe_design"]["target_norms"][1]
            if abs(info["realized_update_norm"] - target) > target * config["probe_design"]["norm_relative_tolerance"]:
                raise ValueError("Matched candidate does not meet the actual stored norm tolerance")
        _native_candidate(record, info, basis, protected, blocks, config, floor)
        states.add(info["tensor_sha256"])
    return report


def _launch(path, sha, material, split, out, manifest, config, *, lock=None, hidden=None):
    args = [sys.executable, "-m", "experiments.phase_iiip.verification", "worker", "--base", str(path.resolve()),
            "--sha256", sha, "--material", str(material.resolve()), "--split", split, "--out", str(out.resolve()),
            "--manifest-sha256", file_hash(material / "manifest.json")]
    if lock is not None:
        args += ["--lock", str(lock.resolve()), "--lock-sha256", file_hash(lock)]
    if hidden is not None:
        args += ["--hidden", str(hidden.resolve())]
    subprocess.run(args, cwd=ROOT, check=True, timeout=config["inference"]["worker_timeout_seconds"])
    return _validate_report(out, sha, material, manifest, config, split, hidden=hidden)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "assess", "finish"))
    for field in ("base", "material", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--split", choices=(*PUBLIC_SPLITS, "hidden"))
    parser.add_argument("--sha256")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--lock-sha256")
    parser.add_argument("--hidden", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if not (args.split and args.sha256 and args.manifest_sha256):
            parser.error("worker requires immutable input identities")
        worker(args.base, args.sha256, args.material, args.split, args.out, args.manifest_sha256,
               lock=args.lock, lock_sha=args.lock_sha256, hidden=args.hidden)
    elif args.command == "assess":
        if any((args.hidden, args.lock, args.lock_sha256, args.split, args.sha256, args.manifest_sha256)):
            parser.error("assess accepts only original checkpoint, frozen material and private trial")
        print(json.dumps(assess(args.base, args.material, args.out), indent=2))
    else:
        if not (args.hidden and args.lock_sha256):
            parser.error("finish requires explicit host-held roster and exact finalist lock")
        print(json.dumps(finish(args.base, args.material, args.out, args.hidden, args.lock_sha256), indent=2))
