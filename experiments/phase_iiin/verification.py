"""Paired-study bindings for the existing independently replayed host verifier.

The trusted Phase III-M functions are rebound to this module's isolated globals;
the imported module is never mutated. N fit/lock identity is explicit. Existing
worker/assessment/outcome receipt wire schemas are retained for compatibility.
No adaptation document or experiment input supplies executable code.
"""
from __future__ import annotations

import types

from experiments.phase_iiim import verification as inherited

# Rebind only repository-defined functions. Imported utilities remain shared,
# while each state-machine function resolves protocol and helpers in N globals.
for _name, _value in vars(inherited).items():
    if not _name.startswith("__"):
        if isinstance(_value, types.FunctionType) and _value.__module__ == inherited.__name__:
            _copy = types.FunctionType(_value.__code__, globals(), _name, _value.__defaults__, _value.__closure__)
            _copy.__kwdefaults__ = _value.__kwdefaults__
            globals()[_name] = _copy
        else:
            globals()[_name] = _value

from experiments.phase_iiin.semantic_roster import PROTOCOL, load_split, read_manifest
from experiments.phase_iiin.source_manifest import ROOT, source_hashes
from experiments.phase_iiin.semantic_metrics import compare

LOCK_SCHEMA = "weightbridge.phase_iiin.stage_lock.v1"


def validate_fit_report(out, material):
    report, config = _read(out / "fit-report.json"), _read(PROTOCOL)
    read_manifest(material)
    if (report.get("schema") != "weightbridge.phase_iiin.fit.v1" or report["sources"] != source_hashes()
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
    for arm, info in candidates.items():
        if (re.fullmatch(r"[ABC][12]", arm) is None
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
        native = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        probe = record["probe"]
        if (hashlib.sha256(native).hexdigest() != info["receipt_sha256"]
                or record["norm_match"]["status"] != "MATCHED"
                or not probe.get("restored_exactly") or probe.get("candidate_tensor_sha256") != info["tensor_sha256"]
                or probe.get("endpoint", {}).get("guard", {}).get("passed") is not True):
            raise ValueError("Candidate lacks matched, restored, practice-qualified stored checkpoint")
        states.add(info["tensor_sha256"])
    return report


def _launch(path, sha, material, split, out, manifest, config, *, lock=None, hidden=None):
    args = [sys.executable, "-m", "experiments.phase_iiin.verification", "worker", "--base", str(path.resolve()),
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
