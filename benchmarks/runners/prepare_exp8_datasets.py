"""Prepare frozen MATH-500, GSM8K and IFEval JSON files from source datasets.

The output embeds source IDs, requested revisions, upstream split names and
dataset fingerprints. The generated files are immutable and content-hashed by
the benchmark runner. They are not downloaded or redistributed by install.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
PROMPT_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."


def _write_json(path: Path, value: dict) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prepared dataset: {path}")
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def prepare(output_dir: Path, *, math500_revision: str | None = None,
            gsm8k_revision: str | None = None, ifeval_revision: str | None = None,
            selected: tuple[str, ...] = ("math500", "gsm8k", "ifeval")) -> dict:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install WeightBridge's benchmark extra to prepare published datasets") from exc
    allowed = {"math500", "gsm8k", "ifeval"}
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= allowed:
        raise ValueError("selected datasets must be unique names from math500, gsm8k, ifeval")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    specs = {
        "math500": ("HuggingFaceH4/MATH-500", "test", math500_revision),
        "gsm8k": ("openai/gsm8k", "test", gsm8k_revision),
        "ifeval": ("google/IFEval", "train", ifeval_revision),
    }
    written = {}
    for name in selected:
        dataset_id, source_split, revision = specs[name]
        if name == "gsm8k":
            dataset = load_dataset(dataset_id, "main", split=source_split, revision=revision)
        else:
            dataset = load_dataset(dataset_id, split=source_split, revision=revision)
        rows = []
        for index, row in enumerate(dataset):
            if name == "math500":
                rows.append({"id": str(row["unique_id"]), "prompt": str(row["problem"]),
                             "answer": str(row["answer"]), "kind": "math"})
            elif name == "gsm8k":
                answer = str(row["answer"]).split("####")[-1].strip().replace(",", "")
                rows.append({"id": f"gsm8k_test_{index}", "prompt": str(row["question"]),
                             "answer": answer, "kind": "math"})
            else:
                rows.append({"id": f"ifeval_{row.get('key', index)}", "prompt": str(row["prompt"]),
                             "kind": "ifeval", "instruction_id_list": list(row["instruction_id_list"]),
                             "kwargs": list(row["kwargs"])})
        metadata = {
            "schema": "weightbridge.prepared_benchmark.v1", "split": "holdout",
            "source_dataset": dataset_id, "source_split": source_split,
            "source_revision_requested": revision or "main (mutable upstream default)",
            "source_version": str(getattr(dataset.info, "version", None)),
            "datasets_fingerprint": str(getattr(dataset, "_fingerprint", "unavailable")),
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "items": rows,
            **({"instruction_suffix": PROMPT_SUFFIX} if name in {"math500", "gsm8k"} else {}),
        }
        path = output_dir / f"{name}.json"
        written[name] = {"path": path.name, "sha256": _write_json(path, metadata),
                         "items": len(rows), "source_dataset": dataset_id,
                         "source_split": source_split, "source_revision_requested": revision or "main"}
    manifest = {"schema": "weightbridge.prepared_datasets.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "datasets": written}
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sets", default="math500,gsm8k,ifeval")
    parser.add_argument("--math500-revision")
    parser.add_argument("--gsm8k-revision")
    parser.add_argument("--ifeval-revision")
    args = parser.parse_args()
    selected = tuple(value.strip() for value in args.sets.split(",") if value.strip())
    result = prepare(args.out, math500_revision=args.math500_revision,
                     gsm8k_revision=args.gsm8k_revision, ifeval_revision=args.ifeval_revision,
                     selected=selected)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
