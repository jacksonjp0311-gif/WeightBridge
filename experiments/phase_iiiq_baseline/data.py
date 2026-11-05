"""Fresh exact-oracle supervised data; selection and hidden remain host-private.

Rows are generated prospectively, never selected by model errors. A target
family contains both opposing comparisons, both orders and both wordings.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import secrets
from typing import TypedDict

from experiments.phase_iiil import semantic_roster as generator
from experiments.phase_iiip import semantic_roster as previous
from weightbridge.adaptation.behavior_endpoint import canonical_decimal
from weightbridge.core import file_hash, json_dump_atomic

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocol.json")
SCHEMA = "weightbridge.phase_iiiq_baseline.data.v1"
SPLITS = ("smoke", "train", "replay", "development", "preservation-validation", "selection", "hidden")
HOST_SPLITS = frozenset(("selection", "hidden"))
DISCLOSED = (*previous.DISCLOSED,
    *(f"results/phase-iii-p/optimized-v1/material/{split}.json" for split in previous.SPLITS
      if (ROOT / f"results/phase-iii-p/optimized-v1/material/{split}.json").is_file()),
    "results/phase-iii-p/reference-v1/roster.json")
DOMAIN = {"target_integer_parts": [10, 9999], "integer_bank_offsets": [0, 9900], "bank_step": 100,
          "scope": "Sign-preserving whole-number translation of inherited exact counterfactual families; close/far gaps and one-digit flips unchanged. Copy/control domains retain their declared separate generators."}
PREFLIGHT_COMPARISONS = (("-1000.1", "-1000.0811"), ("-1000.1", "-1000.1811"),
                        ("-800.0811", "-801.1"), ("-802.1811", "-801.1"))


class BaselineRow(TypedDict):
    id: str
    item: str
    family: str
    group: str
    wording: int
    operands: list[str]
    prompt: str
    answer: str
    oracle_answer: str
    completion: str
    training_weight: float
    task_id: str
    family_id: str


@dataclass(frozen=True)
class PopulationSpec:
    target_families: int = 0
    comparison_controls: int = 0
    copy_controls: int = 0
    addition_controls: int = 0
    successor_controls: int = 0

    def __post_init__(self):
        if (any(type(value) is not int or not 0 <= value <= 1024 for value in asdict(self).values())
                or self.target_families % 2 or not 0 < self.row_count <= 16384):
            raise ValueError("Population counts must be bounded nonnegative integers with balanced even target families")

    @property
    def row_count(self):
        return 8*self.target_families + 4*self.comparison_controls + 2*(self.copy_controls+self.addition_controls+self.successor_controls)


DEFAULT_POPULATIONS = {
    "smoke": PopulationSpec(target_families=4),
    "train": PopulationSpec(target_families=128),
    "replay": PopulationSpec(comparison_controls=128, copy_controls=128, addition_controls=64, successor_controls=64),
    "development": PopulationSpec(target_families=32),
    "preservation-validation": PopulationSpec(comparison_controls=32, copy_controls=32, addition_controls=32, successor_controls=32),
    "selection": PopulationSpec(target_families=32),
    "hidden": PopulationSpec(target_families=32),
}
ARITHMETIC_INSTRUCTIONS = {
    "integer-add": ("Add the two integers. Give only the numerical sum.",
                    "What is the sum of these integers? Write only the number."),
    "integer-successor": ("Write the next integer after this number. Give only the number.",
                          "Increase the integer by one. Output only the result."),
}
ARITHMETIC_DEMOS = {"integer-add": (("3", "4"), ("8", "5")), "integer-successor": (("17",), ("28",))}


def _canonical(value):
    result = canonical_decimal(str(value))
    if result is None:
        raise ValueError("A data operand is not a bounded exact decimal")
    return result


def semantic_key(family, operands):
    values = tuple(Decimal(value) for value in operands)
    return (family, *(sorted(values) if family in ("larger-decimal", "integer-add") else values))


def _family_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def oracle(family, operands):
    values = tuple(Decimal(value) for value in operands)
    if (not values or not all(value.is_finite() for value in values)
            or len(values) != (2 if family in ("larger-decimal", "integer-add") else 1)):
        raise ValueError("Malformed exact task operands")
    if family == "larger-decimal":
        if values[0] == values[1]:
            raise ValueError("Comparison operands must be distinct")
        answer = max(values)
    elif family == "copy":
        answer = values[0]
    elif family == "integer-add":
        if any(value != value.to_integral_value() for value in values):
            raise ValueError("Addition controls require integers")
        answer = sum(values, Decimal(0))
    elif family == "integer-successor":
        if values[0] != values[0].to_integral_value():
            raise ValueError("Successor controls require an integer")
        answer = values[0]+Decimal(1)
    else:
        raise ValueError("Unsupported exact-oracle task")
    return _canonical(format(answer, "f"))


def prompt(family, operands, wording):
    if family in ("larger-decimal", "copy"):
        return generator.renderer._prompt(family, operands, wording)
    if family not in ARITHMETIC_INSTRUCTIONS or wording not in (0, 1):
        raise ValueError("Unknown arithmetic family or wording")
    label = "Integers: {0} and {1}" if family == "integer-add" else "Integer: {0}"
    examples = [f"{label.format(*example)}\nAnswer: {oracle(family, example)}" for example in ARITHMETIC_DEMOS[family]]
    return "\n".join((ARITHMETIC_INSTRUCTIONS[family][wording], *examples, label.format(*operands), "Answer:"))


def _annotate(row, family_id=None):
    raw = row["answer"]
    answer = oracle(row["family"], row["operands"])
    key = semantic_key(row["family"], row["operands"])
    family_id = family_id or _family_hash([key[0], *(_canonical(format(value, "f")) for value in key[1:])])
    return {**row, "schema": SCHEMA, "oracle_answer": raw, "answer": answer,
        "completion": " "+answer+"\n", "training_weight": 1.0,
        "task_id": row["comparison_id"] or row["item"], "family_id": family_id,
        "control_kind": None if row["group"] == "target" else row.get("tier") or row["family"]}


def _blocked():
    used, anchors = previous._blocked()
    for family, examples in generator.renderer.DEMONSTRATIONS.items():
        for operands in examples:
            used.add(semantic_key(family, operands))
            if family == "larger-decimal" and all(Decimal(value) < 0 for value in operands):
                sizes = [len(value.partition(".")[2]) for value in operands]
                if len(set(sizes)) > 1:
                    anchors.add(Decimal(operands[sizes.index(min(sizes))]))
    for operands in PREFLIGHT_COMPARISONS:
        used.add(semantic_key("larger-decimal", operands))
    anchors.update((Decimal("-1000.1"), Decimal("-801.1")))
    for name in DISCLOSED:
        for row in json.loads((ROOT / name).read_bytes())["rows"]:
            operands = row["operands"]
            used.add(semantic_key(row["family"], operands))
            if row["family"] == "larger-decimal" and all(Decimal(value) < 0 for value in operands):
                sizes = [len(value.partition(".")[2]) for value in operands]
                if len(set(sizes)) > 1:
                    anchors.add(Decimal(operands[sizes.index(min(sizes))]))
    for family, examples in ARITHMETIC_DEMOS.items():
        used.update(semantic_key(family, operands) for operands in examples)
    return used, anchors


def generate(split, spec, seed, used, anchors):
    """Generate one prospective split while sharing duplicate guards across all splits."""
    if split not in SPLITS or not isinstance(spec, PopulationSpec) or not isinstance(seed, str) or not seed:
        raise ValueError("Generation requires a named split, typed counts and nonempty seed")
    rng, rows = random.Random(seed), []
    for index in range(spec.target_families):
        tier = "negative-close" if index % 2 == 0 else "negative-far"
        for _ in range(8192):
            offset = Decimal(100*rng.randrange(100))
            pair = tuple(tuple(format(Decimal(value)-offset, "f") for value in operands)
                         for operands in generator._pair(rng, tier))
            identities = {semantic_key("larger-decimal", operands) for operands in pair}
            anchor = Decimal(pair[0][0])
            if len(identities) == 2 and identities.isdisjoint(used) and anchor not in anchors:
                used.update(identities); anchors.add(anchor)
                break
        else:
            raise ValueError("Bounded sampling could not find a fresh target family")
        pair_id = f"baseline-decimal/{split}/target/{index:04d}"
        content = sorted([sorted((_canonical(value) for value in operands), key=Decimal) for operands in pair])
        family_id = _family_hash(["truth-flip", tier, content])
        for side, operands in enumerate(pair):
            rows.extend(_annotate(row, family_id) for row in generator._comparison(pair_id,
                f"{pair_id}/variant-{side}", tier, side, operands, (0, 1), "target"))
    for family, count in (("larger-decimal", spec.comparison_controls), ("copy", spec.copy_controls),
                          ("integer-add", spec.addition_controls), ("integer-successor", spec.successor_controls)):
        for index in range(count):
            tier = ("positive-far", "mixed-sign")[index % 2] if family == "larger-decimal" else None
            for _ in range(8192):
                if family in ("larger-decimal", "copy"):
                    group = "positive" if family == "larger-decimal" else "preserve"
                    operands = generator.renderer._draw(group, tier, ("shorter", "longer")[(index//2) % 2], index % 2, rng)
                else:
                    operands = (str(rng.randrange(1, 100)), str(rng.randrange(1, 100))) if family == "integer-add" else (str(rng.randrange(1, 1000)),)
                identity = semantic_key(family, operands)
                if identity not in used:
                    used.add(identity)
                    break
            else:
                raise ValueError("Bounded sampling could not find a fresh preservation task")
            item = f"baseline-decimal/{split}/{family}/{index:04d}"
            if family == "larger-decimal":
                rows.extend(_annotate(row) for row in generator._comparison(None, item, tier, None, operands, (0, 1), "positive"))
            else:
                for wording in (0, 1):
                    rows.append(_annotate({"id": f"{item}/{wording}", "item": item, "comparison_id": None,
                        "pair_id": None, "counterfactual": None, "order": None, "family": family,
                        "group": "preserve", "wording": wording, "operands": list(operands), "tier": None,
                        "truth_precision": None, "truth_side": None, "answer": oracle(family, operands),
                        "prompt": prompt(family, operands, wording)}))
    validate(rows, split, spec)
    return rows


def validate(rows, split, spec):
    if len(rows) != spec.row_count or len({row["id"] for row in rows}) != len(rows) or len({row["prompt"] for row in rows}) != len(rows):
        raise ValueError("Population row count, identities or prompts are duplicated")
    items, pairs, tasks, comparisons = defaultdict(list), defaultdict(list), defaultdict(set), defaultdict(list)
    for row in rows:
        expected = oracle(row["family"], row["operands"])
        if (row["schema"] != SCHEMA or not row["id"].startswith(f"baseline-decimal/{split}/")
                or row["wording"] not in (0, 1) or row["id"] != f"{row['item']}/{row['wording']}"
                or row["prompt"] != prompt(row["family"], row["operands"], row["wording"])
                or row["answer"] != expected or row["completion"] != " "+expected+"\n" or row["training_weight"] != 1.0
                or _canonical(row["oracle_answer"]) != expected or not isinstance(row["family_id"], str)):
            raise ValueError("A supervised row violates its exact oracle, prompt or completion contract")
        if row["task_id"] != (row["comparison_id"] or row["item"]):
            raise ValueError("Underlying task identity differs")
        expected_group = "target" if row["pair_id"] is not None else "positive" if row["family"] == "larger-decimal" else "preserve"
        if row["group"] != expected_group:
            raise ValueError("Task family and preservation/target role differ")
        items[row["item"]].append(row)
        tasks[row["family"]].add(row["task_id"])
        if row["family"] == "larger-decimal":
            operands = row["operands"]
            correct_side = max(range(2), key=lambda index: Decimal(operands[index]))
            lengths = [len(value.partition(".")[2]) for value in operands]
            precision = "shorter" if lengths[correct_side] == min(lengths) else "longer"
            if row["order"] not in (0, 1) or row["truth_side"] != correct_side or row["truth_precision"] != precision:
                raise ValueError("Comparison order or answer-length stratum differs from the exact operands")
            comparisons[row["comparison_id"]].append(row)
        if row["group"] == "target":
            if (row["family"] != "larger-decimal" or row["counterfactual"] not in (0, 1)
                    or row["order"] not in (0, 1) or not all(Decimal(value) < 0 for value in row["operands"])):
                raise ValueError("Malformed target comparison")
            pairs[row["pair_id"]].append(row)
        elif row["pair_id"] is not None or row["counterfactual"] is not None:
            raise ValueError("Preservation tasks cannot claim target families")
    if any(sorted(row["wording"] for row in forms) != [0, 1] for forms in items.values()):
        raise ValueError("Every item needs both distinct frozen wordings")
    for forms in items.values():
        if len({(row["family"], row["group"], row["answer"], semantic_key(row["family"], row["operands"])) for row in forms}) != 1:
            raise ValueError("Wording variants changed the underlying semantic task")
    for forms in comparisons.values():
        if len(forms) != 4 or {(row["wording"], row["order"]) for row in forms} != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            raise ValueError("A comparison needs both wordings and operand orders")
        original = next(row for row in forms if row["wording"] == row["order"] == 0)
        if any(row["operands"] != (original["operands"][::-1] if row["order"] else original["operands"]) for row in forms):
            raise ValueError("Comparison presentation order changed the operands")
    for pair in pairs.values():
        indexed = {(row["counterfactual"], row["order"], row["wording"]): row for row in pair}
        if len(pair) != 8 or len(indexed) != 8 or len({row["family_id"] for row in pair}) != 1:
            raise ValueError("A truth-flip family must retain all eight distinct presentations")
        originals = [indexed[(side, 0, 0)] for side in (0, 1)]
        for row in pair:
            original = originals[row["counterfactual"]]
            if row["operands"] != (original["operands"][::-1] if row["order"] else original["operands"]):
                raise ValueError("An operand reversal changed the underlying comparison")
        first, second = [row["operands"] for row in originals]
        if (first[0] != second[0] or len(first[1]) != len(second[1])
                or sum(a != b for a, b in zip(first[1], second[1])) != 1
                or originals[0]["answer"] == originals[1]["answer"]
                or {row["truth_precision"] for row in originals} != {"shorter", "longer"}):
            raise ValueError("Counterfactual comparisons do not flip their exact answer")
        for row in originals:
            gap = abs(Decimal(row["operands"][0])-Decimal(row["operands"][1]))
            if not (Decimal(0) < gap < Decimal('.1') if row["tier"] == "negative-close" else Decimal(2) < gap < Decimal(5)):
                raise ValueError("Target gap leaves its declared close/far stratum")
    actual = {"target_families": len(pairs), "comparison_controls": len(tasks["larger-decimal"])-2*len(pairs),
              "copy_controls": len(tasks["copy"]), "addition_controls": len(tasks["integer-add"]),
              "successor_controls": len(tasks["integer-successor"])}
    if actual != asdict(spec):
        raise ValueError("Underlying task counts differ from the frozen population")
    tiers = Counter(pair[0]["tier"] for pair in pairs.values())
    if any(tiers[tier] != spec.target_families//2 for tier in ("negative-close", "negative-far")):
        raise ValueError("Target close/far families are not balanced")


def _sources(protocol):
    return {**previous._sources(), "experiments/phase_iiiq_baseline/data.py": file_hash(__file__),
            "baseline_protocol_document": file_hash(protocol),
            **{name: file_hash(ROOT / name) for name in DISCLOSED}}


def _plan(protocol):
    config = json.loads(Path(protocol).read_bytes())["data"]
    populations = config.get("populations", {name: asdict(spec) for name, spec in DEFAULT_POPULATIONS.items()})
    if set(populations) != set(SPLITS) or not isinstance(config.get("seed_prefix"), str) or not config["seed_prefix"]:
        raise ValueError("Data plan requires all seven named populations and a seed prefix")
    return config["seed_prefix"], {name: PopulationSpec(**populations[name]) for name in SPLITS}


def freeze(material, hidden_path, *, protocol=PROTOCOL, host_seeds=None):
    material, hidden_path, protocol = Path(material), Path(hidden_path), Path(protocol)
    if (material.resolve().is_relative_to(ROOT) or hidden_path.resolve().is_relative_to(ROOT)
            or hidden_path.resolve().is_relative_to(material.resolve())):
        raise ValueError("Initial rosters and distinct sealed evidence require private host paths outside the checkout")
    if material.exists() or hidden_path.exists():
        raise FileExistsError("Use fresh write-once data paths")
    prefix, populations = _plan(protocol)
    supplied = dict(host_seeds or {})
    if set(supplied)-HOST_SPLITS or any(not isinstance(value, str) or not value for value in supplied.values()):
        raise ValueError("Only nonempty host-owned selection and hidden seeds may be supplied")
    seeds = {name: supplied.get(name, secrets.token_hex(32)) if name in HOST_SPLITS else f"{prefix}/{name}" for name in SPLITS}
    if len(set(seeds.values())) != len(SPLITS):
        raise ValueError("Population seeds must be distinct")
    used, anchors = _blocked()
    manifest = {"schema": SCHEMA, "protocol_sha256": file_hash(protocol), "sources": _sources(protocol),
                "domain": DOMAIN, "selection_is_error_based": False,
                "preflight_disclosure": {"purpose": "Timing-only native/bare pilot examples excluded from every study population",
                                         "comparisons": [list(pair) for pair in PREFLIGHT_COMPARISONS]},
                "splits": {}, "artifacts": {}}
    material.mkdir(parents=True)
    for split in SPLITS:
        rows = generate(split, populations[split], seeds[split], used, anchors)
        destination = hidden_path if split == "hidden" else material / f"{split}.json"
        json_dump_atomic(destination, rows)
        manifest["artifacts"][f"{split}.json"] = file_hash(destination)
        manifest["splits"][split] = {"sha256": file_hash(destination), "rows": len(rows),
            "population": asdict(populations[split]), "underlying_tasks": len({row["task_id"] for row in rows}),
            "semantic_families": len({row["family_id"] for row in rows}),
            "seed_commitment": hashlib.sha256(seeds[split].encode()).hexdigest()}
    json_dump_atomic(material / "manifest.json", manifest)
    return manifest


def read_manifest(material, *, protocol=PROTOCOL):
    manifest = json.loads((Path(material) / "manifest.json").read_bytes())
    _, populations = _plan(protocol)
    if (manifest.get("schema") != SCHEMA or manifest["protocol_sha256"] != file_hash(protocol)
            or manifest["sources"] != _sources(protocol) or set(manifest["splits"]) != set(SPLITS)
            or manifest.get("domain") != DOMAIN or manifest.get("selection_is_error_based") is not False
            or any(manifest["splits"][name]["population"] != asdict(spec)
                   or manifest["artifacts"].get(f"{name}.json") != manifest["splits"][name]["sha256"]
                   for name, spec in populations.items())):
        raise ValueError("Data manifest differs from frozen protocol, sources or counts")
    return manifest


def load_split(material, split, manifest, *, protocol=PROTOCOL, host_authorized=False, hidden_path=None):
    if split not in SPLITS or (split == "hidden") != (hidden_path is not None) or (split in HOST_SPLITS and not host_authorized):
        raise ValueError("Selection and hidden data require explicit host authorization and a distinct hidden path")
    material = Path(material)
    if hidden_path is not None and (Path(hidden_path).resolve().is_relative_to(ROOT) or Path(hidden_path).resolve().is_relative_to(material.resolve())):
        raise ValueError("Hidden evidence must stay outside the public checkout and material directory")
    path = Path(hidden_path) if hidden_path is not None else material / f"{split}.json"
    data, expected = json.loads(path.read_bytes()), manifest["splits"][split]
    if file_hash(path) != expected["sha256"] or not isinstance(data, list):
        raise ValueError("Frozen population identity differs")
    validate(data, split, PopulationSpec(**expected["population"]))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--hidden-out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.material, args.hidden_out, protocol=args.protocol), indent=2))
