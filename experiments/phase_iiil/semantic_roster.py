"""Freeze Decimal-oracle comparison families and host-held hidden material."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import secrets

from experiments.phase_iiig import roster_decimal_targeted as renderer
from experiments.phase_iiik import decimal_roster as phase_k
from weightbridge.core import file_hash, json_dump_atomic


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("semantic_protocol.json")
SCHEMA = "weightbridge.phase_iiil.semantic_roster.v1"
SPLITS = ("adaptation", "preservation-fit", "calibration", "transfer-validation", "selection", "hidden")
GROUPS = ("negative-close/shorter", "negative-close/longer",
          "negative-far/shorter", "negative-far/longer")
DISCLOSED = (*phase_k.DISCLOSED, *(
    f"results/phase-iii-k/jspace-v1/material/{split}.json"
    for split in phase_k.SPLITS if split != "hidden"
))


def _read(path):
    return json.loads(path.read_bytes())


def _sources():
    paths = (Path(__file__), Path(renderer.__file__), Path(phase_k.__file__), PROTOCOL,
             *(ROOT / path for path in DISCLOSED))
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths}


def _blocked_identities():
    used = {renderer._identity(family, operands)
            for family, examples in renderer.DEMONSTRATIONS.items() for operands in examples}
    for path in DISCLOSED:
        for row in _read(ROOT / path)["rows"]:
            used.add(renderer._identity(row["family"], row["operands"]))
    return used


def _pair(rng, tier):
    """Hold the short option fixed and change exactly one digit of the long option."""
    short_digit = rng.randint(2, 8)
    suffix = rng.randint(350, 649)
    while suffix % 10 == 0:
        suffix = rng.randint(350, 649)
    if tier == "negative-close":
        whole = rng.randint(10, 98)
        short = f"-{whole}.{short_digit}"
        low, high = f"-{whole}.{short_digit - 1}{suffix}", f"-{whole}.{short_digit}{suffix}"
    elif tier == "negative-far":
        decade = rng.randint(1, 9) * 10
        units = rng.choice((4, 5))
        gap = rng.choice((3, 4))
        short = f"-{decade + units}.{short_digit}"
        tail = f"{rng.randint(1, 9)}{suffix}"
        low, high = f"-{decade + units - gap}.{tail}", f"-{decade + units + gap}.{tail}"
    else:
        raise ValueError("Unknown semantic comparison tier")
    if rng.randrange(2):
        low, high = high, low
    return (short, low), (short, high)


def _comparison(pair_id, comparison_id, tier, counterfactual, operands, wordings, group):
    truth = renderer._answer("larger-decimal", operands)
    lengths = tuple(len(value.split(".")[1]) for value in operands)
    precision = "shorter" if len(truth.split(".")[1]) == min(lengths) else "longer"
    for order, display in enumerate((operands, operands[::-1])):
        item = f"{comparison_id}/order-{order}"
        for wording in wordings:
            yield {"id": f"{item}/{wording}", "item": item, "comparison_id": comparison_id,
                   "pair_id": pair_id, "counterfactual": counterfactual, "order": order,
                   "family": "larger-decimal", "group": group, "wording": wording,
                   "operands": list(display), "tier": tier, "truth_precision": precision,
                   "truth_side": display.index(truth), "answer": truth,
                   "prompt": renderer._prompt("larger-decimal", display, wording)}


def _generate(config, split, seed, used, used_anchors):
    rng = random.Random(seed)
    counts, wordings = config["roster_counts"][split], config["endpoint"]["wordings"]
    pair_count = counts["target_pairs"]
    if type(pair_count) is not int or pair_count < 2 or pair_count % 2 or tuple(wordings) != (0, 1):
        raise ValueError("Frozen semantic roster requires even positive pair counts and both inherited wordings")
    rows = []
    for index in range(pair_count):
        tier = "negative-close" if index % 2 == 0 else "negative-far"
        for _ in range(4096):
            pair = _pair(rng, tier)
            identities = {renderer._identity("larger-decimal", operands) for operands in pair}
            anchor = Decimal(pair[0][0])
            if len(identities) == 2 and identities.isdisjoint(used) and anchor not in used_anchors:
                used.update(identities)
                used_anchors.add(anchor)
                break
        else:
            raise ValueError("Could not draw a fresh semantic comparison family")
        pair_id = f"semantic-decimal/{split}/target/{index:02d}"
        for counterfactual, operands in enumerate(pair):
            rows.extend(_comparison(pair_id, f"{pair_id}/variant-{counterfactual}", tier,
                                    counterfactual, operands, wordings, "target"))
    for group, family in (("positive", "larger-decimal"), ("preserve", "copy")):
        for index in range(counts[group]):
            tier = ("positive-far", "mixed-sign")[index % 2] if group == "positive" else None
            precision = ("shorter", "longer")[(index // 2) % 2]
            for _ in range(4096):
                operands = renderer._draw(group, tier, precision, index % 2, rng)
                identity = renderer._identity(family, operands)
                if identity not in used:
                    used.add(identity)
                    break
            else:
                raise ValueError("Could not draw a fresh semantic preservation control")
            item = f"semantic-decimal/{split}/{group}/{index:02d}"
            if group == "positive":
                rows.extend(_comparison(None, item, tier, None, operands, wordings, group))
            else:
                for wording in wordings:
                    rows.append({"id": f"{item}/{wording}", "item": item, "comparison_id": None,
                                 "pair_id": None, "counterfactual": None, "order": None, "family": family,
                                 "group": group, "wording": wording, "operands": list(operands), "tier": None,
                                 "truth_precision": None, "truth_side": None, "answer": operands[0],
                                 "prompt": renderer._prompt(family, operands, wording)})
    _validate(rows, split, counts)
    return rows


def _validate(rows, split, counts):
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate semantic roster row")
    groups, pairs, items = defaultdict(set), defaultdict(list), defaultdict(list)
    prefix = f"semantic-decimal/{split}/"
    for row in rows:
        family, operands, wording = row["family"], row["operands"], row["wording"]
        group = row["group"]
        if (group not in {"target", "positive", "preserve"} or family != ("copy" if group == "preserve" else "larger-decimal")
                or not row["id"].startswith(prefix) or row["id"] != f"{row['item']}/{wording}"
                or len(operands) != (1 if family == "copy" else 2)
                or any(not Decimal(value).is_finite() for value in operands)
                or row["answer"] != renderer._answer(family, operands)
                or row["prompt"] != renderer._prompt(family, operands, wording)):
            raise ValueError("Semantic roster row violates the frozen Decimal oracle or prompt")
        if family == "larger-decimal":
            if (Decimal(operands[0]) == Decimal(operands[1]) or row["truth_side"] != operands.index(row["answer"])
                    or row["order"] not in (0, 1) or row["item"] != f"{row['comparison_id']}/order-{row['order']}"):
                raise ValueError("Malformed comparison identity")
            lengths = [len(value.split(".")[1]) for value in operands]
            if lengths[0] == lengths[1] or row["truth_precision"] != ("shorter" if lengths[row["truth_side"]] == min(lengths) else "longer"):
                raise ValueError("Incorrect comparison answer-length stratum")
        if group == "target":
            if (row["tier"] not in ("negative-close", "negative-far") or not all(Decimal(value) < 0 for value in operands)
                    or row["counterfactual"] not in (0, 1) or not row["pair_id"]
                    or row["comparison_id"] != f"{row['pair_id']}/variant-{row['counterfactual']}"):
                raise ValueError("Malformed semantic pair metadata")
            pairs[row["pair_id"]].append(row)
        elif row["pair_id"] is not None or row["counterfactual"] is not None:
            raise ValueError("Preservation control cannot claim a truth-flipping pair")
        items[row["item"]].append(row)
        groups[group].add(row["comparison_id"] or row["item"])
    for forms in items.values():
        if sorted(row["wording"] for row in forms) != [0, 1]:
            raise ValueError("Every ordered comparison and copy item requires both wordings")
    for pair in pairs.values():
        indexed = {(row["counterfactual"], row["order"], row["wording"]): row for row in pair}
        if len(pair) != 8 or len(indexed) != 8 or len({row["tier"] for row in pair}) != 1:
            raise ValueError("Each truth-flipping pair must contain both orders and wordings")
        originals = [indexed[(variant, 0, 0)] for variant in (0, 1)]
        for row in pair:
            original = originals[row["counterfactual"]]
            expected = original["operands"][::-1] if row["order"] else original["operands"]
            if row["operands"] != expected:
                raise ValueError("Reversed presentations changed their comparison")
        original, flip = (row["operands"] for row in originals)
        if (original[0] != flip[0] or len(original[1]) != len(flip[1])
                or sum(left != right for left, right in zip(original[1], flip[1])) != 1
                or {row["truth_precision"] for row in originals} != {"shorter", "longer"}):
            raise ValueError("Truth-flipping family must change one digit and reverse the correct option")
        distances = [abs(Decimal(left) - Decimal(right)) for left, right in (original, flip)]
        if not all(Decimal(0) < gap < Decimal('0.1') if pair[0]["tier"] == "negative-close"
                   else Decimal(2) < gap < Decimal(5) for gap in distances):
            raise ValueError("Truth-flipping pair falls outside its declared distance band")
    if (len(pairs) != counts["target_pairs"] or len(groups["target"]) != 2 * len(pairs)
            or len(groups["positive"]) != counts["positive"] or len(groups["preserve"]) != counts["preserve"]):
        raise ValueError("Semantic roster does not match the frozen split size")
    tiers = Counter(pair[0]["tier"] for pair in pairs.values())
    if tiers != {"negative-close": len(pairs) // 2, "negative-far": len(pairs) // 2}:
        raise ValueError("Semantic pair tiers must be balanced")


def freeze(material: Path, hidden_path: Path, *, hidden_seed: str | None = None) -> dict:
    if hidden_path.resolve().is_relative_to(ROOT) or hidden_path.resolve().is_relative_to(material.resolve()):
        raise ValueError("Hidden roster must reside outside the checkout and public material")
    if hidden_path.exists():
        raise FileExistsError(hidden_path)
    config = _read(PROTOCOL)
    seeds = {**config["seeds"], "hidden": hidden_seed or secrets.token_hex(32)}
    if set(seeds) != set(SPLITS) or len(set(seeds.values())) != len(SPLITS):
        raise ValueError("Freeze requires one distinct seed per split")
    used, used_anchors = _blocked_identities(), set()
    manifest = {"schema": SCHEMA, "protocol_sha256": file_hash(PROTOCOL), "sources": _sources(), "splits": {}}
    material.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        rows = _generate(config, split, seeds[split], used, used_anchors)
        path = hidden_path if split == "hidden" else material / f"{split}.json"
        json_dump_atomic(path, {"schema": SCHEMA, "split": split, "rows": rows})
        manifest["splits"][split] = {"sha256": file_hash(path), "rows": len(rows),
                                     "items": len({row["item"] for row in rows}),
                                     "semantic_comparisons": len({row["comparison_id"] for row in rows if row["group"] == "target"}),
                                     "pairs": config["roster_counts"][split]["target_pairs"],
                                     "seed_commitment": hashlib.sha256(seeds[split].encode()).hexdigest()}
    json_dump_atomic(material / "manifest.json", manifest)
    return manifest


def read_manifest(material: Path) -> dict:
    manifest = _read(material / "manifest.json")
    if (manifest.get("schema") != SCHEMA or manifest["protocol_sha256"] != file_hash(PROTOCOL)
            or manifest["sources"] != _sources() or set(manifest["splits"]) != set(SPLITS)):
        raise ValueError("Roster does not bind the frozen protocol and source bytes")
    return manifest


def load_split(material: Path, split: str, manifest: dict, *, hidden_path: Path | None = None) -> list[dict]:
    if split not in SPLITS or (split == "hidden") != (hidden_path is not None):
        raise ValueError("Hidden material requires an explicit host path")
    if hidden_path is not None and (hidden_path.resolve().is_relative_to(ROOT)
                                    or hidden_path.resolve().is_relative_to(material.resolve())):
        raise ValueError("Hidden roster must reside outside the checkout and public material")
    path = hidden_path if hidden_path is not None else material / f"{split}.json"
    digest = manifest["splits"][split]
    if file_hash(path) != digest["sha256"]:
        raise ValueError(f"Frozen {split} roster hash differs")
    data = _read(path)
    rows = data["rows"]
    if (data.get("schema") != SCHEMA or data.get("split") != split or len(rows) != digest["rows"]
            or len({row["item"] for row in rows}) != digest["items"]):
        raise ValueError(f"Malformed {split} roster")
    _validate(rows, split, _read(PROTOCOL)["roster_counts"][split])
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--hidden-out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.material, args.hidden_out), indent=2))
