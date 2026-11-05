"""Pure prospective mixed-population builder; the host alone seals and opens banks.

This module reads no files and has no CLI. Historical Decimal exclusions and
private seeds must be supplied by the host. It never reads unopened R Bank B.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib

from experiments.phase_iiiq_baseline import data as q
from . import data


SPLITS = ("smoke", "train", "replay", "development", "qualification", "confirmation")
HOST_SPLITS = frozenset(("qualification", "confirmation"))
CANONICAL = {"smoke": "smoke", "train": "train", "replay": "replay",
             "development": "development", "qualification": "selection", "confirmation": "hidden"}
TOOL_FAMILIES = {"smoke": 8, "train": 128, "replay": 0, "development": 16,
                 "qualification": 32, "confirmation": 32}
DECIMAL_SPECS = {
    "smoke": q.PopulationSpec(target_families=4),
    "train": q.PopulationSpec(target_families=64),
    "replay": q.PopulationSpec(comparison_controls=64, copy_controls=64, addition_controls=32, successor_controls=32),
    "development": q.PopulationSpec(target_families=16, comparison_controls=8, copy_controls=8, addition_controls=8, successor_controls=8),
    "qualification": q.PopulationSpec(target_families=32, comparison_controls=32, copy_controls=32, addition_controls=32, successor_controls=32),
    "confirmation": q.PopulationSpec(target_families=32, comparison_controls=32, copy_controls=32, addition_controls=32, successor_controls=32),
}


def _seed(seed, label):
    return hashlib.sha256((seed+"/"+label).encode()).hexdigest()


def normalize_fixture_order(fixture):
    """Restore the frozen renderer's construction order after sorted JSON load.

    The calibration renderer is immutable and intentionally not rewritten.
    JSON object order is not a semantic field, but its visible serialized prompt
    is byte-sensitive. Validate exact fields before copying so no unknown field
    can be silently removed. Neither values nor existing prompt bytes change.
    """
    data.contract.validate_fixture(fixture)
    fields = {"request": ("kind", "value", "root"), "roots": ("path", "supported"),
              "documents": ("path", "content", "readable"), "tests": ("target", "component", "result")}
    return {name: ({key: deepcopy(fixture[name][key]) for key in keys} if name == "request"
                   else [{key: deepcopy(entry[key]) for key in keys} for entry in fixture[name]])
            for name, keys in fields.items()}


def normalize_tool_rows(rows):
    return [{**row, "fixture": normalize_fixture_order(row["fixture"])} for row in rows]


def build_populations(seeds, *, used_decimal, decimal_anchors, calibration_rows,
                      historical_exclusion_receipt=None):
    """Return rows and commitments in memory; no authorization is implied.

    Every required population is constructed before model fitting. Caller must
    keep HOST_SPLITS out of worker bundles and public repositories. Inputs are
    copied, so the frozen historical exclusion state is not mutated.
    """
    if (set(seeds) != set(SPLITS) or any(not isinstance(s, str) or not s for s in seeds.values())
            or len(set(seeds.values())) != len(SPLITS)):
        raise ValueError("Every frozen partition requires a distinct nonempty seed")
    if not isinstance(used_decimal, set) or not isinstance(decimal_anchors, set):
        raise ValueError("Host-checked historical Decimal exclusion sets are required")
    calibration_rows = normalize_tool_rows(calibration_rows)
    data.validate_tool_rows(calibration_rows)
    if len(calibration_rows) < 64 or any(r["wording_group"] != "calibration" for r in calibration_rows):
        raise ValueError("Disposable calibration must have at least 64 independent-template presentations")
    used, anchors = set(used_decimal), set(decimal_anchors)
    used_tools = {r["family_id"] for r in calibration_rows}
    populations, entries = {}, {}
    for split in SPLITS:
        canonical = CANONICAL[split]
        tools = (data.generate_tool_families(canonical, TOOL_FAMILIES[split], _seed(seeds[split], "tool"), used=used_tools)
                 if TOOL_FAMILIES[split] else [])
        if split in HOST_SPLITS:
            # Alternate familiarity across repeated mechanism blocks so a
            # mechanism is not permanently synonymous with one wording group.
            families = list(dict.fromkeys(r["family_id"] for r in tools))
            assignments = {family: ("fit" if (index+index//len(data.SCENARIOS)) % 2 == 0 else "confirmation")
                           for index, family in enumerate(families)}
            for row in tools:
                group = assignments[row["family_id"]]
                row["wording_group"] = group
                row["wording_familiarity"] = "familiar" if group == "fit" else "unseen"
                row["prompt"] = data.render_prompt(row["fixture"], group, row["wording"], row["ordering"])
        decimals = data.decimal_rows(q.generate(canonical, DECIMAL_SPECS[split], _seed(seeds[split], "decimal"), used, anchors))
        rows = tools+decimals
        populations[split] = rows
        entries[split] = {"rows": len(rows), "tool_families": TOOL_FAMILIES[split],
            "decimal_population": asdict(DECIMAL_SPECS[split]),
            "task_counts": dict(Counter(r["task"] for r in rows)),
            "category_counts": dict(Counter(r["category"] for r in rows)),
            "tool_action_counts": dict(Counter(r["stratum"] for r in tools)),
            "tool_wording_counts": dict(Counter(r["wording_familiarity"] for r in tools)),
            "rows_sha256": data.digest(rows), "seed_commitment": hashlib.sha256(seeds[split].encode()).hexdigest(),
            "host_private": split in HOST_SPLITS}
    validate_populations(populations, calibration_rows=calibration_rows)
    return populations, {"schema": "weightbridge.phase_iiis.populations.v1", "splits": entries,
        "training_sampling": {"tool_rows": 1024, "negative_decimal_rows": 512,
                              "general_replay_rows": 512, "row_weight": 1.0,
                              "loss_weighting": "supervised-token normalization; row mixture is not token-loss mixture"},
        "historical_exclusion": historical_exclusion_receipt,
        "historical_coverage_claim": "Only the host-supplied exclusion receipt and sets; this builder reads no historical or unopened confirmation files.",
        "format_example_family_id": data.family_key(data.FORMAT_FIXTURES)}


def validate_populations(populations, *, calibration_rows):
    if set(populations) != set(SPLITS):
        raise ValueError("Mixed study must contain every prospectively declared partition")
    tool_pops = {"calibration": normalize_tool_rows(calibration_rows)}
    known_decimal_families, known_tasks, known_ids = set(), set(), set()
    for split, rows in populations.items():
        if len(rows) != 8*TOOL_FAMILIES[split]+DECIMAL_SPECS[split].row_count:
            raise ValueError("Mixed population count differs from the fixed study")
        if any(r.get("training_weight") != 1.0 for r in rows):
            raise ValueError("Prospective mixed examples require unit sampling weights")
        ids = {r["id"] for r in rows}
        if len(ids) != len(rows) or ids & known_ids:
            raise ValueError("Presentation identity is duplicated across partitions")
        known_ids |= ids
        tools = normalize_tool_rows([r for r in rows if r["task"] == "tool"])
        decimals = [r for r in rows if r["task"] == "decimal"]
        if len(tools)+len(decimals) != len(rows):
            raise ValueError("Unknown task category in mixed population")
        if tools:
            data.validate_tool_rows(tools, expected_families=TOOL_FAMILIES[split])
            for r in tools:
                if r["wording_familiarity"] != ("familiar" if r["wording_group"] == "fit" else "unseen"):
                    raise ValueError("Wording familiarity does not match the prospective template assignment")
            if split in HOST_SPLITS and sum(r["wording_familiarity"] == "unseen" for r in tools) < len(tools)/2:
                raise ValueError("At least half of formal families must use unseen wording groups")
            tool_pops[split] = tools
        q.validate(decimals, CANONICAL[split], DECIMAL_SPECS[split])
        families = {r["family_id"] for r in decimals}
        tasks = {q.semantic_key(r["family"], r["operands"]) for r in decimals}
        if families & known_decimal_families or tasks & known_tasks:
            raise ValueError("Decimal family or numeric-equivalent task leaked across partitions")
        known_decimal_families |= families; known_tasks |= tasks
    data.assert_partition_isolation(tool_pops)
    combined = populations["train"]+populations["replay"]
    if len(combined) != 2048 or Counter(r["task"] for r in combined) != {"tool": 1024, "decimal": 1024}:
        raise ValueError("Frozen mixed exposure differs from 2048 equally sampled rows")
    return {"partitions": len(populations), "training_rows": len(combined),
            "tool_families": sum(TOOL_FAMILIES.values()), "decimal_families": len(known_decimal_families)}
