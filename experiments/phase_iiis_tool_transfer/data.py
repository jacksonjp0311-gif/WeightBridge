"""Deterministic tool fixtures and family-safe population builders, without I/O.

No formal private population is created by importing or invoking a CLI. The
host must freeze resource and calibration decisions before writing those banks.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import random

from experiments.phase_iiiq_baseline import data as decimal_data
from weightbridge.adaptation import tool_task as contract


SCHEMA = "weightbridge.phase_iiis.tool_data.v1"
SCENARIOS = ("file_location_known", "missing_test_target", "unsupported_search_root",
             "unsupported_file_type", "ambiguous_test_target", "missing_search_root",
             "ambiguous_file_path", "irrelevant_request", "requested_operation",
             "explicit_unreadable_file")
SPLITS = ("calibration", "smoke", "train", "development", "preservation-validation", "selection", "hidden")
# Independent wording families are assigned prospectively. Held-out wording
# group h appears in half of every formal confirmation population.
TEMPLATES = {
    "calibration": {
        "unsupported": ("Please {value}.", "Your task is to {value}."),
        "contents": ("Find the requested file contents named {value}.", "Obtain file contents named {value}."),
        "search": ("Search for the exact text {value} in the requested root.", "Locate the literal query {value} within the requested root."),
        "test": ("Check the tests for {value}.", "Validate {value} using its tests."),
    },
    "fit": {
        "unsupported": ("I want you to {value}.", "Could you {value}?"),
        "contents": ("Please retrieve contents of the file named {value}.", "I need the contents of the file named {value}."),
        "search": ("Please search the requested root for exactly {value}.", "Find files matching the literal query {value} in the requested root."),
        "test": ("Please run the tests covering {value}.", "Check {value} with the uniquely mapped test target."),
    },
    "development": {
        "unsupported": ("Carry out this request: {value}.", "The requested task is: {value}."),
        "contents": ("Obtain the contents of the file named {value}.", "Get the contents of the file named {value}."),
        "search": ("Find entries containing exactly {value} under the requested root.", "Look for the literal search string {value} within the requested root."),
        "test": ("Execute the mapped verification target for {value}.", "Complete the test check mapped to {value}."),
    },
    "confirmation": {
        "unsupported": ("Handle the following: {value}.", "What I need is for you to {value}."),
        "contents": ("Retrieve the contents of the file named {value}.", "Find the contents of the file named {value}."),
        "search": ("Use the requested root to locate files matching exactly {value}.", "Retrieve search matches for the literal query {value} from the requested root."),
        "test": ("Perform the test-target check associated with {value}.", "Verify {value} using its unambiguous target mapping."),
    },
}
_WORDS = ("amber", "birch", "cobalt", "delta", "elm", "fable", "garnet", "harbor",
          "indigo", "juniper", "kestrel", "linden", "maple", "nectar", "opal", "pebble",
          "quartz", "raven", "spruce", "tulip", "umber", "violet", "willow", "zephyr")
_TOPICS = ("cache recovery", "queue retries", "index refresh", "session expiry",
           "token rotation", "archive import", "search pagination", "stream buffering")
FORMAT_FIXTURES = (
    {"request": {"kind": "contents", "value": "format-example.md", "root": "workspace"},
     "roots": [{"path": "workspace", "supported": True}],
     "documents": [{"path": "workspace/docs/format-example-old.md", "content": "Old format reference.", "readable": True}], "tests": []},
    {"request": {"kind": "contents", "value": "format-example.md", "root": "workspace"},
     "roots": [{"path": "workspace", "supported": True}],
     "documents": [{"path": "workspace/docs/format-example-old.md", "content": "Old format reference.", "readable": True},
                   {"path": "workspace/docs/format-example.md", "content": "Current format reference.", "readable": True}], "tests": []})
FORMAT_EXAMPLE = contract.canonical_action(contract.gold_action(FORMAT_FIXTURES[1]))


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def render_prompt(fixture, wording_group, wording, ordering) -> str:
    contract.validate_fixture(fixture)
    if wording_group not in TEMPLATES or wording not in (0, 1) or ordering not in (0, 1):
        raise ValueError("Unknown frozen tool prompt form")
    request = fixture["request"]
    registry = contract.REGISTRY if ordering == 0 else tuple(reversed(contract.REGISTRY))
    inventory = {key: fixture[key] if ordering == 0 else list(reversed(fixture[key]))
                 for key in ("roots", "documents", "tests")}
    instruction = TEMPLATES[wording_group][request["kind"]][wording].format(value=json.dumps(request["value"]))
    return "\n".join((
        "Select exactly one next local action. All facts are listed; similar names are distractors.",
        'Return only one JSON object: {"tool":"name","arguments":{...}}. No extra fields or text.',
        "Format example (unrelated disposable fixture): "+FORMAT_EXAMPLE,
        "Registry:", *registry,
        "For requested file contents: read a unique matching listed path; clarify path if ambiguous; if not located, search its literal name in the requested root.",
        "Inventory: "+json.dumps(inventory, separators=(",", ":")),
        "Requested root: "+json.dumps(request["root"]), "Request: "+instruction))


def _family(rng, scenario):
    left, right = rng.sample(_WORDS, 2)
    slug = f"{left}-{right}-{rng.randrange(10000):04d}"
    topic = f"{slug} {rng.choice(_TOPICS)}"
    filename = slug+".md"
    doc = {"path": f"workspace/docs/{filename}", "content": f"Reference for {topic}.", "readable": True}
    distractor = {"path": f"workspace/docs/{slug}-legacy.md", "content": f"Legacy {topic} notes.", "readable": True}
    target = {"target": f"tests/{slug}", "component": topic, "result": "PASS"}
    near = {"target": f"tests/{slug}-legacy", "component": topic+" legacy", "result": "FAIL"}
    fixture = {"request": {"kind": "contents", "value": filename, "root": "workspace"},
        "roots": [{"path": "workspace", "supported": True}, {"path": "remote", "supported": False}],
        "documents": [distractor], "tests": [near]}
    first, second = deepcopy(fixture), deepcopy(fixture)
    if scenario == "file_location_known":
        second["documents"].append(doc)
    elif scenario in {"missing_test_target", "ambiguous_test_target"}:
        for item in (first, second):
            item["request"] = {"kind": "test", "value": topic, "root": None}
        first["tests"].append(deepcopy(target))
        if scenario == "ambiguous_test_target":
            second["tests"].extend((deepcopy(target), {**target, "target": f"integration/{slug}", "result": "FAIL"}))
    elif scenario == "unsupported_search_root":
        for item in (first, second):
            item["request"] = {"kind": "search", "value": topic, "root": "workspace"}
        second["request"]["root"] = "remote"
    elif scenario in {"unsupported_file_type", "explicit_unreadable_file"}:
        first["documents"].append(deepcopy(doc))
        second["documents"].append({**doc, "readable": False})
        if scenario == "explicit_unreadable_file":
            for item in (first, second):
                item["request"]["value"] = doc["path"]
    elif scenario == "missing_search_root":
        for item in (first, second):
            item["request"] = {"kind": "search", "value": topic, "root": "workspace"}
        second["request"]["root"] = None
    elif scenario == "ambiguous_file_path":
        for item in (first, second):
            item["documents"].append(deepcopy(doc))
        second["documents"].append({**doc, "path": "workspace/archive/"+filename})
    elif scenario == "irrelevant_request":
        first["request"] = {"kind": "test", "value": topic, "root": None}
        first["tests"].append(deepcopy(target))
        second["request"] = {"kind": "unsupported", "value": f"predict tomorrow's weather in {left} valley", "root": None}
    elif scenario == "requested_operation":
        first["request"] = {"kind": "search", "value": topic, "root": "workspace"}
        second["request"] = {"kind": "test", "value": topic, "root": None}
        for item in (first, second):
            item["tests"].append(deepcopy(target))
    else:
        raise ValueError("Unknown counterfactual fixture mechanism")
    for item in (first, second):
        contract.validate_fixture(item)
    if contract.gold_action(first) == contract.gold_action(second):
        raise ValueError("Counterfactual must change the correct action")
    return first, second


def family_key(fixtures) -> str:
    """Content identity ignores renderer order, split names and row IDs."""
    normalized = []
    for fixture in fixtures:
        contract.validate_fixture(fixture)
        normalized.append({"request": fixture["request"], "roots": sorted(fixture["roots"], key=lambda r: r["path"]),
            "documents": sorted(fixture["documents"], key=lambda d: d["path"]),
            "tests": sorted(fixture["tests"], key=lambda t: t["target"])})
    return digest(sorted(normalized, key=digest))


def generate_tool_families(split, count, seed, *, wording_group=None, used=None):
    """Pure builder. The caller owns host-stage authorization and file sealing."""
    if (split not in SPLITS or type(count) is not int or not 1 <= count <= 1024
            or not isinstance(seed, str) or not seed):
        raise ValueError("Bounded population, named split and nonempty seed required")
    default = "calibration" if split == "calibration" else "fit" if split in ("smoke", "train") else "development" if split == "development" else "mixed-confirmation"
    wording_group = default if wording_group is None else wording_group
    if wording_group not in (*TEMPLATES, "mixed-confirmation"):
        raise ValueError("Unknown prospectively selected wording group")
    if split == "calibration" and wording_group != "calibration":
        raise ValueError("Disposable calibration uses its own independent templates")
    rng, rows = random.Random(seed), []
    used = set() if used is None else used
    # The one fixed schema example belongs only to disposable calibration.
    if split != "calibration":
        used.add(family_key(FORMAT_FIXTURES))
    for index in range(count):
        scenario = SCENARIOS[index % len(SCENARIOS)]
        for _ in range(128):
            pair = deepcopy(FORMAT_FIXTURES) if split == "calibration" and index == 0 else _family(rng, scenario)
            identity = family_key(pair)
            if identity not in used:
                used.add(identity)
                break
        else:
            raise ValueError("Bounded fresh-family generation exhausted")
        group = ("fit" if index % 2 == 0 else "confirmation") if wording_group == "mixed-confirmation" else wording_group
        for cf, fixture in enumerate(pair):
            gold = contract.gold_action(fixture)
            for wording in (0, 1):
                for ordering in (0, 1):
                    rows.append({"schema": SCHEMA, "task": "tool", "category": "tool",
                        "id": f"tool/{split}/{index:04d}/{cf}/{wording}/{ordering}", "family_id": identity,
                        "counterfactual": cf, "wording_group": group, "wording": wording,
                        "ordering": ordering, "scenario": scenario, "stratum": gold["tool"],
                        "wording_familiarity": "familiar" if group == "fit" else "unseen",
                        "fixture": deepcopy(fixture), "gold_action": deepcopy(gold),
                        "prompt": render_prompt(fixture, group, wording, ordering),
                        "completion": contract.canonical_action(gold), "training_weight": 1.0})
    validate_tool_rows(rows, expected_families=count)
    return rows


def validate_tool_rows(rows, *, expected_families=None):
    if not rows or len({r["id"] for r in rows}) != len(rows) or len({r["prompt"] for r in rows}) != len(rows):
        raise ValueError("Tool population must have distinct rows and prompts")
    families = defaultdict(list)
    for row in rows:
        contract.validate_row(row)
        if row.get("schema") != SCHEMA or row["prompt"] != render_prompt(row["fixture"], row["wording_group"], row["wording"], row["ordering"]):
            raise ValueError("Prompt is not the complete frozen visible fixture")
        families[row["family_id"]].append(row)
    expected_forms = {(c,w,o) for c in (0,1) for w in (0,1) for o in (0,1)}
    for identity, forms in families.items():
        if len(forms) != 8 or {(r["counterfactual"],r["wording"],r["ordering"]) for r in forms} != expected_forms:
            raise ValueError("Each tool family requires eight complete forms")
        if len({r["scenario"] for r in forms}) != 1 or len({r["wording_group"] for r in forms}) != 1:
            raise ValueError("Family mechanism or wording assignment changed")
        pair = []
        for cf in (0,1):
            subset = [r for r in forms if r["counterfactual"] == cf]
            if len({digest(r["fixture"]) for r in subset}) != 1:
                raise ValueError("Presentational variation changed visible semantics")
            pair.append(subset[0]["fixture"])
        if family_key(pair) != identity or contract.gold_action(pair[0]) == contract.gold_action(pair[1]):
            raise ValueError("Family identity or opposing action is invalid")
    if expected_families is not None and len(families) != expected_families:
        raise ValueError("Unexpected tool-family count")
    return {"rows": len(rows), "families": len(families),
            "strata": dict(Counter(r["stratum"] for r in rows)),
            "wording_groups": dict(Counter(r["wording_group"] for r in rows)),
            "scenarios": dict(Counter(r["scenario"] for r in rows))}


def assert_partition_isolation(populations):
    seen_families, seen_fixtures, seen_prompts = set(), set(), set()
    for name, rows in populations.items():
        validate_tool_rows(rows)
        families = {r["family_id"] for r in rows}
        fixtures = {digest(r["fixture"]) for r in rows}
        prompts = {r["prompt"] for r in rows}
        if families & seen_families or fixtures & seen_fixtures or prompts & seen_prompts:
            raise ValueError(f"Tool family or transformation leakage in partition {name}")
        seen_families |= families; seen_fixtures |= fixtures; seen_prompts |= prompts
    return {"partitions": len(populations), "families": len(seen_families), "presentations": len(seen_prompts)}


def decimal_rows(rows):
    """Annotate unchanged Q rows; prompts, completion and Decimal oracle persist."""
    result = []
    for row in rows:
        if decimal_data.oracle(row["family"], row["operands"]) != row["answer"]:
            raise ValueError("Decimal answer differs from exact existing oracle")
        category = ("decimal-target" if row["group"] == "target" else
                    {"larger-decimal": row.get("tier") or "decimal-comparison",
                     "copy": "copy", "integer-add": "integer-add", "integer-successor": "integer-successor"}[row["family"]])
        result.append({**row, "task": "decimal", "category": category})
    return result
