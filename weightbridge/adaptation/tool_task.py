"""Small, deterministic tool decisions; no tools or filesystem actions execute.

The oracle and simulator read only the visible fixture. IDs, partitions, stored
gold labels and scenario names never determine the correct action.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
from pathlib import PurePosixPath
import re
from typing import TypedDict


SCHEMA = "weightbridge.tool_task.v1"
TOOLS = ("search_files", "read_file", "run_tests", "clarify", "no_action")
ARGUMENTS = {"search_files": ("root", "query"), "read_file": ("path",),
             "run_tests": ("target",), "clarify": ("field",), "no_action": ()}
REGISTRY = (
    "search_files(root,query): search a supported listed local root for the exact literal query",
    "read_file(path): return the contents of an exact listed readable file",
    "run_tests(target): return the result of the exact test target mapped to the requested component",
    "clarify(field): request a missing or ambiguous required field: root, path, or target",
    "no_action(): the request, scope, or file type is explicitly unsupported",
)
_SPECIAL = re.compile(r"<\|[^<>]*\|>|</?s>|<pad>|\[/?INST\]", re.IGNORECASE)


class ToolAction(TypedDict):
    tool: str
    arguments: dict[str, str]


def action(tool: str, **arguments: str) -> ToolAction:
    result = {"tool": tool, "arguments": arguments}
    validate_action(result)
    return result


def validate_action(value) -> None:
    if (not isinstance(value, dict) or set(value) != {"tool", "arguments"}
            or not isinstance(value.get("tool"), str) or value["tool"] not in TOOLS
            or not isinstance(value.get("arguments"), dict)):
        raise ValueError("A tool action must have exactly tool and arguments")
    if set(value["arguments"]) != set(ARGUMENTS[value["tool"]]):
        raise ValueError("Tool argument fields do not match its exact schema")
    for arg in value["arguments"].values():
        if (not isinstance(arg, str) or not arg or arg != arg.strip() or len(arg) > 240
                or any(ord(c) < 32 for c in arg) or _SPECIAL.search(arg)):
            raise ValueError("Tool arguments must be bounded plain strings")
    if value["tool"] == "clarify" and value["arguments"]["field"] not in {"root", "path", "target"}:
        raise ValueError("Unknown required clarification field")


def canonical_action(value: Mapping) -> str:
    validate_action(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def parse_json_object(text: str) -> dict | None:
    """Parse exactly one full JSON object, including duplicate-key refusal."""
    if not isinstance(text, str) or not text.strip() or len(text) > 4096 or _SPECIAL.search(text):
        return None
    try:
        result = json.loads(text, object_pairs_hook=_pairs,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if not isinstance(result, dict):
            return None
    except (ValueError, TypeError, RecursionError):
        return None
    return result


def parse_action(text: str) -> ToolAction | None:
    """Apply the closed tool schema after whole-response JSON parsing."""
    result = parse_json_object(text)
    try:
        validate_action(result)
    except (ValueError, TypeError):
        return None
    return result


def _plain(value, limit=160):
    return (isinstance(value, str) and bool(value) and value == value.strip()
            and len(value) <= limit and not any(ord(c) < 32 for c in value)
            and _SPECIAL.search(value) is None)


def _path(value):
    return (_plain(value) and not PurePosixPath(value).is_absolute()
            and ".." not in PurePosixPath(value).parts and "\\" not in value)


def validate_fixture(fixture: Mapping) -> None:
    if (not isinstance(fixture, dict) or set(fixture) != {"request", "roots", "documents", "tests"}
            or not isinstance(fixture["request"], dict)
            or set(fixture["request"]) != {"kind", "value", "root"}
            or fixture["request"]["kind"] not in {"contents", "search", "test", "unsupported"}
            or not _plain(fixture["request"]["value"])
            or fixture["request"]["root"] is not None and not _path(fixture["request"]["root"])):
        raise ValueError("Malformed visible request semantics")
    for kind, fields, identity in (("roots", {"path", "supported"}, "path"),
            ("documents", {"path", "content", "readable"}, "path"),
            ("tests", {"target", "component", "result"}, "target")):
        entries = fixture[kind]
        if not isinstance(entries, list) or len(entries) > 8:
            raise ValueError("Bounded visible inventory required")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != fields or not _path(entry[identity]):
                raise ValueError("Malformed visible inventory entry")
            if kind == "roots" and type(entry["supported"]) is not bool:
                raise ValueError("Root support must be explicitly declared")
            if kind == "documents" and (not _plain(entry["content"], 240) or type(entry["readable"]) is not bool):
                raise ValueError("Document content and support must be explicit")
            if kind == "tests" and (not _plain(entry["component"]) or entry["result"] not in {"PASS", "FAIL"}):
                raise ValueError("Test target mapping/result is malformed")
        if len({e[identity] for e in entries}) != len(entries):
            raise ValueError("Fixture inventory identities must be unique")
    req = fixture["request"]
    if req["kind"] == "unsupported" and not re.fullmatch(r"predict tomorrow's weather in [a-z]+ valley", req["value"]):
        raise ValueError("Unsupported-request fixture must use the explicit irrelevant weather task")
    if req["root"] is not None and req["root"] not in {r["path"] for r in fixture["roots"]}:
        raise ValueError("Specified roots must explicitly declare support")


def gold_action(fixture: Mapping) -> ToolAction:
    """Recompute the action from visible semantics; never consult scenario IDs."""
    validate_fixture(fixture)
    req = fixture["request"]
    if req["kind"] == "unsupported":
        return action("no_action")
    if req["kind"] == "test":
        matches = [t for t in fixture["tests"] if t["component"] == req["value"]]
        return action("run_tests", target=matches[0]["target"]) if len(matches) == 1 else action("clarify", field="target")
    if req["kind"] == "contents":
        matches = [d for d in fixture["documents"] if d["path"] == req["value"]
                   or PurePosixPath(d["path"]).name == req["value"]]
        if len(matches) > 1:
            return action("clarify", field="path")
        if matches:
            return action("read_file", path=matches[0]["path"]) if matches[0]["readable"] else action("no_action")
    if req["root"] is None:
        return action("clarify", field="root")
    root = next(r for r in fixture["roots"] if r["path"] == req["root"])
    return action("search_files", root=req["root"], query=req["value"]) if root["supported"] else action("no_action")


def simulate(fixture: Mapping, candidate: Mapping) -> dict:
    """Run arguments on in-memory entries and independently check the effect.

    No stored/recomputed gold action is consulted. Wrong paths, roots, queries
    or targets produce their own result (or an error), never the gold result.
    """
    validate_fixture(fixture)
    try:
        validate_action(candidate)
    except (ValueError, TypeError):
        return {"success": False, "reason": "invalid_action", "result": None}
    req, tool, args = fixture["request"], candidate["tool"], candidate["arguments"]
    success, result = False, None
    documents = [d for d in fixture["documents"] if d["path"] == req["value"]
                 or PurePosixPath(d["path"]).name == req["value"]]
    suites = [t for t in fixture["tests"] if t["component"] == req["value"]]
    roots = {r["path"]: r["supported"] for r in fixture["roots"]}
    if tool == "search_files":
        permitted = roots.get(args["root"], False)
        result = {"matches": sorted(d["path"] for d in fixture["documents"]
            if permitted and d["path"].startswith(args["root"]+"/")
            and (args["query"] in d["path"] or args["query"] in d["content"]))}
        needs_search = req["kind"] == "search" or req["kind"] == "contents" and not documents
        success = needs_search and permitted and args["root"] == req["root"] and args["query"] == req["value"]
    elif tool == "read_file":
        found = [d for d in fixture["documents"] if d["path"] == args["path"]]
        result = {"content": found[0]["content"]} if found and found[0]["readable"] else {"error": "not_readable"}
        success = (req["kind"] == "contents" and len(documents) == 1 and documents[0]["readable"]
                   and args["path"] == documents[0]["path"])
    elif tool == "run_tests":
        found = [t for t in fixture["tests"] if t["target"] == args["target"]]
        result = {"test_result": found[0]["result"]} if found else {"error": "unknown_target"}
        success = req["kind"] == "test" and len(suites) == 1 and args["target"] == suites[0]["target"]
    elif tool == "clarify":
        result = {"requested_field": args["field"]}
        if req["kind"] == "test":
            success = len(suites) != 1 and args["field"] == "target"
        elif req["kind"] == "contents" and len(documents) > 1:
            success = args["field"] == "path"
        elif req["kind"] == "search" or req["kind"] == "contents" and not documents:
            success = req["root"] is None and args["field"] == "root"
    else:
        result = {"effect": "none"}
        if req["kind"] == "unsupported":
            success = True
        elif req["kind"] == "contents" and len(documents) == 1:
            success = not documents[0]["readable"]
        elif req["kind"] in {"contents", "search"} and not (req["kind"] == "contents" and documents):
            success = req["root"] is not None and roots.get(req["root"]) is False
    return {"success": bool(success), "reason": "goal_satisfied" if success else "wrong_operation_or_binding", "result": result}


def validate_row(row: Mapping) -> None:
    if (not isinstance(row, dict) or row.get("task") != "tool" or row.get("category") != "tool"
            or not all(_plain(row.get(k), 512) for k in ("id", "family_id", "wording_group", "scenario"))
            or any(type(row.get(k)) is not int or row[k] not in (0, 1) for k in ("counterfactual", "wording", "ordering"))
            or not isinstance(row.get("prompt"), str) or not row["prompt"].strip()):
        raise ValueError("Malformed tool-task presentation")
    expected = gold_action(row["fixture"])
    if row.get("gold_action") != expected or row.get("stratum") != expected["tool"]:
        raise ValueError("Stored tool label disagrees with independent visible-semantics oracle")
    if row.get("completion") != canonical_action(expected) or not simulate(row["fixture"], expected)["success"]:
        raise ValueError("Gold action fails its independently implemented simulator")


def score_response(row: Mapping, text: str, completed: bool, stop_reason: str | None = None) -> dict:
    validate_row(row)
    if type(completed) is not bool:
        raise ValueError("Completion must come from the native decoding endpoint")
    parsed = parse_action(text)
    schema = parsed is not None
    expected = gold_action(row["fixture"])
    tool_ok = schema and parsed["tool"] == expected["tool"]
    arguments_ok = schema and tool_ok and parsed["arguments"] == expected["arguments"]
    outcome = simulate(row["fixture"], parsed) if schema else {"success": False, "reason": "invalid_action"}
    semantic = bool(tool_ok and arguments_ok and outcome["success"])
    strict = bool(completed and schema and semantic)
    return {"completed": completed, "whole_json_valid": parse_json_object(text) is not None,
        "schema_valid": schema,
        "schema_valid_without_completion": schema, "tool_correct": bool(tool_ok),
        "arguments_correct": bool(arguments_ok), "simulator_success": outcome["success"],
        "semantic_correct": semantic, "strict_correct": strict, "valid": schema,
        "correct": strict, "parsed_action": parsed, "stop_reason": stop_reason,
        "status": "correct" if strict else "incomplete" if not completed else
                  "invalid_schema" if not schema else "incorrect_action"}


def summarize(rows: list[Mapping]) -> dict:
    """Counts retain all eight forms; percentages do not hide failed forms."""
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("A nonempty distinct measured population is required")
    families = {}
    for row in rows:
        validate_row(row)
        forms = families.setdefault(row["family_id"], {})
        key = row["counterfactual"], row["wording"], row["ordering"]
        if key in forms:
            raise ValueError("Duplicate tool family form")
        forms[key] = row
    wanted = {(c, w, o) for c in (0, 1) for w in (0, 1) for o in (0, 1)}
    if any(set(forms) != wanted for forms in families.values()):
        raise ValueError("Tool families require all eight counterfactual forms")
    metrics = ("completed", "whole_json_valid", "schema_valid", "schema_valid_without_completion", "tool_correct",
               "arguments_correct", "simulator_success", "semantic_correct", "strict_correct")
    return {"presentations": len(rows), "families": len(families),
        **{m: sum(bool(r[m]) for r in rows) for m in metrics},
        "complete_families": sum(all(r["strict_correct"] for r in forms.values()) for forms in families.values()),
        "strata": {tool: {"presentations": sum(r["stratum"] == tool for r in rows),
            "strict_correct": sum(r["stratum"] == tool and r["strict_correct"] for r in rows)} for tool in TOOLS},
        "decode_termination": dict(Counter(r.get("stop_reason") for r in rows))}
