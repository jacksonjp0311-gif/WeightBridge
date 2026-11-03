"""Local JSON tool adapter for the host-owned HF edit session."""
from __future__ import annotations

import argparse
import json
import sys

from .hf_session import (
    describe_hf_session,
    finalize_hf_candidate,
    init_hf_session,
    stop_hf_session,
    try_hf_direction,
    try_hf_patch,
)


_MAX_REQUEST_BYTES = 8 * 1024 * 1024
_TOOLS = {
    "describe": (
        "Read the allowed tensors, practice diagnostics, prior attempts and remaining budget.",
        {}, describe_hf_session,
    ),
    "try_direction": (
        "Fit and test a bounded activation-guided edit at an allowed Qwen MLP projection.",
        {"tensor": {"type": "string"}, "strength": {"type": "number"}}, try_hf_direction,
    ),
    "try_patch": (
        "Validate a weightbridge.patch.v1 document and score the separate candidate on practice.",
        {"patch": {"type": "object"}}, try_hf_patch,
    ),
    "finalize": (
        "Freeze an eligible practice candidate for independent evaluation and close the session.",
        {"candidate_id": {"type": "string"}}, finalize_hf_candidate,
    ),
    "stop": (
        "Close the session without a candidate.",
        {"reason": {"type": "string"}}, stop_hf_session,
    ),
}


def hf_session_tools() -> list[dict]:
    return [
        {"name": name, "description": description, "inputSchema": {
            "type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False,
        }}
        for name, (description, properties, _) in _TOOLS.items()
    ]


def call_hf_session_tool(session, request: dict) -> dict:
    if not isinstance(request, dict) or set(request) != {"tool", "arguments"}:
        raise ValueError("Expected an object containing exactly tool and arguments")
    name, arguments = request["tool"], request["arguments"]
    if not isinstance(name, str) or name not in _TOOLS:
        raise ValueError("Unknown HF edit session tool")
    _, properties, handler = _TOOLS[name]
    if not isinstance(arguments, dict) or arguments.keys() != properties.keys():
        raise ValueError(f"{name} expects arguments {list(properties)}")
    return handler(session, **arguments)


def add_hf_session_parser(subparsers) -> None:
    parser = subparsers.add_parser("hf-edit-session", help="Run a local practice-only HF weight editing session")
    actions = parser.add_subparsers(dest="session_action", required=True)
    init = actions.add_parser("init", help="Host only. Pin inputs and create a new session")
    init.add_argument("--session", required=True)
    init.add_argument("--base", required=True)
    init.add_argument("--policy", required=True)
    init.add_argument("--practice", required=True)
    init.add_argument("--max-candidates", type=int, default=8)
    init.add_argument("--min-target-gain", type=int, default=1)
    init.add_argument("--max-sentinel-flips", type=int, default=0)
    init.add_argument("--device", default="cpu")
    init.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    init.add_argument("--cpu-threads", type=int, default=6)
    init.add_argument("--batch-size", type=int, default=8)
    init.add_argument("--max-input-tokens", type=int, default=512)
    actions.add_parser("tools", help="Print the JSON tool schemas")
    call = actions.add_parser("call", help="Read one JSON tool request from stdin and print the result")
    call.add_argument("--session", required=True)


def run_hf_session_command(args: argparse.Namespace):
    if args.session_action == "tools":
        return hf_session_tools()
    if args.session_action == "call":
        raw = sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1)
        if len(raw) > _MAX_REQUEST_BYTES:
            raise ValueError("HF edit session request exceeds 8 MiB")
        return call_hf_session_tool(args.session, json.loads(raw))
    return init_hf_session(
        args.session, args.base, args.policy, args.practice,
        max_candidates=args.max_candidates, min_target_gain=args.min_target_gain,
        max_sentinel_flips=args.max_sentinel_flips, device=args.device, dtype=args.dtype,
        cpu_threads=args.cpu_threads, batch_size=args.batch_size, max_input_tokens=args.max_input_tokens,
    )
