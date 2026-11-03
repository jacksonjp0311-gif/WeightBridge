import asyncio
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from weightbridge.agents.mcp_server import AgentSessionError, ToyMCPToolSession, create_mcp_server
from weightbridge.core import file_hash
from weightbridge.diagnostics import make_defect_benchmark


def make_host_session(tmp_path, *, max_candidates=4):
    benchmark = make_defect_benchmark(seed=1901, per_class=12)
    base_path = tmp_path / "base.npz"
    benchmark["model"].save(base_path)
    practice_path = tmp_path / "practice.npz"
    np.savez_compressed(practice_path, features=benchmark["practice"]["features"],
                        labels=benchmark["practice"]["labels"])
    policy = {"schema": "weightbridge.policy.v1", "allow_tensors": ["output_bias"],
              "max_entries": 1, "max_fraction": 1 / 12, "max_abs_delta": 5.0}
    config = {
        "schema": "weightbridge.agent_session.v1", "session_id": "test-session",
        "base_checkpoint": str(base_path.resolve()),
        "practice_dataset": str(practice_path.resolve()),
        "candidate_directory": str((tmp_path / "candidate-store").resolve()),
        "policy": policy, "max_candidates": max_candidates, "max_failure_examples": 16,
    }
    config_path = tmp_path / "host-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return benchmark, config, config_path


def correction_patch(config):
    return {
        "schema": "weightbridge.patch.v1",
        "base_sha256": file_hash(config["base_checkpoint"]),
        "reason": "Practice errors are one-sided; test the authorized output intercept.",
        "evidence_ids": ["practice-report"],
        "changes": [{"tensor": "output_bias", "index": [1], "op": "add", "value": -4.5}],
        "projections": [],
    }


def test_session_exposes_practice_only_tools_and_freezes_candidate(tmp_path):
    benchmark, config, config_path = make_host_session(tmp_path)
    session = ToyMCPToolSession.from_file(config_path)
    base_hash = file_hash(config["base_checkpoint"])

    report = session.get_diagnostic_report()
    assert report["base_sha256"] == base_hash
    assert report["practice_baseline"]["accuracy"] == 0.5
    assert report["practice_failures"]["total"] > 0
    assert report["hidden_evaluation_accessible"] is False
    assert set(report["allowed_tensors"]) == {"output_bias"}

    submitted = session.submit_patch(correction_patch(config))
    assert submitted["changed_entries"] == 1
    assert submitted["status"] == "candidate_only_not_promoted"
    assert submitted["candidate_sha256"] != base_hash

    score = session.evaluate_practice(submitted["candidate_id"])
    assert score["practice_only"] is True
    assert score["result"]["accuracy"] == 1.0
    assert score["result"]["rescues"] > 0
    finalized = session.finalize_candidate(submitted["candidate_id"])
    assert finalized["finalized"] is True
    assert finalized["hidden_evaluation_performed"] is False
    assert file_hash(config["base_checkpoint"]) == base_hash

    with pytest.raises(AgentSessionError, match="already finalized"):
        session.submit_patch(correction_patch(config))

    # An MCP server restart reconstructs the same final state from append-only events.
    resumed = ToyMCPToolSession.from_file(config_path)
    status = resumed.get_session_status()
    assert status["finalized"] is True
    assert status["candidate_count"] == 1
    assert status["evaluated_count"] == 1
    assert resumed.finalize_candidate(submitted["candidate_id"])["idempotent"] is True


def test_host_policy_and_candidate_budget_cannot_be_expanded_by_tool_caller(tmp_path):
    _, config, _ = make_host_session(tmp_path, max_candidates=1)
    session = ToyMCPToolSession(config)
    invalid = correction_patch(config)
    invalid["changes"][0]["tensor"] = "hidden_bias"
    with pytest.raises(ValueError, match="allowlisted"):
        session.submit_patch(invalid)

    accepted = session.submit_patch(correction_patch(config))
    session.evaluate_practice(accepted["candidate_id"])
    with pytest.raises(AgentSessionError, match="candidate budget"):
        session.submit_patch(correction_patch(config))

    with pytest.raises(AgentSessionError, match="32 KiB"):
        too_large = correction_patch(config)
        too_large["reason"] = "x" * 33000
        session.submit_patch(too_large)


def test_host_config_rejects_caller_controlled_path_fields(tmp_path):
    _, config, _ = make_host_session(tmp_path)
    config["holdout_path"] = str(tmp_path / "hidden.npz")
    with pytest.raises(AgentSessionError, match="unknown fields"):
        ToyMCPToolSession(config)


def test_session_rejects_changed_host_inputs_and_candidate_bytes(tmp_path):
    _, config, _ = make_host_session(tmp_path)
    session = ToyMCPToolSession(config)
    candidate = session.submit_patch(correction_patch(config))
    candidate_path = (session.session_dir / f"candidate-{candidate['candidate_id']}.npz")
    candidate_path.write_bytes(candidate_path.read_bytes() + b"tamper")
    with pytest.raises(AgentSessionError, match="Candidate checkpoint hash changed"):
        session.evaluate_practice(candidate["candidate_id"])

    changed_root = tmp_path / "changed"
    changed_root.mkdir()
    _, changed_config, _ = make_host_session(changed_root)
    ToyMCPToolSession(changed_config)
    changed_config["policy"]["max_abs_delta"] = 4.0
    with pytest.raises(AgentSessionError, match="does not match its original"):
        ToyMCPToolSession(changed_config)


def test_base_checkpoint_change_stops_session_actions(tmp_path):
    _, config, _ = make_host_session(tmp_path)
    session = ToyMCPToolSession(config)
    with open(config["base_checkpoint"], "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(AgentSessionError, match="base checkpoint changed"):
        session.get_diagnostic_report()


def test_stdio_mcp_protocol_lists_and_calls_bounded_tools(tmp_path):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    _, config, config_path = make_host_session(tmp_path)
    # Constructing the server before launch checks the MCP tool registrations.
    create_mcp_server(ToyMCPToolSession(config))

    async def exercise():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "weightbridge.agents.mcp_server", "--config", str(config_path)],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=None,
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as client:
                await client.initialize()
                listed = await client.list_tools()
                names = {tool.name for tool in listed.tools}
                assert {"get_diagnostic_report", "submit_patch", "evaluate_practice",
                        "finalize_candidate", "stop_without_candidate", "get_session_status"} <= names
                response = await client.call_tool("get_diagnostic_report", {})
                assert not response.isError
                report = response.structuredContent
                assert report["hidden_evaluation_accessible"] is False
                assert report["practice_baseline"]["n"] == 24

    asyncio.run(exercise())
