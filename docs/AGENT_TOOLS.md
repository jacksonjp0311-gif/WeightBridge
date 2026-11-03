# WeightBridge agent tools

WeightBridge now has two ways for an AI host to submit a weight proposal:

- The CLI accepts inspection and patch commands. It supports the implemented
  NumPy and safetensors paths, including Hugging Face checkpoints.
- The optional MCP server exposes a typed, local stdio interface for a
  practice-only synthetic MLP diagnosis session. This is the first
  LLM-hosted agent path; it does not yet wrap Hugging Face inference.

## Install and connect

Install the optional SDK adapter:

```bash
python -m pip install -e '.[mcp]'
```

Create a host-owned session JSON file. The paths and policy belong to the host;
they are not MCP tool arguments. `base_checkpoint` and `practice_dataset` must
be absolute `.npz` paths. The practice archive contains only `features` and
`labels`. Do not put a hidden test file in this session directory or config.

```json
{
  "schema": "weightbridge.agent_session.v1",
  "session_id": "toy-session-01",
  "base_checkpoint": "C:/research/task/base.npz",
  "practice_dataset": "C:/research/task/practice.npz",
  "candidate_directory": "C:/research/task/candidates",
  "policy": {
    "schema": "weightbridge.policy.v1",
    "allow_tensors": ["hidden_bias", "output_bias"],
    "max_entries": 1,
    "max_fraction": 0.1,
    "max_abs_delta": 2.0
  },
  "max_candidates": 8,
  "max_failure_examples": 32
}
```

Run `weightbridge-mcp --config <absolute-session-json>` as a local stdio MCP
server. An MCP-capable host can launch it as a subprocess. For example, on
Windows the host entry points `command` at the absolute
`.venv/Scripts/weightbridge-mcp.exe` path and passes
`["--config", "C:/research/task/host-session.json"]` in `args`. Keep the server
local; this pilot does not expose an HTTP endpoint. The Python MCP SDK is an
optional dependency pinned to its v1 maintenance line with `mcp<2`, as its
[official installation guide](https://py.sdk.modelcontextprotocol.io/v1/installation/)
recommends.

The reproducible AI-directed pilot runner creates a safe host session and
holds its hidden examples in memory until the candidate is finalized:

```bash
python experiments/original_weight_edits/run_mcp_agent_experiment.py --seed 28431
```

It prints the tool schemas, then accepts one JSON object per line on stdin:

```json
{"tool":"get_diagnostic_report","arguments":{}}
```

This runner is useful for research and host integration tests. A regular MCP
host should call the MCP tools directly instead of using the JSONL bridge.
The committed pilot evidence includes its synthetic holdout and oracle after
the session; do not reuse that disclosed split as hidden evaluation data.

## Available tools

| Tool | What the caller receives or changes |
|---|---|
| `get_diagnostic_report` | Base hash, bounded tensor statistics, practice-only errors and selected activations |
| `submit_patch` | A host-validated immutable candidate ID and receipt |
| `evaluate_practice` | Candidate results on the host's practice split only |
| `finalize_candidate` | Freezes a practice-evaluated candidate; performs no hidden scoring |
| `stop_without_candidate` | Closes the session with an explicit reason |
| `get_session_status` | Candidate/evaluation budget and session state |

The tool caller cannot supply paths, replace the base model, change the policy,
expand the candidate/evaluation budgets, inspect hidden answers, or promote a
candidate. Candidate artifacts are new files beneath the host-selected
directory. The append-only session log records accepted actions; the pilot
runner also records every tool request and response, including rejected calls.
The server verifies base, practice and candidate hashes before scoring and
finalization.

## Current boundary

The MCP adapter currently understands only the small `ToyMLP` `.npz` task
shape. Real Hugging Face model inspection and patching remain available through
the CLI, but an MCP tool that binds a Hugging Face model and an authorized
practice evaluator is future work. The server runs with the launching user's
operating-system permissions; it is an application-level policy boundary,
not a hardened sandbox. Do not give it a private-memory path or a production
checkpoint.

Test the adapter and the real stdio protocol with:

```bash
python -m pip install -e '.[mcp,test]'
python -m pytest tests/test_mcp_agent_tools.py -q
```
