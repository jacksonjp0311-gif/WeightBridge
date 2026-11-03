# WeightBridge agent tools

WeightBridge has three ways for an AI host to submit a weight proposal.

- A local Hugging Face edit session accepts JSON tool calls for a bounded
  practice loop and returns a candidate for independent evaluation.
- The CLI accepts individual inspection and patch commands for NumPy and
  safetensors checkpoints.
- The optional MCP server exposes a typed, local stdio interface for a
  practice-only synthetic MLP diagnosis session.

## Local Hugging Face edit sessions

Use the [README setup](../README.md#give-an-agent-a-weight-editing-session)
to pin a base checkpoint, host policy and labeled practice suite. Choose
the inference device and token limit when you initialize the session.
Each trial may save a full checkpoint copy. Budget disk space for the
number of attempts and set a subprocess timeout in your host harness.

Register the JSON schemas from `hf-edit-session tools` as local agent
tools. Route each invocation to
`python -m weightbridge hf-edit-session call --session <host-owned-directory>`
with one JSON request on stdin. Read one JSON response from stdout. Keep
the session directory argument fixed in the adapter. A coding agent with
local shell access can issue the same calls directly. An API-backed agent
such as a Grok-backed controller needs a local process to relay those calls.
Treat a nonzero exit as a tool error. A completed trial returns JSON with
its own disposition, so an agent can learn from a rejected or failed edit.

| Tool | Arguments | Use |
|---|---|---|
| `describe` | `{}` | Read the allowed tensors, session status and practice diagnostics. |
| `try_direction` | `{"tensor":"<allowed Qwen MLP projection>","strength":0.03}` | Fit an activation-based direction from the frozen practice examples and measure the candidate. |
| `try_patch` | `{"patch":{...}}` | Submit a `weightbridge.patch.v1` document with exact base and tensor hashes and measure the candidate. |
| `finalize` | `{"candidate_id":"candidate-0001"}` | Freeze a candidate that cleared the practice gate and end the session. |
| `stop` | `{"reason":"No proposal cleared the practice gate"}` | End the session without a candidate. |

A direction request looks like this for the pinned Qwen3.5-0.8B example,
provided the host explicitly allowed this tensor.

```json
{"tool":"try_direction","arguments":{"tensor":"model.language_model.layers.23.mlp.down_proj.weight","strength":0.03}}
```

An example task for the agent is “Inspect the practice failures. Choose a
permitted projection and strength and explain your hypothesis. Use each
practice result to decide what to try next. Finalize a candidate that passes
the practice gate or stop.” The direction tool fits the numeric update.
The raw patch tool lets an agent supply the numeric update itself.

Run the tool process with access to the research checkpoint and session
directory. Keep the holdout, evaluator controls and deployed checkpoint
outside the agent's filesystem permissions. If the agent has unrestricted
shell access, the CLI alone cannot enforce that isolation. In a restricted
harness, expose the five calls and keep the session and its files host-owned.
Check the frozen candidate with `evaluate-hf-edit` on fresh prompts and
unrelated sentinels. A practice PASS authorizes finalization for that test.

The reproducible integration check runs two disclosed choices from the
published WeightBridge pilot through the JSON tool interface. The script
builds the damaged checkpoint, compiles and measures both edits, finalizes
the eligible one, then reloads it for evaluation. It checks the exact
candidate checkpoint hash and every held-out prediction against the archive.
Use the [pinned checkpoint from the README](../README.md#replay-a-measured-qwen-improvement)
and a new output directory. The replay uses roughly 6 GB for three
checkpoint copies and was tested with Transformers 5.22.1 and Torch 2.15.0
on CPU in BF16.

```bash
python experiments/original_weight_edits/replay_hf_edit_demo.py --pristine ./qwen35-base --out ./session-demo --example ai-pilot --session-tools
```

The layer 21 candidate gained one practice answer and flipped two of 16
sentinel predictions, which failed the practice gate. The layer 23
candidate gained one practice answer with zero sentinel flips and was
finalized. After reload, it scored 24/24 versus 21/24 for the damaged
base on the held-out targets, with all 16 sentinel predictions preserved.
The full replay took 166 seconds on the recorded CPU runtime.
[Replay summary, suites and receipts](../results/hf-agent-session-replay/)

## Install and connect synthetic MCP

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

## Multi-task transfer protocol

The v2 cohort template declares 20 blinded tasks: five independent seeds for
each of four single-weight defect mechanisms. The task roster is host-owned;
the model caller must not receive the manifest or access this repository.
Each task requires a fresh agent session with only these MCP tools. The
protocol fixes the prompt, one-entry policy, eight-candidate budget, practice
and holdout split, same-tensor/magnitude random control, and stratified
decision gate. Its current SHA-256 is recorded in
[`mcp_llm_transfer_v2.template.json`](../experiments/original_weight_edits/mcp_llm_transfer_v2.template.json).

The file remains `TEMPLATE_HOLD`: the current Codex host does not expose a
verifiable provider model ID/version or reasoning-sampling receipt, and this
workspace is visible to the current model. The runner rejects this template
before creating a task. A frozen protocol must point to a hash-checked host
identity receipt containing provider, model ID, version, sampling settings and
host call ID; each task record retains that receipt. The receipt is
host-reported and content-hashed, not cryptographically signed by a provider.
Bind an isolated host,
freeze a new protocol version and its source hashes, then invoke one task at a
time with the host-owned manifest. Do not edit the template in place or reuse
the disclosed pilot holdout.

Audit a preflight or completed cohort with:

```bash
python -m weightbridge.evaluation.mcp_cohort \
  --protocol experiments/original_weight_edits/mcp_llm_transfer_v2.template.json \
  --records path/to/experiment_record.json \
  --out results/original_weight_edits/cohort_summary.json
```

The current preflight is [`INCONCLUSIVE`](../results/original_weight_edits/mcp-cohort-v2-preflight.json),
with one prior pilot record and no task bound to the 20-slot protocol. It is a
protocol audit, not a behavioral cohort result.

When a separate host can provide its identity receipt, freeze it into a new
protocol file with `weightbridge-freeze-mcp-cohort` (or
`python -m weightbridge.evaluation.freeze_cohort`). Use a new output name for
every cohort; the command refuses to overwrite protocol or receipt files. Its
identity provenance field is an operator statement, not a provider signature.

The local Transformers host can use the pinned Qwen checkpoint as the
diagnostic model. Run the host with an inference Python environment and pass
the separate MCP-enabled Python executable to its broker. It verifies the
base checkpoint hash, records the full local model file hashes and host code
hash, and constructs a fresh model context for each trial. The model receives
the frozen prompt and the practice-only MCP view. The host handles all file
access and keeps post-finalization results out of the model context.

```bash
python experiments/original_weight_edits/run_local_mcp_cohort.py \
  --model /path/to/pinned/qwen35-base \
  --broker-python /path/to/mcp-enabled/python \
  --tasks 4 --seed-offset 40000 --mode patch-form \
  --out results/wb-qf2
```

Use a fresh output directory and a split-seed offset that does not overlap
an existing run. The four-task prefix has one trial per defect mechanism.
Use a short output path on Windows to leave room for candidate receipts.
In `tool-json` mode, the model chooses each tool call. In `patch-form` mode,
the host calls diagnostics and practice evaluation automatically, while the
model supplies patch proposals and chooses whether to finalize an evaluated
candidate. Treat the interaction modes as separate exploratory conditions.

### Completed direct-proposal qualification

The pinned local Qwen controller completed all 20 blinded patch-form tasks
across four fault types with fresh splits. The host routed practice
evaluation, and the model supplied each patch and its final decision.
The frozen qualification recorded **FAIL** with zero finalized candidates
and zero repairs. All 37 practice evaluations left logits unchanged
within `1e-9`, and 19 proposals duplicated an earlier proposal.
The damaged fallback averaged 2.97 percentage points below the
predeclared random controls. The mechanism-stratified 95% interval ran from
−8.52 to +1.17 points. [Protocol, transcripts and scores](../results/wb-dq/)

Audit the archived result from the repository root. Pass the records in
frozen execution order to reproduce the seeded bootstrap exactly.

```powershell
$protocol = 'results/wb-dq/frozen_protocol.json'
$order = (Get-Content $protocol -Raw | ConvertFrom-Json).execution_order
$records = @($order | ForEach-Object { "results/wb-dq/$_/experiment_record.json" })
python -m weightbridge.evaluation.mcp_cohort --protocol $protocol --records @records --out direct-audit.json
```

The [real-checkpoint CLI pilot](QWEN_AI_EDIT_PILOT.md) used a constrained
controller that chose layers and strengths and a host that fit the edit
direction. It passed one of three real Qwen cases. Use the
[README workflow](../README.md#edit-and-test-a-real-checkpoint) to submit and
evaluate your own saved Hugging Face candidate.

### Paired connectivity probe

For the toy MLP, an input weight `input_weight[i, j]` or hidden bias
`hidden_bias[j]` has no path to either logit when the entire outgoing row
`output_weight[j, :]` is zero. The read-only probe lists those coordinates
from the base checkpoint. A nonzero outgoing row is a structural condition,
and the effect of an edit still depends on the input, weights and edit value.
The host attaches the probe only to the treatment arm's practice view.

```bash
python experiments/original_weight_edits/run_local_mcp_cohort.py \
  --model /path/to/pinned/qwen35-base \
  --broker-python /path/to/mcp-enabled/python \
  --tasks 4 --seed-offset 50000 --mode patch-form \
  --paired-connectivity --out results/wb-cx
python experiments/original_weight_edits/summarize_connectivity_ablation.py \
  results/wb-cx --out results/wb-cx/comparison.json
```

Run the host command in the inference environment and the replay in the
project environment. Use a fresh output directory and unused split-seed
offset for each new study. One pair uses the same base and splits in two
fresh model contexts. Arm order alternates across pairs. The primary outcome
is the fraction of tasks with an accepted edit whose maximum absolute
practice logit change exceeds `1e-9`. The replay also reports prediction
changes, finalized holdout repairs, token counts and CPU time.

### Compare rejection with coordinate search

The three-arm runner shows both AI arms the same connectivity report. In
the rejection arm, an exact match against the disconnected coordinate mask
consumes one model turn and receives an explanation before any checkpoint
write or practice evaluation. The deterministic arm selects one eligible
coordinate per tensor using the task seed and tests deltas of `-3` and
`+3`. It freezes the first candidate with the best practice score only
when that score exceeds baseline. Each arm has an eight-evaluation ceiling.
The AI arms can use fewer evaluations when they stop or repeat.

```bash
python experiments/original_weight_edits/run_local_mcp_cohort.py \
  --model /path/to/pinned/qwen35-base \
  --broker-python /path/to/mcp-enabled/python \
  --tasks 4 --seed-offset 60000 --mode patch-form \
  --paired-search --out results/wb-sr
python experiments/original_weight_edits/summarize_search_comparison.py \
  results/wb-sr --out results/wb-sr/comparison.json
```

The replay checks the shared inputs, budget, search schedule and selection.
Use the source at the original experiment commit to replay an older
comparison whose manifest binds a previous broker version. The paired
connectivity report replays at commit `948d4ae`.

### Generate a practice-supported patch

For the float64 toy MLP, pass an NPZ checkpoint, a separate NPZ practice
split containing `features` and `labels`, and the host patch policy.
The menu contains one practice-improving proposal per eligible tensor
when such a proposal is found. Use `--best-patch` for the deterministic
choice in ordinary WeightBridge patch format.

```bash
python -m weightbridge breakpoint-menu --base base.npz --practice practice.npz \
  --policy policy.json --best-patch --out proposed.json
python -m weightbridge apply --base base.npz --policy policy.json \
  --patch proposed.json --out candidate.npz
```

The first command screens analytical practice margins and writes an
immutable proposal. The second applies the standard policy and writes a
separate candidate. Score the reloaded candidate on a held-out split to
test generalization.

The paired research runner gives Qwen and the deterministic selector the
same practice candidate menu. Qwen chooses among its labels in one model
forward. Each arm validates and evaluates only its chosen candidate, and
finalizes it only when the broker reproduces the predicted practice gain.

```bash
python experiments/original_weight_edits/run_local_mcp_cohort.py \
  --model /path/to/pinned/qwen35-base \
  --broker-python /path/to/mcp-enabled/python \
  --tasks 20 --seed-offset 80000 --mode patch-form \
  --paired-solver --out results/wb-bf
python experiments/original_weight_edits/summarize_solver_selection.py \
  results/wb-bf --out results/wb-bf/comparison.json
```

Choose a fresh output path and split offset for a new experiment. The
historical three-arm search comparison replays at commit `726f668`.

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
