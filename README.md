# WeightBridge

WeightBridge gives an AI a way to propose changes to model weights and gives
the host a way to check and test those changes. The AI receives an inspection
report and submits a patch as JSON. The host enforces a policy and saves the
result as a separate checkpoint with a receipt. You can reload that candidate
and measure whether it improves the behavior you care about.

The CLI supports NumPy and Hugging Face safetensors checkpoints. The
[local HF edit session](#give-an-agent-a-weight-editing-session) lets an
agent choose edits, see practice results and freeze a candidate. The optional
MCP server supports practice sessions on synthetic MLPs.
[Agent tools guide](docs/AGENT_TOOLS.md)

## Install and try it

Use Python 3.10 or later. Clone the repository, install the CLI, and run the
offline demo.

```bash
git clone https://github.com/jacksonjp0311-gif/WeightBridge.git
cd WeightBridge
python -m pip install -e ".[safetensors,test]"
python -m weightbridge demo --dir ./demo-run
python -m weightbridge inspect ./demo-run/base.npz
python -m pytest -q
```

The demo creates a synthetic checkpoint, applies a predefined correction, and
scores the base and saved candidate on the same holdout. The output includes
the policy, patch, receipt, and evaluation JSON.

Install `.[inference]` to load a Hugging Face model locally or `.[mcp]` to
enable the toy MCP workflow. The real-model evaluation examples use a pinned
Qwen3.5-0.8B checkpoint on CPU. See the [runtime notes](docs/LIMITATIONS.md)
for measured compatibility.

## Give an agent a weight editing session

Install `.[inference,safetensors]`. The host creates a new session with a
local checkpoint, a [weight policy](#patch-contract) and a labeled practice
suite. The practice suite uses the same JSON format as the evaluator below.

```bash
python -m weightbridge hf-edit-session init --session ./edit-session \
  --base ./base-model --policy host-policy.json --practice practice.json \
  --max-candidates 6 --max-sentinel-flips 0 --device cpu
python -m weightbridge hf-edit-session tools
echo '{"tool":"describe","arguments":{}}' | python -m weightbridge hf-edit-session call --session ./edit-session
```

Give a local coding agent this tool contract and the behavior you want to
repair. For an API-backed agent, run the calls in its local harness and pass
back the JSON replies. The agent can try a raw patch or ask WeightBridge to
fit a direction on a permitted Qwen3.5 MLP projection. It uses practice
feedback to choose its next attempt, then finalizes an eligible candidate
or stops. The host can reload that candidate with `evaluate-hf-edit` and
score it on a separate holdout. [Host setup and request examples](docs/AGENT_TOOLS.md#local-hugging-face-edit-sessions)

## Edit and test a real checkpoint

1. **Inspect the base.** Use a local Hugging Face model directory. Give the AI
   the inspection report, the behavior you want to change, and practice
   examples. Keep the final evaluation prompts separate.

   ```bash
   python -m weightbridge inspect ./base-model --max-tensors 16
   ```

2. **Review the proposal and apply it.** The AI supplies a
   `weightbridge.patch.v1` JSON document with the report's
   `checkpoint_sha256` as `base_sha256`. A sparse edit specifies a tensor
   name, an index, an operation, and a value. The host supplies a
   `weightbridge.policy.v1` JSON document with allowed tensors, the maximum
   entries and fraction of weights that can change, and the maximum
   absolute delta per entry. The [patch and policy examples](#patch-contract)
   show the schema.

   Ask the AI for one patch grounded in the inspection report and practice
   failures. Give it the allowed tensors and edit budget, and ask it to state
   the expected behavior change in the patch's `reason`. Use practice scores
   for iteration and freeze the chosen patch before opening the holdout.

   ```bash
   python -m weightbridge apply-hf --base ./base-model \
     --patch proposal.json --policy host-policy.json --out ./candidate-model
   ```

   Use a fresh output path. The receipt identifies the base, candidate,
   patch, policy, and affected tensors. The host can keep the base available
   for rollback.

3. **Reload and evaluate.** Keep a host-owned forced-choice suite of unseen
   target prompts and unrelated control prompts, called sentinels. Each
   choice must encode as one distinct token in the model's tokenizer.

   ```json
   {
     "schema": "weightbridge.hf_edit_suite.v1",
     "choices": ["A", "B"],
     "target": [{"id": "t1", "text": "Which is larger? A. 20 B. 30. Reply A or B.", "label": "B"}],
     "sentinel": [{"id": "s1", "text": "Which is a color? A. blue B. chair. Reply A or B.", "label": "A"}]
   }
   ```

   ```bash
   python -m weightbridge evaluate-hf-edit --base ./base-model \
     --candidate ./candidate-model --suite holdout.json --out edit-report.json \
     --min-target-gain 1 --max-sentinel-flips 0 --device cpu
   ```

   The evaluator checks the candidate receipt against both checkpoint hashes,
   reloads both models, and records paired predictions. Here the candidate
   must answer at least one additional target correctly and preserve every
   sentinel prediction. A measured FAIL writes the report and returns exit 1.

## Replay a measured Qwen improvement

Install the inference extras and use the pinned Qwen3.5-0.8B checkpoint.
If you already have that revision locally, pass its directory to
`--pristine`. The replay checks the rebuilt receipt hashes and compares
every prediction with archived case 12.

```bash
python -m pip install -e ".[inference,safetensors]"
hf download Qwen/Qwen3.5-0.8B config.json chat_template.jinja tokenizer.json \
  tokenizer_config.json preprocessor_config.json model.safetensors.index.json \
  model.safetensors-00001-of-00001.safetensors \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir ./qwen35-base
python experiments/original_weight_edits/replay_hf_edit_demo.py \
  --pristine ./qwen35-base --out ./qwen-edit-demo
```

The replay scores **12/24 → 22/24** on targets, with zero changes on
16 sentinels. Use a new output directory and roughly 4 GB of scratch
space for the two checkpoint copies. [Pinned model and case evidence](docs/QWEN_WORDING_ABLATION.md) · [Replay report](results/hf-edit-demo/report.json)

Pass `--example ai-pilot` with a fresh output directory to replay the
AI-selected repair that scored **21/24 → 24/24** on targets and preserved
all 16 sentinel predictions. [AI pilot evidence](docs/QWEN_AI_EDIT_PILOT.md)

## Measured results

| Workflow | Result |
|---|---|
| Deterministic search on real Qwen weights | Guided candidates beat matched random search in 11 of 12 cases. Five passed all per-case requirements. The cohort threshold was eight. [Cohort report](docs/QWEN_REPAIR_COHORT.md) |
| Practice wording on real Qwen weights | One candidate took fresh paraphrase accuracy from 6/12 to 12/12 after reload while preserving all 16 unrelated predictions. The three-wording arm passed 2/12 cases. [Wording study](docs/QWEN_WORDING_ABLATION.md) |
| AI-directed selection on real Qwen weights | Qwen chose which bounded edits to try. One saved finalist improved held-out targets from 21/24 to 24/24 with all 16 unrelated predictions preserved. One of three cases passed the full gate. [Pilot and receipts](docs/QWEN_AI_EDIT_PILOT.md) |
| AI edit selection on toy models | One Codex pilot independently found a blinded repair. Qwen chose successful solver-proposed edits in 19/20 cases. In the separate frozen test of direct patch writing, Qwen finalized 0/20 repairs. [Agent tools](docs/AGENT_TOOLS.md) · [Qualification report](results/wb-dq/cohort_summary.json) |

<details>
<summary>Historical v0.1 guide and patch examples</summary>

## Historical v0.1 guide

> This README preserves the original v0.1 guide. The v0.1 source and its verification artifacts remain in the first commit on `dev/weightbridge-v0.2`; the current additions are summarized in the v0.2 section at the end.

A **working, isolated proof-of-concept** for reading model tensors, accepting an AI-proposed **sparse numerical or targeted directional patch as data**, validating the patch against a host-controlled policy, writing an immutable candidate and comparing the result on a frozen holdout. It runs fully offline on CPU.

**Scope:** reproducible sandbox, not a method for teaching arbitrary LLMs new capabilities. The included synthetic classification task simulates a localized failure. Its successful edit is pre-specified to demonstrate patch application and evaluation; **an AI did not discover it**. No source in this package changes PEGASUS, ChatGPT, your installed models or their memories.

## Quick start

```bash
python -m pip install -e '.[test,safetensors]'
python -m weightbridge demo --dir ./demo-run
python -m weightbridge inspect ./demo-run/base.npz
python -m pytest -q
```

The demo creates `base.npz`, `patch.json`, `policy.json`, `candidate.npz`, `candidate.npz.receipt.json` and `evaluation.json`. The base checkpoint is never overwritten; a successful evaluation never deploys a model or claims learning.

## Give an AI these tools

**Read-only:**
- `inspect` — tensor names, shapes, parameter counts, sampled statistics, checkpoint hash.
- Next: activation probes, rank/spectral summaries, behavioral probes through an isolation runner.

**Candidate-only:**
- `apply` — take a JSON patch (individual parameter edits or a targeted directional projection), enforce the allowlist, parameter-change budget and exact base SHA-256; produce a new checkpoint and receipt.
- `compare-demo` — compare the original and candidate on the **same** frozen synthetic examples.
- Next: editable tensor slices, low-rank and orthogonal projection edits, sharded safetensors adapters, independent benchmark runner.

**Never grant an agent direct production checkpoint write access.** The host controls paths, policy, test data, promotion and rollback. A patch is a proposal, not authorization.

### Patch contract

```json
{
  "schema": "weightbridge.patch.v1",
  "base_sha256": "<exact SHA-256 of checkpoint>",
  "reason": "A specific causal hypothesis, not a verified fact",
  "evidence_ids": ["receipt-or-trace-id"],
  "changes": [{"tensor": "b", "index": [1], "op": "set", "value": 0.0}]
}
```

Policy (host-owned, **not chosen by the AI**):

```json
{
  "schema": "weightbridge.policy.v1",
  "allow_tensors": ["b"],
  "max_entries": 1,
  "max_fraction": 0.20,
  "max_abs_delta": 2.0
}
```

Optional projection primitive: `"projections": [{"tensor":"M", "op":"remove_column_direction", "columns":[3], "direction":[1.0, 0.0, 0.0], "strength":1.0}]`. The direction length must equal the matrix row count; this removes its normalized component only from specified columns, with every touched entry counted against the host budget. This is an experimental operation, not a guaranteed behavior repair.

Apply an externally proposed patch to a **different, new file**:

```bash
python -m weightbridge apply --base demo-run/base.npz \
  --patch demo-run/patch.json --policy demo-run/policy.json \
  --out demo-run/independent-candidate.npz
python -m weightbridge compare-demo --base demo-run/base.npz \
  --candidate demo-run/independent-candidate.npz --out demo-run/independent-eval.json
```

The **v0.1 baseline** supports small unsharded `.safetensors` and NumPy `.npz` files. The v0.2 branch adds indexed sharded safetensors inspection and candidate patching, bounded Hugging Face inference, activation and attention summaries, and controlled activation ablations. Model-specific runtimes remain optional. See `docs/LIMITATIONS.md` for measured compatibility and current HOLDs.

For the optional LLM-hosted practice-only MCP interface, install `.[mcp]` and
see [`docs/AGENT_TOOLS.md`](docs/AGENT_TOOLS.md). It currently supports the
synthetic MLP diagnosis session; real Hugging Face checkpoint inspection and
patching remain available through the CLI.

## What this builds toward

- RecallTree provides eligible, receipt-backed observations. Do not train on or export private memory without separate permission.
- A diagnosis agent proposes a *hypothesis* about a neural mechanism associated with a repeated failure.
- A weight-inspection agent gets scoped tensor summaries and controlled activation/forward-pass probes.
- The agent proposes edits; the **host**, not the AI, applies policy limits and writes candidates.
- ExperimentForge compares baseline, candidate and strong baselines (LoRA, MEMIT/ROME where applicable) on *fresh held-out tasks*, locality tests, adversarial cases and budgets.
- Genome stores separate candidate/evaluation/approval/release/rollback records.
- Only an explicit operator-approved promotion can alter which model the host starts. The main model is immutable, and rollback is a pointer to the known original checkpoint, not a speculative reverse edit.

Review `docs/RESEARCH_PLAN.md` for the experimental roadmap and failure modes.

</details>

## Source inspiration and independence

- Takehi, Teoh, Zhang and Khattab (Nov 2, 2026), *Sparks of Weight-Space Programming*: https://rikiyat.github.io/blog/weight-space-programming/
- Original publicly released logs, not included or copied here: https://github.com/RikiyaT/lm_understand-logs
- Related weight-editing baselines: https://github.com/zjunlp/EasyEdit

This code is original research scaffolding. The authors' published p50c edit has been deterministically reconstructed and its two edited-tensor hashes verified. The bounded CPU smoke is not an independent reproduction of the paper's benchmark results.

The original WeightBridge software is licensed under MIT; see `LICENSE`. The
authors' `p50c` patch artifact is a separate third-party item attributed under
CC BY 4.0 and is excluded from the MIT grant; see `NOTICE.md` and the
[reference provenance record](experiments/published_reproduction/REFERENCE_PROVENANCE.md).
The Qwen checkpoint and full evaluation datasets are not redistributed.

## v0.2 portable research platform

The v0.1 implementation is preserved byte-for-byte in commit `321547d`. v0.2 adds:

- Lazy, header-validated inspection of unsharded and indexed sharded safetensors. Inspection reports model/tokenizer metadata, tensor hashes, tensor roles, bounded numeric summaries and optional small-matrix SVD summaries.
- A host-policy-gated HF patch path supporting sparse BF16/F16/F32/F64 edits, selected row/column slices, selected-column projections and bounded optional low-rank updates. Candidate checkpoints are new files/directories with hashes and receipts; there is no promotion API.
- A separate read-only Transformers inference interface with seed, input, output and time limits; optional activation/attention summaries and reversible activation ablations.
- Append-only experiment records, paired bootstrap intervals, a synthetic diagnostic worker process, and a synthetic-only Memory-to-Weights Consolidation store.
- A deterministic reconstruction script and attributed copy of the authors' p50c sparse patch artifact.
- An optional local MCP server exposing host-pinned inspection, bounded patch
  submission, practice-only scoring and candidate finalization; one Codex
  session found the injected hidden-bias correction on a blinded toy task.
- A hashed 20-task, four-mechanism LLM transfer protocol and stratified cohort
  evaluator. A local Transformers host runs a pinned Qwen diagnostic model
  against the MCP broker on toy weights, with a fresh context for each trial.
  Bounded smoke results and the remaining transfer gate are tracked in
  [the research progress record](docs/PROGRESS.md).
- `breakpoint-menu` generates practice-supported one-scalar patch proposals
  for the float64 toy MLP. The normal `apply` command validates a selected
  proposal under the host policy and writes a separate checkpoint.
  [Tool guide](docs/AGENT_TOOLS.md)

Install the core tools and run the tests/demo:

```bash
python -m pip install -e '.[test,safetensors]'
python -m pytest -q
python -m weightbridge inspect path/to/model-directory --max-tensors 16
python -m weightbridge apply-hf --base path/to/model-directory \
  --patch patch.json --policy host-policy.json --out candidate-directory
python experiments/original_weight_edits/run_toy_discovery.py --seed 842
python experiments/memory_ablation/run_mwc_ablation.py
```

The included toy run seeds, including 842, are already present in the committed
evidence ledger, so use a fresh seed for a new independent run (for example,
`--seed 843`). Historical outputs and hashes are preserved under `results/`; do not
re-label a replay as an independent sample.

For optional local generation, install `.[inference]`; the runtime requires PyTorch and Transformers. The pinned Qwen model was tested on CPU. The CUDA/vLLM benchmark path used by the authors is unavailable in the measured environment.

Install `.[figures]` to regenerate the evidence-only figures and summary:

```bash
python -m pip install -e '.[figures]'
python benchmarks/runners/plot_results.py
```

The generated plots and `results/benchmark_summary.json` are derived from the
append-only experiment ledger and its raw artifacts. The authors' published
scores are never placed in independently measured result columns.

To reconstruct the authors' p50c checkpoint from the pinned original model:

```bash
python experiments/published_reproduction/rebuild_published_edit.py \
  --base path/to/pinned-model.safetensors --out path/to/new-p50c.safetensors
```

The reconstruction verifies the pinned whole-file hash, base tensor hashes, edit bounds, and both published edited-tensor hashes. The original model checkpoint is not stored in this repository. Published scores remain author-reported values; our CPU prompt smoke and synthetic results are separately recorded under `results/`.

Read `docs/ARCHITECTURE.md`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/BENCHMARKS.md`, `docs/FINDINGS.md`, `docs/KNOWLEDGE_LEDGER.md`, `docs/LIMITATIONS.md`, and `docs/ROADMAP.md` for the implementation and evidence status.
