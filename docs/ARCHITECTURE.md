# WeightBridge v0.2 architecture

WeightBridge remains a portable research tool. It has no adapter to PEGASUS,
RecallTree, MemoryAtlas, Genome or a production model host. Private memory is
not read by any module.

```text
local checkpoint
    ├── header-only model map + bounded tensor reads ──> inspection report
    ├── patch proposal ──> host policy gate ──> immutable candidate + receipt
    └── read-only model load in inference process ──> bounded output/probes

synthetic task ──> practice-only agent worker ──> proposed patch
                                         │
                              host patch validator
                                         │
               frozen holdout evaluator, after selection
                                         │
                 append-only experiment ledger + evidence

LLM host ──MCP stdio──> fixed local model/practice session
       ├── diagnostic report (weights + practice failures)
       ├── patch proposal ──> host policy ──> immutable candidate
       ├── practice-only score ──> revise or stop
       └── finalize ──> host opens holdout once

AWL intent + practice ──> bounded direction search ──> one frozen finalist
                                ▲                               │
                        accepted parent                  independent reload
                                │                               │
                       research lineage <── target and retention gates
```

## Modules

- `weightbridge.inspection.safetensors`: validates unsharded or indexed sharded safetensors headers; indexes tensor-to-shard offsets without reading payloads; computes streaming checkpoint and tensor SHA-256 hashes; extracts config/tokenizer identifiers; reads numeric payloads only under fixed byte limits. Layer labels are inferred from tensor names and are reported as labels, not proof of functional role.
- `weightbridge.patches.huggingface`: applies declarative sparse edits, selected row/column slices, selected-column directional projections, and bounded optional low-rank (UV) updates. The host supplies exact tensor allowlists, instruction/entry/fraction/delta/tensor-byte limits and low-rank rank/factor bounds. Patches must include base and per-tensor hashes. File and directory candidates are created at new paths and receive immutable receipts. The function has no promotion or model-serving interface.
- `weightbridge.inference.huggingface`: optional PyTorch/Transformers load and generation path. It does not import the patch engine. It pins the local path, disables remote code, constrains prompt/output/time/RSS/sampling budgets and returns termination/resource metadata. A memory cap preflights local checkpoint bytes, checks RSS after load, and stops generation if the process crosses its host limit. The matrix runner also applies a subprocess wall-time limit. Run the CLI in a dedicated process for stronger model/inference separation.
- `weightbridge.inference.probes`: bounded activation summaries, short-context attention summaries when the backend exposes attention weights, and a context-managed activation zeroing hook. Hook handles are removed even if the probe raises.
- `weightbridge.agents.worker`: child process accepts a strict JSON task with practice data, a checkpoint path, a fixed allowlist and budget. It rejects extra fields, including a holdout field. The parent holds holdout data in memory and only persists it after candidate selection.
- `weightbridge.agents.mcp_server`: optional MCP stdio server for toy diagnosis. Its host config binds one base file, one practice-only dataset, one output directory and one patch policy. Tool callers cannot supply paths, budgets, allowlists, or evaluation data. Candidate calls are append-logged and candidate hashes are checked before scoring and finalization.
- `experiments/original_weight_edits/run_mcp_agent_experiment.py`: local MCP client harness. It keeps holdout arrays in the host process, brokers typed MCP calls, and writes the holdout only after a final candidate or explicit stop.
- `weightbridge.diagnostics.toy_mlp`: a two-layer synthetic classifier with a one-sided output-bias defect. The injected value is not passed to the agent.
- `weightbridge.memory`: append-only synthetic-only MWC experience records keyed by observation ID, with exact failure fingerprint, model hash, intervention and evaluation links. No external memory connector exists.
- `weightbridge.provenance`: append-only JSONL experiment records, duplicate experiment-ID checks, correction links, paired bootstrap intervals and platform resource probes.
- `weightbridge.adaptation.awl` parses versioned adaptation intents. The numerical plan requests scoped `activation_ridge` proposals from the existing Qwen solver. It uses the same patch policy and receipt path as the local HF edit session.
- `weightbridge.adaptation.controller` freezes a sequence of cycles, searches practice-only, and records a single finalist or abstention per cycle. The verifier reloads that finalist and its accepted parent, and checks new targets, current sentinels, and the held-out predictions of active accepted ancestors. Final scoring is a separate command after the protocol closes.
- `weightbridge.adaptation.lineage` stores immutable, hash-linked acceptance, rejection and rollback events in a host-owned directory. Each accepted version points to an isolated checkpoint. Rollback moves the research selection to a verified ancestor without rewriting the checkpoint.
- `weightbridge.adaptation.grounding` parses EOS-completed answers with separate strict and diagnostic results, and calibrates answer scores against token-matched neutral continuations. The Phase III-D runner locks its scoring rule and neutral design on practice. The admission checker independently replays the decision, generation parser, and measurement gates from hash-bound evidence. The optional calibrated content-gradient backend measures both contexts at each rounded probe under the same host guard.
- `weightbridge.adaptation.meta_controller` recommends measurement, local feasibility analysis, a solver, or HOLD from typed practice evidence. It can compare calibrated estimates of verified improvement and harm inside the host's action budget. `weightbridge.adaptation.reachability` seeks witnesses in a small authorized subspace. `weightbridge.adaptation.counterfactual` aligns answer scores and decomposes a paired margin into its changing and shared components. The existing patch and lineage layers retain authority over candidate materialization and acceptance.
- `weightbridge.adaptation.universal_awl` separates behavioral intent, host authorization, and a native execution plan. `weightbridge.adaptation.model_adapter` discovers capabilities and verifies affine roles on supported Qwen3.5 and Llama runtimes. `weightbridge.adaptation.universal_update` constructs bounded selected-column updates for those plans. The separate verifier owns the independent roster and reads materialized checkpoints from disk.
- `weightbridge.adaptation.behavior_endpoint` fixes the prompt representation, decoder limits, output grammar, semantic evaluator and required wordings. The same measurement compares parent, live probe and independently loaded checkpoints. `endpoint_update` applies one rounded affine probe at a time, checks and restores the entire tensor by bytes, and emits patch proposals only after a measured endpoint pass. `endpoint_policy` gives the meta-compiler a separate recommendation based on endpoint, surrogate and independent verification evidence. The recommendation has no checkpoint promotion authority.
- `weightbridge.adaptation.generative_fork` binds an observed generation and aligned accepted next tokens to the checkpoint and prompt. `fork_update` measures shared gradients over authorized affine regions, fits rank-one updates in selected columns, and verifies stored-precision behavior against the shared endpoint. The Phase III-I verifier locks independently scored finalists by hash, checks exact restart and rollback, and recomputes the meta-controller decision after independent outcomes.

## Trust boundaries

Patch documents are data; none of their fields are evaluated as code. Validation checks tensor identity, bounds, data type, coordinate uniqueness, numerical finiteness, dtype rounding, parameter limits and candidate immutability. A valid candidate is still unpromoted. The test suite deliberately covers malformed schema, tensor allowlist/hash violations, duplicate coordinates, path traversal, truncated offsets, nested outputs and budgets.

The toy workers and MCP session get no holdout path or values. The MCP tool session accepts only the host's fixed base and practice paths and refuses a changed base, practice file, candidate hash, or session policy. The Phase III search command receives practice data and input hashes. Only verifier commands receive sealed suite paths. The workers and CLI processes retain the filesystem permissions of the launching user. A harness that exposes tools to an untrusted agent must also isolate the sealed files, the lineage workspace and any deployed checkpoint at the operating-system level.

## Checkpoint identity

Single safetensors files use SHA-256 of all file bytes. Model directories use a deterministic digest over sorted relative paths and each included file hash. Receipts inside a candidate directory are excluded from the candidate identity to avoid a self-referential hash. Nested model symlinks, path escapes, duplicate tensor names and index/shard inconsistencies are rejected. Tokenizer and non-weight asset hashes are retained in inspection output.

See `docs/EXPERIMENT_PROTOCOL.md` for metric definitions and `docs/LIMITATIONS.md` for unsupported paths.

## Patch operations

- Sparse change: `changes` contains `tensor`, `index`, `op` (`set` or `add`)
  and finite `value`.
- Row/column slice: `slices` contains an allowed matrix, `axis`, selected
  `indices`, `op`, and one finite value vector per selected slice.
- Directional projection: `projections` contains a matrix, selected columns,
  direction and bounded strength; it removes the chosen direction from only
  those columns.
- Low-rank update: `low_rank` contains factors `u` and `v` plus a finite scale
  and applies \(W' = W + sUV\). The host policy limits rank (default 8, hard
  maximum 64), factor element count, tensor bytes, changed coordinates,
  parameter fraction and per-entry magnitude. Every touched matrix coordinate
  counts against the change budget, even if target-dtype conversion rounds it
  back to its original value.
