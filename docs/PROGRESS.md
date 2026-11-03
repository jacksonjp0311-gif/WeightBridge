# v0.2 execution progress

Status updated 2026-11-03. Development branch: `dev/weightbridge-v0.2`.
Baseline commit: `321547d` (`Import WeightBridge v0.1 baseline`).

## Milestones

| Milestone | Status | Evidence / outstanding work |
|---|---|---|
| A. Audit and preserve v0.1 | COMPLETE | Source ZIP copied from Desktop without mutation; original 11 tests passed; synthetic result and receipts preserved |
| B. Real-model inspection and patch path | COMPLETE for safetensors/HF local checkpoint path | Lazy shard inspection, exact hashes, policy-controlled edit receipts, bounded Transformers inference and probes |
| C. Published Experiment 8 | PARTIAL / HOLD | p50c reconstructed and hashes match; bounded Qwen CPU smoke truncated; full benchmark not run |
| D. Benchmark laboratory | PARTIAL | Append-only records, paired statistics, runner scripts and evidence-derived figures implemented; full datasets not evaluated |
| E. AI-directed diagnosis | PILOT COMPLETE / TRANSFER HOLD | One Codex LLM discovered a hidden-bias repair; the balanced v2 cohort protocol/evaluator now exists, but execution is HOLD pending verified LLM identity and MCP-only host isolation |
| F. MWC memory ablation | PILOT COMPLETE / general result HOLD | Four conditions across ten synthetic seeds; exact same-family experience was available |
| G. Research knowledge base | COMPLETE for current evidence | Findings, evidence ledger, limitations, roadmap and this progress record |

## Recorded failures and corrections

- The first parser pass counted an empty `\boxed{}` as a completed answer and
  retained reasoning text that the authors' evaluator strips before scoring.
  The parser was corrected to reject empty boxes and grade text after
  `</think>`. The original run and its correction remain linked in the ledger;
  the earlier result is superseded and not counted as independent evidence.
- The first random-control builder attempt created a candidate then failed
  while serializing a relative artifact path. Path normalization was fixed;
  the retry reproduced the same uncompressed patch and receipt hashes. Both
  attempt evidence and the incident remain recorded. The patch rebuild is one
  observation, not two.
- The installed Transformers version rejected a `generator` argument in its
  generation API. A narrow compatibility fallback uses the runtime's seeded
  RNG state and records the runtime version. The initial failure is preserved.
- An initial fake-dataset test fixture lacked a `datasets.info`-shaped
  attribute. The fixture was corrected; product behavior was unchanged.
- An ad hoc parameter-count query used a nonexistent convenience attribute.
  The reported counts were recomputed from the actual safetensors tensor map;
  the checkpoint distinction between MTP auxiliary entries and runtime
  parameters is now documented.
- The first paired-correctness derivation invocation supplied three run roots
  to a CLI that still accepted exactly two. The CLI was expanded to accept
  2–32 frozen runs; the failed invocation created no artifact or ledger row.

## Current evidence boundary

The published p50c edit reconstruction is a hash-verified implementation
result, not a behavioral reproduction. Qwen smokes are INCONCLUSIVE because
all generations hit the token cap. The three-run deterministic discovery and
MWC ablation support only their deliberate output-bias defect. One Codex MCP
session independently proposed the exact hidden-bias correction and improved
one same-task holdout; the multi-mechanism transfer gate remains INCONCLUSIVE.
No Qwen weight edit was discovered or validated, and no private memory was
loaded.

The v2 cohort template contains 20 host-owned task assignments across four
mechanisms and five seeds each. It freezes the prompt, policy, tool/task/evaluator
hashes, matched same-tensor random controls, transcript verification and
stratified gate. The current preflight intentionally returns INCONCLUSIVE: the
host model identity/sampling receipt is unavailable, no tasks are bound to the
template, and this workspace is not isolated from the diagnostic caller. No
new behavioral measurement was collected in this continuation.

## Verification record

Final checks on 2026-11-03:

- Core environment: `.venv\Scripts\python.exe -m pytest -q` — 71 passed,
  2 optional-dependency skips.
- Inference-capable reference environment: 74 passed, 1 optional-dependency
  skip.
- `python -m compileall -q weightbridge benchmarks experiments tests` passed.
- `benchmarks/runners/plot_results.py` completed and generated eight
  evidence-derived figures, including an LLM MCP baseline/candidate/random
  comparison. The deterministic toy aggregate remains fixed to its original
  three-run cohort; later seed-842 and MCP pilots are reported separately.
- Ledger audit: 37 append-only rows, 37 unique experiment IDs, 12 nonempty
  unique measurement keys, and 15 canonical experiments in the summary. Two
  early toy records received versioned artifact-root backfills; canonical
  artifact and metadata-correction paths resolve.
- A fresh seed-842 same-family smoke ran the diagnostic worker on 64 practice
  and 64 hidden items: base 32/64, selected candidate 64/64, random control
  32/64. The worker did not see hidden labels; the run is not included in the
  original fixed three-run aggregate. Direct `inspect` and `apply` CLI calls
  reproduced the recorded candidate hash under a one-entry host policy; the
  CLI replay itself did not score a holdout.
- The earlier seed-842 verification passed 53 core tests (2 skips) and 57
  inference-environment tests; the current full rerun is reported above.
- A Codex LLM used the new MCP interface on a fresh blinded hidden-threshold
  task. Practice improved from 42/64 to 64/64; after finalization, a reloaded
  candidate scored 64/64 on holdout vs baseline 35/64 and matched random 35/64.
  It edited one of 12 parameters (`hidden_bias[0] += -0.7`), exactly matching
  the hidden correction revealed after scoring. Disposition remains INCONCLUSIVE
  for cross-task transfer (one independent task only). Raw run:
  `results/original_weight_edits/mcp-pilot-18d5cab750/`.
- MCP adapter tests exercise host policy, immutable candidates, input/candidate
  hash tampering, state resume, and real stdio tool discovery/calls: 6 passed.
- MCP cohort protocol tests: 12 passed. They cover four forced mechanisms,
  balanced task validation, random-control matching/fallbacks, fixed hashes,
  fail-closed identity binding, task order, transcript integrity and the
  stratified decision gate.
- Cohort preflight: INCONCLUSIVE, one prior pilot record received, zero of 20
  task slots bound. It reports missing provider/model/version/sampling
  evidence and the unpinned template; it is not a new behavioral experiment.
  See `results/original_weight_edits/mcp-cohort-v2-preflight.json`.
- The MCP runtime was Python 3.12.10, Windows 11 build 26200, NumPy 2.5.4 and
  MCP SDK 1.30.0. Host CPU was 0.875 seconds and wall time 83.99 seconds;
  MCP child CPU and provider model-token cost were unavailable.
- `git diff --check` passed; Git emitted only its configured LF-to-CRLF
  working-copy notices. No pretrained Qwen checkpoint is stored in the project;
  the run bundle includes only small synthetic toy checkpoints for replay.
- Canonical experiment records report 454.22 process CPU seconds and 289.05
  summed wall seconds. These exclude coding, tests, reference inspection and
  other unmetered work, and combine distinct experiments. The MCP session's
  elapsed time includes model reasoning and is not a comparable compute cost.

The v0.2 source and research artifacts are published on the public GitHub
repository's `main` and `dev/weightbridge-v0.2` branches. This progress update
records the MCP evolution and its evidence. No edited model is promoted or
deployed; candidates remain isolated research artifacts.

## Cohort protocol continuation checks

- The v2 protocol source bindings and canonical protocol hash were verified.
  The checked-in file remains `TEMPLATE_HOLD`; it cannot launch until a host
  supplies a pinned model identity/sampling receipt and MCP-only isolation.
- Added `weightbridge-freeze-mcp-cohort` to create a new protocol plus a
  colocated, hash-bound identity receipt. It validates source and frozen
  settings and refuses to overwrite either output. Its unit tests use a
  synthetic receipt and do not constitute model runs.
- Preflight against the existing MCP pilot returned INCONCLUSIVE: one prior
  record received, zero of 20 planned tasks bound, and the model identity,
  sampling, and host verification evidence are absent. This is a protocol
  readiness check, not a new behavioral experiment.
- A follow-up audit found that the evaluator declared but did not consume the
  manifest's bootstrap plan. It now validates the frozen seed, resample count
  and confidence, uses those values, and records them in the summary. The first
  focused rerun caught source-hash drift and an obsolete test override; both
  were corrected before final verification.
- Cohort and MCP tests: 22 passed. The cohort module has 16 tests; together
  these cover task binding, source and identity hashes, transcript integrity,
  execution order, random controls, protocol freezing, the analysis plan, and
  the preregistered decision gate.
- Full core suite: 75 passed, 2 skipped. Full inference-capable reference
  environment: 78 passed, 1 skipped. Editable installation exposed the new
  CLI successfully; `compileall` passed.
- No new model-behavior measurements were collected in this continuation.

## Local Qwen diagnostic runs

- The local Transformers host ran the hash-verified Qwen3.5-0.8B checkpoint
  offline with fresh contexts and a separate stdio MCP broker. Trial split
  seeds are spaced so one trial's holdout cannot become another's practice.
- Three four-task smokes completed, each with one task per defect mechanism.
  The direct tool profiles finalized no candidates. The short-path patch form
  accepted eight one-scalar submissions, including duplicates. Reloading
  the candidates found zero change in practice logits. The model finalized
  no candidates in that profile either.
- Two interrupted attempts are archived separately. The initial full
  diagnostic prompted a patch truncated at 192 tokens. The first patch-form
  attempt scored one trial before a Windows receipt-path error interrupted
  the next trial. The launcher now checks the receipt path before loading
  the model, and the short-path rerun completed.
- The measured 20-task gate remains HOLD. Each completed smoke's independent
  cohort audit reports INCONCLUSIVE with 16 unrun slots. Keep the profiles
  separate when interpreting their results.
- The 12 completed smoke trials consumed 3,210.83 model-host process CPU
  seconds and 595.16 summed wall seconds. The host generated 3,358 output
  tokens and processed 77,894 input tokens across its model turns. Partial
  attempts and broker processes are outside these host totals.
- Latest verification included 23 focused tests, 76 passed and 2 skipped in
  the core suite, and 79 passed and 1 skipped in the inference suite.
  The bounded-run reporter replayed the stored holdout arrays and accepted
  practice candidate hashes. See the
  [comparison and raw evidence paths](../results/original_weight_edits/local-qwen-mcp-smoke-comparison.json).
