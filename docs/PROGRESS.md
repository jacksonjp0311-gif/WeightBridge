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
| E. AI-directed diagnosis | COMPLETE for one injected toy failure family | Three-run fixed cohort plus seed-842 operational smoke; built-in worker is deterministic, not an LLM; no Qwen patch proposal |
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
all generations hit the token cap. Three-run synthetic discovery and memory ablation
support only the deliberate one-sided toy defect. No Qwen weight edit was
discovered or validated by the agent, and no private memory was loaded.

## Verification record

Final checks on 2026-11-03:

- Core environment: `.venv\Scripts\python.exe -m pytest -q` — 53 passed,
  2 optional-dependency skips.
- Inference-capable reference environment: 57 passed.
- `python -m compileall -q weightbridge benchmarks experiments tests` passed.
- `benchmarks/runners/plot_results.py` completed and generated seven figures
  from the ledger records. The toy aggregate remains fixed to its original
  three-run cohort; the later seed-842 smoke is retained separately.
- Ledger audit: 32 append-only rows, 32 unique experiment IDs, 11 nonempty
  unique measurement keys, and 14 canonical experiments in the summary. All
  record-listed artifact files exist.
- A fresh seed-842 same-family smoke ran the diagnostic worker on 64 practice
  and 64 hidden items: base 32/64, selected candidate 64/64, random control
  32/64. The worker did not see hidden labels; the run is not included in the
  original fixed three-run aggregate. Direct `inspect` and `apply` CLI calls
  reproduced the recorded candidate hash under a one-entry host policy; the
  CLI replay itself did not score a holdout.
- Full suite rerun after the smoke: core 53 passed, 2 optional skips;
  inference-capable environment 57 passed.
- `git diff --check` passed; Git emitted only its configured LF-to-CRLF
  working-copy notices. No model checkpoint files are stored in the project.
- Canonical experiment records report 453.34 process CPU seconds and 205.06
  summed wall seconds. These exclude coding, tests, reference inspection and
  other unmetered work, and combine distinct experiments.

Implementation and evidence commit: `fd739a6746b01d97dae64248e674f7e4a6985574`.
This verification-note update is committed separately. Both commits remain on
`dev/weightbridge-v0.2`; nothing was pushed or deployed.
