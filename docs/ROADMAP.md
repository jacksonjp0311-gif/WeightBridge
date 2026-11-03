# WeightBridge research roadmap

## Current milestone disposition

| Milestone | Status | Evidence |
|---|---|---|
| 1. Preserve and reproduce v0.1 | COMPLETE | Baseline commit `321547d`; 11 original tests passed; v0.1 demo rerun |
| 2. Real-model inspection and patch reconstruction | COMPLETE for local safetensors/Qwen path | Sharded/unsharded inspection and host-policy patcher; p50c hashes match |
| 3. Bounded original-versus-published benchmark | PARTIAL / HOLD | Small CPU smoke completed but truncated; published datasets/scores not run |
| 4. AI-directed diagnosis and candidate edit | PILOT COMPLETE / TRANSFER HOLD | One Codex MCP discovery; balanced v2 cohort protocol and evaluator implemented, but formal execution is HOLD pending pinned model identity and isolated MCP-only host |
| 5. Memory-guided diagnosis | PILOT COMPLETE / GENERAL CLAIM HOLD | Four-condition synthetic ablation across ten task seeds |

## Immediate next experiment

The balanced 20-task protocol, fixed prompt, host policy, task assignments,
source hashes, matched-control rule and stratified evaluator are implemented
in `experiments/original_weight_edits/`. The current checked-in file is a
template, not a runnable study: preflight marks it HOLD because this Codex
host does not expose a verifiable provider model/version or sampling receipt,
and the current agent context has repository access. Do not run the cohort
from this workspace as if it were isolated. The formal study needs a fresh
session for each task behind an MCP-only host with no filesystem, repository
or network tools; then pin and freeze the identity, sampling configuration,
prompt and hashes before the first task.

The gate remains at least 70% fully repaired holdouts, at most 5% regression
among baseline-correct items, and at least a 20 percentage-point advantage
over matched random edits with a mechanism-stratified paired 95% lower bound
above zero. Archive every call, rejection, failure and candidate receipt. Do
not extend MCP to real Hugging Face inference until this synthetic transfer
gate and the Qwen resource gate are satisfied.

## Qwen benchmark gate

Do not start the full Experiment 8 matrix until all of these are true:

1. The original model and p50c reconstruction are pinned and verified again in
   the intended runtime.
2. Dataset dependencies and exact frozen benchmark data versions are prepared;
   hashes, splits, grading code and license terms are recorded.
3. A representative time and memory estimate supports the planned samples,
   32,768-token cap and 10/25/50-pass experimental history without altering
   the author configuration silently.
4. Base, p50c, matched random and an appropriate conventional baseline share
   the same frozen prompts, sampling settings and paired seeds.
5. Practice problems remain separate from final holdout problems; no outcome
   from the final holdout feeds patch selection.

After the gate, first run a predeclared paired subset with the exact author
settings and enough tokens to finish. Expand only when the recorded runtime
supports it. The full paper result is not reproduced until each suite and
sampling count match the published protocol and independent metrics are
recorded.

## Further work after the gates

- Extend policy-safe patches with optional bounded low-rank edits and compare
  them with the current sparse edits on a task with a frozen evaluator.
- Add multi-writer-safe append semantics, ledger schema migration and
  artifact-content verification for every referenced path.
- Probe activation/attention summaries on Qwen only after a benchmark failure
  is observable and the probe can be tied to a testable hypothesis.
- Test memory retrieval with related and mismatched experiences, equalized
  evaluator calls, multiple defect families and a checkpoint-restart boundary.
- Study edit magnitude and parameter count only with deliberately varied
  interventions and enough independent defects to estimate a response curve.
- Keep integration with RecallTree, MemoryAtlas, Genome, ExperimentForge and
  private PEGASUS data out of scope until separately authorized and backed by
  these isolated results.
