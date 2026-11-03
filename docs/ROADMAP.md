# WeightBridge research roadmap

## Current milestone disposition

| Milestone | Status | Evidence |
|---|---|---|
| 1. Preserve and reproduce v0.1 | COMPLETE | Baseline commit `321547d`; 11 original tests passed; v0.1 demo rerun |
| 2. Real-model inspection and patch reconstruction | COMPLETE for local safetensors/Qwen path | Sharded/unsharded inspection and host-policy patcher; p50c hashes match |
| 3. Bounded original-versus-published benchmark | PARTIAL / HOLD | Small CPU smoke completed but truncated; published datasets/scores not run |
| 4. AI-directed diagnosis and candidate edit | REAL CLI PILOT COMPLETE / DIRECT TRANSFER FAIL | One passing real Qwen CLI case out of three. The frozen 20-task direct-proposal gate failed with zero finalized repairs. |
| 5. Memory-guided diagnosis | PILOT COMPLETE / GENERAL CLAIM HOLD | Four-condition synthetic ablation across ten task seeds |
| Phase III. Continual research adaptation | CONTROLLER IMPLEMENTED / MODEL QUALIFICATION FAIL | One of five natural Qwen cycles retained an edit. Final score fell from 18/24 to 15/24. See the [Phase III study](PHASE_III.md). |
| Phase III-B. Behavioral Interaction Compiler | SYNTHETIC ADMISSION PASS / QWEN PILOT ZERO ACCEPTANCES | Mixed curvature helped constructed dormant cases. Both coordinated Qwen arms reversed every A/B date prediction with no net gain. See the [Phase III-B study](PHASE_III_B.md). |

The separate breakpoint solver study completed 20 paired toy tasks. The
deterministic selector fully repaired 20/20 and the Qwen selector repaired
19/20. Both passed the solver-assisted toy gate. The direct-proposal AI
transfer gate remains held. [Solver evidence](../results/wb-bf/comparison.json)

The deterministic Qwen checkpoint pilot is a separate track. It fits a bounded
weight patch from damaged-model activations and practice scores, then reloads
the patch in a separate evaluator with matched random controls. The next
research gate after a single induced-fault trial is replication on blinded
defect seeds, answer positions and wording. The [12-case Qwen cohort](QWEN_REPAIR_COHORT.md)
defines the frozen protocol and budget-matched comparison. A natural-failure
trial needs a fresh protocol and a held-out generation benchmark.

The [AI-directed CLI pilot](QWEN_AI_EDIT_PILOT.md) completed a bounded
practice-feedback loop on three fresh Qwen faults. Qwen chose layers and
strengths and the host fit each edit from practice activations. One saved
candidate recovered all three lost targets and preserved all unrelated
predictions after reload. One of three cases passed. The README documents
the separate product evaluator for any compatible local Hugging Face
candidate and host-owned forced-choice suite. The [local JSON tool
session](AGENT_TOOLS.md#local-hugging-face-edit-sessions) packages this
bounded practice loop for a local agent harness. The real-model MCP
promotion gate below remains in force.

## Direct-proposal qualification

The balanced 20-task protocol, fixed prompt, host policy, task assignments,
source hashes, matched-control rule and stratified evaluator are implemented
in `experiments/original_weight_edits/`. The local Qwen host pins checkpoint
and runtime identity, freezes the protocol and starts a new model context for
each trial. The model receives the prompt and tool responses. The
[completed qualification](../results/wb-dq/cohort_summary.json) used
one frozen interaction mode and a fresh roster across all 20 tasks.
The local Qwen controller finalized zero repairs. All 37 accepted
practice evaluations left logits unchanged at the measured tolerance.
The earlier exploratory and solver-assisted runs retain separate receipts.

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
