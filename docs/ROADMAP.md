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
| Phase III-C. Behavior-grounded adaptation | MEASUREMENT FAIL / REAL EDITS HOLD | The locked content rule scored 23/48 versus 28/48 for letter scoring on qualification. Only 1/48 content generations passed the parser. See the [Phase III-C study](PHASE_III_C.md). |
| Phase III-D. Counterfactual behavioral grounding | MEASUREMENT FAIL / REAL EDITS HOLD | The locked ranker beat letter scoring 17/24 to 13/24 but got both question directions correct on 11/24 pairs. Strict bullet generation parsed 24/24 and failed its agreement gate. See the [Phase III-D study](PHASE_III_D.md). |
| Phase III-E. Adaptive neural meta-compiler | FAMILIAR SYNTHETIC UTILITY GAIN / SHIFT FAIL / REAL EDITS HOLD | Learned selection reduced harmful finalists on familiar mechanisms. Exact sensitivity reversals produced one harmful finalist per twin pair. See the [Phase III-E study](PHASE_III_E.md). |
| Phase III-F. Model-agnostic adaptation | ENGINEERING PASS ON TWO FAMILIES / LEARNING HOLD | The shared intent compiled to distinct Qwen3.5 and Llama affine plans. Both passed materialization, fresh reload, and source restoration. See the [Phase III-F study](PHASE_III_F.md). |
| Phase III-G. Behavioral qualification | QWEN QUALIFICATION PASS / SINGLE-CYCLE FAIL | Targeted negative-decimal errors replicated. Both fitted candidates failed fresh greedy selection and the hidden roster remained closed. See the [Phase III-G study](PHASE_III_G.md). |
| Phase III-H. Endpoint-aligned adaptation | SHARED ENDPOINT PASS / SINGLE-CYCLE HOLD | All six Qwen probes lowered fitting loss and regressed actual responses. The guard rejected each before materialization. SmolLM2 independently replayed the shared endpoint and rollback. See the [Phase III-H study](PHASE_III_H.md). |
| Phase III-I. Causal fork adaptation | PRACTICE GAINS / SINGLE-CYCLE HOLD | A trajectory-constrained Qwen update gained five targets across both wordings and regressed three previously correct responses. No update qualified for selection. The trained conventional merge exceeded its frozen cap. See the [Phase III-I study](PHASE_III_I.md). |
| Phase III-J. Counterexample-guided adaptation | PRACTICE PASS / SELECTION FAIL / PERSISTENCE HOLD | The rounded E3 update gained three practice items with zero regressions. Fresh selection gained one item and regressed two previously correct responses. See the [Phase III-J study](PHASE_III_J.md). |
| Phase III-K. Robust J-space | PRACTICE AND VALIDATION PASS / SELECTION FAIL | One rounded Qwen correction survived separate preservation validation. Fresh selection preserved every previously correct response and gained no target items. See the [Phase III-K study](PHASE_III_K.md). |
| Phase III-L. Semantic transfer and gradient consensus | PRACTICE, VALIDATION AND SELECTION PASS / SEALED FAIL | The shared-right consensus edit gained three strict practice comparisons and one on selection. The sealed evaluation gained three and regressed one previously correct presentation. Sequential learning remains on HOLD. See the [Phase III-L study](PHASE_III_L.md). |
| Phase III-M. Causal gradient decomposition | MATCHED LARGE-NORM FACTORIAL / RULE TRANSFER HOLD | Six practice candidates gained fresh development comparisons with zero regressions, but no full truth-flip family qualified. Objective choice had zero strict development effect, while UNIT reduced gain on this cohort. Independent selection stayed gated. See the [Phase III-M study](PHASE_III_M.md). |

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


## After Phase III-N

Completed: paired supervision allocation, fixed inherited geometry,
comparison-weighted native gradients, pair-aware numerical synthesis,
matched BF16 probes, full generation, five independent development reloads,
exact B2 reconstruction and original restoration. One complete practice
counterfactual family is recorded.

HOLD: complete-family transfer on fresh development, separate preservation
validation, independent selection, hidden verification, lineage admission,
and sequential learning. Paired allocation did not outperform one-sided
allocation on fresh strict gains at the larger norm.

Next: freeze a small joint per-presentation feasibility experiment with
preservation, within the existing 0.025 maximum. Test whether the current
0.020 dose lies below a selective paired correction threshold. Require an
actual rounded witness before a separately frozen dose or geometry trial.
Use fresh families excluding every disclosed Phase III-N transformation.
Do not refit the finished study or expand rank solely because it held.
[Phase III-N](PHASE_III_N.md)

## After Phase III-O

Completed: fresh per-presentation native fork measurements, averaged-versus-
individual minimum-norm systems, preservation-compatible systems, numerical
dual certificates and independent arithmetic replay. The fixed-right local
individual-target minimum exceeds the 0.025 cap globally and for each fitted
family. The host correctly stops before candidate generation and later data.

HOLD: verified persistent single-cycle improvement, complete-family transfer,
independent selection, sealed evaluation and sequential learning.

Next: a separately preregistered optimized-right-factor rank-one feasibility
study. Keep the authorized columns, layer, norm envelope and complete-generation
guard. Test whether a new unit right factor lowers joint individual-target cost
and survives BF16 plus full decoding with zero prohibited regressions. Do not
raise the norm cap or expand rank in the completed O study. A numerical witness
alone remains insufficient for retention. [Phase III-O](PHASE_III_O.md)

## After Phase III-P

Completed: full authorized block derivatives, bounded Gram relaxation,
independent Frobenius/spectral bound checks, tested right-factor search and
the real-model preflight. Its relaxed target-only bound already exceeds the
unchanged cap, so orientation and candidate generation correctly remain unused.
A separately authorized external patch is exactly reconstructed and evaluated
in two fresh processes on a reference-only roster.

HOLD: selective decimal-rule correction, fresh complete-family transfer,
independent qualification, new lineage admission and sequential learning.
The external edit's isolated gains include a prohibited loss.

Next falsifiable experiment: choose a different authorized sixteen-column
region using joint target/preservation evidence, retaining layer, rank, support
budget, norm cap and endpoint. Compare with the inherited region on fresh
families. Require a below-cap local witness followed by BF16, full generation
and the unchanged independent gates. Rank two inside the unchanged block is
not the first justified response to an all-rank local relaxation failure.
Exclude all disclosed P rosters from new hidden confirmation. No such follow-up
is silently included in the completed trial. [Phase III-P](PHASE_III_P.md)

### Phase III-Q baseline supersedes the unrun column proposal

The operator replaced that proposed column search with a conventional
supervised LoRA baseline. The column experiment remains unrun. The completed
high-rate recipe passed practice, development, preservation, independent
selection, sealed verification, exact BF16 reconstruction and rollback.
The low-rate arm was interrupted, so the learning-rate comparison is
unmeasured. No model was deployed or admitted to lineage.

Next: preregister a replication of the winning recipe on new training and
evaluation families. If the same complete-family improvement and zero-loss
preservation repeat, vary one capacity factor at a time: supervision volume,
module coverage or rank. Sequential learning remains untested. The baseline
does not establish that a restricted direct edit can match it.
[Phase III-Q baseline](PHASE_III_Q_BASELINE.md)
