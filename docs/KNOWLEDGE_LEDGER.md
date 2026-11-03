# Evidence-linked research knowledge ledger

This ledger summarizes current recorded evidence. The authoritative raw
records are append-only in [`results/experiment_ledger.jsonl`](../results/experiment_ledger.jsonl);
per-run JSON, model receipts and raw generations are listed below. Corrections
are appended and do not rewrite earlier records. The plotting runner
canonicalizes only explicit supersession and duplicate measurement keys.

Phase III-H appends one matched Qwen practice experiment and one SmolLM2
engineering replay. The v6 audit derives new diagnostics from the archived
generations and is not counted as a new model experiment. The Qwen probes
all improved the conditional score and all regressed actual generation;
none entered lineage. [Evidence, limitations and next experiment](PHASE_III_H.md)

Phase III-I adds one Qwen adaptation trial and one SmolLM2 portability
run. Token-level and completed practice corrections coincided on the
tested forks. Sampled margin preservation did not cover the three
regressed presentations that blocked the strongest Qwen probe.
[Evidence, limitations and next experiment](PHASE_III_I.md)

## Run observations

| Observation | Hypothesis / intervention | Result and interpretation | Limits / next test | Evidence |
|---|---|---|---|---|
| v0.1 demo was reproducible | A pre-specified sparse correction should alter the targeted synthetic defect | 75.30% to 98.45% on 2,000 seeded examples; confirms the original tooling demo | It was not discovered by an agent; repeat only for code regression checks | [`evaluation.json`](../results/v01_demo_reproduction/evaluation.json), [`candidate.npz.receipt.json`](../results/v01_demo_reproduction/candidate.npz.receipt.json) |
| Published p50c patch reconstructs exactly | Public sparse values should reproduce the author's two target tensor hashes | 130,188 BF16 changes; both tensor hashes and full candidate hash match; source unchanged | Reconstruction alone says nothing about behavior | [`p50c_rebuild_summary.json`](../results/published_reproduction/p50c_rebuild_summary.json), [`p50c_host_receipt.json.gz`](../results/published_reproduction/p50c_host_receipt.json.gz) |
| Qwen paired smoke is unscorable | A low-cost initial smoke can reveal gross generation differences | Both variants hit the 64-token cap on 2/2 prompts and completed 0 answers; no measured accuracy difference | Increase to full protocol only after GPU/runtime and dataset readiness; do not tune on these prompts | [`paired comparison`](../results/published_reproduction/qwen_runner_paired_comparison_v02.json), [`base run`](../results/published_reproduction/qwen_runner_baseline_v02/run.json), [`p50 run`](../results/published_reproduction/qwen_runner_p50c_v02/run.json) |
| Restricted toy diagnosis found the injected correction three times | Activation and error evidence plus bounded candidate evaluation can identify a one-coordinate repair | Practice-only search selected output_bias[1] += -4; three isolated 64-item seeds scored 64/64 vs baseline 32/64; paired audit shows 32 rescues and 0 regressions per seed; random control had 0 rescues and 0 regressions | The allowlist and one-sided defect are narrow; test different defect families and coordinates | [`seed 731`](../results/original_weight_edits/run-20261103T023455Z/experiment_record.json), [`seed 840`](../results/original_weight_edits/run-20261103T024129Z/experiment_record.json), [`seed 841`](../results/original_weight_edits/run-20261103T025154Z/experiment_record.json), [`derived paired correctness`](../results/original_weight_edits/derived_paired_correctness_v2.json) |
| CLI tool path can inspect and materialize a bounded edit | An AI caller with subprocess access can invoke WeightBridge tools while host policy constrains the patch | On operational seed 842, CLI inspection read the 12-parameter toy checkpoint and CLI apply reproduced the worker candidate hash with one authorized `output_bias` change; no new score was collected by the CLI replay | This confirms CLI interoperability, not an LLM integration or new behavioral result; the built-in worker remains deterministic | [`seed 842 run`](../results/original_weight_edits/run-20261103T035435Z/experiment_record.json), [`CLI replay record`](../results/tool_interface_smoke/seed842_cli_application.json), [`policy`](../results/tool_interface_smoke/seed842_cli_policy.json) |
| Codex found a hidden-threshold correction through MCP | An LLM can use scoped inspection and practice scoring to propose a one-entry edit without receiving the correction | From practice-only errors and activation summaries, Codex proposed `hidden_bias[0] += -0.7`; practice rose 42/64 to 64/64; after finalization, the reloaded candidate scored 64/64 on holdout vs base 35/64 and matched random 35/64; the hidden oracle later confirmed -0.7 was exact | One in-distribution synthetic task is a pilot; the task-seed transfer gate remains INCONCLUSIVE; Codex token use and MCP child CPU were unavailable | [`experiment record`](../results/original_weight_edits/mcp-pilot-18d5cab750/experiment_record.json), [`holdout evaluation`](../results/original_weight_edits/mcp-pilot-18d5cab750/evaluation.json), [`MCP events`](../results/original_weight_edits/mcp-pilot-18d5cab750/candidates/sessions/codex-29629e511d7c4647a495fc5ac0d03480/events.jsonl), [`interaction supplement`](../results/original_weight_edits/mcp-pilot-18d5cab750/interaction_audit_supplement.jsonl) |
| Conventional SGD also solves the toy defect | A gradient baseline may solve the same task | 400-step SGD reached 100% on 320 examples and changed all 12 parameters | Not a language-model baseline or compute-matched comparison | [`gradient record`](../results/original_weight_edits/gradient-20261103T030417Z/experiment_record.json) |
| Reused verified memory reduced search cost on the same synthetic pattern | The exact stored repair can shorten diagnosis | External-memory and memory+weight conditions scored 640/640; weight-only scored 628/640; no-edit scored 320/640. Weight-only used 8 candidate evaluations per seed; memory+weight used 2 | Memory condition is a decision-time logit bias; one repeated defect class and ten seeds do not establish general memory benefit | [`MWC batch 1`](../results/memory_ablation/run-20261103T024116Z/mwc_ablation.json), [`MWC batch 2`](../results/memory_ablation/run-20261103T025158Z/mwc_ablation.json) |
| Candidate edits persist after file reload in toy runs | A saved candidate should retain its changed behavior across restart | Toy evaluator loads candidates from disk before scoring; all experiment receipts include candidate hashes | No persistent Qwen improvement has been demonstrated | Same toy records above |
| Randomized Qwen control matches edit size | Matched count and magnitude control can separate targeted edit from arbitrary change | Same 130,188 changed-value count in layers 15 and 19, within the sampled edit-magnitude distribution | No behavioral evaluation; cannot infer a regression or null effect | [`random-control reconstruction`](../results/published_reproduction/qwen35_random_control_rebuild_summary.json) |

## Cross-run patterns supported by current data

- Three isolated toy runs associate the localized failure with one output-bias
  entry; both selected the same -4 change. This is causal for the deliberately
  constructed task because the intervention changed the relevant score and the
  independent holdout improved. It is not evidence that language-model
  transformer circuits are localized in the same way.
- The same 10-seed MWC runs show the external record can provide a useful
  diagnostic shortcut when its provenance exactly matches the defect. Since
  external memory alone scored as well as memory plus weights, the evidence
  does not show that the weight edit adds value over retrieval-time behavior
  correction for this task.
- Weight-only search can select a suboptimal magnitude: one seed retained 12
  holdout errors with -3, while retrieved evidence supplied -4. This is a
  concrete candidate-interference/magnitude-search case, not a general law.
- Published p50c modifies a small fraction of Qwen state entries. No measured
  Qwen accuracy, loop or locality outcome here establishes that its magnitude
  caused a behavioral change.
- Neither layer frequency nor a parameter-count/performance correlation can be
  estimated from one published transformer patch plus a toy bias task.
- Cost per validated improvement is task- and evaluator-dependent. The current
  records are useful for future accounting, not a cross-method ranking.
- A deterministic replay of frozen toy files recovered per-example rescue
  and regression counts. The appended derivation record contributes no new
  independent measurement.
- One Codex MCP session inferred and applied a hidden-bias correction after
  observing practice-only failures. Its matched random edit did not improve
  holdout accuracy. This establishes one AI-directed discovery instance, not
  transfer across random seeds or defect types.

## Metric definitions

For paired binary correctness (b_i,c_i\in\{0,1\}), accuracy is
\(A=N^{-1}\sum_i c_i\), and the candidate difference is
\(\bar d=N^{-1}\sum_i(c_i-b_i)\). Regression rate is
\(\sum_i 1[b_i=1,c_i=0]/\sum_i 1[b_i=1]\); when the baseline has no correct
items it is undefined. For MWC, uncertainty resamples the 10 independently
generated task seeds as clusters, retaining the 64 paired examples inside each
resampled seed. Changed fraction is the number of represented parameter
values whose bits differ after dtype conversion divided by the model state
entry count; touched coordinates that round to their base value are counted
separately. Token-limit rate is the share of generations ending at max-new-
tokens. The smoke's tail-repeat and repeated-block signals are diagnostic
proxies and are not validated reasoning-loop metrics.

## Unrun prior protocol v1

Evaluate the restricted diagnosis worker on 20 newly generated toy defect
instances from four predeclared mechanisms (five independent seeds each),
including defects outside `output_bias`. Freeze the practice data, allowlist,
eight-proposal budget, evaluator, matched random control and holdout before
running the agent. The test passes only if the agent's frozen candidate repairs
at least 70% of holdouts, has no more than a 5% regression rate among baseline-
correct items, and exceeds the random control by at least 20 percentage points
with a seed-clustered paired 95% interval whose lower bound is above zero.
Otherwise record FAIL or INCONCLUSIVE by the predeclared gate. This directly
tests whether the current discovery result transfers beyond its injected
coordinate while retaining provenance and evaluator separation.

## Next falsifiable experiment: MCP LLM protocol amendment v2

The v1 study above was not run. After the single Codex MCP pilot, amend the
agent under test to a pinned LLM host using the new bounded tools. Run 20 new
instances across the same four predeclared single-weight mechanisms, five
independent seeds each. Freeze and hash the model/version, system/tool prompt,
host policy, practice/holdout generator, eight-candidate budget, evaluator,
and random-control procedure before the first instance. Each session gets
practice data only; the host scores the fixed candidate and matched random
edit on holdout once after finalization. Store every MCP request, response,
rejection, candidate receipt and resource record. Keep this cohort separate
from the deterministic worker and from the one-instance pilot.

Use the original v1 gate unchanged: at least 70% of holdouts repaired, at most
5% regressions among baseline-correct items, and at least a 20 percentage-point
advantage over matched random edits with a task-seed-clustered paired 95%
lower bound above zero. If the host cannot pin model identity or retain full
tool traces, mark the result INCONCLUSIVE rather than treating an unrepeatable
LLM response as independent evidence.

### Protocol v2 preflight status

The host task selector, 20-slot balanced manifest template, fixed diagnostic
prompt, matched same-tensor random control, transcript hash checks and
mechanism-stratified cohort analyzer are implemented and covered by tests.
Preflighting the prior one-task pilot against this manifest returns
INCONCLUSIVE: it is not bound to the protocol, 20 task records are absent, and
the current host cannot verify the LLM provider/model/version/sampling data.
The current Codex context can access repository files, so it cannot serve as
the isolated MCP-only diagnostic host required by the protocol. This is a
readiness audit, not a behavioral result; the v2 transfer gate remains
unrun. The preflight output is
[`mcp-cohort-v2-preflight.json`](../results/original_weight_edits/mcp-cohort-v2-preflight.json).

### Local Qwen as the diagnostic model

**Observation.** The same pinned Qwen3.5-0.8B checkpoint drove fresh local
model contexts through the MCP broker. Raw model turns, tool exchanges,
practice candidate checkpoints and hidden scores are linked from the
[bounded-run summary](../results/original_weight_edits/local-qwen-mcp-smoke-comparison.json).

**Hypothesis and intervention.** Separate tool orchestration from choosing
an edit. In the patch form, the host requests diagnostics and evaluates
accepted proposals. The model still chooses the scalar edit, its rationale,
and whether to select a candidate. Each profile has a fresh seed roster.

**Result and interpretation.** The direct tool interface exposed repeated
requests and placeholder copying. The patch form allowed model proposals to
reach practice evaluation. One accepted patch changed an input weight for a
hidden unit with zero outgoing weights. A checkpoint reload confirmed that
the practice logits were unchanged.

**Limitations and next experiment.** These are bounded, exploratory
profiles with different splits and transport rules. A frozen follow-up can
test whether exposing graph reachability from permitted parameters to the
logits improves the fraction of proposed edits that affect practice behavior
under the same budget. Evaluate that change on fresh trials before using the
20-task transfer gate to make a claim about held-out repair.

### Rejection and deterministic search

Rejecting coordinates with zero outgoing connectivity redirected Qwen to
edits that changed logits on three of four tasks. The observed Qwen edits
left all practice classifications unchanged. The fixed coordinate schedule
found three practice gains that survived a checkpoint reload and improved
held-out accuracy. Two of those finalists exceeded their own matched
random control. This is a small comparison of workflows under a shared
evaluation ceiling, with different realized evaluation counts.
[Paired evidence](../results/wb-sr/comparison.json)

### Blind repair on a full Qwen checkpoint

**Observation.** The injected layer 23 edits lowered next-token answer
margins and produced accuracy loss on fresh examples of the same prompt
format. The first paraphrase holdout retained its original accuracy.

**Hypothesis and intervention.** A practice-fitted MLP update can raise the
target answer margin, while fitting and ranking against unrelated activations
can limit collateral changes. The margin-budget protocol fixed that ranking
before evaluating the holdout.

**Result and interpretation.** The margin-budget patch recovered 14/16 lost
answers after reload, outperformed three norm-matched controls, and matched
pristine predictions on 32 unrelated prompts. It also reversed one unrelated
error introduced by the injection. The protocol still scores this as a
locality failure because it uses the damaged checkpoint as the reference.
[Evidence and protocol](QWEN_BLIND_REPAIR.md)

**Limitations and next experiment.** Strength, sample construction and
ranking changed across trials, so their separate effects cannot be estimated
from this series. Freeze both locality references and a compute-matched
search control before a cohort with balanced answer positions, multiple
unknown injection sites and paraphrased holdouts.

### Matched search on a blinded Qwen cohort

**Observation and intervention.** A balanced 12-case study compared nine
signed-activation proposals with nine norm- and support-matched random
proposals per case. Both search arms saw the same practice prompts and the
evaluator loaded the finalists independently.

**Result and interpretation.** Guided search beat random in 11 cases and
passed the full gate in five. The gains concentrated in the practice
wording. One same-wording improvement hid a six-answer paraphrase
regression. All guided finalists edited layer 23, including earlier-layer
faults. Practice gain and the chosen repair layer therefore offer limited
evidence of cross-wording recovery or fault localization.

**Limitations and next experiment.** Both fault patterns alter one
rank-one output direction in the same architecture. On fresh cases, hold
the candidate budget fixed and compare single-wording with multi-wording
practice, then evaluate on wording unseen by either arm. Test whether
this reduces paraphrase regressions while preserving pristine sentinels.
[Cohort evidence](QWEN_REPAIR_COHORT.md)

### Varying practice wording

**Intervention.** Both searches used the same number pairs, sentinels,
patch limits, and nine candidate evaluations. One used a single template
and the other used three. The evaluator tested two novel templates.

**Result and interpretation.** Multi-wording beat single wording by one
of 132 paraphrased answers and changed one more pristine sentinel
prediction. Neither crossed the eight-case gate. A single clean transfer
case shows feasibility, while the case-level comparison gives little
evidence that this mix improves transfer on the tested faults.
[Ablation evidence](QWEN_WORDING_ABLATION.md)
