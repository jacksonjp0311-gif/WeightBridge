# WeightBridge v0.2 findings

## Evidence boundary

This report distinguishes reconstruction, software checks, synthetic behavior,
and language-model benchmarks. The published Qwen scores in the article are
reference claims only. No published MATH-500, GSM8K, or IFEval score was
measured independently in this workspace.

## What was reproduced

- The supplied v0.1 deterministic demo was rerun from its preserved code:
  baseline accuracy was 75.30%; its pre-specified patch candidate reached
  98.45% on 2,000 deterministic examples (seed 813). The patch was known in
  advance and this is a tooling demonstration, not agent discovery.
- The authors' Experiment 8 p50c patch was applied to the pinned
  `Qwen/Qwen3.5-0.8B` checkpoint (revision
  `2fc06364715b967f1860aea9cf38778875588b17`). WeightBridge independently
  reconstructed 130,188 BF16 value changes across two MLP `down_proj` tensors.
  Both edited tensor SHA-256 hashes and the resulting whole-file candidate hash
  matched the recorded values. The original checkpoint was not overwritten.
- A matched-count randomized Qwen control was constructed in the same layers
  with the same per-layer number of changed values and sampled edit-magnitude
  distribution. Its behavior was not evaluated, so it is only a ready control
  checkpoint, not a measured negative result.

The authors report p50 MATH-500 48.1%, GSM8K 67.1%, IFEval 62.3%, and a 33.1%
MATH token-limit rate. These remain author-reported results. Our patch
reconstruction does not reproduce those metrics.

## Qwen behavior smoke

On two hand-authored arithmetic prompts, one generation per prompt, fixed
sampling seed 0/1, CPU BF16 inference and a 64-token cap, both base and p50
candidate reached the imposed limit on both prompts. Neither produced a
complete boxed answer. Accuracy and answer-completion rates were 0/2 for both;
the paired difference is zero and the paired bootstrap interval is [0, 0].
At this cap, the loop detector and 8-gram tail repetition proxy were also zero.
This outcome is INCONCLUSIVE because the answers were truncated before the
task could be scored.

An earlier exploratory two-prompt, 256-token smoke also hit the cap on all
four generations and yielded no boxed answers. Its tail repetition proxy was
0.004132 for base and 0 for p50. The discrepancy in proxy is not evidence of a
behavioral improvement: n is two and neither output completed. Both runs are
far below the published 32,768-token setting and use no benchmark dataset.

## Independent weight-edit discovery on a toy defect

The restricted practice-only diagnostic worker was tested twice on fresh task
seeds. It received a frozen practice set, read-only activation summaries, a
fixed allowlist (`output_bias`), and a maximum of eight bounded candidate
evaluations. It did not receive the hidden holdout or its labels. It selected
`output_bias[1] += -4` without being handed the correction value. The parent
validated the patch, wrote an immutable candidate, reloaded it from disk, and
then scored the hidden holdout once.

Across three isolated 64-item holdouts (192 examples), baseline accuracy was
50% and the diagnostic candidate was 100%; a one-entry random control remained
at 50%. Each run's paired 95% interval was [37.5%, 62.5%] at n=64, and all
three runs selected the same correction. The edit changed one of twelve
parameters. This is a
successful discovery by a deterministic tool-using search agent on a
deliberately localized synthetic defect. It is not an LLM discovery, Qwen
improvement, new mathematical capability, or evidence of general learning.

The seed-731 task was initially recorded twice. An append-only correction
marks the first record as a duplicate replay; the canonical isolated run is
retained with later isolated seeds 840 and 841.

### Additional operational smoke: seed 842

After the original three-run cohort was frozen, seed 842 exercised the same
injected defect through the bounded diagnostic worker. It scored 32/64 at
baseline and 64/64 after selecting `output_bias[1] += -4`; the matched random
control scored 32/64. The worker used eight practice-evaluator calls, did not
access the holdout, and spent zero model-generated tokens. This repeats the
same defect family and does not establish transfer or LLM discovery. The
ledger retains the run, while the three-run aggregate and its figures remain
fixed to the original cohort.

I also called the public `weightbridge inspect` and `weightbridge apply` CLI
directly on that run's base and patch. The host policy allowed one scalar edit
to `output_bias`, and the resulting candidate hash matched the worker's
receipt. This was a patch-application replay only: the CLI replay did not
score the holdout or create another behavioral measurement. See the
[`seed-842 experiment record`](../results/original_weight_edits/run-20261103T035435Z/experiment_record.json)
and [`CLI replay receipt`](../results/tool_interface_smoke/seed842_cli_application.json).

## LLM-directed discovery through MCP

I used the MCP tools as the diagnosing model on a fresh task selected from four
single-weight defect mechanisms. The host exposed a bounded report of tensor
statistics, practice errors and selected activations; the correct checkpoint,
oracle mechanism and holdout labels were not available in the tool session.
From one-sided class-0 failures and the first hidden activation, I hypothesized
a shifted hidden threshold and proposed `hidden_bias[0] += -0.7`. The host
accepted the one-entry patch under its fixed allowlist and policy. Practice
accuracy rose from 42/64 to 64/64, with 22 rescues and zero regressions.

After the candidate was finalized, the host reloaded it from disk and scored
the previously withheld holdout once: baseline 35/64 (54.69%), candidate 64/64
(100%), and a paired 95% bootstrap interval of +32.81 to +57.81 percentage
points over the 64 paired examples. A one-entry random edit matched for
parameter count and absolute edit magnitude scored 35/64. Only after this
evaluation did the host reveal that the injected defect was `hidden_bias[0]`
at -0.3 and that the correction was -0.7. The chosen edit matched it exactly.

One proposed call was rejected because I copied the base hash with a trailing
extra character. The rejection created no candidate; the corrected proposal
was accepted. The server event log and versioned interaction supplement retain
the accepted sequence and the rejected attempt. The MCP session used five
tool calls, eight possible candidate slots, and no gradients. Token usage for
the Codex model was unavailable. This is one successful LLM-directed,
in-distribution synthetic discovery and a saved candidate reload, not evidence
that the method transfers across task seeds, mechanisms, language models, or
Qwen. Its experiment disposition remains INCONCLUSIVE because this one task
does not satisfy the predeclared multi-instance transfer gate.

Evidence: [`experiment record`](../results/original_weight_edits/mcp-pilot-18d5cab750/experiment_record.json),
[`holdout evaluation`](../results/original_weight_edits/mcp-pilot-18d5cab750/evaluation.json),
[`MCP event log`](../results/original_weight_edits/mcp-pilot-18d5cab750/candidates/sessions/codex-29629e511d7c4647a495fc5ac0d03480/events.jsonl),
and [`interaction audit supplement`](../results/original_weight_edits/mcp-pilot-18d5cab750/interaction_audit_supplement.jsonl).
The public evidence bundle includes the synthetic holdout and post-run oracle
for auditability. The MCP caller did not receive them before finalization, but
this disclosed split must never be reused as a hidden test.

## Conventional optimizer and memory ablation

Full-batch SGD on the same toy model, trained for 400 cross-entropy steps on
practice data only, also reached 100% accuracy on 320 held-out examples across
five seeds (baseline 50%; paired 95% interval for the difference
[44.375%, 55.625%]). It changed all 12 parameters. This is a conventional
gradient baseline and is reported separately from gradient-free search. The
toy comparison does not establish compute parity or compare against LoRA on a
language model.

The MWC pilot pools ten independent synthetic task seeds (640 paired examples
per condition):

| Condition | Accuracy | Paired gain vs no-memory/no-edit | 95% cluster bootstrap by seed | Regression rate |
|---|---:|---:|---:|---:|
| No memory, no edit | 50.00% | — | — | — |
| External memory only | 100.00% | +50.00 pp | [50.00, 50.00] pp | 0/320 baseline-correct |
| Weight edit only | 98.125% | +48.125 pp | [44.375, 50.00] pp | 0/320 baseline-correct |
| External memory plus weight edit | 100.00% | +50.00 pp | [50.00, 50.00] pp | 0/320 baseline-correct |

The external-memory-only condition applies a retrieved decision-time logit
bias and leaves the checkpoint unchanged. It is not weight learning. The
weight-only agent selected -3 on one seed and scored 81.25%; the verified
memory-guided candidate selected -4 and scored 100% on that same synthetic
failure family. In the recorded practice budget, the weight-only search used
eight candidate evaluations per seed while memory plus weight used two. The
retrieved experience was the exact prior repair for the same injected defect.
This supports a narrow result: verified external evidence reduced search
effort for this repeated localized failure. It does not show general MWC
benefit, transfer to unrelated defects, or a benefit over simply applying a
known correction.

A deterministic audit reloaded the three frozen toy candidates, controls and
already scored holdouts. It recovered 32 rescues and zero regressions for the
diagnostic edit on each seed; the random control had zero rescues and zero
regressions. This is a metric derivation for the same observations, not a new
experiment.

Every selected toy and MWC candidate was reloaded from a new checkpoint file
before scoring. That demonstrates persistence of those synthetic edits after
serialization/reload. A fresh Qwen process also reloaded the original model
successfully for rollback; no post-restart Qwen performance gain was shown.

## Weight-level observations and limits

The published patch targets 64 columns in layers 15 and 19, each tensor shaped
1024 × 3584. Its sparse artifact contains 65,260 and 64,928 changed BF16
values, respectively; 884 of the 131,072 targeted positions round to the base
value. The changed-value fraction is about 0.014905% of the 873,438,784
checkpoint state entries. These are structural observations of one published
patch. They do not identify the causal role of those columns.

No layer-frequency, magnitude-response, or cross-task interference pattern is
established by the available experiments. The toy edit touched an output bias,
whereas the published patch touches transformer MLP tensors; they are not
comparable intervention classes. The randomized Qwen control remains
unevaluated. No chart or aggregate in this report converts published metrics
into independent measurements.

## Recorded compute

The canonical ledger records 454.22 process CPU seconds and 289.05 summed
wall-clock seconds across the experiment records with resource fields. This
includes two Qwen CPU smoke runs, patch reconstruction, synthetic discovery,
MWC and the toy SGD baseline. The sum excludes development time, test time,
reference inspection, dataset downloads and other unmetered work. Process CPU
seconds can exceed wall time on a multithreaded run. See
[`benchmark_summary.json`](../results/benchmark_summary.json) for record-level
resource entries and evidence links. The total is refreshed from the ledger
after each recorded experiment; the MCP pilot's host CPU and elapsed time are
included, while its MCP child-process CPU and model-token use were unavailable.

## Open questions

- Can the diagnostic tool find a repair when the defect location and mechanism
  vary, rather than when a single permitted bias coordinate is known?
- Does a retrieval advantage persist when the stored experience is related but
  not identical, and when the weight-only agent receives the same number of
  evaluator calls?
- Do sparse weight edits survive broad locality and regression suites?
- Does the published p50c edit reduce repetition under the authors' exact
  context, prompt, sampling, and dataset protocol on this pinned checkpoint?
- Do any Qwen-directed edits discovered by the diagnostic system generalize
  beyond the practice failure and survive restart without hidden-set tuning?

See [`KNOWLEDGE_LEDGER.md`](KNOWLEDGE_LEDGER.md) for evidence-linked
observations and the next controlled experiments.

## Local Qwen diagnosis of synthetic weights

Qwen3.5-0.8B served as the diagnostic model. Each condition used fresh
synthetic splits and a frozen host receipt. The model saw practice evidence
and selected its own proposed weight edits.

| Interaction | Scored tasks | Accepted proposals | Finalized candidates | Host CPU seconds |
|---|---:|---:|---:|---:|
| Compact JSON tools with an example | 4 | 0 | 0 | 1,241.11 |
| Compact JSON tools with field descriptions | 4 | 0 | 0 | 959.33 |
| Host-routed patch form with a repeated-response stop | 4 | 8 | 0 | 1,010.39 |

The first profile copied a placeholder into 28 rejected requests. The
second repeatedly requested diagnostics. In the third, all accepted edits
targeted `input_weight[0, 1]`, which feeds a hidden unit with zero outgoing
weights in these toy networks. The affected unit has no contribution to
either logit, so those scalar edits could change its activation while leaving
classification unchanged. Reloading every accepted candidate confirmed a
maximum absolute practice logit change of zero. Eight submissions represent
two copies of one proposal per task.

Three fallback random controls reached 64/64 on individual holdouts. Their
magnitudes were set by the fallback rule because the agent stopped without
a candidate. The descriptive smoke outcomes offer no matched comparison
against a finalized AI patch. The profiles use different splits and
interaction rules, so the table does not estimate a causal effect of the
interface change.

The full cohort remains HOLD. An actionable follow-up is to expose a
read-only connectivity probe and test whether it increases the fraction of
proposals that alter practice logits under a fixed call budget on fresh tasks.
The earlier Codex pilot remains the only recorded LLM-proposed persistent
repair. The current runs use Qwen as a diagnostic caller and edit synthetic
checkpoints.

The [comparison artifact](../results/original_weight_edits/local-qwen-mcp-smoke-comparison.json)
links per-task scores and checkpoint hashes. Raw traces with local absolute
paths are retained locally. The published record includes their failure
summaries and, for the stopped Windows attempt, a hash of the omitted trace.
The comparison counts finalized candidates from scored candidate artifacts
because the broker's legacy `candidate_finalized` flag can also be true on a
stop event.

## Paired test of the connectivity probe

The control and probe each ran the patch form on the same four fresh
synthetic tasks. Each task used a fresh model context in both arms. The
schedule alternated arm order, and only the probe arm saw the disconnected
coordinate list. The scorer treated a task as active if at least one
accepted edit changed any practice logit by more than `1e-9`.

| Condition | Active tasks | Accepted submissions | Submissions to disconnected coordinates | Finalized candidates |
|---|---:|---:|---:|---:|
| Control | 0/4 | 8 | 8 | 0 |
| Connectivity probe | 0/4 | 10 | 10 | 0 |

All four pairs tied. Every accepted edit targeted `input_weight[0, 1]`,
and checkpoint replay found zero practice logit and prediction changes.
On the first probe trial the model mentioned the zero outgoing weights
but still submitted an edit to the disconnected coordinate. The result
shows a failure to use the supplied evidence in this particular model
and workflow. The 20-task transfer gate remains HOLD.

The study used 2,216.09 model-host CPU seconds and 402.73 wall seconds
across eight sessions. The two arms processed 32,538 input tokens and
generated 3,158 output tokens. The replay verifies equal paired data,
the frozen execution order, diagnostics sent, and checkpoint scores.
See the [paired comparison](../results/wb-cx/comparison.json) and
[frozen plan](../results/wb-cx/paired_plan.json).

## Rejection versus coordinate search

The same four fresh checkpoints and splits were used in three arms.
Both Qwen arms received the connectivity probe. Only the rejection arm
received a refusal for a disconnected proposal, which consumed a model
turn. Coordinate search used a fixed, seeded schedule of eight proposals
and chose the first best practice improvement.

| Outcome | Qwen with advice | Qwen with rejection | Coordinate search |
|---|---:|---:|---:|
| Decision attempts | 14 | 19 | 32 |
| Practice evaluations | 7 | 8 | 32 |
| Tasks with a practice logit change | 0/4 | 3/4 | 4/4 |
| Tasks with a practice prediction change | 0/4 | 0/4 | 4/4 |
| Tasks with a practice accuracy gain | 0/4 | 0/4 | 3/4 |
| Finalized holdout improvements | 0/4 | 0/4 | 3/4 |

Rejection blocked six proposals. It redirected the model to weights that
changed logits, but its edits did not cross a decision boundary. Search
used more of the shared evaluation ceiling. In a descriptive replay of
its stored practice scores truncated to the rejection arm's realized
evaluation count for each task, search found gains on two of four tasks.
That prefix comparison uses practice scores only.

| Search candidate | Original holdout | Reloaded candidate | Matched random |
|---|---:|---:|---:|
| Hidden threshold | 34/64 | 62/64 | 62/64 |
| Output collapse | 32/64 | 64/64 | 27/64 |
| Output intercept | 32/64 | 59/64 | 32/64 |

The search arm stopped on input reversal because its practice scores
never exceeded baseline. In the hidden-threshold task, the selected
edit subtracted 3 from output bias 1 while the matched control added 3
to output bias 0. Both edits reduce the class-1 versus class-0 logit
margin by exactly 3, which accounts for the tied scores.

These outcomes favor the deterministic schedule for this small test.
The paired comparison is exploratory, and the 20-task AI transfer gate
remains HOLD. The experiment used 543.21 seconds of elapsed time and
2,957.89 model-host CPU seconds. The deterministic arm used 14.11
seconds of elapsed time including its broker calls. See the
[frozen plan](../results/wb-sr/paired_plan.json) and
[comparison with raw evidence paths](../results/wb-sr/comparison.json).

## Practice breakpoint search and candidate selection

For one scalar change `d`, the binary logit margin is the class-1 logit
minus the class-0 logit. An output-layer edit makes this margin affine
in `d`. An edit to an input weight or hidden bias gives a constant plus
a scaled `tanh(z + s*d)`. The solver finds where each practice margin
crosses zero inside the allowed delta range. It scores the interval
midpoints and cap endpoints, skips ambiguous near ties, and keeps one
improving proposal per eligible tensor. The deterministic ranking uses
correct predictions, minimum signed margin, and then edit magnitude.

In a four-pair smoke, both Qwen menu selection and deterministic
selection produced 64/64 holdout scores after checkpoint reload on
all four tasks. Input reversal was repaired by adding about 4.013 to
`input_weight[0, 0]`. The earlier fixed-step search used a step of 3.
Qwen picked A on all four menus, including the two with multiple
choices. It chose a different edit from the deterministic selector on
one task and tied its holdout score. The solver calculated every
candidate delta from practice data.

The menu is screened analytically before the one broker practice
evaluation. The smoke screened 1,204 scalar deltas per arm, so its
one-evaluation ceiling is distinct from the eight-candidate search
budget in the earlier study. See the [smoke comparison](../results/wb-bs/comparison.json).

A fresh 20-pair confirmation used five split seeds per synthetic defect
mechanism. Both arms passed the frozen repair, regression, and matched
random thresholds for this solver-assisted workflow.

| Selector | Original correct | Reloaded correct | Matched random correct | Full repairs | Mean lead over random |
|---|---:|---:|---:|---:|---:|
| Qwen | 677/1,280 | 1,274/1,280 | 737/1,280 | 19/20 | 41.95 percentage points |
| Deterministic | 677/1,280 | 1,280/1,280 | 738/1,280 | 20/20 | 42.34 percentage points |

Both arms had zero regressions among examples the original classified
correctly. The stratified paired 95% intervals for the advantage over
random were 34.77 to 48.05 points for Qwen and 35.39 to 47.97 for the
deterministic selector. These intervals apply to the tested task
distribution, with repeated splits of the four injected mechanisms.

Qwen picked A on 17 of 20 menus. Eleven menus had multiple options, and
the two selectors disagreed six times. On the sole partial repair, Qwen
chose an output-weight edit at the policy cap that scored 54/64 on
practice and 58/64 on holdout. The deterministic selector chose an
output-bias edit that scored 64/64 on both. Candidate selection offered
no measured advantage over deterministic ranking on these tasks.

The confirmation used 169.49 seconds of wall time and 198.97 host
CPU seconds across 40 sessions. Each arm screened 6,005 analytical
deltas and used 20 broker practice evaluations. Qwen used 20 model
forwards and 11,407 input tokens. The original direct-proposal AI transfer
study remains a separate held experiment. Qwen served as the selector
in this study and the edited checkpoints were toy MLPs.
[Confirmation and raw evidence](../results/wb-bf/comparison.json)

## Blind weight repair on Qwen3.5-0.8B

The search independently chose an edit from damaged-checkpoint activations
and practice scores. It found the injected layer in both initial completed
trials without receiving the pristine checkpoint or the injected patch. A
separate evaluator loaded the saved repair and exactly replayed its practice
scores. In the first same-format trial the edit recovered both held-out errors
and outscored the three norm-matched random edits. It also changed three
unrelated predictions, including one regression, so it failed the frozen
locality gate. In the paraphrase trial the injected fault changed logit margins
without reducing target accuracy, leaving no accuracy loss to recover.

The sentinel-constrained follow-up recovered 13/13 lost target answers
and beat the best random control by 12/24. Its layer 23 patch changed
59,265 stored BF16 values over 64 columns. It introduced no new sentinel
errors and corrected one existing sentinel error, so it still failed exact
prediction preservation. Reloading the pristine checkpoint afterward
reproduced every original recorded holdout prediction and margin.

The final margin-budget trial recovered 14/16 lost target answers and
outscored the best norm-matched random edit by 14/24. The saved patch
reproduced practice scores after reload. All 32 unrelated predictions
matched pristine Qwen. The sole change relative to the damaged checkpoint
restored the pristine answer, and the original exact-preservation rule
still returned FAIL. That difference separates two plausible experimental
goals. One holds all non-target predictions fixed relative to the broken
model, while the other restores the behavior of the known good model.
Future injected-fault studies should preregister both measures. Natural
failure studies need independent expected behavior or regression tests
because there may be no known good checkpoint.

These are interventions on a full transformer checkpoint under a synthetic
localized fault. They support a narrow behavioral recovery claim. The design
uses one correct target answer position and related injection and repair
families. It does not estimate transfer to arbitrary defects, alternative
answer positions, or natural generation failures. The search was deterministic
and did not use an LLM to formulate or select its patch.
[Protocol and raw evidence](QWEN_BLIND_REPAIR.md)

## Blinded Qwen cohort

On the revised 12-case cohort, saved activation-guided patches won 11
paired comparisons against equally budgeted random search. They recovered
52 of the 86 net answers lost to damage. Only five met the full recovery,
locality and paraphrase gate, short of eight. Two cases had insufficient
held-out damage to judge repair.

The same-wording gain accounted for 46 of the 52 additional correct
answers. One case gained five same-wording answers and lost six
paraphrased answers. Layer 23 was selected in all 12 cases, including
eight faults injected in earlier layers. Behavioral recovery can therefore
occur through downstream compensation in this setup. The results motivate
a comparison that divides the same practice budget across several
wordings and preserves new paraphrases for final evaluation.
[Cohort protocol and evidence](QWEN_REPAIR_COHORT.md)

## Qwen practice wording ablation

Splitting the same practice examples and candidate budget across three
wordings gained one answer on 132 fresh paraphrases relative to the
single-wording control. Both improved over the damaged checkpoints, and
both had one case with a paraphrase regression. The paired interval
includes zero. Multi-wording passed two of 12 cases and single wording
passed one, short of eight for either arm. Their respective 14 and 13
pristine sentinel prediction changes leave locality unresolved.
[Ablation report and evidence](QWEN_WORDING_ABLATION.md)

## Endpoint-aligned Qwen adaptation

All six rounded BF16 probes reduced the conditional fitting loss. None
improved the number of targets solved across both wordings. The smallest
tested dose (realized norm about 0.025) regressed three previously correct
responses even with corrected comparison rivals. Exact live rollback held
on every probe. The real generation guard prevented materialization and
the hidden roster remained unopened. A separate SmolLM2 replay verified
the endpoint contract across runtime paths without claiming a model gain.
[Phase III-H evidence and replay](PHASE_III_H.md)

## Causal fork adaptation

The trajectory-constrained Qwen update changed six of eight fitted
greedy decisions to accepted tokens and raised completed practice scores
from 9/16 to 14/16 across both wordings. Three previously correct
presentations outside the sampled preservation trajectories regressed.
The guard rejected the update and exact parent restoration passed. The
layer scan chose the same region as the fixed baseline. The conventional
LoRA merge exceeded its frozen norm cap before decoding. The held-out
candidate outcome remains unmeasured. [Phase III-I evidence](PHASE_III_I.md)

## Counterexample-guided preservation

Group-aware sampling covered all eight group and wording cells but still
missed four sensitive decisions with higher parent margins. Counterexample
refitting and a smaller constrained update produced a practice pass with
two repeatable errors corrected, three net targets gained, and zero
regressions. The exact checkpoint gained one target on fresh selection
and regressed two previously correct responses. The frozen gate rejected
it. [Phase III-J evidence](PHASE_III_J.md)

## Robust J-space

Group-stratified directional sampling covered all eight required cells
across eight underlying items. At the two tested radii the historical,
directional and buffered fits produced identical rounded weights. The
smaller update corrected a repeatable error and preserved every protected
response on separate development validation. Fresh selection preserved
prior responses and gained no target items. The held residual allowance
covered five of seven items. [Phase III-K evidence](PHASE_III_K.md)

## Semantic transfer and gradient consensus

The Phase III-L parent solved 1/16 comparisons across both wordings and
operand orders. Equal subgroup counts still produced much larger historical
gradient effects on close comparisons with shorter correct answers. A
shared-right consensus update produced distinct BF16 weights and improved
practice to 4/16 with zero losses of prior correctness. It preserved all
25 previously correct responses on separate validation and improved fresh
selection from 0/16 to 1/16. The sealed evaluation gained three comparisons
and regressed one previously correct response. That violation rejected the
candidate. The public truth-flip families still had no jointly correct pair.
The bounded maximin witness coincided with a simpler equal-group mean at
the larger dose. [Phase III-L evidence](PHASE_III_L.md)

## Objective and normalization decomposition

Phase III-M kept one comparison gradient per semantic item and compared
CE/RAW, CE/UNIT, MARGIN/RAW and MARGIN/UNIT in the same selected-column
rank-one space. The larger-dose BF16 norms differed by only 0.08615% of
the target. Practice gains were 3, 3, 4 and 3 strict comparisons. Fresh
development gains were 3, 1, 3 and 1, with zero prohibited regressions.
Normalization changed correction allocation and reduced measured strict
development gain here. Objective choice supplied no strict development
gain advantage. All complete truth-flip gains were zero. The family gate
held before independent selection and hidden access.
[Phase III-M findings](PHASE_III_M.md)


## Paired counterfactual supervision

Phase III-N kept the exact Phase III-M columns and right factor. At the
larger matched norm, paired supervision gained thirteen strict practice
comparisons and completed one previously incomplete opposing-outcome family;
one-sided supervision gained ten comparisons and completed no family. Both
preserved every required correct fitting response. The newly complete family
was outside the paired gradient fit but remained observed practice data.

On six fresh development families, both larger-dose candidates gained two
strict comparisons and neither completed a new family. The smaller paired
mean and pair-aware candidate had zero strict development gain. All five
reloaded candidates had zero prohibited development regressions. C2 was
withheld at calibration, so its generated behavior is unmeasured. No candidate
reached independent selection or hidden verification. This supports a
practice correction and rejects qualification of complete rule transfer in
this trial. It does not establish that paired supervision is generally useless.

A batch-packing change exposed a BF16 tie during fitting. Strict refusal,
unchanged-parent diagnosis, and replay using the original diagnostic context
resolved it without relaxing data or endpoint gates. The failed attempt is
retained. [Phase III-N evidence](PHASE_III_N.md)

## Joint individual decisions versus averages

Phase III-O measured sixteen repeatably wrong first-fork decisions from two
fresh opposing-outcome families. The approximate minimum norm is 0.023666
when four comparison averages are constrained, 0.035805 when every individual
decision is constrained, and 0.036063 after adding eight protected decisions.
Numerical dual bounds exceed the frozen 0.025 cap for the latter two systems.
Every individual target halfspace fits below the cap alone. Both close and far
families exceed it when their eight decisions are required jointly.

The result supports a local coupling limitation hidden by averaging.
Preservation contributes a much smaller additional cost in this cohort.
No candidate qualified for full generation; transfer, precision effects on
candidate target correction and persistent improvement remain untested.
This does not establish general neural or rank-one unreachability. The next
recommended test changes the fixed right factor under a separately frozen
rank-one study while keeping the layer, columns, cap and behavioral gates.
[Report and raw numerical evidence](PHASE_III_O.md)
