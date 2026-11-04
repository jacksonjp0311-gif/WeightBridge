# Phase III-L semantic transfer and gradient consensus

The Phase III-G and III-H studies found that higher teacher-forced scores could accompany worse complete answers. Phase III-I moved fitting to the first wrong generative token and found real practice corrections with regressions. Phase III-J passed practice and failed preservation on fresh selection. Phase III-K passed practice and separate preservation validation, then produced no independent correction. Phase III-L asks whether a direction built from agreement across semantic error groups transfers better than the historical mean fork-loss direction. [Phase III-K evidence](PHASE_III_K.md)

An audit of the Phase III-K archive found that all four fitted errors involved close values. Mean predicted slopes were 112.73 for shorter correct answers and 21.81 for longer correct answers. Two repeatable errors on far comparisons were outside the fit. On the sole corrected item, both wordings switched from a digit to newline. These observations motivate a test of answer-length and termination effects. [Hash-bound audit](../results/phase-iii-l/phase-k-fork-audit.json)

## Frozen comparison

An exact Decimal oracle labels every comparison. A truth-flip family contains a fixed short operand and two long operands differing in one digit. The greater value changes between variants. Both variants run in both operand orders and both inherited wordings. Close and far comparisons are balanced, and each family has one shorter and one longer correct answer. A strict comparison counts as correct only when all four complete answers are correct. One family is the unit of split assignment and uncertainty resampling.

The study pins the original Qwen3.5-0.8B checkpoint and the native feed-forward output region at layer 23. Both arms share the same selected columns, right factor, preservation anchors, local constraints, BF16 representation, decoding and two nominal radii. Preparation uses adaptation, preservation fit, and calibration. A separate process handles transfer validation and independent selection after fitting has written its receipt. The hidden roster remains with the host and requires a hash-bound selection winner. [Protocol](../experiments/phase_iiil/semantic_protocol.json) · [Roster commitments](../results/phase-iii-l/semantic-v1/material/manifest.json)

For each fitted comparison, the backend averages projected corrective-margin VJPs across eligible order and wording presentations. It streams one full affine gradient at a time. The exact agreement matrix uses the projected margin rows, and the wider affine CE agreement estimate uses a seeded signed-column sketch. The historical raw mean fork-CE establishes the shared columns and right factor. Arm A projects its left direction into the allowed set. Arm B seeks a positive witness for the largest minimum subgroup effect, using subgroup averages of unit-normalized item rows. An equal-weight mean of subgroup rows is a numerical comparator. The behavioral A/B comparison changes both the local objective and its aggregation.

The common constraints are the rank-one left-factor norm cap and sampled preservation inequalities. We rank protected positions by the worse measured BF16 secant from the two preview directions and select across behavior groups and wordings. A positive margin floor and a finite empirical allowance from separate calibration are shared. A linear forecast supplies an estimated interval for each fork. The rounded live model supplies the actual margin and full response evidence. Both are measured because positive projected group means can coexist with uncorrected individual forks.

For a fixed unit right factor `v`, the projected slope of a fork margin is `a_i = grad_W(h_i) v`, averaged over fitted presentations of comparison `i`. The update is `u vᵀ`. The group mean uses unit-normalized item rows and includes zero rows in its denominator. For any convex combination `c` of group rows, the local maximin is bounded above by the product of the radius and `||c||₂`. The study records this bound separately from the best feasible witness found by the bounded numerical search.

## Replay repair

The first fitting run used the original sampling rule and held before candidate generation. Its selected target anchor had a margin of +0.125 on the complete parent trajectory and a tie on individual prefix replay, even though the selected token matched. Seven of eight anchors survived. The repair keeps every replay-qualified anchor and replaces the failed one in its original behavior group and wording. It allows up to three sampling rounds, applies the same rule to calibration, and records each refusal. The rerun reused only hash-verified unchanged-parent fitting measurements. The frozen rosters, edit budgets, and behavioral gates are shared across both runs. [First fit archive at its source revision](https://github.com/jacksonjp0311-gif/WeightBridge/tree/7f750b2/results/phase-iii-l/semantic-v1/initial-fit-hold) · [Repair protocol](../experiments/phase_iiil/replay_repair.json) · [Interrupted executions](../results/phase-iii-l/semantic-v1/interruptions.json)

## Parent behavior and gradient agreement

The unchanged Qwen parent completed all 88 adaptation responses and every response parsed as a displayed answer or valid copy. It solved 13 of 64 target presentations and 1 of 16 strict comparisons. Both variants were correct in zero of eight truth-flip families. One family had only its counterfactual correct and seven had neither correct. Fourteen comparisons were repeatably wrong across both wordings in at least one order, including nine wrong on all four forms. Order and wording consistency were 10/16 and 11/16 respectively, including consistent wrong answers.

On close negative comparisons, the model selected the wrong smaller operand on 31 of 32 presentations. It did so when the smaller operand was shorter and when it was longer. Far-negative performance differed by length group in this sample. The model solved 10/16 far presentations with a shorter correct answer and 2/16 with a longer correct answer. Those are behavioral associations under the frozen prompts.

The fork replay matched 50 of 51 observed wrong target continuations. The diagnostic found 35 first distinguishing digit forks and 16 answer-length boundaries. The frozen selection rule retained 24 forks from eight distinct truth-flip families, with two fitted comparisons per semantic subgroup. Computing the shared basis and per-item evidence took seven forwards and 32 backwards. Exact projected subgroup cosines ranged from −0.353 to +0.167 off the diagonal. The strongest opposition was between close and far cases with longer correct answers.

| Group | Fitted comparisons | Predicted raw margin slope under the unconstrained historical unit direction |
| --- | ---: | ---: |
| Close, shorter correct | 2 | 113.79 and 119.84 |
| Close, longer correct | 2 | 1.62 and 1.71 |
| Far, shorter correct | 2 | 3.79 and 5.86 |
| Far, longer correct | 2 | 18.91 and 21.18 |

The initial teacher-forced diagnostic refused all 64 target presentations. The displayed negative answer without preceding whitespace changed the frozen prompt tokenization at the colon. The separately recorded aligned probe supplies the same leading space and newline to both displayed options and checks the native prefix boundary before scoring.

The aligned parent-only probe scored all 64 target presentations. Summed likelihood preferred the correct answer on 10/64 and agreed with greedy choice on 61/64. Mean per-token likelihood preferred the correct answer on 16/64 and agreed with greedy on 41/64. Both are distinct from the parent's 13/64 complete-generation correctness. The probe took 16 forwards, 48.17 wall seconds and 194.48 CPU seconds. It ran after candidate fitting and supplied no optimization data. [Aligned scoring receipt](../results/phase-iii-l/semantic-v1/aligned-teacher.json)

## Stored-precision practice outcomes

| Arm | Nominal radius | Realized BF16 norm | Changed values | Strict target comparisons | Repeatable comparisons corrected | Regressed adaptation / preservation-fit presentations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 historical | 0.0125 | 0.01255199 | 14,631 | 1 → 1 of 16 | 0 | 1 / 1 |
| B1 consensus | 0.0125 | 0.01255014 | 14,643 | 1 → 2 of 16 | 0 | 0 / 0 |
| A2 historical | 0.0225 | 0.02187657 | 15,391 | 1 → 5 of 16 | 4 | 1 / 1 |
| B2 consensus | 0.0225 | 0.02039389 | 15,223 | 1 → 4 of 16 | 2 | 0 / 0 |

All four stored tensors were distinct, and every live probe restored the exact parent. The historical and consensus directions were 43.66° apart at the smaller dose and 40.18° at the larger dose. Every sampled candidate margin exceeded the selected floor of 0.125. The common empirical error allowance was 0.16572. The eight active constraints cover four behavior groups and two wordings, while the target anchors draw from one family. The allowance also has limited independent coverage.

A2 corrected all close cases with a shorter correct answer but still solved no complete truth-flipping family. Its two regressions were outside the selected anchors. B1 gained a previously incomplete comparison whose parent already solved three of four presentations. That gain failed the frozen repeatable-error condition. B2 corrected three far-negative comparisons across both orders and wordings, including two repeatable errors. Its complete truth-flipping family count also remained zero. B2 was the only practice passer and was materialized for independent verification.

The maximin search improved the minimum normalized projected group effect over A. At the larger dose its best numerical witness coincided with the simpler equal-group mean comparator. That observation limits attribution to the maximin search procedure. The historical arm also used a larger realized norm at that dose, so the outcomes describe the measured pipelines within the same maximum radius.

## Independent public gates

The exact B2 checkpoint was reloaded in separate processes. Transfer validation preserved 25/25 previously correct presentations and 56/56 valid completions. Its strict target score stayed at 0/8. Independent selection improved from 0/16 to 1/16 strict comparisons with zero previously correct regressions and zero validity losses. The new selection success had been partly correct before the update, while the declared selection gate permits any positive strict gain. That gate passed and locked B2 for the sealed test.

The number of fully correct truth-flipping families remained zero on these public populations. These gates measure selected task improvements and preservation. Establishing the complete negative-comparison rule would require success on both opposing questions and broader independent replication.

## Sealed result and interpretation

The locked candidate gained three strict comparisons on the sealed 16-comparison population and regressed one previously correct presentation. It lost zero valid completions. The zero-regression gate failed, so the result is FAIL and the controller returned HOLD. The research lineage gained no accepted checkpoint. The raw host-held questions and verifier rows remain separated from the public aggregate. [Independent outcome](../results/phase-iii-l/semantic-v1/trial/independent-outcome.json)

The descriptive family-cluster interval for sealed target accuracy change was 0 to +37.5 percentage points across eight independent families. It includes zero, and the frozen regression rule independently rejects the candidate. The study supplies one seed and one checkpoint. It establishes that the selected direction can transfer some gains, with incomplete preservation under new prompts.

| Question | Observed evidence | Remaining uncertainty |
| --- | --- | --- |
| Does the mean hide gradient imbalance? | All four groups had equal item counts, but historical projected effects were much larger on close-short errors. | The experiment changes CE to margin gradients as well as aggregation. |
| Does a common local direction exist? | The constrained solver found positive subgroup witnesses and distinct rounded candidates. | Group-average improvements leave individual nonlinear decisions unresolved. |
| Can consensus improve the tradeoff? | B2 preserved all fitting and separate validation responses, then gained on selection and sealed targets. A2 had two practice regressions. | The bounded maximin witness matched a simpler equal-group mean at the larger dose. The realized norms differed. |
| Did it learn the full numerical rule? | No public truth-flipping family became jointly correct. | The transferred gains could reflect a narrower distance or answer-length effect. |
| Is the update safely generalizable? | One prohibited sealed regression rejected the candidate. | The tested region may contain another direction, and this trial cannot establish that none exists. |

## Next experiment

Freeze a fresh study comparing the simpler equal-group margin mean with the historical mean-margin direction at the same shared right factor and matched realized BF16 norm. Require improvement on at least one complete truth-flipping family during development, with all earlier preservation rules. This separates gradient aggregation from the CE-versus-margin change and tests whether the correction can support opposing numerical decisions. The current sealed outcome supplies no refitting data to that future study.

## Verification and cost

Independent patch reconstruction reproduced checkpoint SHA-256 `9f48b191217fd7cabc5935aacd39a127ffb6c06be3878181b2a8b1fca1b68cfd`, its tensor hash, 15,223 changed values and realized norm. The original checkpoint hash was rechecked. The failed sealed gate ended evaluation before the four-process acceptance replay and lineage stage. [Reconstruction receipt](../results/phase-iii-l/semantic-v1/reconstruction.json)

The complete pinned-runtime suite passed 402 tests and skipped the optional MCP test because that environment lacks MCP. All six MCP tool tests passed in the separate repository environment. The original Phase III-L HOLD replayed from a clean checkout at its recorded source commit. [Test receipt](../results/phase-iii-l/semantic-v1/tests.json)

The repaired fitting run consumed 499.98 wall seconds and 2,627.31 CPU seconds, including four unique decoded probes. It performed 96 explicit forwards, 48 backwards and 864 batched greedy decode steps. Its reused parent measurements are charged to the initial fitting run. Recorded experimental CPU across completed stages and observed interrupted-process samples totals 6,054.00 seconds, about 100.90 CPU minutes. Recorded stage wall time totals 1,357.92 seconds and peak RSS was 2.80 GiB. Interrupted samples make the CPU total a lower bound. Test time, assistant tokens and tool costs are excluded. [Compute accounting](../results/phase-iii-l/semantic-v1/compute.json) · [Environment](../results/phase-iii-l/semantic-v1/environment.json)

## Reproduction

Use the pinned CPU checkpoint and runtime from [Phase III-I](PHASE_III_I.md). Run fit and assessment in a private workspace. The original fit-only HOLD replays against commit `7f750b2`. The commands below replay the separately recorded anchor repair.

```bash
material=results/phase-iii-l/semantic-v1/material
initial=results/phase-iii-l/semantic-v1/initial-fit-hold
python -m experiments.phase_iiil.run_semantic --base ./qwen35-base-cpu --material "$material" --reuse-initial "$initial" --out ../semantic-trial
python -m experiments.phase_iiil.verification assess --base ./qwen35-base-cpu --material "$material" --out ../semantic-trial
python -m experiments.phase_iiil.aligned_teacher_probe --base ./qwen35-base-cpu --material "$material" --fit-dir ../semantic-trial --out ../aligned-teacher.json
python -m experiments.phase_iiil.archive_trial --source ../semantic-trial --material "$material" --out ../semantic-archive
python -m experiments.phase_iiil.analyze --trial ../semantic-archive --material "$material"
```

The host-only hidden verifier requires the digest of a public selection lock and the separately held roster. It recomputes the public gate before any hidden worker can access its inputs. A hidden pass requires an exact candidate replay and original-checkpoint rollback for research lineage acceptance.
