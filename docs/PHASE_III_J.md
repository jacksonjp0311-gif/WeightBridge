# Phase III-J counterexample-guided adaptation

Phase III-I's best live update gained five complete practice items and lost three previously correct presentations. This trial asks whether sampling across preservation groups, allowing bounded margin movement, and refitting on observed practice regressions can keep the gains while passing the same full-generation gate.

The historical audit resolves the selected anchors by joining the scan's trajectory indices with the frozen response IDs. All four anchors came from two previously correct targets, one negative-far and one positive-close, in both wordings. The sampled pool also included positive and copy controls, but neither group received a selected anchor. All three observed regressions were outside the sampled trajectory pool. Incomplete coverage is a plausible contributor. The archive does not establish that coverage was the primary cause of the regressions. [Phase III-I coverage audit](../results/phase-iii-j/phase-i-coverage-audit.json)

## Design and geometry

The frozen Qwen protocol binds fresh practice, selection and hidden rosters. The unchanged parent qualified with 48/48 valid responses on each public split. It solved 10/16 practice targets and 9/16 selection targets across both wordings. Five practice errors and four selection errors repeated across both wordings. Both public splits had four solved copy controls, with four solved positive items on practice and three on selection. The fitter receives only practice data. The original checkpoint was pinned by hash and the hidden roster by a host-held commitment. [Main protocol](../experiments/phase_iiij/decimal_protocol.json) · [Secondary protocol](../experiments/phase_iiij/secondary_protocol.json)

The engine replays complete correct trajectories and records strictly positive decisions consistent with the cached greedy output. The historical sampler selects the weakest eligible token per presentation and takes the lowest margins in a restricted trajectory pool. Stratified sampling reserves a token position for each group and wording. It reports missing cells instead of crediting unavailable coverage. The risk diagnostic ranks measured anchors by their predicted adverse directional change relative to their parent margin. Risk ranking covers the measured gradient subset.

For each affine region, the fork cross-entropy gradient chooses columns and a rank-one direction. The solver fixes the right factor `v`, with unit norm, and operates on the left factor `u`. The rigid baseline removes components along the preservation-gradient rows and then scales the direction to the nominal radius. The inequality solver projects the preferred factor onto the intersection of `gᵢ + aᵢᵀu ≥ εᵢ` and `||u||₂ ≤ ρ`. It uses bounded cyclic projections with Dykstra corrections and checks numerical residuals. Failure to find a witness within the iteration limit remains distinct from a proven conflict.

Along a normalized direction `d`, a target with parent margin `h < 0` and slope `s > 0` first reaches its tie at `α = −h/s`. The target must move strictly past that boundary. A protected decision with parent margin `g` and negative slope `s` imposes an upper bound `(g − ε)/−s`. The engine intersects each target's allowed interval with all sampled preservation bounds and the radius, then also reports the joint interval for all measured targets. The linear margins are local estimates. BF16 rounding and actual complete decoding can change the outcome.

Arms A and B use the same target objective, native layer, rank and nominal radius. A uses the historical four-anchor pool and B uses eight anchors to cover all four groups and both wordings, so their comparison changes anchor count as well as sampling policy. C uses B's evidence with inequalities. D may add decisions from observed practice regressions and refit from the original checkpoint, with two refits and at most two new anchors per refit. E measures predeclared radii `0.005`, `0.0125`, `0.025`, and `0.05` using the frozen practice ranking rule. Layer 17 and projected full-width rank-four LoRA follow the separately frozen secondary protocol.

A candidate's first divergence becomes an eligible counterexample only if the full endpoint records a regression and the measured candidate logits reproduce the competing token on the parent prefix. The report binds the parent and candidate tokens and margins. This locates the first differing decision on a failing trajectory. The evidence cannot isolate that token as the sole cause of the wrong final answer. The engine records each attempted refit, rejects BF16 no-ops, deduplicates exact rounded tensor states, and restores the parent affine bytes after every live probe.

The dose comparison refits the same formulation at each radius. Binding inequalities can change the direction, so these results describe the response to the formulation's radius rather than scaling a single vector.

The initial process reached D1 and stopped when the write-once artifact store refused an attempt to add metadata after the receipt had already been written. The retry writes the full receipt in one operation and reruns from the same parent, with the protocol intact. The interrupted attempt accounts for at least 158.79 wall seconds and 822.08 CPU seconds in recorded, nonoverlapping stages. The archive retains its hashes, traceback and raw practice evidence. It is a failed execution of the same experiment, not a second independent cohort. [Interrupted execution receipt](../results/phase-iii-j/preservation-v1/interrupted-attempt.json) · [Interrupted raw evidence](../results/phase-iii-j/preservation-v1/interrupted)

## Practice results and feasibility

All ten diagnosed error presentations were semantic forks with exact greedy replay agreement. The first eight presentations, from four underlying items, entered the fit. The historical sampler again selected only target anchors. Stratification filled all eight group and wording cells. For the positive and mixed-sign groups, the two wordings happened to come from different underlying items. The complete-generation guard continued to score both wordings of every item.

A, B, and C each corrected the same two repeatable wrong items. B raised net target gain from zero to one and C tied B. All three lost the two wordings of the same two previously correct target items, even though the sampled anchor margins remained positive. Their four regressions came from token decisions with original margins between 1.625 and 2.625, well above the 0.125 target anchors chosen by lowest margin. The outcome supports measuring susceptibility to a proposed update. Low parent margin alone did not identify the regressions.

The initial risk diagnostic had directional slopes for ten decisions, with one adverse slope. Within that restricted pool it selected the same eight cells and exact decisions as stratified sampling. The trial did not compare a separately decoded risk-selected update. A deterministic solver fixture shows that inequalities can retain a feasible nonzero factor when rigid projection removes it. In the real matched B/C comparison, C corrected more individual fitted forks and tied B on completed target gain and prohibited regressions.

For C, the eight individual minimum crossing estimates were approximately 0.0143, 0.0135, 0.0135, 0.0089, 0.0865, 0.0688, 0.0881, and 0.0514. Four lay inside the sampled preservation limit of 0.05. The joint interval for all eight was empty, even though a subset could be corrected. Replay and complete generation provide separate counts because the linear approximation can miss actual fork crossings.

D1 added the two wordings of the first regressed item. Its factor moved from the nominal 0.05 ball to a realized norm of about 0.0254. The refit cut regressions from four presentations to one and increased net gain from one to two items. The added decision that still failed sat at approximately zero in the linear solution and at −0.125 on the rounded live model. The frozen floor of zero left no buffer there. D2 stopped because the remaining regression occurred at an already active decision, so the protocol found no novel anchor to add. The refit improved the tradeoff, with a change in actual norm as well as the addition of constraints. The next dose stage uses the D1 formulation.

| Arm | Nominal radius | Realized BF16 norm | BF16 entries changed | Net target gain | Repeatable items corrected | Regressed presentations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A. Historical nullspace | 0.05 | 0.050011 | 15,920 | 0 | 2 | 4 |
| B. Stratified nullspace | 0.05 | 0.050010 | 15,942 | +1 | 2 | 4 |
| C. Stratified inequalities | 0.05 | 0.050016 | 15,957 | +1 | 2 | 4 |
| D1. Counterexample refit | 0.05 | 0.025395 | 15,259 | +2 | 2 | 1 |
| E1. Refitted dose | 0.005 | 0.005075 | 12,386 | +1 | 0 | 0 |
| E2. Refitted dose | 0.0125 | 0.012556 | 14,627 | +1 | 0 | 0 |
| E3. Refitted dose | 0.025 | 0.016687 | 14,962 | +3 | 2 | 0 |
| E4. Refitted dose | 0.05 | Same as D1 | Same | Same | Same | Same |

All measured main arms completed and produced valid answers on 48/48 practice responses. E1 and E2 gained one underlying item without correcting a repeatable wrong negative. E3 met the complete practice gate. It corrected both wordings of repeatable items 00 and 01 and gained a third item outside the four fitting items. All four fitted presentations that switched to an accepted fork token completed correctly. The paired bootstrap over 16 underlying practice items gives a descriptive accuracy change interval of 0 to 37.5 percentage points. It reflects practice exposure and is not independent evidence.

E3 changed 14,962 BF16 entries. The constrained solution bound at a norm below its nominal radius. The previously regressed active decision replayed at a zero margin, while cached generation preserved the correct completed answer. Exact behavioral preservation at that point was measured. A strict positive local-margin certificate was unavailable. E4 produced the same rounded tensor as D1 and reused its observation.

The alternate layer 17 probe used C's original stratified anchors, solver, and nominal 0.05 radius. It changed 14,644 BF16 entries at norm 0.050029, gained one target item, and preserved all previously correct responses. It corrected no repeatable wrong-negative item and switched one of eight fitted fork decisions to an accepted token. Layer 23 under C switched six of eight and corrected two repeatable items, with four regressions. The result shows a different tradeoff at one measured dose. It does not establish an optimal layer or rule out useful directions at either layer.

The conventional arm trained 18,432 rank-four factors for four Adam steps on 32 paths and 254 tokens per step. On step two the proposed stored norm was 0.442. Projection reduced it to 0.331 while retaining the 0.35 cap. The final BF16 merge norm was 0.307297, with 2,872,109 changed entries. All 48 generations were valid. It lost one target item overall, regressed both wordings of target 05, and corrected no repeatable negative error. The optimizer used 16 forwards, 16 backwards, 106.28 wall seconds and 598.25 CPU seconds. Its wider geometry and larger norm were measured and accounted for separately from the direct probes. This conventional configuration offered no additional benefit on the frozen practice gate.

## Independent result

E3 was the only practice passer. Its saved checkpoint hash was `6895c7ceec2b8c401af0db75a5765c2ea3ce878bb3593d4ed98a4d1cb2333177`. A fresh process reloaded those exact bytes and improved selection from 9/16 to 10/16 targets. Two target items became correct across both wordings, and one previously correct target lost a wording. A positive control also lost one wording. All 48 selection responses remained valid. The descriptive paired selection interval for the accuracy change was −12.5 to +25 percentage points. The two prohibited regressions independently failed the frozen gate regardless of the net gain.

| Evidence level | Result |
| --- | --- |
| Local mathematical feasibility | Individual crossing witnesses for a subset of forks. The joint interval for all fitted forks was empty. |
| Stored BF16 practice correction | PASS. E3 corrected two repeatable errors across both wordings, gained three target items, preserved every protected response, and restored the live parent exactly. |
| Independent selection | FAIL. The exact reloaded checkpoint gained one target item and regressed two previously correct responses. |
| Verified persistent improvement | HOLD. No finalist qualified for hidden evaluation or lineage acceptance. |
| Sequential learning | HOLD. |

The meta-controller returned `hold` for `selection_failure`. The parent checkpoint hash was rechecked at the end of fitting. Live restoration passed for all nine unique probes. The rerun matched the interrupted run's four rounded tensor hashes and all complete practice response rows. That exact reproduction measures determinism on the same practice cohort. The independent selection remains a separate result.

The completed host run took 500.04 wall seconds, including the candidate verifier's 37.80 seconds, and used 2,344.38 CPU seconds excluding its subprocess. The practice parent, selection parent, and candidate verifier used another 160.52, 159.11, and 147.28 CPU seconds, respectively. The completed Qwen trial therefore used 2,811.28 CPU seconds and peaked at 6.54 GiB RSS. This is separate from the interrupted attempt's recorded lower bound. The nine unique live practice probes used 648 batched greedy decode steps. The initial layer-23 basis and sensitivities used three forwards and 19 backwards. The layer-17 basis and sensitivity scan used two forwards and 17 backwards. Raw receipts also account for the diagnosis, parent replay, refits, probe diagnostics, and candidate reload.

[Raw trial and checkpoint receipt](../results/phase-iii-j/preservation-v1/trial) · [Replayed analysis](../results/phase-iii-j/preservation-v1/analysis.json) · [Measured dose response](../results/phase-iii-j/preservation-v1/dose-response.png)

## Next experiment

Freeze a new cohort and compare the current sampling method with coverage based on measured adverse directional response across the full correct practice pool. Predeclare positive margin slack at active anchors to address the boundary seen in D1 and E3. Keep a separate preservation-validation slice within development and keep fresh selection and hidden cases outside every refit. The falsifiable test is whether broader sensitivity coverage and slack retain a repeatable correction while eliminating prohibited regressions on fresh selection. The current failure does not establish that the model or tested layers lack a feasible update.

## Reproduction and portability

The same sampler ran on the unchanged SmolLM2 checkpoint through its Llama-family runtime. It found 203 eligible token decisions and filled all eight group and wording cells. SmolLM2 produced 34/48 valid responses and solved 5/16 target items across both wordings. The run is an engineering check of native trajectory replay and coverage. It took 15.74 wall seconds and 46.67 CPU seconds. [SmolLM2 receipt](../results/phase-iii-j/smollm2-preservation.json)

Use the pinned CPU checkpoints and runtime in [Phase III-I](PHASE_III_I.md). The commands below use POSIX shell syntax. Keep the Qwen trial workspace outside the checkout.

```bash
material=results/phase-iii-j/preservation-v1/material
python -m experiments.phase_iiij.audit_phase_i
python -m experiments.phase_iiij.baseline practice --base ./qwen35-base-cpu --material "$material" --out ../preservation-practice.json
python -m experiments.phase_iiij.baseline selection --base ./qwen35-base-cpu --material "$material" --out ../preservation-selection.json
python -m experiments.phase_iiij.run_preservation --base ./qwen35-base-cpu --material "$material" --practice ../preservation-practice.json --selection-base ../preservation-selection.json --out ../preservation-trial
python -m experiments.phase_iiij.analyze --trial results/phase-iii-j/preservation-v1/trial --material "$material"
```

`verification finish` requires the host-held roster and the candidate-lock SHA-256. It recomputes selection gates from the frozen roster and worker receipts, replays finalist ranking, and opens hidden data only after the lock is verified. A passing candidate still requires exact fresh-process replay and rollback before lineage acceptance. The public archive contains raw practice and permitted selection receipts, plus aggregate hidden evidence only if that stage ran.
