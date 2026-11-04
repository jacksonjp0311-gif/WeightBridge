# Phase III-K robust J-space discovery

The Phase III-J E3 patch improved practice from 10/16 to 13/16 targets. It reloaded at the expected checkpoint hash and gained one target in independent selection, with two prohibited regressions. At an active practice constraint the replay margin was zero. This study tests whether sensitivity-based preservation sampling and positive slack improve the correction-to-regression tradeoff on a fresh population. [Phase III-J findings](PHASE_III_J.md)

## Frozen design

The protocol pins the original Qwen checkpoint, separate development roles, a fresh independent selection roster and a host-held hidden commitment. Adaptation has 24 underlying items. Preservation fit, residual calibration and preservation validation each have 12. Selection and hidden have 24 each. Each underlying item has two wordings confined to one partition. The generator excludes disclosed Phase III-G through III-J question identities. All acceptance criteria were frozen before baseline measurements. [Protocol](../experiments/phase_iiik/decimal_protocol.json) · [Roster commitments](../results/phase-iii-k/jspace-v1/material/manifest.json)

The unchanged parent produced valid completions on all public responses. It solved 10/16 adaptation targets with six repeatable errors and 9/16 selection targets with five repeatable errors. It passed all four positive and four copy items on both splits. The preservation-fit roster had 24 correct presentations across both wordings, and each other small development partition produced 24/24 valid responses. Those smaller partitions serve preservation and calibration roles rather than supplying target errors.

The target loss, editable region, fixed right factor, rank and radii are shared across the decoded primary arms. A uses historical low-parent-margin sampling with eight anchors. B changes the selection policy to cover group and wording cells while ranking by measured directional vulnerability. C uses those same anchors and adds the calibrated positive floor and residual allowance. D reuses the C observations to rank qualified candidates by measured slack and feasible-window width, with a maximum-practice-gain comparator. E tests only the locked D candidate on separate preservation validation before independent selection. None of the held outcomes can change the direction, constraint set or radius for this cohort. Low-margin stratification and unstratified directional ranking are additional sampling diagnostics on the same pool.

## Mathematical representation

In authorized parameter space, `J_practice` contains bounded updates that improve the repeatable target errors and preserve all required previously correct responses under actual generation. For the local approximation `J_linear`, each selected target requires `h + JₜΔθ > 0` and each sampled preservation decision requires `g + JₚΔθ ≥ ε + δ`. Here `h` and `g` are original logit margins, `ε` is the positive floor, and `δ` is the empirical adverse residual allowance. These sets are evaluated separately. A local witness, a rounded state, a complete-generation practice pass, a validation pass and an independent pass each have distinct evidence.

The sensitivity scan replays every eligible token of every previously correct preservation-fit trajectory. It applies a small rounded BF16 probe along the preferred direction, measures actual margin movement on the original prefixes and restores exact parent bytes. This gives a finite secant, including rounding and nonlinear response at the scan dose. Ranking uses adverse movement divided by available slack, and coverage allocation precedes diversification across underlying items. Selected constraints use individual VJPs for the local solver. The rank-one solver projects a preferred left factor onto the bounded intersection of halfspaces with the right factor fixed. It reports checked constraint residuals and distinguishes an exhausted solver from a demonstrated conflict.

For a normalized factor direction, each target's crossing threshold intersects the sampled preservation bound and radius. Width is the nonnegative length of that intersection. A target tie is excluded. Positive width is a direction-specific local witness. The required behavioral checks run on the actual BF16 state after rounding and include complete decoded responses.

Calibration estimates adverse residual `max(0, predicted − measured)` on the prescribed reference direction. The allowance uses the frozen quantile of each underlying item's worst residual across sampled token decisions. The positive floor comes from the frozen grid and original margins. A separate preservation-validation audit estimates residuals only after practice choice is locked. These small-sample empirical quantiles describe prediction error for the observed direction and dose. They do not certify arbitrary directions, checkpoints or distribution shifts.

## Measured sampling and calibration

The Qwen scan measured 206 eligible positions. Its nominal radius was 0.0125 and its realized BF16 norm was 0.012546, changing 14,295 entries. No scanned replay argmax changed. One measured margin reached a tie. The scan took six forwards and 13.91 wall seconds.

| Sampling rule | Group and wording cells covered | Underlying items represented |
| --- | ---: | ---: |
| Historical low margin | 5/8 | 5 |
| Stratified low margin | 8/8 | 4 |
| Directional | 6/8 | 8 |
| Stratified directional | 8/8 | 8 |

The historical sample omitted copy coverage. Stratified directional sampling covered all required cells and doubled the underlying-item coverage of the stratified low-margin comparator at the same eight-anchor capacity. The decoded A/B contrast changes both stratification and sensitivity ranking. The additional two diagnostic samplers separate their effects on coverage, but were not used for separate decoded interventions.

The calibration grid selected `ε = 0.125` from a cap of 0.375. The six calibration items supplied eight measured decisions and set `δ = 0.038547`, making the C constraint threshold about 0.163547. Applying this allowance to the constrained factor or to the larger dose is a transfer hypothesis. The runner checks the active margins again on each rounded state and always checks complete generation.

The held residual audit covered five of seven different underlying items. The two adverse residuals outside the allowance were 0.164877 on a positive control and 0.160533 on a mixed-sign target. Both exceeded the calibrated allowance by more than four times on the same preferred direction and dose. This is direct evidence that the small calibration sample did not provide a reliable uniform allowance across these development examples.

## Rounded outcomes and admission

| Arms | Nominal radius | Realized norm | BF16 entries changed | Net target gain | Adaptation regressions | Preservation-fit regressions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1, B1, C1 | 0.0125 | 0.012546 | 14,295 | +1 | 0 | 0 |
| A2, B2, C2 | 0.0225 | 0.022526 | 15,096 | −1 | 3 | 0 |

At each radius A, B and C used the preferred factor because their sampled linear inequalities did not bind. The rounded tensor hashes matched exactly across the three arms. The study therefore measured two unique complete practice probes. It establishes coverage differences between samplers and compares two radii. It cannot attribute a behavioral change to the added sensitivity or slack at these points because those interventions produced no change in stored weights.

The smaller state corrected the repeatable negative item 01 across both wordings, improving adaptation from 10/16 to 11/16 with no regression on either fitting partition. The larger state corrected that same item but regressed three adaptation presentations. Those three were outside the preservation-fit sensitivity pool. Neither state regressed a preservation-fit response, so the recorded sampler flags cannot establish detection of actual preservation-fit regressions. Net gain regret relative to the best measured adaptation gain was zero at the smaller radius and two target items at the larger radius.

C1 had a positive paired local window for at least one item. Its maximum paired width was 0.003087, versus a maximum individual fork width of 0.005238. Its smallest predicted sampled margin was 0.191352 and its smallest measured margin was 0.125. Seven of eight rounded active margins reached the combined floor, and all eight stayed positive. The eighth margin fell short of the buffered constraint despite a feasible linear prediction. The frozen D rule requires positive measured margins and a positive individual width. It therefore selected C1, which also won the maximum-gain comparison. Its excess above `ε` was zero. This is a practice-compatible checkpoint, with no demonstrated benefit from ranking for interior slack on these two probes.

Arm E independently reloaded the D checkpoint at `e9b0be56d4d48d71121212f27c970cb80798214c3077b3bfa6ba3f9752ec151f`. It preserved all 23 previously correct validation presentations and kept all 24 responses valid. Four previously correct presentations changed token sequences while preserving the accepted semantic result. This meets the frozen preservation-validation gate and permits independent selection.

## Independent result

Independent selection reloaded the same checkpoint and scored 9/16 correct target items, exactly the parent score. All 48 presentations stayed valid, every previously correct response was preserved, and the paired endpoint recorded no semantic changes. The candidate failed the required positive improvement. There was no selection winner or hidden evaluation, and the meta-controller returned `hold` for `selection_failure`.

| Evidence level | Result |
| --- | --- |
| Local linear feasibility | A positive paired window existed for the smaller candidate. All six projection calls kept the preferred update. |
| Measured buffered margin | The smaller C state fell short on one of eight active margins. |
| Rounded practice correction | PASS. One repeatable correction across both wordings, no losses in either fitting partition. |
| Separate preservation validation | PASS. Twenty-three previously correct responses retained. |
| Independent selection | FAIL. Zero regressions and zero target gain. |
| Verified persistent improvement and sequential readiness | HOLD. |

The two unique practice probes used 216 batched greedy decode steps and restored the live parent affine bytes exactly. The host rechecked the pinned original checkpoint hash. The main process took 263.48 wall seconds including the candidate verifier subprocesses, and used 856.05 CPU seconds excluding them. The three fitting parent measurements used 291.05 CPU seconds. Parent preservation validation and selection used 253.59 CPU seconds. The two edited-checkpoint verifier workers used 233.58 CPU seconds. Total measured Qwen compute was 1,634.27 CPU seconds, with peak process RSS of 2.83 GiB. Shared calibration, gradient, reference and residual work is accounted for once in the raw receipts. The two aliases per radius consumed no additional complete decodes.

The exported patch independently rebuilt the exact candidate checkpoint and tensor hashes. Reconstruction used 14.88 CPU seconds. The pinned original still matched its original hash. [Reconstruction receipt](../results/phase-iii-k/jspace-v1/reconstruction.json)

The primary study used a fixed rank-one region and two matched radii. The optional secondary geometry comparison was outside the frozen CPU pilot. This region still yields a practice correction at small norm. The experiment found no independently beneficial point at the tested doses and no distinct rounded treatment from the sensitivity or slack changes. It does not settle whether broader coverage or another feasible direction would produce an independent correction.

[Raw study and verifier receipts](../results/phase-iii-k/jspace-v1/trial) · [Replayed analysis](../results/phase-iii-k/jspace-v1/analysis.json) · [Actual practice outcomes](../results/phase-iii-k/jspace-v1/arms.png)

## Next experiment

Freeze fresh cohorts and use development-only sensitivity scans to predeclare two small-norm directions for which sampled preservation inequalities actually bind. Compare each direction with and without a positive buffer under the same radius and decode budget. Measure residual allowance coverage on multiple directions, then require both paired target correction and positive measured excess above the floor before independent selection. This tests whether a changed stored update improves the correction and preservation tradeoff. The independent selection and hidden gates remain the final tests.

## Portability and reproduction

The same BF16 secant ran through the native SmolLM2 adapter at its deepest supported feed-forward output. The engineering probe used a deterministic alternating output factor and four input columns rather than a learned correction. It measured all 138 eligible positions and restored the affine tensor exactly. The sampler reported both target wording cells unavailable. The model retained its pinned checkpoint identity. The probe took 14.64 wall seconds and 39 CPU seconds. [SmolLM2 receipt](../results/phase-iii-k/smollm2-sensitivity.json)

Use the pinned CPU checkpoint and model runtime from [Phase III-I](PHASE_III_I.md). Keep trial output outside the checkout. These commands use POSIX syntax.

```bash
material=results/phase-iii-k/jspace-v1/material
python -m experiments.phase_iiik.baseline fit --base ./qwen35-base-cpu --material "$material" --out ../jspace-fit
python -m experiments.phase_iiik.baseline preservation-validation --base ./qwen35-base-cpu --material "$material" --out ../jspace-validation.json
python -m experiments.phase_iiik.baseline selection --base ./qwen35-base-cpu --material "$material" --out ../jspace-selection.json
python -m experiments.phase_iiik.run_jspace --base ./qwen35-base-cpu --material "$material" --fit-dir ../jspace-fit --parent-validation ../jspace-validation.json --selection-base ../jspace-selection.json --out ../jspace-trial
python -m experiments.phase_iiik.analyze --trial results/phase-iii-k/jspace-v1/trial --material "$material"
python -m experiments.phase_iiik.reconstruct --base ./qwen35-base-cpu --archive results/phase-iii-k/jspace-v1/trial --scratch ../jspace-reconstructed --receipt ../jspace-reconstruction.json
```

The hidden closure takes the host-held roster path and finalist-lock SHA-256. It replays preservation validation and selection from their receipts before any hidden load. The archive is hash-checked and excludes hidden rows, worker output and model checkpoints.
