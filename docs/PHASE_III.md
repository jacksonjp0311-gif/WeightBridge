# Phase III continual adaptation

## What the controller does

An [AWL intent](AWL.md) describes a failure hypothesis and requests a
bounded numerical search. The host freezes the original model digest, policy,
inference configuration, ordered cycle IDs and the exact intent, practice,
cycle holdout and final suite hashes. Each search takes the current accepted
checkpoint as its parent. It fits rank-one updates on practice activations,
applies the existing policy and scores independently saved candidates.

The solver selects up to 64 columns of a Qwen3.5 MLP down projection by signed
target activation contrast. It normalizes those columns by their root mean
square activation. With normalized target matrix \(X\), sentinel matrix
\(S\), and signed target labels \(y\), the coefficient vector before undoing
the normalization is \(v=(X^TX+256S^TS+I)^{-1}X^Ty\). The update is the
outer product of the normalized answer embedding difference and the resulting
column coefficients, scaled by the requested strength. This is a deterministic
solver on activations. It does not train the original model or assume that the
named projection caused the original error.

For an eligible practice candidate, the controller ranks
\(J=G-\lambda F-\mu C\), where \(G\) is the paired gain in correct target
answers, \(F\) is the number of changed practice sentinel predictions, and
\(C\) is the fraction of model parameters changed at stored precision. The
host freezes both penalties and all hard gates. A proposal with unchanged
stored weights or a previously scored checkpoint hash consumes its attempt
slot without another evaluation. An unchanged prediction is a behavioral
observation at the chosen test, with several possible causes. If an earlier
cycle of the same family was accepted, its tensor is tried earlier when the
new intent also authorizes that tensor. The coefficients are fit again from
the current checkpoint.

The verifier opens the cycle holdout only after practice has selected one
finalist. The evaluator reloads both checkpoints, verifies their receipts,
and requires the frozen target gain, target regression and unrelated behavior
limits. Every active accepted ancestor contributes its verified targets and
sentinels as retention rows. The parent must reproduce those predictions and
the candidate must preserve them. A failed candidate and its evidence remain
on disk while the accepted parent stays active. A stopped search records
`NO_FINALIST` and leaves its cycle holdout unopened. The final suite is opened
only after every declared cycle closes and compares the actual head with the
unchanged original. The paired prompt bootstrap summarizes this suite. Closely
related prompts limit its use as an estimate of transfer to other tasks.

Each accepted or rejected result is recorded in a hash-linked lineage event.
Rollback selects an earlier accepted ancestor by checkpoint hash and records
that choice as another event. It does not rewrite a model file. Research
operations assume a host-owned workspace. Keep sealed evaluation files and
checkpoint write permissions outside any untrusted agent account.

## Run the host stages

Use `python -m weightbridge phase-iii --help` for the CLI. Initialize a fresh
workspace with `init --root <workspace> --base <local-hf-model> --protocol
<frozen-protocol>`. Run `search` with `--cycle`, `--intent` and `--practice`.
Run `verify` in a separate process with `--cycle`, `--holdout` and one
`--retain` for each active accepted cycle in lineage order. The `status`
command shows the active head. Use `final --suite` after the protocol closes.
The `rollback` command requires an accepted `--ancestor` SHA-256 and a
`--reason`. Start a new research workspace for a new evaluation protocol.

## Qualification

The integration test uses real safetensors files, the patch policy, receipts,
lineage and rollback, with numerical proposal generation and inference stubbed.
It accepts two sequential edits, rejects an intervention that harms an earlier
verified prediction, verifies a fresh final suite, recovers the head from a new
process and rolls back to an accepted ancestor. This establishes controller
behavior in the test fixture.

Real-model qualification requires sequential accepted improvements and a
separate cumulative result. The natural Qwen study uses the pinned original
checkpoint and authored, disjoint practice and evaluation variants. The
five-cycle gate requires at least three active accepted improvements, a
positive gain on the final suite, and zero final target regressions or
sentinel flips. Each cycle permits two candidates and requires at least one
new held-out answer, no target regressions, and zero sentinel and retention
flips. The two-cycle pilot is a separate, smaller transfer test.
[Frozen full protocol](../benchmarks/configs/phase_iii_natural_qwen_v1.json)
· [Qualification criteria](../benchmarks/configs/phase_iii_natural_qwen_v1_qualification.json)
· [Reproduction instructions](../experiments/phase_iii/README.md)

### Two-cycle pilot

The word-order practice winner improved 5/12 to 9/12. On unseen inputs it
fixed one answer and broke another, leaving 7/12 correct. The verifier
rejected it. Neither date-order candidate passed practice. The original
checkpoint stayed active and scored 7/16 on the separate final suite.

| Word-order comparator | Practice correct | Holdout correct | Holdout target regressions | Sentinel flips |
|---|---:|---:|---:|---:|
| Original | 5/12 | 7/12 | 0 | 0 |
| First activation-ridge proposal | 9/12 | 7/12 | 1 | 0 |
| Matched random row permutation | 5/12 | 7/12 | 0 | 0 |
| Two-step rank-one Adam | 8/12 | 5/12 | 3 | 0 |

The first proposal was also the practice-selected finalist. Its comparator
reevaluation is a repeat measurement of the same intervention. The gradient
arm trained two factors on the same selected columns and capped the update
norm at the activation-ridge proposal's requested norm. The three arms used
the same evaluation prompts, with all fitting performed on practice data.
[Pilot suites, decisions, controls and timings](../results/phase-iii-natural-qwen-v1/prefix-pilot/)

### Five-cycle natural Qwen study

**Qualification failed.** One of five cycles retained an edit, against a
threshold of three. The final sealed suite measured **18/24 for the original
and 15/24 for the retained head**. There was one correction, four regressions,
and zero changes on 12 unrelated sentinels. The paired prompt bootstrap
interval for the accuracy difference was −29.17 to +4.17 percentage points.
The original is the better checkpoint on this suite.

| Cycle | Family | Finalist practice gain | Current holdout | Prior prediction flips | Decision |
|---|---|---:|---:|---:|---|
| 01 | Word order | +3 | 4/12 → 7/12 | 0 | Accept |
| 02 | Date order | +5 | 10/12 → 11/12 | 3/24 | Reject |
| 03 | Fraction order | None | Unopened | — | No finalist |
| 04 | Word order | +8 | 4/12 → 3/12 | 6/24 | Reject |
| 05 | Date order | +1 | 10/12 → 10/12 | 3/24 | Reject |

All four independently evaluated finalists preserved their current unrelated
sentinel predictions. The two date-order edits changed the same three prior
word-order predictions, including two correct answers that became incorrect.
The repeated word-order attempt also interfered with the accepted edit even
though it had the largest practice gain. The two fraction proposals changed
stored values while leaving all measured practice predictions unchanged.

All four finalists used layer 23. The two-slot schedule tested layers 23
and 21, so these results do not establish the best layer. The accepted
candidate changed 64,004 of 873,438,784 parameters and was used as the parent
of cycles 02 through 05. The final word-order score was 5/8 for both models.
Date order fell from 7/8 to 6/8 and fraction order from 6/8 to 4/8. The generic
sentinels missed all four final target regressions. The three task families
share the A/B answer tokens and the compiler edits along their embedding
difference. A shared change in choice margins is a plausible explanation
for interference. The experiment did not establish that mechanism or locate
the source of the original errors.

| Cycle 01 comparator | Practice correct | Holdout correct | Holdout target regressions | Sentinel flips |
|---|---:|---:|---:|---:|
| Original | 6/12 | 4/12 | 0 | 0 |
| First activation-ridge proposal | 9/12 | 7/12 | 0 | 0 |
| Matched random row permutation | 6/12 | 4/12 | 0 | 0 |
| Two-step rank-one Adam | 8/12 | 5/12 | 3 | 1 |

The first proposal was the retained edit, so its comparator reload repeats
the same measurement. The random arm matched the rank, selected columns,
requested coordinate count and requested norm. Its realized Frobenius norm
was 2.252296 versus 2.252224 for the accepted edit, and 63,998 versus 64,004
stored entries changed. Adam used the same tensor and columns with a norm cap.
Its realized norm was 0.535281. These controls compare one cycle and one seed.

The study ran on the pinned Qwen3.5-0.8B checkpoint with six CPU threads per
inference process and BF16 storage. The stages totaled 832.1 wall seconds
and 1,586.6 process CPU seconds, with a 2.43 GB peak RSS for a single process.
The full study controls took 173.9 wall seconds and 387.0 process CPU seconds.
The separate pilot totaled 325.1 stage wall seconds. Its controls took
183.0 wall seconds and 404.1 process CPU seconds. The four runs account for
1,514.1 summed wall seconds and 2,377.7 measured process CPU seconds. The CPU
total excludes calibration and uninstrumented pilot stages. Some experiments
ran concurrently, so summed wall time differs from elapsed clock time.

Exact checkpoint replay reconstructed the accepted model from the archived
numeric patch and verified its receipt. A separate rollback audit selected
the original checkpoint in a copied research lineage and verified both
checkpoints remained byte-identical. The measured study retains its original
decision history and final head for reproducibility.
[Full study evidence](../results/phase-iii-natural-qwen-v1/five-cycle/)
· [Replay and rollback evidence](../results/phase-iii-natural-qwen-v1/accepted-lineage-replay.json)

The inference environment passed 130 tests and skipped the optional MCP test.
The core environment passed 126 and skipped three Torch-dependent tests.
The first wheel check used a core environment without the build backend.
The standard isolated build succeeded and the wheel contained the Phase III
modules.

### Next experiment

Freeze fresh examples and paraphrases across all three task families. Compare
the current compiler with one that includes authorized prior-family practice
examples in the preservation penalty. Match the candidate and prompt budgets
by adding the same number of generic practice sentinels to the control arm.
The hypothesis is that task-specific preservation lowers interference enough
to retain at least three edits in five cycles and improve the separate final
suite without target regressions. Keep the prior-cycle and final holdouts
outside both search arms. Random and gradient controls need separate lineage
runs for a cumulative comparison. The current controls cover cycle 01 only.
