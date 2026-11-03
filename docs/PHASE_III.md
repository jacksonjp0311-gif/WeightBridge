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
