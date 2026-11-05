# Phase III-R: fresh replication and conditional rank reduction

**Status: prospective implementation and validation. No Phase III-R learning result yet.**

This isolated study starts at `d0e0ea9e4cc1687d5ed0a70a321ca0af66b24c8a`.
It preserves the successful [Phase III-Q baseline](PHASE_III_Q_BASELINE.md),
its original checkpoint and historical evidence. Main, accepted lineage and
deployed models remain unchanged.

## Questions and declared scope

First, can the successful rank-eight recipe qualify again on fresh semantic
families at the same initialization seed? Only after exact independent
qualification does one rank-four run test retention of the rank-eight result.
This is a comparison of two declared adaptation configurations. It is not a
multiple-seed robustness study, a new optimizer, or a column search.

Both arms start from the original Qwen checkpoint
`7d4532858cf839d05ec34da088114df47238322f86f5e6e5c6693594a21a03b8`.
The attached directive omitted two characters from this hash; its 62-character
literal is not a SHA-256. A new read-only audit verified the existing original
directory against the valid Phase III-Q identity. No replacement was used.
The preserved Q merged reference is
`ddf9d6c72eab4cc6cbd7448a5d9c73a429881a26c721ef5726709df7ddf08191`.
Neither arm starts from it.

## Frozen design

The [protocol](../experiments/phase_iiir_replication/protocol.json) fixes the
two configurations, all populations, endpoint, budgets and gates before
training. The same 186 supported text linear modules across 24 layers receive
FP32 adapters while original BF16 parameters remain frozen. Rank eight has
alpha 16 and 5,411,328 trainable parameters; rank four has alpha 8 and
2,705,664. Both use the original randomized-A, zero-B initialization convention.
An independent data-order generator controls identical exposure despite
shape-dependent adapter initialization.

The shared population has 128 target truth-flipping families, 1,024 target
presentations and 1,024 equally weighted replay presentations. Development
has 32 fresh families. Each family retains both opposing outcomes, both operand
orders and both wordings. Exact Decimal arithmetic validates every label.
Host-side historical exclusion checks keep protected data private.

Bank A contains 32 selection families, 32 sealed families and 320 controls
for rank-eight replication. Disjoint Bank B has the same sizes and compares
the unchanged parent and both locked merged finalists. Its evaluation is
unavailable until both finalists pass the declared development checks.

AdamW uses learning rate 3e-4, betas (0.9, 0.999), epsilon 1e-8, weight decay
0.01, dropout zero, clipping 1.0, effective batch 16 through microbatches of
four, one epoch, 128 optimizer steps and seed 606117. The existing schedule
has seven warmup steps then linear decay over the 128-step horizon. Only steps
64 and 128 are evaluated. Supervision includes one leading space, the entire
canonical correct answer and newline; prompt and padding positions are masked.
The next-token causal shift and token-normalized loss are unchanged.

Each arm has one main attempt bounded by 5,400 wall seconds, 30,000 observed
process-tree CPU seconds, 24 GiB memory, 40,000 answer-token exposures and
400,000 input-token exposures. One disposable smoke per rank is limited to
32 optimizer steps. Exact interrupted training continuation is not validated;
an interrupted run cannot provide an eligible checkpoint or be silently
replaced. Temporary Windows sleep prevention is released after each run.

## Behavioral acceptance and persistence

All evaluations use native-tokenizer bare prompts, unrestricted greedy
generation, batch eight, six CPU threads, 128 input tokens and sixteen output
tokens. The frozen parser scores the completed first answer line. A valid
first line does not mean the full decode reached EOS. Full-continuation
differences and termination reasons are reported separately.

Practice must correct a repeatable error and gain a strict comparison.
Development must gain a strict comparison and a complete opposing family.
Every gate prohibits loss of required parent-correct or parent-valid responses.
Finalist ranking prefers family gain, strict gain, earlier step, then smaller
stored norm. A selected adapter must survive fresh loading, separate BF16
merging and remeasurement before independent evaluation.

Bank A runs preservation, selection, then sealed verification. Selection
requires positive strict gain; sealed verification requires at least two
strict gains and one complete-family gain. Exact reconstruction, independent
output replay and original-checkpoint rollback are also required. Only this
verified result permits rank-four main training.

Before Bank B opens, the selected rank-four checkpoint must retain all
rank-eight-correct and required valid development answers, controls, strict
comparisons and complete families. Bank B reports ordinary adaptation and
paired retention separately. Any applicable preservation or selection failure
stops access to the sealed stage. Neither result authorizes deployment or
automatic lineage admission.

## Implementation and validation

Phase III-R reuses the Q LoRA backend, completion loss, resource monitor,
generation endpoint, scoring, BF16 merge and independent verifier. Its additions
are versioned study bindings, fresh shared data, confirmation-bank isolation,
rank-specific authorizations and paired retention reporting. Historical Q
source and protocols remain unchanged. Software tests and model learning
outcomes will be reported separately.

## Results and interpretation

Pending the prospectively frozen execution. Unopened stages remain unmeasured.
Successful rank reduction would establish fewer trained factor parameters,
not automatically lower runtime, a smaller stored delta or fewer changed BF16
entries. A failed reduction would not erase a verified replication or prove
that all smaller solutions are impossible.
