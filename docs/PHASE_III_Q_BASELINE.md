# Phase III-Q: working learning baseline

This study supersedes the unrun column-selection proposal. It does not run a
column search. It starts from published Phase III-P commit
`3b925f4485c7708e87d9a898eb308d8b6688b3a5` on the isolated
`dev/phase-iii-q-baseline` branch. Historical evidence, accepted lineage and the
original checkpoint remain unchanged.

## Question and scope

Can conventional supervised LoRA training produce a persistent improvement in
complete greedy decimal-comparison answers, with no prohibited regressions on
the frozen measured suites? This is a practical baseline. It changes data
volume, editable capacity and training method together and cannot attribute
the failures of earlier custom editors to any one restriction.

The I and J controls were earlier adapter experiments. Both used rank four in
one affine matrix, 18,432 trainable parameters, 32 training paths and four
optimizer steps. I's merged norm exceeded its frozen limit and received no
generation evaluation. J stayed within its separate projected envelope but
lost a strict target comparison and caused two presentation regressions. Those
results are preserved; they did not test this broader conventional recipe.
Phase III-P's local full-block lower bounds also exceeded the 0.025 edit cap
in its authorized region. That result concerns the sampled linear constraints,
not global trainability. Its separate external repetition edit is not used
here, either as a parent or a source of training examples.

## Frozen recipe

The parent is the original Qwen3.5-0.8B checkpoint with SHA-256
`7d4532858cf839d05ec34da088114df47238322f86f5e6e5c6693594a21a03b8`.
Runtime inspection found 186 supported text attention and MLP linear modules
across 24 layers. The explicit allowlist excludes vision, embeddings, the
output head, tied tensors, normalization and non-linear/custom operations.

Each eligible matrix receives the additive adapter from
[Hu et al., LoRA (v2)](https://arxiv.org/abs/2106.09685v2):

\[
W_{\mathrm{effective}}=W+\frac{16}{8}BA,
\qquad A\in\mathbb{R}^{8\times d_{in}},\quad
B\in\mathbb{R}^{d_{out}\times8}.
\]

Both factors train in FP32. A is deterministically randomized and B starts at
zero. All original BF16 parameters stay frozen. The adapter has 5,411,328
parameters. No previous 0.025 norm, sixteen-column or single-layer restriction
applies. A separate, exact-module and exact-data training authorization leaves
the historical patch policy untouched.

The supervised target is one leading space, the canonical correct decimal,
and a newline. Native tokenization must preserve the bare frozen prompt
prefix exactly. Prompt and padding labels are masked. Logits at position t
predict the label at t+1. Cross-entropy covers every answer token and the
delimiter. Restricted output-head computation is checked against full logits;
it does not constrain the vocabulary or greedy generation.

For prompt \(x_i\) and complete supervised continuation \(y_i\), the batch
loss is the negative sum of \(\log P(y_{i,t}\mid x_i,y_{i,<t})\), divided by
the number of unmasked continuation tokens. The two sources have equal
example exposure; longer answers contribute more token terms. No loss term
sees a development, selection or hidden example.

Fresh data contain 128 target training families (1,024 presentations) and
1,024 equally weighted preservation-replay presentations. Targets contain both
opposing outcomes, both operand orders and both wordings. Replay includes
positive/mixed-sign comparisons, decimal copying, integer addition and
successor controls. Exact Decimal arithmetic supplies labels; inference has
no oracle. No training example is selected by parent error. A prospectively
declared wider integer bank supplies enough disjoint families; the domain is
10 through 9,999 in magnitude. Historical and timing-preflight identities are
excluded. All relatives remain in one partition.

Development, selection and hidden evaluation each contain 32 fresh target
families. A separate preservation-validation population has 320
presentations. Selection/hidden seeds and rosters remain host-private.

After a disposable four-family pipeline smoke passes, two fresh runs use
AdamW, learning rates 1e-4 and 3e-4, one preregistered seed, effective batch 16,
microbatch four, clipping 1.0, 5% warmup and linear decay. Each run has one
epoch and 128 optimizer steps. Checkpoints 64 and 128 are evaluated, at most
two per run. The one-epoch limit was chosen from CPU timing before candidate
accuracy. It is below the permitted three-epoch maximum and limits the
conclusions of a failure. Each run is bounded by 5,400 wall seconds, 30,000
process CPU seconds and 24 GiB RSS.

## Measurement and gates

All parent, adapter and merged-model comparisons use the same bare prompts,
native tokenizer, unrestricted greedy decoding, sixteen-token budget and
exact completed-first-line parser. Individual correct responses, all-four
strict comparisons and all-eight complete counterfactual families are
reported separately. Newline completion does not imply that the entire
decode reached EOS; EOS and budget stops are recorded separately.

The practice probe uses sixteen complete training families and eighty replay
presentations. Development uses its 32 fresh families and the same declared
replay probe. Practice requires a corrected repeatable error and positive
strict gain. Development requires positive strict gain and a newly complete
family. Both reject every lost parent-correct response or lost parent-valid
completion. Finalist ordering is frozen in the protocol; all four evaluated
checkpoints remain in the record.

Only a locked qualifying finalist can be merged under a separate
authorization. Fresh-process adapter loading, BF16 merging, merged practice
and development checks precede preservation validation. Selection requires
positive strict gain and no prohibited losses. Hidden verification requires
at least two strict-comparison gains, at least one complete-family gain and
no prohibited losses. Failed or incomplete stages cannot authorize later
ones. Exact reconstruction and rollback precede a verified-baseline claim.
Lineage admission and deployment remain operator-controlled.

## Preflight evidence

The historical K–N numerical archive and Phase III-P analysis replayed
without modifying their files. The pinned runtime is Python 3.12.10, PyTorch
2.15.0, Transformers 5.22.1, NumPy 2.5.4 and safetensors 0.8.0. No package was
installed or changed, and no replacement checkpoint was downloaded.

The initial cost-only pass accidentally used the native chat template.
Its immutable receipt is retained as a format mismatch. A separately bounded
repeat used the correct bare prompt. This measured 6.824 seconds for a
microbatch-four forward/backward and estimated 27.483 seconds per effective
batch-sixteen optimizer step, about 59 minutes for 128 steps. This is one
timed microbatch, not a precise runtime promise. It excludes evaluation,
checkpoint statistics and startup. Neither timing pass scored generated
answers. Both discarded their adapters and verified unchanged parent bytes.

## Execution status

The disposable smoke passed. The unchanged parent answered 2/32 completed
presentations correctly; the trained adapter answered 32/32, including all
eight strict comparisons and four complete opposing families. Two fresh
processes reproduced the exact generated token sequences. Disabling the
adapter restored the parent predictions exactly, and original checkpoint
bytes stayed unchanged. Every A and B factor received nonzero gradients and
changed. The same-population token NLL fell from 0.1758512 to 0.00000876256.
All 32 decodes still reached the sixteen-token budget rather than EOS; their
first answer lines were completed and correct. This is training-set fit,
not independent semantic transfer.

The smoke used 32 optimizer steps, 32 forward/backward training calls and
1,216 supervised answer-token exposures. The adapter artifact SHA-256 is
`119f2375044a26c723befa23000897c8be81c507bd760e692bcb73c5c34a19d6`.
Its unrounded effective update norm was 8.52609. The BF16 merge preview norm
was 8.54197, with 463,949,792 changed entries and rounding residual norm
0.522482. This preview was measured without installing or exporting a merged
smoke checkpoint. These are conventional-training magnitudes, not authorized
edits under the older direct-edit policy.

Two host integration issues were found before main training. The raw smoke
gate stores protocol identity under `identity`, while the main admission API
expects a top-level field. A separate host bridge independently replays the
raw checks and provides that binding without changing the frozen files.
Also, the Windows venv launcher spawned the real model process, so the raw
launcher CPU/RSS aggregate undercounted work. That receipt is retained and
flagged. An append-only audit uses actual worker-body/cleanup measurements:
586.716 wall seconds, a CPU lower bound of 2,881.0 seconds and a conservative
upper bound of 3,461.10 seconds, below the 4,500-second smoke limit. Exact total
CPU is unknown. Known lifetime peak RSS is 7,020,761,088 bytes. The main runs
use process-tree monitoring. No behavioral threshold was changed.

The first main launch omitted the required smoke-admission argument and was
refused before model load. The corrected launch used the unchanged frozen
training configuration. Windows then entered idle Modern Standby. Seventeen
optimizer steps were recorded before the original wall-time guard stopped
the owned process tree on resume. No scheduled checkpoint was reached. The
observed process tree used 3,054.625 CPU seconds; its 22,020.967 elapsed wall
seconds include standby and are not active compute. At least 68 training
forward/backward calls were completed. Partial work beyond the last journal
entry is unmeasured. The original model hash was verified again after the stop.

The resource interruption consumed the authorized `1e-4` arm. It was not
retrained. A [prospective resource continuation](../results/phase-iii-q-baseline/baseline-v1/resource-continuation.json)
keeps the remaining `3e-4` recipe, checkpoint steps, populations and every
behavioral threshold unchanged. Its two checkpoints may be assessed only if
the run completes. The lower-rate arm has no behavioral observation, so the
study cannot compare learning rates. The temporary Windows sleep request
is cleared on exit; no persistent power setting is changed.

Development and independent stages remain unopened while the surviving run
trains. The accepted lineage remains unchanged.

Receipts: [smoke report](../results/phase-iii-q-baseline/smoke-report.json),
[raw gate](../results/phase-iii-q-baseline/baseline-v1/smoke/smoke-gate.json),
[independent admission and cost audit](../results/phase-iii-q-baseline/baseline-v1/smoke-admission.json),
[frozen-source tests](../results/phase-iii-q-baseline/tests/frozen-v1/tests.json).

The [interruption archive](../results/phase-iii-q-baseline/baseline-v1/interruptions)
preserves both failed launch attempts and the relevant Windows power events.

The full protocol is
[`experiments/phase_iiiq_baseline/protocol.json`](../experiments/phase_iiiq_baseline/protocol.json).
Training curves are diagnostic; only actual complete generated answers can
establish behavioral improvement. A baseline pass would establish one
bounded learning result, not a universal adaptation algorithm or a success
of the earlier custom editor.

## Reproduction

Use an existing local checkpoint with the exact pinned directory hash and the
recorded Windows runtime. The host utilities do not download a model. Set
`$wbPython` to that runtime's `python.exe`, `$wbBase` to the immutable checkpoint,
and `$wbStudy` to a new private directory outside the checkout and checkpoint.
In PowerShell, from the branch checkout:

```powershell
$wbInventory = 'results/phase-iii-q-baseline/preflight/inventory-v1/architecture-inventory.json'
& $wbPython -m experiments.phase_iiiq_baseline.initialize freeze `
  --base $wbBase --inventory $wbInventory --out $wbStudy
& $wbPython -m experiments.phase_iiiq_baseline.execute_smoke `
  --base $wbBase --protocol "$wbStudy/protocol.json" `
  --authorization "$wbStudy/smoke-authorization.json" --bundle "$wbStudy/smoke-bundle.json" `
  --rows "$wbStudy/material/smoke.json" --out "$wbStudy/smoke"
```

Replay the smoke evidence with `experiments.phase_iiiq_baseline.admit_smoke`
before `initialize authorize-main`. Use the resulting `smoke-admission.json`
as the `--smoke-gate` for both main authorization and `run_training`. Main
training receives only `training-bundle.json`; it does not receive the
material directory. The host chooses a new output directory for each attempt.
For the surviving recipe, the command is:

```powershell
& $wbPython -m experiments.phase_iiiq_baseline.run_training `
  --base $wbBase --protocol "$wbStudy/protocol.json" `
  --authorization "$wbStudy/main-authorization.json" --bundle "$wbStudy/training-bundle.json" `
  --smoke-gate "$wbStudy/smoke-admission.json" --run-index 1 --out "$wbStudy/run-1"
```

The ordinary `study assess` command expects both completed runs. The separate
`resource_limited_assessment` command exists for the recorded standby cutoff;
it binds the published continuation and both process receipts while reusing
the same evaluation and gates. It cannot convert another kind of failure
into an eligible checkpoint. No public command opens selection or hidden
without the host's frozen inputs and prior receipts. Reproducing a private
qualification score requires that operator-held material; the public archive
contains commitments and permitted evidence, not private rosters or weights.
