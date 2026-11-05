# Phase III-Q: working learning baseline

This study supersedes the unrun column-selection proposal. It does not run a
column search. It starts from published Phase III-P commit
`3b925f4485c7708e87d9a898eb308d8b6688b3a5` on the isolated
`dev/phase-iii-q-baseline` branch. Historical evidence, accepted lineage and the
original checkpoint remain unchanged.

**Disposition: `VERIFIED_BASELINE_PASS`.** The completed `3e-4` recipe
produced a locked BF16 checkpoint that corrected every tested independent
opposing-answer family, preserved every required response, and reproduced
its complete greedy outputs after exact reconstruction. The low-rate arm
was interrupted and supplies no learning-rate comparison.

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

The surviving run completed all 128 steps. It exposed 2,048 presentations,
17,384 supervised answer tokens and 172,804 unpadded input tokens, using 512
training forwards and 512 backwards. All adapter factors received nonzero
gradients and changed; the parent remained byte-identical. Observed process
tree plus host cost was 3,216.263 wall seconds and 20,068.656 CPU seconds.
The conservative sum of observed lifetime peak working sets was
7,283,310,592 bytes. The temporary sleep request was released. No external
compute was used.

| Saved adapter | Effective norm | BF16 merge preview norm | Preview changed entries | Rounding residual norm |
|---|---:|---:|---:|---:|
| Step 64 | 3.86612 | 3.89828 | 425,661,705 | 0.516876 |
| Step 128 | 3.92829 | 3.96006 | 426,620,449 | 0.517094 |

These are round-trip BF16 previews computed from actual saved FP32 factors.
The [training journal](../results/phase-iii-q-baseline/baseline-v1/run-1)
retains every step and both adapter receipts.

## Generated answers and stored checkpoint

Both saved adapters answered 208/208 practice presentations and 336/336
development presentations correctly. The latter includes 256 target responses
and 80 controls. Neither candidate lost a previously correct response or
valid answer. They tied on all behavioral ranks, so the frozen earlier-step
tie-break locked step 64. Its saved adapter SHA-256 is
`23f5209e5cbebf9ab4e7284023776548dce9a4be0265ead86b83a20cf8890b56`.
A fresh process exactly reproduced all 208 practice token sequences, text,
and stop reasons before the merge.

The separately authorized merged BF16 checkpoint has SHA-256
`ddf9d6c72eab4cc6cbd7448a5d9c73a429881a26c721ef5726709df7ddf08191`.
An independent byte audit confirmed 425,661,705 changed BF16 entries in 186
authorized tensors. The realized norm is 3.8982769786, or 0.418575% of the
full parent parameter norm. All other 302 tensors and six nonweight assets
matched the parent byte for byte. The effective update is dense across its
authorized text regions even though only 5,411,328 factor parameters trained.

The merged model preserved all 544 scored first lines from the adapter's
practice and development evaluations. It changed the full continuation on
28 rows, all after the first line. Thus the merged model passed the declared
endpoint but did not produce identical full continuations to the adapter.
The merged checkpoint, with its own identity, is the only representation
used for independent behavioral qualification.

| Merged-model evaluation | Correct target presentations, parent → merged | Strict comparisons | Complete opposing families | Prohibited losses |
|---|---:|---:|---:|---:|
| Practice target probe | 23/128 → 128/128 | 0/32 → 32/32 | 0/16 → 16/16 | 0 |
| Fresh development targets | 30/256 → 256/256 | 0/64 → 64/64 | 0/32 → 32/32 | 0 |
| Independent selection | 31/256 → 256/256 | 0/64 → 64/64 | 0/32 → 32/32 | 0 |
| Sealed verification | 26/256 → 256/256 | 0/64 → 64/64 | 0/32 → 32/32 | 0 |

The 80 replay control presentations in each screening roster stayed correct.
The independent preservation suite retained **320/320** correct responses,
64 presentations each for copying, addition, successor, mixed-sign and
positive comparisons. There were zero lost valid responses in any measured
gate. The strict comparison unit is all four wording/order presentations;
a complete family requires both strict comparisons, eight correct responses.
Every reported behavioral pass here concerns a completed first answer line.
The full greedy decodes reached the sixteen-token budget, with zero EOS stops.

All frozen behavioral gates passed. A fresh process rebuilt the entire merged
checkpoint to the same SHA-256 and reproduced the numerical merge receipt
byte for byte. Fresh independent processes then replayed all 832 protected
outputs from the rebuilt model and all 832 original outputs after rollback
by reselecting the immutable parent. Token IDs, text and stop reasons matched
exactly in all three suites. These replayed rows are verification, not new
independent samples. No checkpoint was admitted to lineage or deployed.

Public [screening receipts](../results/phase-iii-q-baseline/baseline-v1/screening)
include the raw scored practice/development outputs, numerical and ranking
receipts, and [measured figures](../results/phase-iii-q-baseline/baseline-v1/screening/figures).
The development examples are now disclosed and cannot serve as independent
data. Protected rosters and per-row protected outputs remain host-private.

See the [protected aggregate gates](../results/phase-iii-q-baseline/baseline-v1/qualification),
[independent outcome figure](../results/phase-iii-q-baseline/baseline-v1/qualification/independent-outcomes.png),
and [final verification receipt](../results/phase-iii-q-baseline/baseline-v1/final-verification/final-verification.json).
The first independent receipt's HOLD indicates that reconstruction was
pending at that time. The later write-once final receipt resolves it without
overwriting the earlier evidence. The final receipt SHA-256 is
`735807c9f56c560660004fa60bd5c110fb84afb8354544c88c646b54e56c6cdf`.

## Computation and validation

The [compute ledger](../results/phase-iii-q-baseline/baseline-v1/compute-accounting.json)
accounts for each enclosing experiment process tree once. The observed lower
bound is 49,338.828125 CPU seconds across preflight, smoke, both attempted
training arms, generation, merging and final replay. Exact total CPU is
unknown because some startup and interrupted work was not observed. The
summed elapsed scope time is 29,937.694 seconds, including the interrupted
run's standby interval. It is not active compute time. The completed training
run used 3,216.263 elapsed seconds and 20,068.656 observed CPU seconds;
screening used 1,103.539 and 6,475.391 respectively; final reconstruction and
replay used 1,236.793 elapsed seconds and at least 6,903.656 CPU seconds.

The model work ledger records at least 648 explicit forward calls and 620
backward calls, 183 recorded optimizer steps including the disposable runs,
and 5,904 complete generation records across 11,808 batched decode steps.
Decode steps are reported separately from explicit loss forwards. A partial
interrupted step is unmeasured. CPU accounting excludes the separately
recorded software validation and read-only archival audit. No paid or remote
model compute was used; electricity cost was not measured.

The Windows host has a Ryzen AI Max+ 395, 16 physical and 32 logical cores,
and 132,766,306,304 bytes of reported RAM. The pinned model runtime used six
CPU threads and native BF16 with FP32 adapter factors. CUDA was unavailable.
The reference PyTorch fallback kernels ran without changing dependencies.

The final [runtime suite](../results/phase-iii-q-baseline/tests/final-q-v1/tests.json)
passed 832 tests with two optional-dependency skips, for Matplotlib and MCP.
The separate MCP suite passed all 23 tests. Figures were rendered with the
existing plotting environment. The 127 frozen source files verified before
and after the study. The original Phase III-P archive was not rewritten.

## Interpretation

The complete-answer training recipe transfers to the tested fresh families
and preserves the measured controls. This shows that a conventional adapter
can produce the desired behavior under the narrow, frozen endpoint. It does
not identify which of the earlier geometry, data, and objective restrictions
caused their failures. It also does not establish a new general mathematical
ability, preservation outside the measured suites, or continual learning.
The interrupted arm leaves the learning-rate contrast unmeasured. The next
scientific step is a fresh, prospectively frozen replication of the winning
recipe before reducing rank, module coverage, or supervision.

Receipts: [smoke report](../results/phase-iii-q-baseline/smoke-report.json),
[raw gate](../results/phase-iii-q-baseline/baseline-v1/smoke/smoke-gate.json),
[independent admission and cost audit](../results/phase-iii-q-baseline/baseline-v1/smoke-admission.json),
[frozen-source tests](../results/phase-iii-q-baseline/tests/frozen-v1/tests.json).

The [interruption archive](../results/phase-iii-q-baseline/baseline-v1/interruptions)
preserves both failed launch attempts and the relevant Windows power events.

The full protocol is
[`experiments/phase_iiiq_baseline/protocol.json`](../experiments/phase_iiiq_baseline/protocol.json).
Training curves are diagnostic; only actual complete generated answers can
establish behavioral improvement. The recorded pass supplies a bounded
reference for later direct-edit comparisons.

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

Replay the smoke evidence and authorize the exact training bundle. Main
training receives only that bundle; it does not receive the material
directory. Use a fresh output directory for each attempt. The surviving
recipe is run index 1:

```powershell
& $wbPython -m experiments.phase_iiiq_baseline.admit_smoke `
  --protocol "$wbStudy/protocol.json" --authorization "$wbStudy/smoke-authorization.json" `
  --bundle "$wbStudy/smoke-bundle.json" --rows "$wbStudy/material/smoke.json" `
  --smoke "$wbStudy/smoke" --out "$wbStudy/smoke-admission.json"
& $wbPython -m experiments.phase_iiiq_baseline.initialize authorize-main `
  --out $wbStudy --smoke-gate "$wbStudy/smoke-admission.json"
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

For the recorded finalist, the [host qualification command](../results/phase-iii-q-baseline/baseline-v1/qualification-command.py)
replays the saved adapter, merges it under the separate authorization, and
runs the original gated `study finish` verifier. After all behavioral gates
pass, run the final verifier against the locked stored representation:

```powershell
& $wbPython -m experiments.phase_iiiq_baseline.verify_final `
  --base $wbBase --merged "$wbStudy/merged-checkpoint" `
  --merge-receipt "$wbStudy/merge-receipt.json" --merge-authorization "$wbStudy/merge-authorization.json" `
  --protocol "$wbStudy/protocol.json" --material "$wbStudy/material" `
  --assessment "$wbStudy/assessment" --out "$wbStudy/final-verification"
```

This produces an append-only verification receipt without accepting or
deploying the candidate. Exact replay establishes checkpoint persistence,
not bitwise reproducibility of a new training run on different hardware.
