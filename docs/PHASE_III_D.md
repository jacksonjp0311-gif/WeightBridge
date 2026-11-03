# Phase III-D counterfactual behavioral grounding

Phase III-C held real-model editing because the locked content score
underperformed the letter score and strict free generation rarely produced
a valid answer. Its display and neutral-prior diagnostics were exploratory.
Phase III-D tests them prospectively on new examples from the same three
task families. [Phase III-C evidence](PHASE_III_C.md)

## Measurement design

The pinned Qwen3.5-0.8B checkpoint is revision
`2fc06364715b967f1860aea9cf38778875588b17`, with checkpoint SHA-256
`7d4532858cf839d05ec34da088114df47238322f86f5e6e5c6693594a21a03b8`.
The roster has six underlying questions in each of practice, selection,
qualification, and the adaptation reserve. Each split has two word-order,
two date-order, and two fraction-order questions. Every question has two
paraphrases crossed with two presentation orders, for 24 presentations per
split. An opposite-direction question makes the other option correct
without changing the options. Practice chooses the endpoints. Selection
checks that choice before independent qualification. The adaptation reserve
is excluded from the measurement runner. [Frozen protocol](../experiments/phase_iiid/protocol.json)

The primary comparison crosses displayed labels with bullets and raw summed
content likelihood with neutral-prior subtraction. The instruction asks
for answer text under both displays. The letter-score baseline uses a
separate label-response instruction on the same labeled options. Raw
summed, mean per-token and EOS-inclusive summed likelihood are all scored.
For calibrated scoring, practice chooses a multiplier of 0.5, 1 or 1.5
and a neutral question template that either omits the question or uses a
generic direction to consider the options. All scored contexts keep the
answer options, display and response format fixed. The neutral subtraction
requires matching answer token IDs across contexts and excludes EOS. A
separate irrelevant statement supplies a negative control.

Let `d(q)` be the first-minus-second answer margin for question `q`.
The directional contrast multiplies `d(q) - d(q′)` by +1 if the first
answer is correct for `q` and −1 otherwise, then subtracts the absolute
margin difference between the irrelevant and neutral contexts.
The paired neutral term cancels when two contexts share the same prior.
Calibration can change which answer wins, while the contrast measures
whether the raw margins respond in the expected direction. The gate also
requires correctness for the original and reversed questions and stability
under option-order swaps. A correct answer preference under one prompt
alone gives weaker evidence.

Greedy free generation is measured separately on each display under both
questions. Strict scoring preserves the Phase III-C parser behavior and
requires an EOS-completed normalized option string or frozen alias.
Diagnostic semantic scoring also accepts a printed label or an agreeing
label with exact answer content. Ambiguous, conflicting, unrecognized and
incomplete output have explicit categories. A completed label can reveal
an unambiguous semantic choice while failing the answer-text instruction.
Only the strict score is eligible for the generation gate. The frozen
generation gate sets marginal accuracy thresholds for the original and
counterfactual questions. Joint correctness and order consistency are
reported separately. Agreement is conditional on parsed task responses,
using the selected scoring rule on the same display as generation.
Teacher-forced ranking and greedy generation are separate deployment
endpoints.

Intervals resample underlying questions within each family, keeping all
renderings of each sampled question together. Per-family intervals have
only two underlying questions and are descriptive. The two label sets are
balanced across items, so their strata mix label effects with item
difficulty. The experiment tests literal ordering on a small synthetic
question population using a bounded 12-token generation budget.

## Research gate

A ranker or generation endpoint must pass its own frozen thresholds.
An admitted ranker can establish headroom using repeatable incorrect
decisions. Generation headroom requires a completed, mapped answer and
keeps parse failures separate. Headroom also requires enough consistently
wrong items from the same family on practice and selection, with solved
cross-family controls. The practice lock cannot be revised using
qualification evidence. Selection and qualification data stay outside the
compiler.

The optional calibrated content-gradient backend differentiates through
both the question and neutral continuations under candidate weights and
remeasures the same objective after BF16 rounding. It accounts for four
continuations per row, keeps raw task and neutral margins, and requires the
existing measured preservation guard. The software path is exercised on
toy checkpoints. Any real-model update needs prospective admission and a
separate verifier.

## Measured result

Practice locked bullets with the generic neutral context and α = 1.5 for
ranking, and bullets for strict generation. The calibration rule was
chosen for joint accuracy across question directions and overall gate
performance. It scored 14/24 on the original practice prompts versus
15/24 for raw bullet scoring.

| Split | Locked ranker original | Reversed question | Both correct | Stable under order reversal | Ranker gate | Strict bullet generation original | Reversed question | Generation gate |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Practice | 14/24 | 17/24 | 10/24 | 4/12 | FAIL | 13/24 | 9/24 | FAIL |
| Selection | 17/24 | 19/24 | 14/24 | 9/12 | FAIL | 17/24 | 20/24 | PASS |
| Qualification | 17/24 | 17/24 | 11/24 | 5/12 | FAIL | 17/24 | 16/24 | FAIL |

The independent letter baseline scored 13/24. The locked ranker gained
16.7 percentage points against it, with a question-clustered 95% interval
of +8.3 to +25.0 points. The ranker missed joint correctness, correctness
in all four presentations, and option-order stability. Bullet generation
met the independent parsing and correctness thresholds and fell short on
ranker agreement, at 16/24 against the required 70%. These are separate
measurements of the same unchanged checkpoint.

| Qualification teacher-forced scoring | Labeled | Bullets |
| --- | ---: | ---: |
| Raw sum | 13/24 | 18/24 |
| Mean | 13/24 | 17/24 |
| EOS-inclusive sum | 14/24 | 18/24 |
| Locked prior subtraction | 16/24 | 17/24 |

Calibration gained 3/24 on labeled prompts and lost 1/24 on bullets. The
clustered intervals for the prior-minus-raw differences were 0 to +25.0
points and −33.3 to +25.0 points, respectively. The factorial interaction
interval also included zero. The locked ranker solved all eight date
presentations in both directions, five of eight word presentations in
each direction with three joint successes, and four of eight fraction
presentations in each direction with zero joint successes. Of 24 bullet
presentations, the raw margin moved in the expected direction in 17 and
exceeded the absolute irrelevant margin shift in 14. The mean signed
question shift was 1.534 log-score units, while the mean absolute
irrelevant shift was 0.642. The paired difference was +0.892
with a question-clustered 95% interval of +0.110 to +1.675. This
shows sensitivity to the question on this roster, with inconsistent
choices under reversal. The calibrated score has no clear advantage
over raw scoring on the locked display.

| Qualification free generation | Labeled | Bullets |
| --- | ---: | ---: |
| EOS completion on original prompts | 20/24 | 24/24 |
| Strict parses on original prompts | 0/24 | 24/24 |
| Strict correct on original prompts | 0/24 | 17/24 |
| Semantic correct on original prompts | 11/24 | 17/24 |
| Semantic correct on reversed prompts | 13/24 | 16/24 |
| Semantic correct on both | 4/24 | 13/24 |

Nineteen completed labeled responses on original prompts used a label
with its matching content, and one used the label alone. The paired gain
in original-prompt strict correctness was 70.8 points (95% interval +41.7
to +100.0), while the semantic gain was 25.0 points (−8.3 to +58.3).
The display effect was clearest for instruction compliance. Strict bullet
generation solved 8/8 date, 4/8 word, and 5/8 fraction presentations on the
original prompts. The letter baseline picked the first displayed option on
21/24 original presentations. The label strata and answer-length strata
are available in the raw analysis. Only six independent questions inform
each of these split-level estimates.

The admission replay classified the six practice items for the locked
ranker as one solved and five unstable. Selection had three solved, two
unstable and one persistently wrong item. No family reached the minimum
of two persistent errors in both splits. The adaptation decision is
**HOLD** with zero candidate attempts, zero modified parameters and no
inference on the adaptation reserve. The source hash was verified again
at the end of measurement.

The run took 1,068.45 wall seconds, 6,181.30 process CPU seconds, and
6.27 GiB peak RSS on the Windows CPU BF16 path. It loaded the checkpoint
once, scored 1,584 continuations in 198 forward batches, and generated
288 responses in 36 batches. The runtime used reference PyTorch kernels.
[Measured report](../results/phase-iii-d/grounding/run/report.json) ·
[Admission receipt](../results/phase-iii-d/grounding/run/admission.json) ·
[Question-level analysis](../results/phase-iii-d/grounding/run/description.json) ·
[Archive manifest](../results/phase-iii-d/grounding/artifact_manifest.json) ·
[Runtime](../results/phase-iii-d/runtime.json)

The next experiment is a larger fresh comparison of the raw and
neutral-subtracted rankers with a paired score based on the margin
difference between a question and its semantic opposite. Freeze that
score on practice and cross A/B and X/Y within the same question. The
falsifiable prediction is higher joint accuracy and order stability on
unseen items without larger preference shifts under irrelevant or
meaning-preserving question changes. Keep free generation separate and
leave the adaptation gate in force.

## Replay

Use the pinned local checkpoint and a Python environment with the optional
inference dependencies. Each run needs a fresh output directory.

```powershell
python experiments/phase_iiid/roster.py --base $Base --out $Material
python experiments/phase_iiid/run_grounding.py --base $Base --material $Material --out $Run
python experiments/phase_iiid/describe_grounding.py --run $Run --out "$Run/description.json"
python experiments/phase_iiid/admission.py --material $Material --measurement $Run --out "$Run/admission.json"
python experiments/phase_iiid/archive_grounding.py --material $Material --run $Run --out $Archive --admission "$Run/admission.json" --description "$Run/description.json"
```

The admission and description commands also work on the archived material
and run directories with fresh output paths, without loading the model.
The archive binds exact source, protocol, prediction and decision hashes.
Admission replays the decisions from saved model outputs and stop reasons.

The synthetic generator and seed make this experiment reproducible. Its
reserve is unopened by the experiment, and can be regenerated by someone
with the source. A future blind study exposed to public repository access
needs host-private evaluation examples.
