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
