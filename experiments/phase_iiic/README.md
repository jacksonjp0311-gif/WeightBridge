# Phase III-C behavioral measurement

The factorial holds each presented question and pair of visible options fixed
while changing the response instruction. The two scorers evaluate the next
letter or the whole answer text under each instruction. Both answer orders
are crossed with two paraphrases. A/B and X/Y label pairs are balanced across
underlying items, and the practice invariance probe switches both label pairs
within the same six items. Question clusters retain all four presentations in
the paired bootstrap. The reported percentile intervals preserve family strata
and describe this small generated sample.

The roster has four underlying items per family and split. The new word
strings use at least eight letters, dates come from years 2210 through 2270,
and fractions lie near three or above. Exact string ordering, calendar dates,
and rational arithmetic supply the correct options. Each item's semantic
hash is independent of option order and equivalent date or fraction spelling.
The answer tokens may differ in length even for equal character lengths.

Mean and sum exclude EOS. The separate terminated sum includes the score of
the tokenizer EOS after the last answer token. Crossed instructions are
diagnostic conditions. Each score uses the same rendered prompt prefix and
refuses any continuation whose joint tokenization would change that prefix.
Greedy generation runs from the same prefix and records raw text, token IDs,
and completion. Parsing uses exact case-insensitive, whitespace-normalized
matches against the presented option texts or their frozen aliases. The
generation correctness denominator includes unparsed and incomplete responses.

The practice decision ranks the two eligible content scores by passing the
practice gate, number of passing checks, number of fully correct underlying
items, accuracy, and unconditional agreement with generated answers. Mean
breaks remaining ties. The decision is written before selection or measurement
qualification data are opened. Both practice and independent qualification must
pass the frozen gate. The separate headroom check counts persistent errors and
cross-family solved controls in practice and selection only. Each qualifying
error needs a wrong likelihood choice and a completed, parsed, wrong generated
answer in all four presentations. The adaptation holdout stays unopened.

Use the pinned local Qwen checkpoint and an inference-enabled environment.
The generated roster uses the same seeds on every replay. The published
material includes exact measured bytes for practice, selection, and measurement
qualification. A new independent qualification requires new sealed material.

```powershell
$base = Join-Path $env:TEMP weightbridge-reference-inspection/qwen35-base-cpu
$python = Join-Path $env:TEMP weightbridge-reference-inspection/venv/Scripts/python.exe
$material = Join-Path $env:TEMP wb-phase-iiic-replay-material
$run = Join-Path $env:TEMP wb-phase-iiic-replay-run
$sourceReceipt = Join-Path $env:TEMP wb-phase-iiic-replay-sources.json
& $python -m experiments.phase_iiic.measurement_roster --base $base --out $material
& $python -m experiments.phase_iiic.archive_measurement --capture-sources --material $material --out $sourceReceipt
& $python -m experiments.phase_iiic.run_measurement --base $base --material $material --out $run
Copy-Item -LiteralPath $sourceReceipt -Destination (Join-Path $run execution-sources.json)
& $python -m experiments.phase_iiic.probe_invariance --base $base --material $material --out (Join-Path $run practice-invariance.json)
& $python -m experiments.phase_iiic.analyze_measurement --measurement $run --out (Join-Path $run factorial-analysis.json)
& $python -m experiments.phase_iiic.archive_measurement --material $material --run $run --out (Join-Path $env:TEMP wb-phase-iiic-replay-archive)
```
