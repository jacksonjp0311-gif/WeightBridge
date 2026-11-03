# Blinded Qwen repair cohort

The cohort tests whether a practice-only, activation-guided edit can repair
unseen induced faults in the pinned Qwen3.5-0.8B checkpoint. It spans layers
19, 21 and 23, with two seeds for each layer and each of two fault patterns.
Both patterns are rank-one MLP down-projection edits. Reversal suppresses
the correct label contrast for both answer positions. The one-position
pattern suppresses the contrast for one answer position, alternating
positions across its two replicates.

## Frozen protocol

- Each case has distinct construction, practice and holdout prompts.
  Targets balance A and B in each split and in each of the two holdout
  forms. The evaluator uses construction data to calibrate the fault.
  The search receives the damaged checkpoint and labeled practice data.
  The trusted stages run in separate processes under the same OS account.
- Each arm scores nine proposals, three strengths at each of three layers.
  The arms share one damaged baseline sweep over practice, which
  also captures activations. Each random proposal uses its guided pair's
  layer, column support, rank and requested Frobenius norm. The realized
  BF16 norms must match within 5%. The arms select finalists independently,
  so the two selected norms may differ.
- The guided proposal fits a ridge regression to signed correct answer
  positions, with a squared penalty on unrelated activations. Both arms
  select a positive practice gain with zero practice sentinel flips, then
  minimize sentinel margin drift and edit norm as tie breakers. An arm
  that finds no eligible edit falls back to the damaged checkpoint.
- The evaluator independently loads each saved finalist, verifies its
  hash and replays its practice scores before scoring holdout. To pass a
  case, guided repair must recover at least half the pooled accuracy loss,
  beat random search strictly, preserve every pristine sentinel
  prediction, and avoid decreasing paraphrase accuracy relative to damage.
  Report damaged-relative sentinel flips separately. A loss of fewer than
  two target answers makes the case inconclusive.
- At least eight of the original twelve slots must pass for a cohort PASS.
  Calibration holds and inconclusive cases remain in the denominator.
  Unrun cases or execution failures keep the cohort on HOLD.

The first protocol stopped at its construction floor in slot 01. Pristine
Qwen answered A on all eight targets and scored 4/8. A construction-only
probe on the same number pairs found 8/8 with a three-digit comparison
prompt and 8/8 with an inline alternative. The revised protocol uses
these two wordings, fresh seeds and a new numeric range. The first
protocol's remaining eleven slots were unrun. No holdout result informed
this change. [Retired protocol evidence](../results/qwen-repair-cohort-v1/summary.json)

## Run

Use a local copy of the model and the CPU BF16 runtime from the
[single-checkpoint experiment](QWEN_BLIND_REPAIR.md). The runner checks
the pinned weight shard and the manifest's exact source hashes. Allow
roughly 65 GB of scratch space for a full run. The output directory can
be reused to run unstarted slots without overwriting completed cases.

```powershell
$runner = 'experiments/original_weight_edits/run_qwen_repair_cohort.py'
$base = 'C:/path/to/pinned/Qwen3.5-0.8B'
$run = Join-Path $env:TEMP 'weightbridge-qwen-cohort-replay'
python $runner run --run $run --pristine $base --slots 1 2 --jobs 2
python $runner run --run $run --pristine $base --jobs 2
python $runner export --run $run --out results/qwen-repair-cohort-local
python experiments/original_weight_edits/verify_qwen_repair_cohort.py results/qwen-repair-cohort-local
```

The export includes raw practice and holdout scores, fault and finalist
patches, checkpoint receipts and stage resource measurements. Model
shards remain in scratch space. The archived cases measure two forced-choice
fault patterns in one checkpoint. Natural generation failures and direct
LLM patch selection remain separate experiments.
