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
probe on three-digit versions of those pairs found 8/8 with a comparison
prompt and 8/8 with an inline alternative. The revised protocol uses
these two wordings, fresh seeds and a new numeric range. The first
protocol's remaining eleven slots were unrun. No holdout result informed
this change. [Retired protocol evidence](../results/qwen-repair-cohort-v1/summary.json)

## Completed result

The revised cohort finished all 12 slots. Five passed, five failed, and
two were inconclusive because damage cost only one held-out answer. The
cohort disposition is **FAIL** against the frozen eight-pass threshold.

| Held-out target prompts | Pristine | Damaged | Guided | Random |
|---|---:|---:|---:|---:|
| Same wording | 144/144 | 92/144 | 138/144 | 92/144 |
| Paraphrase | 140/144 | 106/144 | 112/144 | 107/144 |
| Total | 284/288 | 198/288 | 250/288 | 199/288 |

Guided repair beat random on 11 of 12 cases. Its mean paired advantage
was 17.7 percentage points, with a case-resampled descriptive 95%
interval of 10.1 to 25.7 points. The interval reflects variation across
these synthetic cases on one model. Both arms used 324 candidate forward
batches, in addition to the shared practice baselines. Four random arms
selected a finalist. The remaining eight used the damaged checkpoint.

Case 7 gained five answers on the original wording and lost six on the
paraphrase. Across the cohort, guided patches changed eight pristine
sentinel predictions, including five correct-to-incorrect changes.
Every guided finalist targeted layer 23. Four faults were injected there
and eight were injected elsewhere. The result supports compensatory
repair and leaves fault localization unverified.

The 12 cases used 6,689.13 process CPU seconds and 1,955.16 summed
stage wall seconds with two cases running at a time. Peak recorded
single-process RSS was 2,542,981,120 bytes. The original weight shard
matched its pinned SHA-256 after the run.

[Per-case scores, policies and receipts](../results/qwen-repair-cohort-v2/)
· [Comparison by wording](../results/figures/qwen_repair_cohort.png)

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
