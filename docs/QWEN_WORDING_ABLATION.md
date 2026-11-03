# Qwen repair across practice wordings

The [previous cohort](QWEN_REPAIR_COHORT.md) recovered 46 answers on
the practice wording and six on a paraphrase, with two cases losing
paraphrase accuracy. This follow-up compares single-wording and
multi-wording practice on the same fresh faults.

## Frozen design

The 12 slots cross layers 19, 21 and 23 with the two prior fault patterns
and two replicates. The evaluator calibrates each injected fault once on
construction prompts. Both arms receive the same damaged checkpoint,
eight practice number pairs with the same answer positions, and 16
practice sentinels. Single-wording practice uses the standard form eight
times. Multi-wording practice uses it four times, an inline question twice,
and a bracketed comparison twice. Every practice wording balances A and B.

Each arm gets nine candidate evaluations, or 27 forward batches, plus its
baseline sweep. Each has its own random control with the same candidate
budget and practice prompts. The random proposals share the layer, column
support, rank, and requested Frobenius norm of their paired guided
proposals. The prior proposal grid, selection rule, and patch policy apply.
Token lengths can differ with wording. The budget is matched in evaluated
prompts and forward batches.

The holdout contains 12 standard-form prompts, six comparison paraphrases,
six maximum paraphrases, and 16 sentinels. Each held-out number pair is new
to both searches. The two paraphrase templates scored 12/12 on a separate
pristine-only wording probe with disjoint numbers. Neither template is
present in the construction or practice prompts. The evaluator verifies
the saved finalist's checkpoint hash and replays practice before scoring
holdout. Search and evaluation run in separate trusted processes.

The primary comparison is the paired accuracy difference on the 12 fresh
paraphrases per scored case. For each guided arm the previous gate requires
recovering at least half of the pooled target loss, strictly beating its
matched random control, preserving every pristine sentinel prediction, and
avoiding any aggregate paraphrase accuracy drop versus the damaged model.
Loss below two answers leaves a case inconclusive. A cohort pass still
requires eight of the 12 frozen slots, including calibration holds in the
denominator.

## Completed result

Both arms failed the eight-pass threshold. Multi-wording passed two
cases and single wording passed one. One fault failed construction
calibration, and one scored case was inconclusive because the held-out
fault caused fewer than two errors. Eleven cases, including the
inconclusive case, contributed paired scores.

| Held-out target prompts | Pristine | Damaged | Single wording | Multi-wording | Random single | Random multi |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 132/132 | 78/132 | 120/132 | 125/132 | 83/132 | 85/132 |
| Fresh paraphrases | 128/132 | 104/132 | 119/132 | 120/132 | 105/132 | 108/132 |
| Comparison paraphrase | 62/66 | 50/66 | 58/66 | 57/66 | 50/66 | 52/66 |
| Maximum paraphrase | 66/66 | 54/66 | 61/66 | 63/66 | 55/66 | 56/66 |

Multi-wording gained one correct fresh paraphrase over single wording,
a paired difference of +0.76 percentage points. The case-resampled
descriptive 95% interval ran from −4.55 to +6.82 points. Multi-wording
won two cases, tied seven, and lost two on that measure. Both arms lost
paraphrase accuracy relative to damage in one case. Of the six extra
target answers from multi-wording, five came on the standard wording.

Single wording changed 13 pristine sentinel predictions, including
eight correct-to-incorrect changes. Multi-wording changed 14, including
nine correct-to-incorrect changes. In case 12, multi-wording recovered
10 of 12 lost target answers, went from 6/12 to 12/12 on fresh
paraphrases, and preserved all 16 pristine sentinel predictions.
The aggregate comparison gives little evidence that the wording mix
improves paraphrase transfer at this budget.

Each arm used 297 candidate forward batches. Candidate input tokens
were 88,182 per single-wording arm and 90,162 per multi-wording arm,
about 2.25% more with varied wording. Retokenization matched the logged
combined search input tokens for every scored case. The run used
11,069.20 process CPU seconds and 2,907.92 summed stage wall seconds
with up to three cases running at a time. The original weight shard
matched its pinned SHA-256 after the run.

[Raw scores and patch receipts](../results/qwen-wording-ablation-v1/)
· [Comparison chart](../results/figures/qwen_wording_ablation.png)
· [Pristine wording probe](../results/qwen-wording-probe-v1.json)

## Replay

Use the pinned Qwen3.5-0.8B shard and CPU BF16 environment described in
the [original cohort](QWEN_REPAIR_COHORT.md).

```powershell
$runner = 'experiments/original_weight_edits/run_qwen_wording_ablation.py'
$base = 'C:/path/to/pinned/Qwen3.5-0.8B'
$run = Join-Path $env:TEMP 'weightbridge-qwen-wording-replay'
python $runner run --run $run --pristine $base --jobs 3
python $runner export --run $run --out results/qwen-wording-ablation-local
python experiments/original_weight_edits/verify_qwen_wording_ablation.py results/qwen-wording-ablation-local
```
