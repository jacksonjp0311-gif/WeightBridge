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
