# WeightBridge v0.2 findings

## Evidence boundary

This report distinguishes reconstruction, software checks, synthetic behavior,
and language-model benchmarks. The published Qwen scores in the article are
reference claims only. No published MATH-500, GSM8K, or IFEval score was
measured independently in this workspace.

## What was reproduced

- The supplied v0.1 deterministic demo was rerun from its preserved code:
  baseline accuracy was 75.30%; its pre-specified patch candidate reached
  98.45% on 2,000 deterministic examples (seed 813). The patch was known in
  advance and this is a tooling demonstration, not agent discovery.
- The authors' Experiment 8 p50c patch was applied to the pinned
  `Qwen/Qwen3.5-0.8B` checkpoint (revision
  `2fc06364715b967f1860aea9cf38778875588b17`). WeightBridge independently
  reconstructed 130,188 BF16 value changes across two MLP `down_proj` tensors.
  Both edited tensor SHA-256 hashes and the resulting whole-file candidate hash
  matched the recorded values. The original checkpoint was not overwritten.
- A matched-count randomized Qwen control was constructed in the same layers
  with the same per-layer number of changed values and sampled edit-magnitude
  distribution. Its behavior was not evaluated, so it is only a ready control
  checkpoint, not a measured negative result.

The authors report p50 MATH-500 48.1%, GSM8K 67.1%, IFEval 62.3%, and a 33.1%
MATH token-limit rate. These remain author-reported results. Our patch
reconstruction does not reproduce those metrics.

## Qwen behavior smoke

On two hand-authored arithmetic prompts, one generation per prompt, fixed
sampling seed 0/1, CPU BF16 inference and a 64-token cap, both base and p50
candidate reached the imposed limit on both prompts. Neither produced a
complete boxed answer. Accuracy and answer-completion rates were 0/2 for both;
the paired difference is zero and the paired bootstrap interval is [0, 0].
At this cap, the loop detector and 8-gram tail repetition proxy were also zero.
This outcome is INCONCLUSIVE because the answers were truncated before the
task could be scored.

An earlier exploratory two-prompt, 256-token smoke also hit the cap on all
four generations and yielded no boxed answers. Its tail repetition proxy was
0.004132 for base and 0 for p50. The discrepancy in proxy is not evidence of a
behavioral improvement: n is two and neither output completed. Both runs are
far below the published 32,768-token setting and use no benchmark dataset.

## Independent weight-edit discovery on a toy defect

The restricted practice-only diagnostic worker was tested twice on fresh task
seeds. It received a frozen practice set, read-only activation summaries, a
fixed allowlist (`output_bias`), and a maximum of eight bounded candidate
evaluations. It did not receive the hidden holdout or its labels. It selected
`output_bias[1] += -4` without being handed the correction value. The parent
validated the patch, wrote an immutable candidate, reloaded it from disk, and
then scored the hidden holdout once.

Across three isolated 64-item holdouts (192 examples), baseline accuracy was
50% and the diagnostic candidate was 100%; a one-entry random control remained
at 50%. Each run's paired 95% interval was [37.5%, 62.5%] at n=64, and all
three runs selected the same correction. The edit changed one of twelve
parameters. This is a
successful discovery by a deterministic tool-using search agent on a
deliberately localized synthetic defect. It is not an LLM discovery, Qwen
improvement, new mathematical capability, or evidence of general learning.

The seed-731 task was initially recorded twice. An append-only correction
marks the first record as a duplicate replay; the canonical isolated run is
retained with later isolated seeds 840 and 841.

## Conventional optimizer and memory ablation

Full-batch SGD on the same toy model, trained for 400 cross-entropy steps on
practice data only, also reached 100% accuracy on 320 held-out examples across
five seeds (baseline 50%; paired 95% interval for the difference
[44.375%, 55.625%]). It changed all 12 parameters. This is a conventional
gradient baseline and is reported separately from gradient-free search. The
toy comparison does not establish compute parity or compare against LoRA on a
language model.

The MWC pilot pools ten independent synthetic task seeds (640 paired examples
per condition):

| Condition | Accuracy | Paired gain vs no-memory/no-edit | 95% cluster bootstrap by seed | Regression rate |
|---|---:|---:|---:|---:|
| No memory, no edit | 50.00% | — | — | — |
| External memory only | 100.00% | +50.00 pp | [50.00, 50.00] pp | 0/320 baseline-correct |
| Weight edit only | 98.125% | +48.125 pp | [44.375, 50.00] pp | 0/320 baseline-correct |
| External memory plus weight edit | 100.00% | +50.00 pp | [50.00, 50.00] pp | 0/320 baseline-correct |

The external-memory-only condition applies a retrieved decision-time logit
bias and leaves the checkpoint unchanged. It is not weight learning. The
weight-only agent selected -3 on one seed and scored 81.25%; the verified
memory-guided candidate selected -4 and scored 100% on that same synthetic
failure family. In the recorded practice budget, the weight-only search used
eight candidate evaluations per seed while memory plus weight used two. The
retrieved experience was the exact prior repair for the same injected defect.
This supports a narrow result: verified external evidence reduced search
effort for this repeated localized failure. It does not show general MWC
benefit, transfer to unrelated defects, or a benefit over simply applying a
known correction.

A deterministic audit reloaded the three frozen toy candidates, controls and
already scored holdouts. It recovered 32 rescues and zero regressions for the
diagnostic edit on each seed; the random control had zero rescues and zero
regressions. This is a metric derivation for the same observations, not a new
experiment.

Every selected toy and MWC candidate was reloaded from a new checkpoint file
before scoring. That demonstrates persistence of those synthetic edits after
serialization/reload. A fresh Qwen process also reloaded the original model
successfully for rollback; no post-restart Qwen performance gain was shown.

## Weight-level observations and limits

The published patch targets 64 columns in layers 15 and 19, each tensor shaped
1024 × 3584. Its sparse artifact contains 65,260 and 64,928 changed BF16
values, respectively; 884 of the 131,072 targeted positions round to the base
value. The changed-value fraction is about 0.014905% of the 873,438,784
checkpoint state entries. These are structural observations of one published
patch. They do not identify the causal role of those columns.

No layer-frequency, magnitude-response, or cross-task interference pattern is
established by the available experiments. The toy edit touched an output bias,
whereas the published patch touches transformer MLP tensors; they are not
comparable intervention classes. The randomized Qwen control remains
unevaluated. No chart or aggregate in this report converts published metrics
into independent measurements.

## Recorded compute

The canonical ledger records 453.05 process CPU seconds and 204.23 summed
wall-clock seconds across the experiment records with resource fields. This
includes two Qwen CPU smoke runs, patch reconstruction, synthetic discovery,
MWC and the toy SGD baseline. The sum excludes development time, test time,
reference inspection, dataset downloads and other unmetered work. Process CPU
seconds can exceed wall time on a multithreaded run. See
[`benchmark_summary.json`](../results/benchmark_summary.json) for record-level
resource entries and evidence links.

## Open questions

- Can the diagnostic tool find a repair when the defect location and mechanism
  vary, rather than when a single permitted bias coordinate is known?
- Does a retrieval advantage persist when the stored experience is related but
  not identical, and when the weight-only agent receives the same number of
  evaluator calls?
- Do sparse weight edits survive broad locality and regression suites?
- Does the published p50c edit reduce repetition under the authors' exact
  context, prompt, sampling, and dataset protocol on this pinned checkpoint?
- Do any Qwen-directed edits discovered by the diagnostic system generalize
  beyond the practice failure and survive restart without hidden-set tuning?

See [`KNOWLEDGE_LEDGER.md`](KNOWLEDGE_LEDGER.md) for evidence-linked
observations and the next controlled experiments.
