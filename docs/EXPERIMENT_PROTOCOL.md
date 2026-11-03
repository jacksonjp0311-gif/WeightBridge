# Experiment protocol

## Required experiment record

Each `weightbridge.experiment.v1` record identifies the run, source model and
hashes, hypothesis, patch, changed parameter count/fraction, practice/holdout
separation, evaluation suite, sampling settings, baseline/candidate outcomes,
uncertainty, resources, failure/termination rates, artifact paths and final
disposition. A correction is a new ledger record with `supersedes`; old lines
are never edited. One experiment ID refers to one run. Replays on the same
examples are repeatability checks, not additional independent samples.

The ledger is append-only, not a database with multi-writer locking. Current
runners are single-writer command-line tools. Duplicate experiment IDs and
duplicate non-empty measurement keys are rejected. Corrections and identity
metadata are new ledger events. Explicit duplicate-replay corrections are
excluded from figures; one replay cannot count as a new independent run.
New `weightbridge.experiment.v1` records require a deterministic measurement
key, source model, proposed modification, affected-parameter record, resource
record, artifact references, and an evaluation block that names its dataset or
measurement type and explicitly records split/separation and seeds/sampling
or marks those items not applicable. Legacy records remain preserved; missing
fields are handled by versioned addenda rather than silent edits.

## Evaluation sequence

1. Pin and hash the base model, config, tokenizer and relevant assets.
2. Freeze the practice suite and evaluator; record its version and seed.
3. Give the diagnostic worker only the practice data and an authorized scoring
   tool. It receives neither the holdout file nor its answers.
4. Validate every proposal against a host-owned policy and write a new
   candidate. Rejected proposals are evidence and must be retained.
5. Select/freeze the candidate on practice data.
6. Evaluate the frozen baseline, controls and candidate once on paired holdout
   examples. Do not use those results to revise a patch.
7. Reload the candidate from disk, verify hashes, score again if restart
   persistence is being tested, and test rollback by reloading the original.
8. Append the record and raw evidence paths. Generate visualizations from the
   ledger and raw measurements only.

## Metric definitions

For (N) labeled examples with correctness indicator (c_i\in\{0,1\}):

- Accuracy: \(A=N^{-1}\sum_i c_i\).
- Paired accuracy difference: \(\bar d=N^{-1}\sum_i(c_i^{candidate}-c_i^{base})\),
  evaluated on the same examples.
- The reported paired interval is a percentile bootstrap over paired examples
  with replacement. The experiment record fixes the seed, confidence level and
  number of resamples. It describes sampling uncertainty for this suite; it
  does not account for dataset shift or repeated adaptive use of holdout data.
- Changed parameter fraction: number of stored values whose represented bits
  differ after target-dtype conversion divided by total model parameters.
  Touched coordinates are reported separately because a requested edit may
  round back to the original value.
- Completion rate: fraction of examples for which the task-specific parser
  extracts a final answer before the limit. Classification toy tasks terminate
  by construction; this is not a language-model completion measurement.
- Token-limit rate: fraction of generations ending at the configured generation
  cap rather than an EOS/stop condition.
- Repetition rate in the Qwen smoke is an 8-token n-gram overlap fraction
  between a generated tail window and the immediately preceding window. It is
  a diagnostic proxy only, not a validated reasoning-loop detector.
- Inference cost includes recorded load time, generation time, tokens/s and
  process peak RSS/VRAM when available. The interface defaults to an 8 GiB RSS
  limit, preflights local checkpoint bytes, checks RSS after load and stops
  generation if current RSS crosses the limit. The matrix runner enforces a
  subprocess wall-time timeout. Transient allocations during model load still
  need an OS/container memory limit for hard containment. The benchmark
  reconstruction currently lacks CPU-energy or dollar-cost metering.

## Published Experiment 8 protocol reference

The authors report Qwen3.5-0.8B with thinking enabled; temperature 0.6, top-p
0.95, top-k 20, presence penalty 1.5 and up to 32,768 generated tokens. They
used practice data from GSM8K/MATH training splits, 10/25/50 forward-pass
budgets, and multiple benchmark samples (MATH-500 4, GSM8K 2, IFEval 2 among
others). The 50-pass edit targets 64 MLP output columns at layers 15 and 19.
See `benchmarks/configs/published_exp8_reference.json` and
`experiments/published_reproduction/REFERENCE_PROVENANCE.md`.

The current paired CPU smoke uses two hand-authored prompts, one sample each,
and a 64-token cap. A prior exploratory run used a 256-token cap. Neither
completed an answer, so both are inconclusive. The full frozen datasets and
paper runtime settings are required before any benchmark-replication claim.

## Published Experiment 8 protocol captured

The local reference configuration and provenance live in
`benchmarks/configs/published_exp8_reference.json` and
`experiments/published_reproduction/REFERENCE_PROVENANCE.md`. The authors use
Qwen3.5-0.8B, thinking, temperature 0.6, top-p 0.95, top-k 20, presence
penalty 1.5 and up to 32,768 generated tokens. The cited benchmark settings
use four samples per MATH-500 item, two per GSM8K test item and two per IFEval
instruction. Math answers use the last boxed answer and `math-verify==0.9.0`;
IFEval scoring uses the original Google instruction registry after the
reasoning section. These settings are represented in the full-matrix runner.
The paper's 10/25/50 editing passes and four-run intervention history are not
recreated by a tensor rebuild; only the public p50c result is reconstructed.

Dataset revisions must be frozen before a full run. The preparation script
records upstream IDs, source split, dataset fingerprint and output hashes. If
an upstream `main` revision is used, that run is identifiable by the frozen
output hash but cannot be described as a fully revision-pinned reproduction.
