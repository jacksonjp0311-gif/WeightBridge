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
3. Give the diagnostic worker or MCP caller only practice data and an
   authorized scoring tool. The hidden holdout remains in the host process and
   its path is absent from the agent session config.
4. Validate every proposal against a host-owned policy and write a new
   candidate. Retain every tool request and response, including rejected
   proposals, budget failures and invalid hashes.
5. Select/freeze the candidate on practice data. The agent's finalize call
   records a hash and does not run hidden scoring.
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
- LLM token/tool costs are recorded when the host exposes them. If the model
  host does not provide usage totals or a pinned model ID, store null and state
  the limitation; do not estimate or backfill invented costs.

## MCP LLM transfer protocol v2

The checked-in host-only template declares 20 tasks: four single-weight defect
mechanisms, five unique seeds each, randomized execution order, and a fresh
agent context per task. The pinned prompt, task generator, MCP server, cohort
evaluator, policy, candidate budget and decision gate are hash-bound. The
caller must have MCP tools only, without filesystem, repository or network
tools. The host keeps task assignments, practice/holdout data and the oracle
out of the caller's context.

A complete repair means 100% holdout accuracy when the baseline is below 100%.
The gate requires repairs on at least 70% of tasks, no more than 5% regressions
among baseline-correct items, a mean candidate advantage of at least 20
percentage points over a one-entry random control matched to the same tensor
and absolute magnitude, and a 95% stratified paired-bootstrap lower bound
above zero. The bootstrap resamples task seeds within each mechanism.

The current file is `TEMPLATE_HOLD`, because this environment cannot verify an
LLM provider model/version or sampling receipt and the current workspace is
visible to the model. A runnable protocol must point to a hash-checked host
identity receipt containing provider, model ID, version, sampling settings and
host call ID; each run copies the receipt into its raw evidence. Hashes detect
content changes but do not prove provider identity cryptographically; the
receipt remains host-reported evidence. The runner
rejects the current template before task creation. Bind a separate isolated
host, create and hash a new frozen protocol version, and retain each task's
full request/response transcript before starting the study. The current
one-record preflight is INCONCLUSIVE and is not a behavioral result.

Freeze a host receipt into a new protocol rather than editing the template:

```bash
python -m weightbridge.evaluation.freeze_cohort \
  --template experiments/original_weight_edits/mcp_llm_transfer_v2.template.json \
  --identity-receipt path/to/host_identity_receipt.json \
  --out experiments/original_weight_edits/mcp_llm_transfer_v2.cohort-a.json \
  --cohort-id mcp-llm-transfer-v2-cohort-a \
  --identity-provenance "host identity checked from the recorded inference session"
```

The command copies and hashes the receipt beside the new protocol, verifies
all bound sources and frozen settings, and refuses existing output paths. The
operator-provided provenance remains an attestation, not cryptographic proof.

## Generative endpoint adaptation

Phase III-H separates conditional fitting scores from completed greedy
behavior. Score and count correctness only after parsing the frozen first
line. A comparison result must belong to the actual displayed options.
Count target gain over underlying items, requiring every specified wording
to pass. Count preservation losses over individual presentations so that
losing one correct wording cannot be hidden by an item-level aggregate.

Each effective dose first runs as a stored-precision live probe and must
restore the parent tensor exactly. A no-op after rounding is not evidence
of a failed behavioral intervention. The host screens practice generation
before any materialization. Separate checkpoint workers own selection and
hidden scoring. A hidden pass requires a previously locked finalist,
hash-bound private material, exact reload and exact parent replay. See the
[Phase III-H design](PHASE_III_H.md) and [frozen ablation](../experiments/phase_iiih/decimal_protocol.json).

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
