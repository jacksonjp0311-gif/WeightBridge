# Blind repair of an injected Qwen checkpoint fault

This experiment uses the pinned `Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17`. The evaluator injects a localized
fault into a copy of the model. The deterministic search receives that copy,
practice prompts, and a bounded patch policy. It does not receive the pristine
weights, the fault patch, or holdout answers through its input interface.
The stages run in separate processes under the same local account. This is
input separation for a trusted runner and does not provide filesystem security
for an untrusted agent.

## Margin-budget result

| Checkpoint | Correct targets | Correct unrelated prompts |
|---|---:|---:|
| Pristine | 23/24 | 26/32 |
| Damaged | 7/24 | 25/32 |
| Reloaded repair | 21/24 | 26/32 |
| Random control 853021 | 6/24 | 25/32 |
| Random control 853022 | 7/24 | 25/32 |
| Random control 853023 | 5/24 | 25/32 |

The search selected layer 23 at strength 0.03. It changed 59,826 stored
BF16 values and recovered 14 of 16 lost target answers after reload (87.5%).
The repair matched all 32 pristine unrelated predictions. Compared with the
damaged checkpoint, it corrected one unrelated prediction and introduced
zero new unrelated errors. Because the frozen gate compares every unrelated
prediction to the damaged checkpoint, its recorded disposition is FAIL.
Pristine agreement is a separate post-score analysis.

[Raw scores and receipts](../results/qwen-blind-repair-v6/)
· [Comparison chart](../results/figures/qwen_blind_repair.png)

The [rollback check](../results/qwen-blind-repair-v6/evaluation/rollback.json)
loaded the pristine checkpoint in a fresh process and reproduced all 56
original predictions and margins. The final trial used 339.20 wall seconds
and 1,361.78 process CPU seconds, including rollback. Across timed trials,
including the retired calibration and both rollback checks, the total was
1,510.79 summed stage wall seconds and 5,816.20 process CPU seconds. Peak
process RSS was 2,541,518,848 bytes. The first calibration logged forwards
and RSS without CPU or wall time. These are local CPU measurements and the
forced-choice protocol generated zero model tokens.

## Intervention and gate

The search reads the final-position inputs to the MLP down projections of
layers 19, 21, and 23. The first trials select 16 features by the absolute
target-versus-sentinel mean contrast divided by feature RMS. On the resulting
RMS-normalized matrix, split into target rows `T` and sentinel rows `S`, it
fits `(TᵀT + λ SᵀS + I)⁻¹Tᵀ1`. The initial trials use `λ = 1`. The
sentinel-constrained trial uses 64 columns and `λ = 64`, with 40 practice
sentinels. It converts the coefficients `v` back to the raw feature scale
and tries `ΔW = s u vᵀ`. Here `u` is the normalized difference
of the answer-token output embedding vectors and `s` is one of eight frozen
strengths. Selection maximizes target gain among candidates with zero practice
sentinel flips. It breaks ties by smaller realized BF16 norm, then larger
target margin gain. The low-drift trial uses `λ = 256` during search,
keeps candidates with zero practice sentinel flips and at least half the
largest eligible target gain, then minimizes the sum of absolute practice
sentinel margin changes. It breaks ties using the initial ranking.
The margin-budget trial instead maximizes practice target gain under a
total drift budget of `0.25` across 40 practice sentinels and zero practice
sentinel flips. It breaks ties by lower drift, smaller realized norm, then
larger target margin gain.

The evaluator independently loads the saved repair and verifies its practice
scores. Held-out recovery is `(repair correct − damaged correct) / (pristine
correct − damaged correct)` when the denominator is positive. PASS requires at
least 50% recovery, strictly better target accuracy than all three seeded
random controls, and zero unrelated prediction changes relative to the damaged
checkpoint. Controls share the repaired tensor, column support, and requested
rank and Frobenius norm. The realized BF16 norms must match within 5%.

The target is a forced choice between two next-token logits. The initial
protocol fixes the correct target label to A to induce and test one directional
failure. Sentinels balance A and B. This measures recovery on a narrow task and
does not establish balanced-choice or generation behavior. The injection and
the search use related intervention families. A discovered repair is a new
practice-fitted patch and is not evidence of identifying the exact original
weights. The three controls compare individual interventions on the same
holdout. They do not give random search the same 24 practice trials or support
a population-level estimate from a single damaged checkpoint.

## Prior attempts

- [Calibration HOLD](../results/qwen-blind-repair-v1-calibration/experiment_record.json).
  At the smallest tested strength, seven of eight target predictions and
  three of eight sentinel predictions changed. No checkpoint was saved.
- [Paraphrase trial INCONCLUSIVE](../results/qwen-blind-repair-v2/evaluation/summary.json).
  Pristine, damaged and repaired checkpoints each scored 24/24. Damage
  lowered the mean target margin from 0.7604 to 0.3490 and the repair raised
  it to 0.8698. Four of 32 sentinel predictions changed relative to damaged.
  This outcome led to a fresh same-format trial with separate three-digit
  number ranges for construction, practice and holdout.
- [Same-format trial FAIL](../results/qwen-blind-repair-v3/evaluation/summary.json).
  Pristine, damaged and reloaded repair scored 24/24, 22/24 and 24/24.
  Random controls scored 22/24, 23/24 and 22/24. The repair recovered both
  errors and changed three of 32 sentinel predictions. Two became correct
  and one became incorrect. The strict locality gate failed.
- [Sentinel protocol HOLD](../results/qwen-blind-repair-v4-retired/experiment_record.json).
  A pre-score overlap audit found six sentinel prompts reused from the first
  same-format holdout. The source-hash gate stopped the retired trial before
  search loaded a model. The replacement holdout has zero exact overlaps
  with both earlier scored holdouts.
- [Sentinel-constrained trial FAIL](../results/qwen-blind-repair-v4/evaluation/summary.json).
  The saved repair recovered 13/13 lost target answers and scored 24/24.
  Random controls scored 10/24, 12/24 and 9/24. One unrelated prediction
  changed from incorrect to correct, which failed exact preservation.
  The [rollback replay](../results/qwen-blind-repair-v4/evaluation/rollback.json)
  independently reloaded the pristine model and matched every recorded
  prediction and answer-logit margin.
- [Low-drift trial FAIL](../results/qwen-blind-repair-v5/evaluation/summary.json).
  Pristine scored 23/24. Damaged and all three random controls scored 0/24.
  The saved repair recovered 8/23 losses and changed zero of 32 unrelated
  predictions. The recovery threshold remained at least 50%.

## Run or replay

Use Python 3.11+ with NumPy, safetensors, PyTorch and a Transformers version
that supports Qwen3.5. The saved runs record their exact package versions.
Obtain the pinned local model directory, including its tokenizer and model
configuration. The runner checks the original weight shard SHA-256 before use.
Use a new scratch directory with at least 11 GB free for checkpoint copies.
Run the three stages in sequence and stop on a nonzero exit.

```powershell
$runner = 'experiments/original_weight_edits/run_blind_qwen_repair.py'
$base = 'C:/path/to/pinned/Qwen3.5-0.8B'
$run = Join-Path $env:TEMP 'weightbridge-blind-qwen-replay'
python $runner prepare --run $run --pristine $base --fault-seed 919650
python $runner search --public "$run/public" --out "$run/search"
python $runner evaluate --run $run --pristine $base
```

Use `1264cb6` for the margin-budget trial with seed `919650`, `804c8d5` for
the low-drift trial with seed `919643`, and `e49dc82` for
the sentinel-constrained trial with seed `919630`.
The first same-format trial used `f433517` and seed `919623`.
The paraphrase trial used `1f3990c` and seed `919611`. The protocol binds the
runner's exact source bytes. On Windows, use `git -c core.autocrlf=false
archive` to extract historical source with the original LF line endings.
To audit a completed export without model files, run the verifier from the
current revision.

```powershell
python experiments/original_weight_edits/verify_blind_qwen_evidence.py results/qwen-blind-repair-v2
```

Pass `--frozen-source` with the runner from the corresponding source commit
to include its byte hash in the audit. Pass `--pristine` with the local
model directory to re-evaluate the original checkpoint after using the repair.

The evaluator writes raw per-prompt predictions, true-label logit margins and
stage CPU, wall-time and peak RSS measurements. The evidence export contains
patches, protocols and JSON receipts. Checkpoint shards stay in scratch space.
To record a new completed run in your checkout, choose a new result ID and
run `python $runner export --run $run --out results/<new-result-id>`. This
also appends a ledger entry. A replay of an existing experiment should use
the scratch output for comparison and keep the original ledger entry.
