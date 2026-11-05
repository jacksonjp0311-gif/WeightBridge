# Phase III-S: grounded tool transfer after a verified decimal improvement

**Preflight snapshot: calibration and numerical timing completed; main training
has not run. No Phase S adapter, merged model or independently qualified tool
improvement is reported.** The planned sequential and original-parent arms use
the same new curriculum and fixed rank-eight recipe. Formal gates remain strict.

This work starts from `baa336e7c517139876a1a5d6309af7199ea1e661` on an isolated
`dev/phase-iii-s-tool-transfer` branch. Main, original and R checkpoint bytes,
historical evidence, deployed models and accepted lineage remain unchanged.
It builds on the [verified Phase III-R replication](PHASE_III_R_REPLICATION.md),
not its rejected rank-four reduction.

## Question and comparison

Can a fresh adapter add grounded, single-action tool behavior to the verified
decimal model while retaining its measured decimal capability? The sequential
arm starts from the locked R8 merged checkpoint. The control starts from the
unchanged original model and receives the same tool examples and decimal replay;
it is **original-parent joint training**, not a tool-only control. The inherited
R training cost belongs to the sequential model's history and is separate from
the additional matched S training budget.

Both starting identities were recomputed through the existing checkpoint path:

| Starting representation | SHA-256 | Identity check |
|---|---|---|
| Original Qwen3.5-0.8B | `7d4532858cf839d05ec34da088114df47238322f86f5e6e5c6693594a21a03b8` | PASS |
| Verified R8 merged model | `2d9102982dc0f0fdf3397308cd397bc2d5239eeb99ef6297e5f09f6f6deba158` | PASS |

Both contain 488 checkpoint tensors. Their tokenizer, configuration and native
chat-template files match. The R acceptance receipt was validated against its
pinned historical checkout. Separately, public archive checking verified 422
published R artifacts and 141 frozen source files; that archive check did not
execute a model or replay private confirmation responses.

## Grounded single-action task

The versioned registry permits exactly five actions:

| Tool | Exact arguments | In-memory effect |
|---|---|---|
| `search_files` | `root`, `query` | Search fixture paths and content for the exact literal query |
| `read_file` | `path` | Return the selected readable fixture's content |
| `run_tests` | `target` | Return the selected test target's recorded result |
| `clarify` | `field` | Request the missing or ambiguous `root`, `path` or `target` |
| `no_action` | none | Decline an explicitly unsupported request, scope or file type |

No generated action executes a shell command or accesses a filesystem, network
or real repository. Each prompt supplies the registry, inventory, target mapping
and natural request. Close paths and targets are distractors. Wrong queries,
paths or targets are executed against their own fixture entries and do not
silently return the intended result.

The independent label checker uses visible request semantics and inventory
facts. Family IDs and scenario labels do not determine its answer. The separate
simulator does not compare an action to a stored gold label. Counterfactuals
change availability, required fields, support, or the requested operation; both
outcomes appear under two wordings and two inventory/registry orders. One family
therefore has eight presentations. This tests a narrow single-action decision,
not multistep planning or general software engineering.

The complete response must be one JSON object with exactly `tool` and
`arguments`. The parser rejects unknown fields or tools, duplicate keys,
wrong argument types, extra objects, prose and unexpected special tokens.
Ordinary surrounding whitespace is allowed. Full JSON syntax, registered-schema
validity, semantic correctness and natural completion are separate measurements.
Strict success requires all of them, including the correct simulated outcome.
There is no extraction, repair, reranking, forced brace, logit bias or oracle
routing during inference.

Tools use the pinned native non-thinking chat template and greedy decoding,
six CPU threads, batch eight, a 512-token input ceiling and 96-token output
budget. Only the declared native terminator, `<|im_end|>` with ID 248046, is
removed as transport. Its occurrence is required for strict success. Legacy
decimal and general controls retain the existing bare-prompt, sixteen-token,
completed-first-line evaluator. A valid decimal first line is not an EOS stop.

## Measured disposable calibration

Calibration was committed before its first model output: ten independent tool
families, 80 presentations and exactly sixteen examples of each action. Its
family identities and wording templates are excluded from formal populations.
A single fixed JSON format example comes from this disposable population and
is identical in both models' prompts. Native tool prompts ranged from 257 to
402 tokens; the longest supplied answer including terminator used 27 tokens.
The separate legacy screen contains 32 negative-decimal target presentations
and forty positive, mixed-sign, copying, addition and successor controls.

| Disposable tool endpoint | Original parent | Verified R8 parent |
|---|---:|---:|
| Whole-response JSON syntax valid | 71/80 | 17/80 |
| Registered schema valid | 54/80 | 17/80 |
| Correct tool and exact arguments | 16/80 | 6/80 |
| Correct simulated outcome | 16/80 | 6/80 |
| Strict completed actions | 16/80 | 6/80 |
| Complete eight-presentation families | 0/10 | 0/10 |
| Natural EOS termination | 80/80 | 17/80 |
| Output budget reached | 0/80 | 63/80 |

All successful tool actions in this screen are reads. The original model has
38 schema-valid semantic errors across ten families, including erroneous
search and test decisions. The R8 parent has eleven such errors across four
families, with search as its only represented real-action error type. It also
has extensive completion failure. Thus the R8 deficiency cannot be described
solely as an already qualified semantic-choice problem.

The conservative calibration plausibility screen passes for the original and
fails for R8: R8 misses the 50% schema-validity condition and the requirement
for semantic errors in two real action types. At least one parent provides
useful headroom, so the directive's overall no-headroom stop does not apply.
This does not grant R8 a semantic pass or relax its subsequent formal gains,
action-diversity, completion or retention gates. A formal bank without sufficient
qualifying opportunities must produce an insufficiency result, not a redraw.

Compared presentation by presentation, original-to-R8 tool behavior loses ten
strict and semantic successes, gains none, loses 39 schema-valid responses and
gains two. Those were not measured by the earlier decimal qualification. This
underscores the declared scope of preservation claims rather than revising R's
historical result.

| Disposable legacy endpoint | Original parent | Verified R8 parent |
|---|---:|---:|
| Negative-decimal target responses | 4/32 | 32/32 |
| Strict comparisons | 0/8 | 8/8 |
| Complete opposing families | 0/4 | 4/4 |
| General/positive controls | 40/40 | 40/40 |
| Valid completed first lines | 72/72 | 72/72 |
| Full decode reached EOS | 0/72 | 0/72 |

These are fresh disposable measurements of unchanged parents, not a Phase S
learning result, independent confirmation or a population-level competence claim.

## Training implementation and prospective population

The implementation reuses the existing LoRA factors, parent inspection, complete
answer loss, Decimal generator, checkpoint merger, resource monitor and replay
machinery. New code consists of the grounded task contract and simulator, mixed
completion preparation, explicit-parent bindings, data builders and staged host
verification. Historical loaders, protocols and endpoints are preserved.

Both arms are planned to use 186 supported text attention/MLP matrices across
24 layers, rank eight, alpha sixteen, no dropout and 5,411,328 trainable FP32
factor entries over frozen BF16 parents. Embeddings, output head, tokenizer and
vision components remain outside the adapter. The fresh adapter is initialized
from seed 606118; matched initialization and row-order hashes must be verified.
The sequential model must not attach the original R adapter a second time.

The fixed recipe is AdamW at 3e-4, betas (0.9, 0.999), epsilon 1e-8, weight decay
0.01, clipping 1.0, microbatch four and effective batch sixteen. One epoch gives
128 steps, with seven warmup steps then the existing linear decay. Step 64 is
diagnostic only. Step 128 is the sole candidate; no alternative checkpoint may
be selected after observing outcomes. Each parent permits one disposable smoke
of at most sixteen steps and one main attempt, with full state reset afterward.

| Prospective partition | Tool presentations | Negative-decimal presentations | General replay/control presentations |
|---|---:|---:|---:|
| Training plus replay | 1,024 | 512 | 512 |
| Development | 128 | 128 | 80 |
| Independent qualification | 256 | 256 | 320 |
| Sealed confirmation | 256 | 256 | 320 |

Training uses 128 tool families and 64 new decimal families. Its general replay
contains 128 positive, 128 mixed-sign and 128 copying presentations, plus 64
addition and 64 successor presentations. Every row has unit sampling weight.
Training loss is the sum of supervised-token losses divided by supervised
tokens in the effective batch, not an equal mean of unequal microbatch means.
The 50/25/25 row mixture therefore does not imply a 50/25/25 loss mixture.

Tool supervision covers the complete canonical JSON and native terminator.
Decimal supervision remains leading space, canonical exact answer and newline.
Prompt/padding tokens are masked, and the causal one-token shift is explicit.
A real-model preflight compared full-position logits to the optimized answer-tail
loss on a mixed-length batch: both losses were 1.1987115145, absolute loss
disagreement was zero and maximum factor-gradient disagreement was zero at
the recorded tolerances. Four forward/backward timing probes performed zero
optimizer steps; adapter factors and base bytes stayed unchanged.

All transformations of one family stay together. Development uses separately
authored request wording; half of each formal tool bank uses wording groups
absent from training. Familiarity assignments rotate across repeated mechanism
blocks, rather than permanently tying a mechanism to one wording group. New
values in the same template are reported separately from unseen wording.

Freshness checking excluded 2,893 numeric task identities and 710 negative-family
anchors from the recorded available sources, including disposable calibration.
It did not open R's locked 832-case battery or unopened Bank B. No disjointness
claim is made against their inaccessible numeric identities. The old R battery
is a later locked retention test, not new fitting material or fresh transfer.

## Required admission and persistence

Each formal tool bank requires at least 90% strict completed actions, a gain
of at least ten percentage points over the arm's own parent, six new complete
families out of 32, and no lost parent-complete family. At least eight corrected
parent-schema-valid semantic errors must span four families and two real action
types: search, read or tests. Clarify/no-action gains cannot substitute for this
semantic condition. Development applies the declared rate and preservation
requirements; its smaller family count does not inherit the formal absolute
six-family and eight-error counts.

Tool preservation includes own-parent strict successes, schema-valid semantic
successes and schema validity. Sequential decimal preservation uses R8 as its
reference. Sequential general controls preserve the union of original-correct
and R8-correct responses; the original-joint control uses the original parent.
All applicable legacy validity requirements remain active.

Eligible adapters must be merged from their actual parents, pass development
again, and be reconstructed exactly before independent data opens. Both arms'
locked artifacts or terminal failures precede access to the old R battery and
formal qualification. One arm's failure cannot tune or block the other's
predeclared recipe. Qualification precedes sealed confirmation; no private
failure may supply a repair or new checkpoint choice in this trial.

Sequential reconstruction is original to rounded R8, then rounded S. Adding
deltas and rounding once is not an equivalent reconstruction. Both incremental
and original-to-final norms must be measured. Exact replay compares each stored
representation with its own prior full native outputs, followed by rollback
to its actual parent. Endpoint correctness and full-token identity are distinct.

## Measured resource preflight

The existing local runtime was reused: Python 3.12.10, torch 2.15.0,
transformers 5.22.1, safetensors 0.8.0, numpy 2.5.4 and psutil 7.2.2. No package,
model or remote compute purchase was introduced.

| Completed enclosing scope | Wall seconds | Observed CPU seconds | Conservative peak bytes |
|---|---:|---:|---:|
| Original tool calibration | 135.035 | 765.922 | 2,983,460,864 |
| R8 tool calibration | 266.370 | 1,587.156 | 2,990,567,424 |
| Original legacy calibration | 53.609 | 256.203 | 2,582,654,976 |
| R8 legacy calibration | 53.860 | 257.484 | 2,589,024,256 |
| Four forward/backward timing probes | 90.800 | 428.781 | 14,084,444,160 |
| Total enclosing observed work | 599.674 | 3,295.547 | Maximum shown above |

Nested worker resources are not added again. The peak is the conservative sum
of observed processes' lifetime working-set peaks, not simultaneous aggregate
memory. Polling cannot exclude unobserved short-lived process cost. Software
tests, reporting and unlisted startup checks are outside this table. Temporary
Windows sleep prevention was released for every recorded action.

The longest measured microbatch had shape 4 by 415 tokens. Its forward and
backward work required about nineteen wall seconds; short legacy batches were
substantially cheaper. This is a timing observation, not proof that both full
runs and every verification stage fit. The operator ceilings are four hours
per main attempt, eighteen cumulative model-execution hours and 32 GiB under
the conservative monitor. Tighter stage budgets and exposure ceilings must be
committed from these measurements before a main run begins. Exhaustion remains
a resource outcome, not evidence that the task is unlearnable.

## Tests, evidence and reproduction

The current pre-freeze runtime receipt records **1,000 passed and two skips**
(Matplotlib and MCP unavailable in that runtime). The separately configured MCP
suite records **23 passed**. Forty-two focused simulator/population tests cover
whole-response rejection, exact arguments, actual search effects, irrelevant
requests, completion separation, eight-form scoring, mixed counts and split
isolation. Software passes do not demonstrate a learned model improvement.

The first formal-freeze attempt stopped before writing populations: sorted-key
JSON serialization changed inventory-object insertion order, while the frozen
calibration renderer reproduces its original construction order. All eighty
loaded calibration prompts then failed bytewise re-render validation despite
unchanged values and matching frozen source hashes. A host-only population
normalizer now validates exact fields and restores their construction order.
It preserves every frozen prompt byte and semantic value; no calibration was
rerun or reinterpreted. Sorted-file round-trip and extra-field refusal tests
pass in the [dedicated receipt](../results/phase-iii-s-tool-transfer/tests/pre-freeze/population-serialization.xml).

Public preflight evidence contains only schemas, commitments, aggregates and
resource receipts:

- [Export manifest](../results/phase-iii-s-tool-transfer/preflight-v1/export-manifest.json)
- [Calibration measurements and headroom](../results/phase-iii-s-tool-transfer/preflight-v1/calibration-summary.json)
- [Numerical timing and gradient equivalence](../results/phase-iii-s-tool-transfer/preflight-v1/training-timing.json)
- [Compute accounting](../results/phase-iii-s-tool-transfer/preflight-v1/compute-accounting.json)
- [Historical exclusion coverage](../results/phase-iii-s-tool-transfer/preflight-v1/freshness-coverage.json)
- [Runtime tests](../results/phase-iii-s-tool-transfer/tests/pre-freeze/runtime.xml)
- [MCP tests](../results/phase-iii-s-tool-transfer/tests/pre-freeze/mcp.xml)

No raw populations, private seeds, private per-row outputs, adapters or model
weights are published. Recheck public evidence from the repository root with:

```powershell
python results/phase-iii-s-tool-transfer/export_preflight.py --self-test
python results/phase-iii-s-tool-transfer/export_preflight.py `
  --verify results/phase-iii-s-tool-transfer/preflight-v1
```

That validates archive integrity, not private behavioral replay. Reproduction
of actual inference requires the recorded local checkpoints, runtime, private
calibration commitments and host authorization. The pure population builder
does not authorize training or open protected stages. No Phase S learning,
merged-model demo, retention pass or continual-learning success is claimed at
this preflight milestone.
