# Phase III-F model-agnostic adaptation

WeightBridge now has a shared intent, an architecture-specific translation
step, and a host-controlled execution and verification step. The experiment
uses the pinned Qwen3.5-0.8B checkpoint from Phase III-E and
SmolLM2-135M-Instruct, a Llama model. Phase III-E's measurement and edit
disposition remains part of the research record.
[Phase III-E evidence](PHASE_III_E.md) ·
[SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)

## Interface and execution

`weightbridge.adaptation.model_adapter` exposes the versioned capability
contract and the typed local adapter interface. Metadata inspection works
before loading a runtime. Direct editing requires a supported safetensors
layout and a live match to the adapter's official model implementation.
Each adapter inspects the native feed-forward block and locates a single
untied affine contraction from intermediate width to hidden width. It
checks the serialized shape and precision. Its numerical tensor name is
an output of discovery. A bounded runtime probe must observe a finite,
nonzero weight derivative before the contract advertises gradients.

The restricted contract exposes only verified operations. Unknown dense
architectures can be inspected from compatible safetensors metadata.
Recognized quantized or compressed layouts require a separate backend for
adaptation. The inference-only adapter accepts a host-owned evaluator and
has no checkpoint identity or weight access.

`weightbridge.adaptation.universal_awl` parses the experimental AWL v3
document. It names a behavioral family and metric, preservation families,
data roles, risk limits, allowed methods, budgets, and fresh verification
requirements. Parsing creates typed data. Compilation intersects the
request with the host's model hash, capability contract, tensor and method
allowlists, and authorized datasets. The resulting plan records the native
affine region, source checkpoint, method, and numerical bounds. The
meta-controller accepts architecture and checkpoint-bound evidence and
rejects learned predictions bound to another model.

`weightbridge.adaptation.universal_update` fits one bounded rank-one
update on selected columns of a discovered affine output. The direct
engineering probe applies a fixed analytic direction. The constrained
solver selects columns from the target gradient of
`mean(softplus(-signed_answer_margin))`. It projects away an adverse
component along the mean preservation gradient and takes the leading
rank-one direction. That projection is a local approximation. The
measured practice guard requires a target correction with no previously
correct target or preservation decision flip. The conventional
comparison uses projected rank-one Adam on the same column support with
a parent-distribution penalty. The run records realized forward and
backward calls for each method. The measured guard remains in place after
rounding to checkpoint precision.

The backend uses the existing patch engine to write validated column
slices in a new candidate. The intended pre-rounding product has rank one.
Stored rounding can change the realized numerical rank. Candidate receipts
bind the base and candidate hashes. The independent evaluation reloads
from disk. An engineering rollback selects and reloads the source
checkpoint without replacing any deployed model. Scientific admission
requires evidence beyond these engineering checks.

## Study

The shared objective measures parity and preserves integer-sign
classification. Exact greedy answer checking is the behavioral endpoint.
Teacher-forced likelihood on the two semantic answers is an auxiliary
measurement. Programmatic labels, two prompt templates, disjoint practice
and evaluation populations, and a frozen admission gate separate
format compliance, classification, and repeatable failures. The host
scores candidate behavior only after candidate construction closes.
[Frozen protocol](../experiments/phase_iiif/matrix_protocol.json)

The two pinned source checkpoints are Qwen/Qwen3.5-0.8B at revision
`2fc06364715b967f1860aea9cf38778875588b17`, SHA-256
`7d4532858cf839d05ec34da088114df47238322f86f5e6e5c6693594a21a03b8`,
and HuggingFaceTB/SmolLM2-135M-Instruct at revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, SHA-256
`4e4984f36a1a8566aceb2d0baf01e267042dc34d087fc4609f64b80de249624a`.
The study uses local CPU inference with Transformers 5.22.1 and
PyTorch 2.15.0. It requires the optional `inference` and `safetensors`
dependencies. Model files stay outside the evidence archive.

## Matrix results

The same behavioral intent and analytic probe compiled through both
runtime adapters. The verifier independently loaded the original,
candidate, and restored original in separate subprocesses.

| Model | Discovered editable affine regions | Selected region | Changed BF16 entries | Realized update norm | Original prediction replay |
| --- | ---: | --- | ---: | ---: | --- |
| Qwen3.5-0.8B | 24 | `model.language_model.layers.23.mlp.down_proj.weight` | 2 | 0.0349454 | Exact |
| SmolLM2-135M-Instruct | 30 | `model.layers.29.mlp.down_proj.weight` | 2 | 0.0349386 | Exact |

Both models passed the engineering check. Each yielded a finite, nonzero
gradient in a one-forward, one-backward smoke probe. The engineering
patches produced zero exact-answer or ranker item gain on the independent
parity set and zero measured preservation changes. The physical weight
modifications therefore verify the interface, serialization, and restore
path. The isolated engineering candidates have no promotion receipt.

Both parity adaptation gates recorded **HOLD**. Qwen parsed and agreed
with its ranker on 24/24 practice presentations, with three solved sign
controls. Its parity ranker was stable across wordings on 0/8 practice
items. SmolLM2's ranker was stable on 8/8 but only 1/24 generations passed
strict parsing and none of its sign controls qualified. Neither model
had a persistent error confirmed by both measurements on both wordings.
The controller requested more measurement for each model. The optimizer
and its conventional comparison were held on the pretrained checkpoints,
and sequential learning remains unqualified.

The concurrent matrix runs took 113.89 wall seconds for Qwen and 55.36
for SmolLM2. Summed host and verifier process CPU time was 266.86 and
100.69 seconds respectively. Peak single-process RSS was 2.39 GiB and
0.68 GiB. The runtime probes and independent inference receipts record
forward work separately. Each engineering update used zero training
passes. [Matrix evidence](../results/phase-iii-f/matrix)

## Alternate behavioral screen

A separately frozen practice screen tested the last digit of
multiplication and used digit echo as a positive control. It scored all
ten digits and measured unrestricted greedy one-token completion from
the same bare prompt. Each item had two wordings. A completed one-token
answer is the declared output for this protocol. It differs from the
parity study's full response check.

Both models parsed 20/24 presentations. Qwen agreed with the conditional
ranker on 20/20 parsed presentations and had three persistent target
errors. SmolLM2 agreed on 17/20 and had five persistent errors. Neither
solved any target item or digit-echo control under both wordings and
measurements. Both emitted a whitespace token on every arrow-style echo
prompt. Qwen ranked all eight echo prompts correctly and generated the
right digit on all four prompts in the other echo template. This isolates
a format effect on the positive control. The screen held both models before any edit or independent
evaluation. Qwen used 16.69 wall seconds and 33.44 CPU seconds. SmolLM2
used 5.93 wall seconds and 9.36 CPU seconds. Each used three ranking
batches and three generation batches.
[Digit evidence](../results/phase-iii-f/digit-screen)

The next model trial should first establish a positive control for the
chosen output interface on both model families. A fresh practice roster
should show stable correct answers and confirmed errors under the same
endpoint before the host authorizes any optimizer comparison. The
architecture-specific derivative probe establishes available gradients.
Behavioral reachability and cross-family learned method selection still
need independent study.

## Reproduce

Use the pinned local checkpoints. The published material directory
contains the disclosed rosters after the study. Generate new rosters
with `matrix_data.py --out` for a new experiment.

```powershell
python experiments/phase_iiif/run_matrix.py run --base $Qwen --checkpoint-sha256 7d4532858cf839d05ec34da088114df47238322f86f5e6e5c6693594a21a03b8 --material results/phase-iii-f/matrix/material --out $QwenRun
python experiments/phase_iiif/run_matrix.py run --base $Smol --checkpoint-sha256 4e4984f36a1a8566aceb2d0baf01e267042dc34d087fc4609f64b80de249624a --material results/phase-iii-f/matrix/material --out $SmolRun
python experiments/phase_iiif/archive_matrix.py --material results/phase-iii-f/matrix/material --first-run $QwenRun --second-run $SmolRun --out $Evidence
python experiments/phase_iiif/digit_screen.py --model qwen --checkpoint $Qwen --out $QwenDigitScreen
python experiments/phase_iiif/digit_screen.py --model smollm2 --checkpoint $Smol --out $SmolDigitScreen
```
