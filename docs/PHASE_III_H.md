# Phase III-H endpoint-aligned adaptation

The Phase III-G negative-decimal study exposed a split between fitting and use: both rank-one weight updates improved eight of 12 teacher-forced contrasts, yet both failed greedy decoding on a fresh selection population. This phase tests whether valid comparison rivals, much smaller rounded changes and actual generation can find or safely reject an update. It keeps AWL v3, the native model adapters and the independent verifier boundary.

## What failed in v6

The [read-only audit](../results/phase-iii-h/v6-audit.json) checks 20 archived artifact hashes and 22 source hashes. All 12 target contrasts used the real displayed rival. All 40 comparison preservation contrasts used `correct + 1`, which was outside the displayed pair in every case. The remaining eight rows were copy tasks with auxiliary `+1` decoys. In the original code, `_fit_rows` treated the comparison and copy preservation cases alike. The solver preserved the sign of those margins; it did not measure probability relative to the actual comparison rival or exercise greedy completion before writing a candidate.

On the original independent selection, the parent solved 18/24 target items across both wordings. The constrained edit solved 3/24 and regressed 43 previously correct presentations. It produced 36 completed, grammatically valid decimals outside the options. The Adam edit solved 0/24, regressed 53 and completed no first lines. Its output first diverged in the first or second generated token on all 64 presentations. Constrained output first diverged in the first or second token on 37/64. These are diagnostic counts over the original traces, not a new trial or a change to the historical gate. Candidate norms were 0.712507 and 1.424994; both candidates failed and the hidden roster stayed closed.

## Endpoint and numerical method

The shared endpoint records completion, grammar validity, option membership and semantic correctness independently. A newline or EOS must finish the first answer line. An exact decimal outside the displayed options is neither a valid task answer nor a genuine reasoning error. The two-option contrast validator checks the final displayed operands against the roster and exact decimal semantics. Comparison preservation uses the actual competing operand. Copy `+1` rows remain explicitly labeled auxiliary; a separate legacy mode retains comparison decoys only for the diagnostic ablation.

For teacher-forced answer scores `s`, the fitting margin is `m = s(correct) − s(rival)`; the surrogate loss is the mean of `softplus(−m)` on authorized error presentations. This is a two-answer preference. Greedy decoding instead selects from the full vocabulary at every step and can depart from both answers before completion. The practice guard requires positive gain across both frozen wordings, correction of a previously repeatable wrong item in this study, and zero lost validity or previously correct presentations. A changed score alone cannot authorize materialization. The same semantics apply in independent workers.

The constrained direction is constructed in the adapter's native affine module. Only selected columns receive a rank-one update. Each requested radius is converted to stored precision, the live tensor must match those bytes, and the entire parent tensor is restored and verified even if measurement raises an error. The runner records any rounded no-op, actual norm, affected entries, teacher-forced scores, generation traces and first differing tokens. Live probes are isolated research operations; they are not accepted checkpoints.

On a frozen set of completed parent response prefixes, the full-vocabulary diagnostic computes `KL(P || Q) = Σ P(t) [log P(t) − log Q(t)]` over the next token. The selection regularizer adds `0.2 × mean KL + 0.1 × realized_norm²` to the measured target loss. It ranks already endpoint-qualified measured points. This pilot does not backpropagate through KL or change the update directions with it. A small average KL over these prefixes does not bound greedy changes elsewhere; near a token tie even a small change can change the next context.

## Frozen Qwen pilot

The protocol and public populations were committed before scoring in `f75e699`.
The [roster manifest](../results/phase-iii-h/decimal-endpoint-v1/material/manifest.json)
binds fresh practice, selection and host-held hidden populations. Each has
16 target, four positive and four copy items with two wordings per item.
The question generator excludes disclosed v5 and v6 identities. Qwen's
unchanged checkpoint passed both scored baseline populations: all 48
responses were valid on each, all positive and copy items passed, and
11/16 practice targets and 8/16 selection targets were solved in both
wordings. Practice found five repeatable negative reversals; the separate
selection parent measurement found six. Only practice examples reached
the fitter. A target item and each of its two presentations always stay
together when counting item gains.

Arms A and B fit the same practice rows with legacy or actual preservation
rivals respectively. Both use the same first-order constrained solver,
native layer, rank, three doses, and endpoint budget. The old margin
gate is computed for each arm as a diagnostic control. Arms C and D reuse
B's exact probe traces: C selects by measured endpoint gain and then norm;
D ranks eligible points by target loss, prefix KL and realized norm. Thus
C versus B tests a selection gate without spending additional forwards;
D versus C tests only a different ordering of eligible measured points.
The unchanged model is the primary behavioral baseline. The frozen pilot
does not train a new conventional baseline.

The two fits each used 40 presentations and the same 16 columns of Qwen's
native layer 23 feed-forward output. Four of the eight target fitting
presentations already favored the correct answer under conditional scoring
even though the parent generated the wrong displayed option. This mismatch
was present before any edit.

| Fitting rivals | Realized norm | BF16 entries changed | Conditional loss reduction | Target item gain | Regressed responses | Completed / valid (of 48) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy | 0.02503 | 15,271 | 0.326 | −1 | 3 | 48 / 48 |
| Legacy | 0.09999 | 16,083 | 0.617 | −1 | 8 | 48 / 46 |
| Legacy | 0.24999 | 16,258 | 0.714 | −8 | 25 | 32 / 21 |
| Displayed | 0.02502 | 15,434 | 0.308 | −1 | 3 | 48 / 48 |
| Displayed | 0.10002 | 16,142 | 0.622 | −2 | 9 | 48 / 46 |
| Displayed | 0.25000 | 16,292 | 0.734 | −9 | 28 | 32 / 16 |

Every dose changed stored BF16 values. The smallest tested radius already
changed the conditional objective and generated responses. Some probes
corrected a repeatable wrong item in both wordings, but every probe lost
more target items than it gained and regressed previously correct responses.
The old A rule would have selected its highest dose; B rejected all three
corrected points on its conditional preservation check. Correcting the
rivals made the conditional veto more selective in this run, while the
resulting weight direction did not improve endpoint preservation or target
gain. C found no generation-qualified probe. D had no eligible point to
rank, so its incremental benefit is unmeasured. Without the hard guard,
the combined score for the three corrected doses still preferred the most
harmful largest dose (0.575, 0.264, 0.170; lower is better).

Define the observed fidelity as `F = count(conditional loss reduction > 0
and completed target item gain > 0) / count(conditional loss reduction > 0)`.
It was 0/6. A descriptive Wilson interval is about [0, 0.39]; the shared
practice population and two directions make those six probes dependent.
This is evidence against using the surrogate alone on this tested scope,
not a population estimate or proof that this layer is uneditable.

The distribution traces make one failure concrete. At the largest legacy
dose the first changed token on the second wording of the first positive
control was inside the answer. At the shared prefix the top digit changed
from `5` to `2`. KL at that token was 0.762, versus a mean of 0.061 over
34 diagnostic positions. The smallest legacy dose regressed three
previously correct responses while its sampled mean KL was only 0.000354;
none of those three responses was among the four KL control prompts.
This shows why a small KL on sampled prefixes cannot replace the guard.

**Disposition: HOLD before candidate selection.** The host restored the
live parent tensor exactly after all six probes and rechecked the pinned
source checkpoint hash. No Qwen candidate was materialized or reloaded,
no candidate entered independent selection, and the hidden roster and
lineage stayed closed. The independently scored selection receipt is for
the unchanged model only. The meta-controller recommended HOLD for a
surrogate-to-endpoint mismatch.

The Qwen fit and parent practice run took 435.99 s wall and 2,429.30 s CPU.
The separate selection parent worker added 38.09 s wall and 147.84 s CPU.
Peak process RSS was 6.73 GiB. Each fitted direction used 40 contrast
forwards, four distribution forwards, two backward calls and 216 batched
decode steps for its three probes. Practice and selection parent decoding
used 72 steps each, separate from the candidate counts. The native gradient
capability preflight added one forward and backward. The SmolLM2 replay
took 50.21 s wall and 120.55 s worker CPU. No GPU or energy cost was measured.

[Trial report and raw evidence](../results/phase-iii-h/decimal-endpoint-v1/trial)
· [Measured dose-response figure](../results/phase-iii-h/decimal-endpoint-v1/dose-response.png)

## Next test

Use the first wrong token on the actual greedy path to construct a
prefix-level target, including the competing continuation the model
actually chose. Compare that objective against answer-pair loss at the same
rank, radii and decoding budget. The falsifiable prediction is that at
least one point fixes a repeatable target across both wordings with zero
practice regressions and retains its gain after independent reload on a
fresh roster. Keep the endpoint gate fixed. To isolate the effect of KL,
test it inside synthesis in a separate matched arm; this phase only tested
its use for ranking. Do not move to sequential learning before a fresh
single-cycle pass. Before another hidden-enabled run, bind the lineage
modules into the source manifest and recompute the meta recommendation
after independent verification, including a failed replay.

## Cross-model integration

The archived Phase III-F portability report revalidated Qwen3.5 and Llama engineering PASS. A new [SmolLM2 engineering replay](../results/phase-iii-h/cross-model-smollm2/report.json) also reapplied the exact historical two-entry BF16 patch, verified its checkpoint and tensor hashes, and ran the same Phase III-H endpoint in fresh processes on the parent, candidate, and restored parent. Rollback reproduced all 48 raw rows exactly. Both the base and patch solved 3/16 targets across both wordings. Both completed all 48 first lines, but eight target generations were outside the options. They solved only two of four positive items. This verifies shared endpoint execution and restoration; the historical patch was not fitted to this task and is not an adaptation or qualification claim.

## Reproduction

Use Python 3.12, CPU Torch 2.15.0, Transformers 5.22.1, and the existing
[pinned model directories](PHASE_III_F.md). The local run used six CPU
threads with native BF16 storage. The full test suite does not require
downloading model checkpoints. On Windows, use a short checkout path or
`git -c core.longpaths=true clone ...` if Git cannot unpack older evidence
paths.

```bash
python -m pytest -q
python -m experiments.phase_iiih.audit_v6 --out ./v6-audit-replay.json
python -m experiments.phase_iiih.cross_model run --base ./smollm2-135m-instruct-cpu --out ./smollm2-endpoint-replay
python -m experiments.phase_iiih.run_decimal_endpoint run --base ./qwen35-base-cpu --material results/phase-iii-h/decimal-endpoint-v1/material --out ../qwen-endpoint-replay
```

The Qwen command does not receive a hidden roster path. A separate host
`finish` command is eligible only after a selection winner exists, and
requires the exact previously committed private hidden roster and a workspace
outside the checkout. Passing selection is an intermediate state; only an
independent hidden pass, exact candidate replay and parent rollback can
advance the research lineage. Never publish the private worker artifacts.

To reproduce the separately measured selection parent even when fitting
produces no candidate, run the same module's `worker` command first with
`--split selection`, the same `--base` and `--material`, a new `--out`
JSON path, `--sha256` from the protocol, and `--manifest-sha256` equal to
the SHA-256 of `material/manifest.json`. Pass the resulting JSON to `run`
as `--selection-base`. The fitter reads that receipt only after practice
probes complete. Any report referencing private verifier artifacts requires
those artifacts for a complete host-side evidence replay.

To replay the archived practice decisions without model weights, use:

```bash
python -m experiments.phase_iiih.analyze results/phase-iii-h/decimal-endpoint-v1/trial --plot ./phase-iii-h-doses.png
```

The analyzer checks the bound sources and artifacts, reparses every
candidate completion, and recomputes the selection rules and the regularized
objective. Plotting uses the optional `figures` dependencies.
