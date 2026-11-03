# Phase III-I causal fork adaptation

Phase III-H found six BF16 updates that reduced answer-pair loss and worsened completed generation. This trial tests whether the loss misses the actual token decision, whether another affine region offers a more selective update, and whether rank-one geometry interferes with correct trajectories. The unchanged Qwen checkpoint qualified on both public populations. All 48 responses on each split were valid, all positive and copy items passed, and five target errors on each split repeated across both wordings. The model solved 9/16 practice targets and 10/16 selection targets across both wordings.

## What the experiment measures

The [frozen protocol](../experiments/phase_iiii/decimal_protocol.json) binds the pinned parent, three disjoint rosters, the Phase III-H endpoint, and the practice, selection, and hidden gates. The fitter receives the practice roster. An independent worker evaluates selection after a practice candidate is saved. The precomputed selection parent receipt is opened by the runner after fitting. Candidate identities and selection observations must match the lock before the host can access the private hidden roster.

A `GenerativeFork` binds the native prompt prefix, generated tokens, exact semantic answer, and a finite set of aligned accepted continuations. A valid equivalent representation is not treated as an error. For a wrong answer the latest shared prefix across the bounded accepted continuations locates the first token that leaves that set. The diagnosis validates prompt and answer boundaries, records EOS or newline completion, and checks that a replay at the recorded prefix reproduces the actual greedy token. This is a decision about the recorded continuation and representation set. It does not identify a unique internal reasoning circuit.

For logits `z` on a shared prefix and an accepted token set `A`, the fork cross-entropy is `logsumexp(z) − logsumexp(z[A])`. The margin variant uses `softplus(max(z[V\A]) − logsumexp(z[A]))`. Diagnostic margins also compare the best accepted token to the actually observed wrong token and the strongest rejected token. A correct argmax under replay is not eligible as a newly discovered wrong decision. The first fork is a local objective. A greedy path may diverge at a later token even when the measured fork flips.

On a protected parent path, the decision margin is the observed token logit minus the largest alternative logit at that prefix. Strictly positive margins along the entire path suffice for the same decisions under the same deterministic inference semantics. The projection only constrains a few sampled replay-consistent decisions. Candidate measurements also log per-token margin changes and parent-to-candidate KL along the sampled paths. The endpoint guard checks actual complete generation across the practice roster and permits a different answer surface that the frozen semantic parser accepts.

The region scan uses shared vector-Jacobian products on three authorized affine modules. It restricts the rank-one edit to selected columns and fixes the right factor before projecting the left factor away from a small preservation-gradient subspace. The local witness intersects the intended radius with sampled linear preservation inequalities along that direction and scores predicted mean target margin gain. This is not a solver for the full feasible set. Aggregate margin crossing cannot establish per-item or full-generation correction.

## Observed Qwen behavior

The parent supplied ten repeatable-error presentations. All ten were semantic forks under the bounded accepted set, and uncached replay matched all ten observed greedy tokens. The first fork fell after one generated token on two presentations, after four on four presentations, and after five on four. The protocol used the first four complete underlying error items for fitting. These eight presentations cover close negative pairs. The fifth item was a separated pair and stayed outside the fit. The sampled preservation trajectories contained 58 decisions and all had positive parent margins under replay.

| Arm | Layer | BF16 values changed | Realized norm | Fork loss reduction | Accepted fork argmaxes | Net target gain | Regressed presentations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A. Whole-answer fit | 23 | 15,879 | 0.050024 | 0.763 | 4/8 | +1 | 4 |
| B. First-fork fit | 23 | 15,918 | 0.050023 | 1.005 | 5/8 | +1 | 4 |
| C. Selected region | 23 | Same state as B | Same | Same | Same | Same | Same |
| D. Trajectory constraints | 23 | 15,761 | 0.050028 | 1.058 | 6/8 | +5 | 3 |
| E. Conventional LoRA | 23 | 3,280,684 in proposed merge | 0.386476 | — | — | — | — |

The scan predicted raw feasible mean margin gains of 0.725, 0.568, and 3.283 for layers 11, 17, and 23. Only layer 23 was predicted to cross zero from the parent mean of −1.281. After projecting out the sampled constraints, the output factor retained 0.780, 0.955, and 0.638 of its norm in those layers, respectively. The ranking selected layer 23. Its rounded C state was identical to B, so C reused B's observation. Layers 11 and 17 were measured through gradients only, leaving their real generation effects unmeasured.

A and B compare two practical pipelines. A includes the existing mean preservation contrast, B uses the raw fork gradient, and D adds a different preservation projection. The comparison holds the region, rank, dose, target items, and endpoint constant. B improved more fork decisions than A and tied A on net target gain and regressions. The frozen run fits cross-entropy in B through D and reports the strongest-rival and observed-rival variants as diagnostics.

D raised the practice count from 9/16 to 14/16. It corrected four repeatable errors, including the separated pair outside the fit. It also lost both wordings of one previously correct target and one wording of a positive control. Those three lost presentations disqualified it. Neither affected item belonged to the sampled trajectories. D kept the four sampled anchor margins positive and reduced mean sampled KL from B's 0.0954 to 0.0243. The corresponding mean fork margin was lower for D (0.859) than B (1.672), yet D fixed more individual decisions and complete answers. The mean margin alone missed that difference. A moved two of the 58 sampled path margins from positive to nonpositive. B and D moved none across zero.

Actual candidate decoding reached the original shared prefix on all eight fitted presentations in A, B, and D. Each accepted fork token in those runs led to a correct completed answer on the fitted presentation. Among the three unique decoded updates, all three reduced fork loss and gained complete target items, and zero passed the hard endpoint gate. The three probes share a parent and a practice cohort. These counts describe the observed relationship and do not estimate generalization.

E trained 18,432 parameters over the full layer-23 affine matrix using four Adam steps on 32 paths and 262 tokens per step. Training used 16 forwards and 16 backwards. The natural merge's rounded norm of 0.386476 exceeded the preregistered 0.35 limit. The runner stopped before applying or decoding it. E establishes the training cost and budget failure. Its generation performance and any benefit of broader update geometry remain unmeasured.

**Disposition is HOLD.** All three unique applied Qwen probes restored the exact parent affine bytes. The runner rechecked the original checkpoint hash. There was no saved Qwen candidate and no candidate selection, hidden, or checkpoint reload outcome. The selection receipt measures the original checkpoint only. The meta-controller returned `hold` for `preservation_conflict`.

The trial process used 305.00 wall seconds and 1,597.67 CPU seconds. The two parent workers added 312.08 CPU seconds. Peak process RSS was 6.68 GiB. The joint three-layer scan used three forwards, six backwards, and 70.05 CPU seconds. A used 16 scoring forwards, two extra fork and path forwards, two backwards, and 72 batched decode steps. B and D each added two diagnostic forwards and 72 decode steps after the shared scan. C reused B's rounded state. E used 584.14 CPU seconds for training and no decode. The engineering run on SmolLM2 used 39.22 CPU seconds. [Raw trial report](../results/phase-iii-i/causal-fork-v1/trial/report.json) · [Replayed analysis](../results/phase-iii-i/causal-fork-v1/analysis.json)

## Next experiment

Freeze a fresh population and broaden the preservation sample before fitting. Select low-margin positions from previously correct target, positive, mixed-sign, and copy responses, with explicit representation from each group. Compare that projection with the existing four-anchor method at predeclared radii. Separately predeclare a norm-constrained conventional fit whose optimizer can stay inside its merge envelope. The test is whether an edit retains a repeatable correction across both wordings with zero practice regressions and then survives independent selection, hidden evaluation, and exact reload. An actual alternate-layer probe is needed to isolate parameter location.

## Prior research and scope

DeepSeek-R1 uses verifiable outcome feedback in reinforcement learning. Here an exact deterministic evaluator checks the completed decimal answer after each numerical probe. The scope is a CPU-sized weight edit trial with a fixed endpoint and no reinforcement-learning loop. The authors release their code and model under MIT. [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948) · [Repository and license](https://github.com/deepseek-ai/DeepSeek-R1)

Generalized Knowledge Distillation studies training on student-generated sequences. The fork here similarly starts from an observed student trajectory, with a task-derived accepted token set at one prefix. It uses neither an external teacher distribution nor a continual on-policy training loop. Hugging Face's TRL implementation is Apache-2.0. The pilot uses local Torch operations, with no TRL dependency. [GKD paper](https://arxiv.org/abs/2306.13649) · [TRL license](https://github.com/huggingface/trl/blob/main/LICENSE)

ROME studies localization and rank-one interventions on factual associations. MEMIT distributes edits for many associations across model layers. This motivates comparing several native regions and measuring their endpoint effects. A factual edit result does not establish editability of an arithmetic comparison. Their MIT repositories use GPU-oriented example environments, so this trial uses small Torch VJPs on the existing CPU adapter. [ROME project](https://rome.baulab.info/) · [ROME repository](https://github.com/kmeng01/rome) · [MEMIT project](https://memit.baulab.info/) · [MEMIT repository](https://github.com/kmeng01/memit)

AlphaEdit motivates testing projections that reduce interference. Its knowledge-preservation null space differs from the sampled next-token margin gradients used here. EasyEdit motivates reporting target effectiveness, generalization, and locality separately. Both repositories use MIT licensing. This trial imports no code or data from either project. [AlphaEdit paper](https://arxiv.org/abs/2410.02355) · [AlphaEdit license](https://github.com/jianghoucheng/AlphaEdit/blob/main/LICENSE) · [EasyEdit repository](https://github.com/zjunlp/EasyEdit)

The conventional comparator freezes the parent and trains low-rank factors over a full authorized affine matrix using Adam on accepted continuations. It tests the merged weights in their native storage precision. The training step count and parameter budgets are reported separately from the direct edits. [LoRA paper](https://arxiv.org/abs/2106.09685)

## Portability and replay

The same fork and trajectory diagnostics ran on the pinned SmolLM2 parent
without training. Four sampled wrong-option decisions matched the cached
greedy choices under replay, and the two sampled correct trajectories kept
strictly positive replay margins. On the full practice endpoint, SmolLM2
produced 39/48 valid answers, solved 5/16 target items and passed 2/4 positive
items and 4/4 copy items. This run exercises the Llama-family tokenizer and
causal runtime. The baseline did not qualify for adaptation on this roster.
[SmolLM2 engineering receipt](../results/phase-iii-i/cross-model-smollm2.json)

Use the pinned Qwen and SmolLM2 checkpoint directories and the same CPU
Torch and Transformers environment as [Phase III-H](PHASE_III_H.md).
Set `material` to `results/phase-iii-i/causal-fork-v1/material`. Keep the
new Qwen trial workspace outside the repository. The commands below use
POSIX shell syntax. Use the same arguments with PowerShell if preferred.

```bash
material=results/phase-iii-i/causal-fork-v1/material
python -m experiments.phase_iiii.baseline practice --base ./qwen35-base-cpu --material "$material" --out ../fork-practice.json
python -m experiments.phase_iiii.baseline selection --base ./qwen35-base-cpu --material "$material" --out ../fork-selection.json
python -m experiments.phase_iiii.run_causal_fork --base ./qwen35-base-cpu --material "$material" --practice ../fork-practice.json --selection-base ../fork-selection.json --out ../fork-trial
python -m experiments.phase_iiii.analyze --trial results/phase-iii-i/causal-fork-v1/trial --material "$material"
```

The analyzer checks source and artifact hashes and recomputes the public
practice and selection gates. The host `verification finish` command
requires the precommitted private roster and finalist lock. It verifies
the locked selection gate from raw rows before opening that roster. An
incomplete verifier, failed hidden gate, changed replay, or failed
rollback cannot produce a research lineage acceptance.
