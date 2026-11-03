# Phase III-B behavioral interaction compiler

The Phase III-B backend extends WeightBridge's practice-only synthesis. The
existing activation-ridge backend, qualification data, lineage records and
patch receipts remain available. AWL v2 describes a binary choice margin
objective, permitted Qwen MLP projections, practice preservation families,
and a requested solver and radius. The host decides which practice families,
tensors, edit limits and forward batch budgets to authorize. The compiler
returns a patch or abstains. The host independently applies its patch policy
and verifies the saved candidate before it can advance a research lineage.

```json
{
  "schema": "weightbridge.awl.v2",
  "objective": "binary_choice_margin",
  "target": [
    "model.language_model.layers.23.mlp.down_proj.weight",
    "model.language_model.layers.21.mlp.down_proj.weight"
  ],
  "preserve": ["generic", "word-order"],
  "representation": "rank_one_slices",
  "optimize": {"method": "bic_interaction", "radius": 1.0}
}
```

The same two rank-one directions define the search space for the first-order
and interaction methods. BIC measures the derivative of the answer logit
difference with respect to each selected projection's last-token output and
records the parent margin for each practice example. It uses target errors
and low margins to construct the directions. The directions are normalized
in parameter space and cover at most 32 columns per layer on this checkpoint.
The joint coefficient radius bounds the requested Frobenius norm. The fitter
uses logistic loss on signed target margins and a weighted binary KL penalty
against the parent on preservation rows. A quadratic cost penalizes the size
of the proposed update. This KL is over the two authorized answer tokens.

The first-order method estimates the local objective slope from four axial
probes. The interaction method adds four corner probes to estimate the mixed
derivative in that same two-dimensional space. Both methods record axial
curvature and omit it from selection, so the mixed term is the change in the
surrogate. No model-sized Hessian is formed. The probes use the checkpoint's
stored precision. BIC separately measures the selected update and abstains
if it rounds away or fails to lower the measured practice objective. A host
then reloads the persisted candidate and checks behavior under the frozen
practice and evaluation rules. Rounded coefficient probes, locally predicted
margins and reloaded scores are distinct measurements.

## Synthetic admission

For \(L(a,b)=v(ab-c)^2\), the origin has zero gradient and an off-diagonal
Hessian entry \(-2vc\). With \(v,c>0\), increasing both parameters along
the negative-curvature direction can reduce the loss. The lab recovered this
effect and reduced the constructed loss from 0.08096 to numerical zero.
Across 200 dormant-pathway cases, guided selection found the designated
feature 200 times versus 54 for random paired selection. Both arms received
17 nonlinear candidate evaluations. The guided arm also used a mixed
derivative contraction. An exact one-feature least-squares method matched
the guided result on this separable family.

In the active fixture, the first-order solver reached zero loss with four
probes while the mixed solver left residual loss with eight. In the harmful
fixture, a favorable local prediction led to an actual loss increase from
0.16 to 1.1595. Verification rejected it. Raw transfer to a permuted network
gave MSE 0.4329 and coordinate-aware recompilation recovered equivalent
behavior. On 20 five-cycle runs, soft preservation reduced prior output
drift MSE from 0.5782 to 0.1220. Every strict nullspace run exhausted its
permitted directions. These tests passed the frozen gate for a bounded Qwen
pilot. They do not establish an advantage on general learning tasks.
[Synthetic protocol, results and reproduction](../experiments/phase_iiib/SYNTHETIC.md)

## Retrospective Phase III diagnosis

The old accepted checkpoint was reconstructed by exact hash and rolled back
in an isolated lineage. A separate retrospective inference run reproduced
the original 18/24 and edited 15/24 target predictions on the disclosed
Phase III final set. Those prompts did not enter the new protocol.

Across all eight historical fraction examples, the patch increased the A
minus B logit margin under both answer orders. The average shift was +0.414
logits. The same examples with X/Y choices had a +0.023 average X minus Y
shift. Date and word comparisons had more varied A/B changes. This supports
a token-specific bias on that fraction suite. A uniform shift across every
family would be inconsistent with the measured changes. The original chose
Y on every historical X/Y prompt, so that format provides a weak behavioral
test despite the measurable logits. Counterbalancing distinguishes a fixed
preference for one token from some order-sensitive effects, while the paired
prompts still share the same underlying questions.

The measured cosine between the A/B embedding difference and the functional
margin derivative at layer 23 was about 0.992 in six parent examples. That
alignment makes a poorly aligned layer-23 direction an unlikely explanation
for these six cases. It does not establish that the fitted activation gate
is selective for the intended task or that the same alignment holds earlier
in the model. The margin diagnosis took 25 forward batches and one backward
batch, totaling 174.0 process CPU seconds and 53.2 summed wall seconds.
[Retrospective margins and replay evidence](../results/phase-iii-b/retrospective/)

On the fresh common practice screen, none of the tested alternate token pairs
met the predeclared accuracy, response balance and same-answer consistency
criteria. A/B scored 12/20 and selected the same answer content in 2/10 paired
variants. X/Y chose X on 19/20 rows and was consistent in 1/10. The preferred
token changed across prompt sets. The calibration used 18 forward batches,
32.9 wall seconds and 118.8 process CPU seconds. Renaming the answer tokens
alone did not produce a reliable behavioral measurement.
[Practice-only choice calibration](../results/phase-iii-b/retrospective/choice-calibration.json)

An exploratory probe kept the two actual answer strings in the prompt and
scored the model's likelihood of each full answer. The length-normalized
rule scored 18/20 and chose the same answer content in 8/10 paired variants.
Raw summed likelihood reached 19/20 and 9/10. Length normalization changed
one decision, and the single fraction item still failed the normalized
consistency check. This probe changed both the instruction and the scoring
rule. It used practice data only, five forward batches, 42.9 wall seconds,
174.6 process CPU seconds and 6.53 GB peak RSS.
[Whole-answer practice probe](../results/phase-iii-b/retrospective/answer-content-probe.json)

## Measured probe guard

A follow-up prototype selects only among already measured practice probes.
It requires a lower observed objective, preserves every prior prediction,
and forbids regressions on previously correct target rows. The host can also
require a positive target gain. It abstains when no evaluated point qualifies.
In two synthetic fixtures with eight measured probes each, selection by the
soft loss gained one answer while flipping two preservation predictions. The
guard kept a one-answer gain with zero preservation flips in one fixture and
abstained in the other. It used the same probes and added zero model
evaluations. This prototype did not participate in the frozen Qwen pilot.
[Synthetic guard evidence](../results/phase-iii-b/synthetic/measured-probe-guard-v1.json)

## Fresh Qwen ablation

The frozen exploratory pilot used one date-order cycle with four underlying
holdout items. Each item was evaluated in both answer orders and in both A/B
and X/Y formats. The compiler fit on authorized practice only. Every arm had
one candidate slot, 16 fitting rows, the same 65,536-coordinate limit and the
same requested norm cap. The common practice screen had 20 rows. The host
froze all six practice decisions before opening any cycle holdout. It scored
rejected saved candidates for diagnosis after those decisions.

| Method | Date practice | Practice target regressions / preservation flips | A/B holdout | A/B target regressions / sentinel flips | X/Y holdout |
|---|---:|---:|---:|---:|---:|
| A. Activation ridge | 4/8 → 4/8 | 0 / 0 | 4/8 → 4/8 | 0 / 0 | 6/8 → 6/8 |
| B. Task-aware activation ridge | 4/8 → 4/8 | 0 / 0 | 4/8 → 4/8 | 0 / 1 | 6/8 → 6/8 |
| C. First-order coordinated | 4/8 → 6/8 | 2 / 6 | 4/8 → 4/8 | 4 / 1 | 6/8 → 6/8 |
| D. Mixed-curvature coordinated | 4/8 → 6/8 | 2 / 6 | 4/8 → 4/8 | 4 / 1 | 6/8 → 6/8 |
| Matched random | 4/8 → 4/8 | 0 / 0 | 4/8 → 4/8 | 0 / 0 | 6/8 → 6/8 |
| Two-step rank-one Adam | 4/8 → 4/8 | 0 / 1 | 4/8 → 4/8 | 0 / 1 | 6/8 → 6/8 |

No candidate was eligible. Both coordinated edits switched every A/B target
prediction from A to B, trading four corrections for four regressions. They
left every X/Y prediction unchanged. The task-aware single-layer arm showed
no improvement over activation ridge at these settings. The two-layer arm
also changes the basis and objective, which prevents attributing its result
to the extra layer alone. The deterministic first-choice control is Arm A
under a one-candidate budget. Adam is a fixed-column gradient comparator.

First order used 12 fitting forward batches and two backward batches. Mixed
curvature used 20 forward batches and two backward batches. Both realized
norms were about 1.0, with 64,556 and 63,282 stored values changed respectively.
The mixed term reversed the selected layer 21 coefficient, with identical
practice and holdout decisions. The parent objective was 1.040. The first-order
and mixed surrogates predicted −0.972 and −0.924, outside the range of the
nonnegative objective. Their measured rounded objectives were 0.817 and
0.861. The measured scalar improvements still allowed per-example regressions.

Every arm ended at the original checkpoint and scored 26/48 on the final
suite of 12 underlying items. The pilot took 578.9 wall seconds and 1,246.1
process CPU seconds across the host and compiler workers. Peak RSS in a
single process was 2.44 GB. The two frozen five-cycle rosters remain unrun.
At the pilot rate those rosters would take about 97 minutes before accounting
for larger retention suites. This pilot used one roster, one target family,
and a small number of correlated answer presentations. It gives a specific
counterexample to the current objective and provides no evidence of cumulative
learning or a mixed-curvature advantage on Qwen.
[Qwen pilot archive](../results/phase-iii-b/qwen-pilot/)
· [Per-candidate failure classifications](../results/phase-iii-b/qwen-pilot-feedback.json)

The next experiment should first qualify a whole-answer measurement on fresh
practice from all three task families. Then compare guarded measured selection
with the existing soft objective using the same authorized preservation rows
and probe budget. For the mixed-curvature comparison, give first order the
same number of probes at additional axial radii. A positive result requires
more accepted edits under the original independent retention and final gates.
[Frozen design and reproduction](../experiments/phase_iiib/README.md)

The inference environment passed 149 tests and skipped the optional MCP test.
