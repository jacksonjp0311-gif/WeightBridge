# Phase III-E adaptive neural meta-compiler

WeightBridge tests whether choosing a solver from practice evidence improves
the number of verified repairs, reduces harmful proposals, or spends less
compute than a fixed solver. Phase III-D supplied the motivation. Its
counterfactual ranker responded to changes in question meaning but failed
joint correctness and option-order stability. Its generation endpoint
also failed qualification. [Phase III-D record](PHASE_III_D.md)

## Decision model

The selector accepts a typed evidence record and emits a versioned action
recommendation. Measurement validity, persistent errors, local reachability,
authorized methods, and budget determine which actions are eligible.
Missing evidence can request measurement, a local feasibility check, or a
bounded curvature diagnostic. The host can refuse an action when its cost
exceeds the remaining budget. A support flag and optional sensitivity
disagreement threshold allow abstention under uncertainty.

For each eligible method, the learned model estimates the probability of
independent verified improvement and the probability of a harmful proposed
modification. It fits separate regularized logistic models for those events
and calibrates their outputs on a distinct split. Its utility is the
estimated improvement probability minus a host-selected harm penalty and
an action cost penalty. It refuses methods beyond the harm limit and can
HOLD when every eligible action has nonpositive utility. With learned
predictions omitted, the same selector uses a fixed evidence rule.
The selector's recommendation has no checkpoint authority. The caller
owns hard practice preservation checks, stored-precision materialization,
independent evaluation, and lineage.

Local feasibility searches at most 64 permitted coordinates for a witness
whose target residual is within tolerance, whose Euclidean norm is within
the edit radius, and whose signed preservation margins meet their floors.
Each preservation inequality has the form
`sign * (current_margin + local_margin_change) >= floor`.
The solver intersects target slabs, preservation halfspaces, and the norm
ball using bounded Dykstra projections. A failed search can be
indeterminate. A feasible witness certifies the stated linear constraints
and still needs remeasurement after a real edit.

## Paired question geometry

Each margin is answer 0 minus answer 1 in canonical content coordinates.
Printed order and label assignment map back to those coordinates. With a
question margin `m` and an opposite-question margin `r`, the odd component
`(m - r) / 2` measures the change across the pair and the even component
`(m + r) / 2` measures shared preference. Adding the same answer bias to
both margins leaves the odd component invariant. The decomposition is an
algebraic diagnostic. Other differences between the prompts can also
affect the score.

The prototype loss adds `softplus(-m)`, `softplus(r)`, and a nonnegative
multiple of squared shared preference. Its derivatives with respect to
the margins are `-sigmoid(-m) + weight * preference` and
`sigmoid(r) + weight * preference`. Finite differences verify both.
The separate directional terms penalize a wrong answer even when the
margin difference is large. The Qwen study evaluates scores on the
unchanged model. Its paired choice predicts opposite answers by construction,
so it also checks the raw standalone margins in both directions.

## Study separation

The synthetic benchmark separates method training, calibration, a familiar
test, and held mechanisms. Exact observational twins expose the limit of
any policy given shared practice evidence and opposite hidden outcomes.
A conservative policy can trade missed repairs for fewer harmful
proposals. It cannot identify the hidden response of an indistinguishable
twin. The benchmark reports diagnostic and update costs separately.

The Qwen study uses new question populations and a practice-only rule
selection. Its ranking and free generation endpoints have separate gates.
Host-private examples remain outside the checkout until measurement is
complete. Any real-model adaptation requires both a qualified ranker
and persistent errors with adequate preservation controls.

The synthetic tasks use two editable coordinates and four target margins
of the form `m + Jx + c*x[0]*x[1]`. Three preservation margins have the
form `p + Px`. First order follows the normalized mean Jacobian for
incorrect targets. Constrained search tests the linear surrogate on 64
angles at four radii. The interaction solver adds the mixed term on the
same grid. The solver gets up to four stored FP16 probes. The practice
guard requires a target correction, retains earlier correct targets, and
keeps preservation margins above 0.025. The verifier independently
loads the parent and finalist and requires a new correct prediction with
zero hidden sign regressions. A harmful finalist rejected at that stage
still counts as a harmful proposal.

The study assigns proxy work units to diagnosis, forward evaluation,
analytic derivative requests, curvature, grid trials, candidate slots,
and reloads. It reports these counts with measured wall and process CPU
time. Training and calibration profile multiple methods and are charged
separately from per-policy inference. The familiar test has 1,560 tasks
across six mechanisms. Training and calibration have 1,320 and 600
different tasks, respectively. Two complete mechanisms are held out for
the 520-task shift set. The external laboratory's exact fixtures and code
were unavailable. This benchmark independently tests the hypothesis with
new fixtures and its own frozen budget.

The Qwen protocol crosses A/B and X/Y labels with both option orders
within each of 30 questions. It allocates nine questions each to practice
and selection and 12 to qualification. The rankers share teacher-forced
scores on original, reversed, paraphrased, neutral, and irrelevant
contexts. A standalone raw decision uses one directional context. Prior
subtraction needs the matched neutral context, and paired ranking needs
the opposite question. The runner records the pooled context costs.
Greedy generation uses bullets and runs once per direction and order.
Differences between ranker and generation outcomes therefore include a
change in option display. [Frozen Qwen protocol](../experiments/phase_iiie/qwen_protocol.json)

## Synthetic findings

The learned selector with all diagnostic features had the highest
observed utility among the implementable policies on the familiar test.
It traded repairs for fewer harmful finalists relative to fixed interaction
search and the rules controller. It beat fixed first order on both measured
events. The utility uses a harm penalty of 2 and a work penalty of 0.006.
Those weights matter when comparing the policies.

| Policy, 1,560 familiar tasks | Verified repairs | Harmful finalists | Attempts | Work units per repair | Mean utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed first order | 438 | 93 | 1,406 | 99.66 | −0.0064 |
| Fixed constrained | 537 | 267 | 1,406 | 67.28 | −0.1370 |
| Fixed interaction | 787 | 281 | 1,406 | 62.35 | −0.0445 |
| Deterministic rules | 782 | 280 | 1,076 | 90.42 | −0.1296 |
| Same features, no learning | 782 | 280 | 1,076 | 85.76 | −0.1156 |
| Learned A, behavioral inputs | 194 | 33 | 372 | 174.26 | −0.0480 |
| Learned B, preservation added | 227 | 12 | 253 | 135.51 | +0.0118 |
| Learned C, reachability added | 272 | 24 | 364 | 127.14 | +0.0106 |
| Learned D, curvature added | 514 | 33 | 610 | 99.04 | +0.0914 |
| Learned, adaptive diagnostics | 513 | 35 | 616 | 106.88 | +0.0731 |
| Learned, support check and half radius | 215 | 19 | 364 | 157.83 | −0.0171 |
| Fixed HOLD | 0 | 0 | 0 | — | 0 |
| Unattainable oracle | 821 | 0 | 821 | 35.51 | +0.4142 |

The preregistered paired comparison of D against rules gave a utility
difference of +0.221 with a mechanism-stratified 95% bootstrap interval of
+0.183 to +0.258. Repairs decreased by 17.2 percentage points and harmful
finalists decreased by 15.8 points. D made 96 attempts without verified
improvement and held 259 of the 821 tasks with a verified path at either
authorized radius. In the post-run paired comparison with fixed first
order, D gained 4.9 repair points (interval +3.3 to +6.3), reduced harmful
finalists by 3.8 points (−4.9 to −2.9), and gained 0.098 utility (+0.073 to
+0.125). The oracle examines independent outcomes and illustrates the
searched opportunity set.

A, B and C compare predictor inputs. They all pay for the common
reachability gate. B lowered the number of harmful finalists relative to
A. C increased repairs over B and had similar utility. D provided enough
additional gains in the familiar task mix to cover its extra cost under
the frozen utility. The adaptive policy requested 593 curvature feature
checks versus 1,406 for D. It still spent 54,832 work units versus 50,904
because its expanded measurement repeated earlier probes. Its lower
utility gives no support for that implementation of selective acquisition.
Training and calibration together cost 183,288 work units before fitting.
The full run, including evaluator-only profiles, took 32.34 wall seconds
and 31.70 process CPU seconds.

The exact shift counterexample held on all 130 observational pairs.
Learned D chose the same method within each pair and produced 130
verified repairs and 130 harmful finalists on the 260 reversal tasks.
The support check flagged none of those inputs, and halving the radius
did not change the outcome. The holdout harm estimates also lost
calibration. For the D interaction prediction, harm Brier score went
from 0.114 on 1,406 profiled familiar tasks to 0.265 on the 520 held
tasks. Ten-bin expected calibration error increased from 0.026 to 0.284.
Learned A avoided both edits in 52 pairs and gave up the paired repair
opportunities along with the harms.

The expensive-diagnosis mechanism illustrates the other failure mode.
Every deployed policy held all 260 cases. The learned rule estimated
0.377 benefit from a new measurement and charged 0.594 for it. The oracle
found 248 cases with a verified path under the work cap. This was a
failure of the development estimate to transfer, with real opportunity
cost inside the fixture. No proposed harmful finalist advanced through
the independent verification gate.
[Synthetic report](../results/phase-iii-e/synthetic/report.json) ·
[Frozen protocol](../experiments/phase_iiie/synthetic_protocol.json) ·
[Raw outcomes](../results/phase-iii-e/synthetic/outcomes.jsonl.gz)

## Replay

Use new output directories. Synthetic replay uses the standard research
dependencies. Qwen inference additionally requires the pinned local
checkpoint and its inference dependencies.

```powershell
python experiments/phase_iiie/run_synthetic.py --out $Synthetic
python experiments/phase_iiie/qwen_roster.py --out $Material
python experiments/phase_iiie/qwen_measure.py run --base $Base --material $Material --run $Run
python experiments/phase_iiie/qwen_measure.py archive --material $Material --run $Run --archive $Archive
python experiments/phase_iiie/qwen_handoff.py --archive $Archive --out $Decision
```

A new freeze randomizes the private split populations. To replay the
published measurement exactly, use the archived `material` directory.
The archive command can replay the published decisions without inference
by using the published Qwen directory as `--run` and a new export path.
