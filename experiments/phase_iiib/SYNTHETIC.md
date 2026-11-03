# Synthetic interaction laboratory

The lab independently constructs six mechanisms from the supplied research
claims. The frozen protocol records seeds, row counts, budgets and the gates
for bounded real-model exploration. Its numerical results refer to these
fixtures. The original private experiment data and exact architectures were
unavailable.

```powershell
python -m experiments.phase_iiib.interaction_lab --out phase-iiib-replay/report.json
python -m pytest -q tests/test_interaction_lab.py
```

The writer refuses to overwrite an existing report. The archive contains
per-seed measurements, elapsed time, and exact source and protocol hashes.
Replays use a new output path. Compare measurements independently of timing
and floating-point differences across NumPy builds.
[Frozen protocol](synthetic_protocol.json) · [Archived results](../../results/phase-iii-b/synthetic/report.json)

At the origin of \(L(a,b)=v(ab-c)^2\), the gradient vanishes. The off-diagonal
Hessian entries are \(-2vc\), so with \(v>0\) and nonzero \(c\) the Hessian has
one negative eigenvalue. For the selected \(v=0.506\), \(c=0.4\), the loss starts
at 0.08096. The shared BIC grid picks a coupled step that brings it to zero.
The first-order grid abstains. The active version starts at \(a=1,b=0\), so a
first-order step reaches zero loss with four probes. The mixed method uses
eight and leaves loss 0.000382. The harmful fixture uses the output
\(ab-8(ab)^2\). The mixed surrogate predicts a decrease from 0.16 to about
−0.8745, while the actual loss rises to 1.1595 and fails verification.

The dormant family uses independent Gaussian inputs and a separate practice
and test draw per seed. Each of 48 features has two dormant factors, and a
single randomly designated feature determines the label up to Gaussian noise.
The mixed derivative at the origin is \(-2E[x_j y]\) for the two factors on
feature \(j\). The guided arm computes one contraction over all 48 features,
then tests magnitudes on the highest ranked feature. Random search samples
16 features without replacement and fits a separate coefficient on each.
Both arms include the origin and measure 17 candidates, with the derivative
and scalar fitting costs recorded separately. The nine finite difference
audit calls per case run after selection.

| Dormant method | Feature hits | Mean unseen MSE |
|---|---:|---:|
| First order | 0/200 | 0.177987 |
| Mixed guided | 200/200 | 0.000001001 |
| Random paired | 54/200 | 0.137478 |
| Exact one-feature least squares | 200/200 | 0.000001001 |

The exact one-feature solution uses the known separable architecture to
evaluate candidate improvement from sufficient statistics. It matches the
guided method on this fixture without spending the nonlinear trial budget.
The randomized comparison supports an advantage from identifying interactions
when the parameters are dormant. The exact comparison bounds the claim to
cases where the architecture does not already supply a cheaper closed form.

The coordinate fixture permutes hidden neurons and the matching readout
weights. Baseline predictions agree within \(2.3\times10^{-16}\). Raw transfer
of the same readout delta raises test MSE from 0.3285 to 0.4329. A one-feature
refit on practice activations in the permuted coordinates restores the intended
output on unseen inputs to numerical precision.

| Five-cycle shared bottleneck | Final target MSE | Final prior-output drift MSE | Mean accepted cycles |
|---|---:|---:|---:|
| Target only | 0.4828 | 0.5782 | 5 |
| Soft retention | 0.2033 | 0.1220 | 5 |
| Strict nullspace | 0.4120 | Below 1e-30 | 4 |

The sequential fixture uses one linear bottleneck with four independent
directions and a fifth target that conflicts with the first. The strict
nullspace removes all permitted directions in cycle five on all 20 starts.
The soft penalty reduces drift and balances the conflicting objectives on
this construction. Hidden rows are scored after all practice decisions.
The experiment tests preservation objectives independently of curvature.

All nine gates passed in 0.358 wall seconds and 0.344 process CPU seconds
with NumPy 2.5.4. The gate permits a bounded Qwen ablation. The next model
test must compare mixed information with first-order synthesis using the
same directions and report the extra probe cost.
