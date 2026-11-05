# Limitations and held work

## Current HOLDs

- **Published benchmark reproduction: HOLD.** No MATH-500, GSM8K or IFEval
  benchmark dataset was downloaded or evaluated. The dataset preparation and
  matrix scripts are present, but the `benchmark` optional dependencies are
  not installed in the project environment. The Qwen checkpoint and candidate
  were kept in an external temporary cache and are not included in the project.
- **Published Qwen generation reproduction: HOLD / INCONCLUSIVE.** The paired smoke used two
  hand-authored development prompts, one output each, and a 64-token cap. Every
  output was truncated. A separate exploratory 256-token smoke also truncated
  every generation. Neither can support an accuracy or loop-reduction claim.
- **Constrained Qwen AI-directed edit: pilot.** The controller chose layers
  and strengths and a fixed compiler fit edit directions. One of three
  induced-fault cases passed the full reload and locality gate.
  [Pilot](QWEN_AI_EDIT_PILOT.md)
- **Direct LLM transfer: FAIL.** The frozen four-mechanism, 20-task
  Qwen cohort completed with zero finalized repairs. One separate Codex
  session found a blinded hidden-bias repair.
  [Qualification](../results/wb-dq/cohort_summary.json)
- **Published random Qwen control behavior: HOLD.** The control checkpoint has matching
  changed-value counts but was not run through inference.
- **LoRA or language-model conventional baseline: HOLD.** The measured
  conventional baseline is full-batch SGD on the small synthetic MLP only.
- **Private-memory integration: intentionally absent.** MWC uses generated
  synthetic records only. There is no RecallTree connector or private notebook
  ingestion path.

## Environment and compute constraints

The captured machine is Windows 11 Pro on an AMD Ryzen AI MAX+ 395 (16 physical
cores, 32 logical processors), with 132.8 GB RAM and an integrated Radeon
8060S. The graphics adapter reports shared memory, not dedicated VRAM; CUDA is
unavailable. Local Qwen inference ran with CPU PyTorch and Transformers and
fallback attention kernels at about 10 generated tokens/second in the smoke.
No vLLM or CUDA runtime was available. The pinned single-file checkpoint is
1.747 GB and remains outside the repository.

At 10 tokens/second, generating the full 32,768-token cap for just the cited
MATH-500 (500 × 4), GSM8K test (1,319 × 2), and IFEval (541 × 2) settings would
take roughly 217 days of generation time before model loading, evaluation or
other overhead. This is a cap-based resource estimate, not a measured
runtime. A real evaluation may terminate earlier, but the 64-token smoke shows
why the current CPU configuration cannot estimate that behavior reliably.

The exact captured values and dependency versions are in
[`SYSTEM_CAPABILITIES.md`](SYSTEM_CAPABILITIES.md) and
[`system_capabilities.json`](../results/system_capabilities.json). RAM and disk
availability are time-specific snapshots.

## Scientific limits

- The bundled `DiagnosticAgent` remains deterministic finite-difference
  search code; it is not an LLM and spent zero model-generated tokens. A
  separate Codex MCP pilot made one LLM-directed proposal without being given
  the correct value. Codex token use was not available from the host.
- The toy defects were deliberately injected and the permitted tensor set was
  specified by the host. The MCP task withheld its mechanism and correction
  from the tool session, but one synthetic task does not establish broad
  discovery or transfer.
- External-memory-only changes inference-time decision behavior; it is not
  parametric learning. The memory record contains the exact repair for the
  same synthetic task family.
- A successful patch application, matching hash, or restart load is not a
  behavioral improvement. Toy persistence is demonstrated only on the toy
  model; the Qwen rollback smoke reloads the original checkpoint but does not
  prove a Qwen candidate improvement persisted.
- The Qwen repetition estimator uses a repeated-block heuristic and tail
  n-gram proxy. The small run has insufficient output and sample count to
  validate either against the paper's loop metrics.
- The 10-seed MWC confidence interval is clustered by seed, but only ten
  independent synthetic seeds are available. Its exact memory transfer and
  repeated defect structure limit inference to that task family.
- No final holdout has been used for patch selection in the current canonical
  toy runs. The first seed-731 record was marked as a duplicate replay; the
  second isolated seed-731 run remains in aggregate plots with seeds 840/841.

## Dataset, license and runtime notes

The checkpoint is Qwen3.5-0.8B under its published Apache-2.0 terms. The
authors' code/logs are attributed in
[`REFERENCE_PROVENANCE.md`](../experiments/published_reproduction/REFERENCE_PROVENANCE.md).
MATH-500's dataset card does not present a clear license in its metadata;
dataset artifacts were not redistributed. GSM8K and IFEval have their source
license metadata recorded in the prep documentation. Users must verify current
terms before downloading, using or redistributing data and model weights.

The safetensors scanner accepts unsharded and indexed-sharded model layouts;
model-specific remote code is disabled in the inference adapter. Activation
and attention probes are bounded and depend on backend support. The toy worker
validates its JSON boundary but shares the current user's OS account and is not
a hardened OS sandbox. A separate inference process limits code coupling but
does not replace OS/container isolation.

The inference adapter rejects local checkpoint files above its configured RSS
budget, checks current RSS after model load, and stops generation if the budget
is crossed. The matrix runner can terminate a child process at its wall-time
limit. The Python interface cannot prevent a transient memory spike during
model loading before the post-load check; use an OS/container memory limit when
the process must be contained against allocation spikes.

## Remaining engineering gaps

- Actual benchmark downloads, exact-sampling full generation, MATH/GSM grading,
  and IFEval official scoring have not run end-to-end here.
- No GPU/ROCm backend support is implemented or validated in this environment.
- No long-context, restart-equivalence or rollback-performance trial was run
  on the p50 Qwen candidate.
- No layer-specific activation intervention, singular-value survey or
  causal circuit analysis has identified why the published patch works.
- The MCP tools bind a synthetic `ToyMLP` practice evaluator. Real Hugging
  Face checkpoint edits and held-out scoring use the CLI. The MCP server is
  an application boundary under the launching user's OS permissions.
- The completed v2 20-task cohort covers local Qwen in patch-form mode.
  The local Transformers host records
  checkpoint, tokenizer, sampling and runtime identity and uses a fresh
  context per trial. The model's only action channel is the broker's MCP
  interface. Bounded runs use separate protocols and seed rosters, so they
  cannot be pooled into the registered cohort.
- The host identity receipt and cohort records are content-hashed, not
  cryptographically signed or provider-attested. They make accidental or
  post-freeze changes detectable when the original protocol is retained, but
  a host operator could fabricate the identity assertion. Treat model identity
  as host-reported evidence until an auditable attestation/signature mechanism
  is added.
- The MCP pilot measured host-process CPU and elapsed time, but not MCP child
  CPU or LLM token costs. A future multi-task study needs a pinned model/harness
  identity, full tool transcript and provider cost records where available.
- The ledger is append-only for a single writer but does not implement
  multi-process locking or signed remote provenance.
- There is no selection correction across repeated adaptive cohorts or model
  selection on a common task family. The v2 cohort freezes one roster, order,
  and decision gate; it does not authorize reusing its holdout to tune later
  candidates or correct significance across multiple cohorts.

Phase III-E's work units are a declared proxy for the two-coordinate
fixtures. They do not predict GPU or training costs. The synthetic
controller's harm estimates lost calibration on held mechanisms.
All exact sensitivity-reversal twins fell inside calibrated support, and
a half-radius edit still produced opposite hidden outcomes within each
pair. Independent verification remains necessary even when the solver is
locally feasible and the controller estimates low risk.
[Phase III-E evidence](PHASE_III_E.md)

Phase III-F validates one editable affine role in supported dense
Qwen3.5 and Llama safetensors layouts. Quantized weights, external
adapter training, arbitrary model derivatives, and model-independent
risk calibration remain outside the verified capability set. The
engineering study verified source restoration in fresh processes.
The pretrained models have yet to demonstrate behavioral improvement or
retention through this interface.
[Phase III-F evidence](PHASE_III_F.md)

Phase III-G showed that the conditional two-answer practice guard can
miss severe regressions in completed greedy answers. Both tested
rank-one candidates passed that guard and failed fresh generation after
reload. The observed failure is scoped to the selected layer, fitting
data, update limits and task. The two realized perturbation norms
differed, so the study does not isolate a solver-only effect.
[Phase III-G evidence](PHASE_III_G.md)

The Phase III-H guard can reject a probe that degrades the real decoding
endpoint, even when its fitted scores improve. Passing practice still does
not establish generalization. Full-vocabulary KL in this pilot is measured
on a few observed parent prefixes and only ranks measured eligible probes;
it does not constrain the derivative fit or bound behavior on other inputs.
The probe space covers selected columns of one affine module and cannot
exclude useful updates in other regions. SmolLM2's replay establishes
shared endpoint compatibility, not behavioral qualification.
[Phase III-H evidence](PHASE_III_H.md)

Phase III-I's fork diagnostic is relative to a bounded set of verified
answer surfaces. Uncached logits can disagree with the cached greedy path.
The diagnostic excludes such cases from fitting. Sampled preservation
gradients and average KL describe selected prefixes. Neither substitutes
for the complete generation guard or independent evaluation. The regional
scan estimates reachability along constructed directions in three affine
regions. Other regions and update geometries remain untested by that scan.
[Phase III-I evidence](PHASE_III_I.md)

Phase III-J samples a finite set of token decisions even when each practice
group and wording has an anchor. Local intervals assume linear logit margins
along a particular direction. Rounding, changing competitors and subsequent
decoding can invalidate those predictions. Practice counterexamples are
eligible for refitting and cannot serve as independent confirmation.
[Phase III-J evidence](PHASE_III_J.md)

Phase III-K measures a finite directional secant at one rounded dose.
The resulting ranking may change for other update directions and radii.
The residual allowance describes a small calibration cohort and the
reference probe. Both paired local feasibility and semantic generation
must be measured before drawing conclusions about correction behavior.
[Phase III-K evidence](PHASE_III_K.md)

Phase III-L evaluates one roster seed and one Qwen checkpoint. It changes
the historical fork-CE direction to normalized margin gradients while
balancing groups, so the behavioral comparison cannot isolate aggregation
alone. The projected cosine matrix is exact in the fixed-right subspace.
The wider affine CE matrix is a seeded sketch. Positive subgroup-average
slopes and eight protected anchors left individual errors and a prohibited
sealed regression. Rejected BF16 states retain exact tensor hashes, norms
and support counts, while a full elementwise cross-arm delta is unavailable
from their archive. The sealed data remain host-held.
[Phase III-L evidence](PHASE_III_L.md)

Phase III-M normalizes projected fixed-right gradients, with eight fitted
comparisons and four development families. This is one small study. Its
complete larger-dose factorial does not establish which factor caused the
old Phase III-L transfer or which will generalize elsewhere. The smaller
CE/RAW cell was withheld after a precision-sensitive measured-margin
refusal. Full truth-flipping family correction never qualified. Four mean
directions leave most of the allowed rank-one space unexplored, and the
fitter selected one comparison per family rather than paired supervision.
[Phase III-M evidence](PHASE_III_M.md)


Phase III-N matches eight comparisons per main arm, not family diversity or
qualified presentation counts: A uses eight families/thirty forks, B/C four
families/twenty-six forks. One checkpoint and roster seed are tested. There
are only six fresh development families, and none became completely correct.
The pair-aware objective uses average fork margins within comparisons, not
a guarantee on every token or all complete trajectories. Its larger-dose
state was not decoded because calibration failed. No same-family four-item
supplemental arm, new LoRA comparison, broader geometry or sequential learning
study ran.

BF16 rounding makes stored deltas depart from the unrounded rank-one
subspace. Two-row local norm bounds exclude this rounding, preservation,
nonlinear effects and alternative valid generation trajectories. They cannot
certify neural unreachability. Earlier task archives and disclosed evaluation
examples must not be reused as new hidden tests. The initial failed run lost
whole-process and partial-gradient CPU totals; overall recorded compute is
a lower bound. The final unit suite also recorded a large unexplained wall
anomaly in one existing passing BIC test. Test CPU and assistant costs are
not included in experimental totals. [Phase III-N](PHASE_III_N.md)

Phase III-O's numerical norm certificates concern a supplied finite local
linear system with a fixed right factor and positive target margin. They use
floating-point residuals and conservative padding, not interval arithmetic.
They do not establish nonlinear infeasibility, impossibility across other
right factors, or inability to learn the task. Native BF16 rounding can leave
the exact unrounded rank-one subspace. A locally required token correction
may not cover all semantically valid alternative generation paths.

Two fresh fitted families and eight sampled preservation decisions are a
small diagnostic study. Neither candidate generation nor any later evaluation
partition opened. The two restored BF16 preservation secants are not model
improvements. No candidate target-precision failure, generation failure,
independent regression or new persistent success can be inferred from this
numerical HOLD. The average-constrained QP diagnostic is distinct from the
historical mean-direction proposal. [Phase III-O](PHASE_III_O.md)

Phase III-P's real-model right-factor search is unperformed because the
preregistered relaxed bound blocks it. Synthetic free-right successes validate
implementation paths, not Qwen improvement. The full-block numerical bound
covers every rank within these sixteen columns under the recorded linear
approximation; it does not cover nonlinear changes, other columns, another
layer, or every valid output trajectory. Numerical padding is not interval
arithmetic. Native AD and BF16 execution remain approximations to a local
behavioral model. The fourteen wrong fitted presentations are not independent
semantic samples; they belong to two families.

The external reference has different layers, support and intervention norm,
and its test numbers use a disjoint magnitude range. It cannot isolate the
causal effect of optimizing the right factor. Six reference families are a
small exploratory cohort. Six gains and one loss still violate zero-regression
qualification, and no strict comparison or complete family is gained. The
authors' held-out-practice and published benchmark results are separate from
our short greedy measurements. No full benchmark, newly learned semantic rule,
accepted checkpoint or continual-learning result follows. Preliminary source
inspection and assistant costs are not completely metered; recorded CPU totals
are scoped, not an exhaustive cost claim. [Phase III-P](PHASE_III_P.md)
