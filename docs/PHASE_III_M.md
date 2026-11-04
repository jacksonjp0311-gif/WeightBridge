# Phase III-M causal gradient decomposition

The Phase III-L audit reproduces the recorded practice gain from 1/16 to 4/16, preservation of all 25 parent-correct validation responses, independent selection gain from 0/16 to 1/16, and the sealed result of three comparisons gained with one prohibited regression. The candidate was rejected. Its larger-dose maximin solution matched an equal-group normalized mean and had a different realized norm from the historical comparator. No public opposing-question family became jointly correct. [Source-bound previous-study audit](../results/phase-iii-m/previous-audit.json)

## Frozen mathematical comparison

For comparison i, first average its eligible replay-qualified forks across the two wordings and operand orders. Let C_i be the negative gradient of full-vocabulary accepted-set cross-entropy and let M_i be the positive gradient of the accepted-versus-strongest-rejected decision margin. The shared selected columns and unit right factor v come from the equally weighted comparison-level CE mean.

The permitted update is u vᵀ. Its local comparison gradients are c_i = C_i v and m_i = M_i v in the selected-column region. RAW averages these vectors with equal weight per comparison. UNIT normalizes each nonzero projected vector before averaging. Zero rows remain zero. This explicitly reproduces normalization in the authorized left-factor coordinates. Normalizing a full affine gradient before projection is a different intervention.

| Arm | Corrective objective | Per-comparison scaling |
| --- | --- | --- |
| A | Full-vocabulary accepted-set CE | RAW |
| B | Full-vocabulary accepted-set CE | UNIT |
| C | Greedy accepted-versus-rejected margin | RAW |
| D | Greedy accepted-versus-rejected margin | UNIT |

Each of the four semantic groups must contribute exactly two comparisons from different truth-flip families. Equal group sizes make global item averaging and equal-group averaging equivalent. The primary study supplies no independent group-reweighting treatment.

For an accepted token set S, the CE is logsumexp(z) minus logsumexp(z over S). The margin is max(z over S) minus max(z outside S). Native full-vocabulary indices determine ties. The item gradient averages its eligible fork derivatives before any normalization, so a comparison with four forks and one with two receive the same aggregate raw weight.

For any outcome Y measured on four distinct, matched stored candidates, the descriptive normalization effect is ((Y_B − Y_A) + (Y_D − Y_C)) / 2, the objective effect is ((Y_C − Y_A) + (Y_D − Y_B)) / 2, and the interaction is Y_D − Y_C − Y_B + Y_A. The analyzer withholds these contrasts if the states do not meet its matching conditions. Practice effects describe the fitted population. Independent causal transfer attribution also requires eligible independent observations for the relevant treatments.

The two target stored norms are 0.0125 and 0.0200, with a 0.025 host cap. A bounded search measures the actual BF16 norm, support and tensor hash and replays active preservation prefixes after every admissible numerical adjustment. Individual and pairwise norm differences must meet the frozen two-percent tolerance for a causal matched contrast. Distinct directions that round to the same tensor share one generation observation.

Eight prospective anchors span correct targets, positive comparisons, mixed-sign comparisons and copying, with both wordings. The same anchor-selection policy uses adverse sensitivity over all four directions. Calibration estimates adverse local error separately for each arm and target norm. Up to two calibration and norm-matching passes are permitted. A failed second calibration check blocks that cell before full candidate decoding.

## Data and admission

Fresh truth-flip families bind both opposing comparisons, both operand orders and both wordings to a single partition. Labels use exact Decimal arithmetic. Adaptation has twelve families, preservation fit and calibration have four each, development transfer and preservation validation have four each, and selection and hidden verification have eight each. Positive, mixed-sign and copy controls accompany every partition. [Frozen protocol](../experiments/phase_iiim/semantic_protocol.json) · [Manifest](../results/phase-iii-m/factorial-v1/material/manifest.json)

Fitting receives only adaptation, preservation fit and calibration. A practice candidate must correct a repeatable error with positive strict gain and zero prohibited losses. Development transfer must add a fully correct eight-presentation opposing-question family as well as positive strict gain. Only a locked survivor may access preservation validation and independent selection. A hidden pass requires a separate operator decision before any lineage admission.

## Parent and gradient preflight

The unchanged parent completed all 120 adaptation presentations validly. It solved 2/24 strict comparisons and no complete truth-flip family among twelve. The fit selected eight comparisons, two per subgroup, with 28 replay-qualified forks. All observed wrong next tokens were digits. Twenty accepted sets contained only digits and eight also allowed newline or EOS. Computing the item gradients took eight forward batches and 44 backwards.

The objective change produced a 6.49° rotation under RAW and a 1.67° rotation under UNIT. Normalization produced a 35.43° rotation for CE and a 29.49° rotation for margin gradients. The close-short group supplied 88.62% of the signed raw CE aggregate projection and 81.53% under raw margin. UNIT reduced those shares to about 34%. These are measured geometric contributions in the selected subspace.

A simple limiting case explains why normalization can reduce objective differences. At a single fork with one accepted token and one rival, CE = softplus(z_rival − z_accepted), so its improving gradient is sigmoid(z_rival − z_accepted) times the corrective margin gradient. Unit normalization removes that positive scalar at that fork. Averaging differently scaled forks can still change the comparison-level direction. The actual experiment uses the full vocabulary and accepted token sets. Its measured near-alignment after normalization is consistent with the limiting case and supplies no proof that only two logits matter.

At the smaller target norm, CE/RAW reached a rounded norm of 0.01249999056, but every numerical attempt replayed one active target margin at exactly the 0.125 floor. None changed the token, and the linear constraints predicted a larger margin. The strict positive-slack policy rejected all those states before decoding. Its UNMATCHED status describes the absence of an admissible state in the bounded search, even though the numerical norm itself was reachable.

## Complete-generation outcomes

Every strict comparison requires all four presentations. Every complete family requires both comparisons, for eight correct presentations. All seven generated stored states preserved every previously correct and valid fitting response. Six passed the positive-gain and repeatable-error practice gate.

| Cell | Realized BF16 norm | Changed values | Local practice responses corrected | Strict practice gain | Complete practice families gained | Fresh development strict gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 CE/RAW | 0.01249999056 closest numerical attempt | No admissible state | Not generated | Not measured | Not measured | Not opened |
| B1 CE/UNIT | 0.01254998459 | 14,533 | 12 | +3 | 0 | +1 |
| C1 MARGIN/RAW | 0.01254438774 | 14,743 | 5 | 0 | 0 | Not eligible |
| D1 MARGIN/UNIT | 0.01254338288 | 14,520 | 12 | +3 | 0 | +1 |
| A2 CE/RAW | 0.02003807123 | 15,349 | 25 | +3 | 0 | +3 |
| B2 CE/UNIT | 0.02002734874 | 15,161 | 18 | +3 | 0 | +1 |
| C2 MARGIN/RAW | 0.02002084148 | 15,358 | 28 | +4 | 0 | +3 |
| D2 MARGIN/UNIT | 0.02002507457 | 15,191 | 17 | +3 | 0 | +1 |

The larger-dose four-arm cohort is matched within 0.08615% of its intended norm, comfortably inside the two-percent criterion. All four states are distinct. The sampled linear inequalities were inactive at all seven decoded solutions. Their selected margins retained strict excess above the 0.125 floor after actual rounded execution. Thus the solver did not project those eligible mean directions onto a binding preservation boundary. [Raw trials and checkpoint receipts](../results/phase-iii-m/factorial-v1/trial) · [Analysis](../results/phase-iii-m/factorial-v1/analysis.json)

At the larger dose, practice strict-count main effects were +0.5 comparisons for objective choice, −0.5 for normalization and −1 for their interaction. On the fresh development population, CE and margin each gained three comparisons under RAW and one under UNIT. The corresponding effects were 0, −2 and 0. The development normalization effect was −25 percentage points, with a descriptive family-bootstrap interval of −50 to 0 across only four families. These estimates describe this frozen cohort. They provide limited population-level inference.

The smaller full factorial is unavailable because A1 did not reach generation. Its valid matched MARGIN contrast gained three strict comparisons under UNIT relative to RAW. At the larger norm, RAW margin gained one more comparison than UNIT margin on practice and two more on development. Normalization's behavioral effect therefore depends on the measured radius and population.

CE/UNIT and MARGIN/UNIT produced identical generated token sequences on all 120 adaptation presentations at the smaller dose. At the larger dose they agreed on 119/120 semantic answers. CE/RAW and MARGIN/RAW agreed on 117/120. CE/RAW and CE/UNIT agreed on 95/120 despite equal aggregate strict gains. Numerical and behavioral agreement are reported separately.

## Counterfactual and termination qualification

All six saved practice passers were independently reloaded and measured on the development population. Each gained strict comparisons and preserved all previously correct and valid presentations. None added a complete opposing-question family. The declared family gate stopped the study with HOLD before preservation validation, independent selection or hidden access.

The token evidence supports several kinds of local correction. B1 changed six accepted next tokens to digits and all six prefixes led to complete correct responses. A2 changed five accepted tokens to digits, three with correct full responses, and seven to newlines, all with correct full responses. C2 changed six to digits, five fully correct, and seven to newlines. These are fitted fork observations, not independent task samples.

The unchanged-parent runtime diagnostic applied fixed ±0.125 bias to native EOS and newline logits. Both conditions stayed at 2/24 strict comparisons and zero complete families, with zero prohibited losses and exactly preserved affine bytes. This tested termination bias did not reproduce the weight-edit gains. It leaves other possible runtime biases untested. The numerical-token probability diagnostic covers a declared native subset of digit, decimal-point, newline and EOS tokens. It does not measure every token that could extend or complete a number. [Runtime diagnostic](../results/phase-iii-m/factorial-v1/runtime-termination.json)

## Scientific conclusions

The larger-dose controlled comparison supplies no strict development-gain advantage for replacing CE with margin. CE/UNIT reproduced the margin/UNIT practice corrections, and normalization rotated the aggregate much more than objective choice. However, normalization reduced strict development gains in this cohort. These results prevent a general claim that normalization was the beneficial cause of Phase III-L's transfer. They isolate its effect on correction allocation in the new study.

All generated methods improved individual answers, and six improved complete four-form comparisons. None repaired a full truth-flipping family on practice or development. The aggregate gains concentrated in shorter-correct groups. The evidence supports limited response corrections under the frozen endpoint and leaves complete numerical-rule learning unqualified.

Each tested larger-dose direction had positive predicted slopes for many target forks and a full sampled preservation interval, yet no joint interval covering every fitted target fork. The active inequalities did not bind at the decoded states. This narrows the measured obstruction to the tested target directions and finite budget rather than an observed binding preservation conflict there. A1 separately shows a precision-sensitive measured-margin refusal. Four mean directions do not exhaust the rank-one subspace, so the study cannot establish that the geometry is globally insufficient.

## Next executable study

Freeze fresh families and compare the present one-comparison-per-family fit with a paired-family fit using both opposing comparisons. Keep eight comparison gradients, the same representation, measured norm envelope and preservation gates. Test whether paired supervision produces a complete-family correction. If its local target directions remain incompatible within the allowed subspace, record that evidence before freezing a separate rank-two or alternate-layer feasibility study.

## Integrity, cost and reproduction

The complete runtime suite passed 452 tests with one optional MCP skip. All six MCP tests passed in the separate MCP environment. The published Phase III-K and Phase III-L analyses replayed exactly. The test receipt records an initial PowerShell argument-formatting error that ran no tests and the corrected successful invocation. [Tests](../results/phase-iii-m/factorial-v1/tests.json)

C2, the largest strict practice gain, was independently reconstructed from the base and patch. Checkpoint SHA-256 `7e97531f102338f2c98ef40ed73927c31cc9052cc5309693dc464dcb788a1b58`, its tensor digest, 15,358 changed entries and norm 0.02002084148 matched exactly. All six practice candidates reloaded at their committed hashes for development evaluation, and every live numerical and generation probe restored parent bytes. No candidate reached a lineage-admission stage. [Reconstruction](../results/phase-iii-m/factorial-v1/reconstruction-C2.json) · [Candidate identities](../results/phase-iii-m/factorial-v1/trial/fit-report.json)

Recorded experiment work totals 3,351 model forwards, including greedy steps, 60 backwards, 2,096 generated response records and 25,152 generated native tokens. Fitting consumed 1,156.93 wall seconds and 6,199.91 CPU seconds. Complete measured experimental CPU was 8,091.73 seconds, about 134.86 CPU minutes. Sequential experimental stages took 27.09 wall minutes, with peak RSS of 2.83 GiB. Counts include numerical failures, cached-state handling, development reloads, the runtime diagnostic and reconstruction. Tests, tool orchestration and assistant costs are outside those experiment measurements. [Compute accounting](../results/phase-iii-m/factorial-v1/compute.json) · [Environment](../results/phase-iii-m/factorial-v1/environment.json)

Use the pinned CPU checkpoint and package versions from the environment receipt. The following commands use a private trial directory and the public frozen fitting material.

```bash
material=results/phase-iii-m/factorial-v1/material
python -m experiments.phase_iiim.audit_previous --out ../previous-audit.json
python -m experiments.phase_iiim.run_factorial --base ./qwen35-base-cpu --material "$material" --out ../factorial-trial
python -m experiments.phase_iiim.verification assess --base ./qwen35-base-cpu --material "$material" --out ../factorial-trial
python -m experiments.phase_iiim.runtime_termination --base ./qwen35-base-cpu --material "$material" --fit-dir ../factorial-trial --out ../runtime-termination.json
python -m experiments.phase_iiim.reconstruct --base ./qwen35-base-cpu --material "$material" --trial ../factorial-trial --arm C2 --scratch ../reconstructed-C2 --receipt ../reconstruction-C2.json
python -m experiments.phase_iiim.archive_trial --source ../factorial-trial --material "$material" --out ../factorial-archive
python -m experiments.phase_iiim.analyze --trial ../factorial-archive --material "$material"
```

The sealed roster is host-held and committed by hash. The host-only hidden command requires the successful independent lock. It records any qualifying hidden result for operator review without accepting a checkpoint automatically. This run stopped before those inputs were authorized.

The unscored preservation-validation and selection rows are also retained in the private host workspace. Their original file hashes remain in the public manifest. The public study can replay its actual fitting and development result without opening them. A future authorized verifier must use a private material copy containing the committed files. The development branch was sanitized before its first push so unscored rows do not appear in its published commits.
