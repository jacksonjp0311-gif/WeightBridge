# Phase III-M causal gradient decomposition

The Phase III-L audit reproduces the recorded practice gain from 1/16 to 4/16, preservation of all 25 parent-correct validation responses, independent selection gain from 0/16 to 1/16, and the sealed result of three comparisons gained with one prohibited regression. The candidate was rejected. Its larger-dose maximin solution matched an equal-group normalized mean and had a different realized norm from the historical comparator. Neither public opposing-question family became jointly correct. [Source-bound previous-study audit](../results/phase-iii-m/previous-audit.json)

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

The two target stored norms are 0.0125 and 0.0200, with a 0.025 host cap. A bounded search measures the actual BF16 norm, support and tensor hash and replays active preservation prefixes after every admissible numerical adjustment. Individual and pairwise norm differences must meet the frozen two-percent tolerance for a causal matched contrast. Distinct directions that round to the same tensor share one generation observation.

Eight prospective anchors span correct targets, positive comparisons, mixed-sign comparisons and copying, with both wordings. The same anchor-selection policy uses adverse sensitivity over all four directions. Calibration estimates adverse local error separately for each arm and target norm. Up to two calibration and norm-matching passes are permitted. A failed second calibration check blocks that cell before full candidate decoding.

## Data and admission

Fresh truth-flip families bind both opposing comparisons, both operand orders and both wordings to a single partition. Labels use exact Decimal arithmetic. Adaptation has twelve families, preservation fit and calibration have four each, development transfer and preservation validation have four each, and selection and hidden verification have eight each. Positive, mixed-sign and copy controls accompany every partition. [Frozen protocol](../experiments/phase_iiim/semantic_protocol.json) · [Manifest](../results/phase-iii-m/factorial-v1/material/manifest.json)

Fitting receives only adaptation, preservation fit and calibration. A practice candidate must correct a repeatable error with positive strict gain and zero prohibited losses. Development transfer must add a fully correct eight-presentation opposing-question family as well as positive strict gain. Only a locked survivor may access preservation validation and independent selection. A hidden pass requires a separate operator decision before any lineage admission.
