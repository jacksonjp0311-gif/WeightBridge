# Phase III-B Qwen comparison

The study starts from the pinned Qwen3.5-0.8B checkpoint. The frozen design
defines a one-cycle exploratory pilot and two independent five-cycle rosters.
The five-cycle order is word, date, fraction, word and date comparison. The
materializer assigns canonical underlying items to disjoint hash partitions.
It uses fresh domains for the old task families and separates practice, cycle
verification and cumulative final items.

Every underlying item appears with the correct answer in both printed
positions. Label order alternates across items. Practice uses A/B. Each
independent suite renders the same items with both A/B and X/Y and uses
different wording from practice. Statistical summaries resample underlying
items as clusters because the variants share content.

The numerical arms each fit 16 practice rows and use one candidate slot per
cycle with the same 65,536-coordinate cap. A uses generic preservation
examples. B swaps in two non-target family practice items while keeping the
same rank-one ridge representation. C and D each use two rank-one slices and
fit on B's examples. C uses the local first-order estimate. D adds a mixed
curvature term. Both measure axial curvature as a diagnostic. Probe counts and process time are
measured separately. The common cap on requested Frobenius norm is 1.0. The
study measures the realized norm and changed entries after storing BF16
weights. The same one-slot schedule makes the fixed deterministic comparator
identical to arm A. The other controls use a fixed random row permutation and
a two-step Adam fit of rank-one factors on the reference columns.

Every practice finalist is picked against the same practice screen, which
includes generic and other-family examples. The study completes and freezes
every arm's practice decision for a cycle before opening that cycle's holdout.
It also independently measures stored practice-rejected candidates for the
ablation. Only practice-eligible finalists can advance the lineage. A
qualifying cycle requires at least one new A/B answer, no target regressions
or unrelated prediction changes in either format, and no changes to earlier
accepted predictions. All cumulative final suites stay sealed until the
last cycle closes. The full gate requires at least three accepted cycles,
positive final gain beyond the strongest baseline, and zero final target
regressions or sentinel flips. The exploratory pilot has one cycle and cannot
qualify.

Use an inference-enabled Python environment and the exact local checkpoint.
Create new output directories for each command.

```bash
python experiments/phase_iiib/qwen_material.py --base /path/to/base --out /path/to/material --pilot
python experiments/phase_iiib/run_qwen_ablation.py run --base /path/to/base --material /path/to/material --admission results/phase-iii-b/synthetic/report.json --out /path/to/run
python experiments/phase_iiib/export_qwen_ablation.py --base /path/to/base --material /path/to/material --run /path/to/run --out /path/to/evidence
```

Omit `--pilot` and pass `--roster-index 0` or `--roster-index 1` for the
independent five-cycle rosters. Freeze one study before opening its
verification data. Use the archived material for byte-exact replay across
operating systems. Released prompts can serve as reproducibility inputs.
New independent evaluation requires fresh sealed items.

The compiler subprocess receives only an input job containing authorized
practice rows, numerical intent and edit policy. Keep verification files
outside any untrusted process account's readable directories. The evidence
export omits model shards, redacts local paths in lineage metadata and stores
the exact patch bytes inside gzip files.
