# Phase III-N paired counterfactual adaptation

This study tests supervision allocation in the existing rank-one neural adaptation engine. It begins at Phase III-M commit `178db583894278b9fc52150f8789a1a5f2eadb49`. Historical protocols, checkpoints and results remain unchanged. Sequential learning remains on HOLD until independently verified single-cycle improvement exists.

## Preregistered comparison

All arms use the original pinned Qwen checkpoint, the exact Phase III-M selected columns and unit right factor, corrective decision-margin gradients, BF16 storage, and the same complete-generation endpoint. For each comparison, its replay-qualified erroneous presentations contribute one averaged gradient. Correct parent responses remain preservation obligations.

Arm A averages eight comparisons from eight families, balanced across distance and correct-answer length. Arm B averages both opposing comparisons from four families, with two close and two far families. Both have eight distinct comparison gradients. The comparison tests supervision allocation, because A also has greater family diversity. No within-family supplemental arm is included in this CPU trial.

Arm C uses the same eight comparisons as B. Its bounded numerical solver maximizes the minimum predicted absolute corrective margin, subject to sampled preservation inequalities and a radius ball. The intercept and gradient for each comparison average the same eligible forks. An average predicted margin does not establish that every constituent decision is correct. The report retains per-fork diagnostics, and actual complete decoding determines admission. Bounded projection failure is not a proof that the permitted parameter space is infeasible.

The target stored update norms are 0.0125 and 0.0200, with a hard maximum of 0.025 and a two-percent matching tolerance. At most six unique complete candidate practice probes are allowed. Identical stored tensors share observations. Every numerical probe restores the exact parent weights.

## Data and gates

Each exact-Decimal truth-flipping family contains two opposing comparisons, two operand orders, and two wordings. All eight presentations remain within one partition. Seven partitions separate adaptation fit, preservation fit, calibration, development transfer, preservation validation, independent selection, and host-held hidden verification. Initial material remains in a private host workspace. Only scored, authorized partitions are published.

A practice candidate must gain a strict four-presentation comparison, correct a repeatable semantic error, preserve every required previously correct response, retain valid completions, exceed its measured positive preservation floors, and satisfy the numerical policy. Development transfer additionally requires a newly complete eight-presentation truth-flipping family. Only a locked survivor can reach preservation validation and independent selection. Hidden verification and exact replay remain separate requirements. No automatic lineage admission or production deployment is permitted.

## Execution status

Implementation and prospectively frozen execution are in progress. No new behavioral or scientific success is claimed by this preregistration.
