# Phase III-G behavioral qualification

Phase III-F established that one AWL intent could compile to native
affine updates, reconstruct exact candidate hashes, and restore the
original checkpoint on Qwen3.5 and SmolLM2. Phase III-G tests the
measurement needed to authorize learning on those two pinned models.
[Phase III-F evidence](PHASE_III_F.md)

## Whitespace and completion

The format diagnostic crossed two answer prefixes, a trailing prompt
space, and greedy generation caps of one, two, and six tokens. It used
four fresh digit-echo items and six fresh last-digit multiplication
items. Each model generated a whitespace token first on 20/20 prompts
without a trailing space and on 0/20 matched prompts with one. The
additional prompt token exactly matched the token the model otherwise
generated first. Its native ID was 220 on Qwen and 216 on SmolLM2.
The generated token sequences agreed through the common prefix of every
cap. This distinguishes a generated separator from a failed digit
prediction at the one-token boundary.

At six tokens, both models completed all 16 echo presentations correctly.
On multiplication, Qwen produced 18 completed single-digit lines in 24
presentations and got two right. SmolLM2 produced zero completed lines
under the single-digit grammar. The longer cap fixes the echo truncation.
It still leaves wrong multiplication answers and invalid formats on
the harder prompts. The published Phase III-F scores retain their
original endpoint.
[Frozen format protocol](../experiments/phase_iiig/format_protocol.json) ·
[Token traces and reports](../results/phase-iii-g/format)

## Arithmetic experiment

The second protocol uses a bare `Answer:` prefix and unrestricted greedy
completion. It scores an integer only after the first line ends with a
newline or EOS. Output captured at a token cap without a line boundary
remains unparsed. The prefix and candidate answer strings use the native
tokenizer on each model. For fitting, the host checks that the two
contrasting answer strings preserve the frozen prefix token IDs.

Both model-specific trials receive the same independently generated
populations. Practice and selection each contain 36 underlying items.
The hidden population has 44. Every item has two fixed wordings.
The two target families test multi-digit addition with carries and
subtraction with borrows. Each has easy one-digit positive controls.
Integer copying serves as a separate preservation family.

The practice gate requires high parse reliability, at least three
solved positive and copy controls, two solved target items, and two
repeatable wrong target answers. A solved or repeatably wrong item must
meet its criterion on both wordings. Practice chooses a family
separately for each model. Selection checks the locked family on fresh
examples. The fitting steps use practice evidence only. Independent
selection remeasures the original and every materialized candidate,
then locks a finalist before access to the hidden evaluation. Acceptance
requires at least two newly correct target items, zero regressions on
previously correct target and control presentations, and reproducible
checkpoint reload and restoration.
[Frozen arithmetic protocol](../experiments/phase_iiig/protocol.json)
