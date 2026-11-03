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

## First two qualification rounds

Both model-specific scopes parsed every scored practice presentation
in the first two rounds. The remaining gate failures depended on
task behavior.

| Trial and model | Practice target | Solved across both wordings | Same wrong answer across both wordings | Solved positive controls | Qualification |
| --- | --- | ---: | ---: | ---: | --- |
| v1 Qwen | Addition | 10/12 | 0/12 | 4/4 | HOLD |
| v1 Qwen | Subtraction | 11/12 | 0/12 | 4/4 | HOLD |
| v1 SmolLM2 | Addition | 0/12 | 0/12 | 2/4 | HOLD |
| v1 SmolLM2 | Subtraction | 0/12 | 0/12 | 1/4 | HOLD |
| v2 Qwen | Multiplication | 10/12 | 1/12 | 4/4 | HOLD |
| v2 SmolLM2 | First digit | 1/12 | 9/12 | 4/4 | HOLD |
| v2 SmolLM2 | Last digit | 6/12 | 0/12 | 0/4 | HOLD |

All copy-control sets passed. Qwen's first multiplication roster fell
one repeatable error short of admission. SmolLM2's sole solved
first-digit target was 44, for which choosing the last digit also gives
the correct output. Most of its repeatable wrong first-digit answers
were the last digit of the input. These rounds held before fitting.
Their hidden rosters remain outside the public archive.
[Arithmetic v1 evidence](../results/phase-iii-g/arithmetic-v1) ·
[Model-specific v2 evidence](../results/phase-iii-g/arithmetic-v2)

## Expanded multiplication qualification

The Qwen-only v3 study drew fresh 18-item practice and selection
populations across three multiplication width tiers. Practice passed
with 52/52 parses, all eight positive and copy controls solved, 10/18
target items solved, and three repeatable errors. Selection parsed
51/52, solved all eight controls and 8/18 target items, and found only
one repeatable error. The frozen two-error gate held fitting. The hidden
population remained unopened.

The v3 scope declared Qwen alone under eligible families. Its checkpoint
table still contained the SmolLM2 hash, so the generic finalizer required
an additional lock. The closure receipt records the already measured
HOLD from the sole declared participant, with zero fits and no hidden
access. The v3 protocol and original runner remain byte-identical.

The larger v4 replication parsed 74/76 practice presentations, solved
all eight controls, and solved 13/30 targets across both wordings. One
item had the same wrong answer in both forms. Thirteen items had two
valid but different wrong answers, one was correct in only one form,
and two had an unparsed form. Difficulty exposed errors, but changing
the wording often changed the wrong result. The model held before
selection scoring or fitting.
[v3 selection HOLD](../results/phase-iii-g/arithmetic-v3) ·
[v4 practice HOLD](../results/phase-iii-g/arithmetic-v4)

## A repeatable semantic error

The independently frozen v5 trial asks Qwen to return the larger decimal
value itself. It balances display side and which option has more
decimal places. The parser uses exact decimal arithmetic and accepts
equivalent representations of a completed value.

Both practice and selection parsed 64/64 presentations, solved 16/24
target items across both wordings, and reproduced eight wrong answers.
In every repeatable error the model chose the smaller of the two
displayed negative numbers. The errors covered both display sides and
both precision directions. The four copy controls passed in both
splits. Practice solved three of four positive controls. Selection
solved two of four because the two easy negative-only controls also
reversed their operands. That missed the frozen minimum of three.
The v5 trial held before optimization. Its hidden set remains unopened.
[Signed-decimal v5 evidence](../results/phase-iii-g/decimal-v5)

## Replay

The local model prerequisites and pinned checkpoint hashes match
[Phase III-F](PHASE_III_F.md). Each output directory must be new. The
two-model drivers use both candidate locks before hidden scoring.
Their archive commands accept the measured run directories, or the
published model evidence directories for offline replay.

```powershell
python experiments/phase_iiig/format_diagnostic.py --model qwen --checkpoint $Qwen --out $Format
python experiments/phase_iiig/run_trial.py run --model qwen --base $Qwen --material $Material --out $QwenRun
python experiments/phase_iiig/run_trial.py run --model smollm2 --base $Smol --material $Material --out $SmolRun
python experiments/phase_iiig/run_trial.py finish --base $Qwen --material $Material --out $QwenRun --peer $SmolRun
python experiments/phase_iiig/run_trial.py finish --base $Smol --material $Material --out $SmolRun --peer $QwenRun
python experiments/phase_iiig/archive_trials.py --material $Material --qwen $QwenRun --smollm2 $SmolRun --out $Archive
```

Use `run_followup.py` and the archive flag `--followup` for v2.
The Qwen-only v4 and v5 drivers are `run_replication.py` and
`run_decimal.py`. They take `run` and `finish` with the same base,
material and output paths, and require no peer. Their archive flags
are `--replication` and `--decimal`. The v3 HOLD archive replays
with `--scaling`. A fresh independent study needs a new freeze.
