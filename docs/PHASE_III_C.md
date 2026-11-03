# Phase III-C behavior-grounded adaptation

Phase III-B's whole-answer probe changed the prompt labels, response
instruction and score at the same time. It measured conditional likelihood
on the unchanged model. It did not score the stop token or test generated
answers. The Phase III-C factorial measures those effects separately before
admitting real-model updates.

For every underlying item, both instructions see the same labeled options.
The experiment scores the two label tokens and the two full answer strings
under each instruction, including the crossed diagnostic conditions. Both
option orders appear in each of two fixed paraphrases. A/B and X/Y are
balanced between items. Printed date and fraction forms vary between items.
An additional practice probe changes labels and representations within an
item. The base model is the pinned Qwen3.5-0.8B checkpoint from Phase III.

Teacher forcing evaluates the joint tokenization of the rendered prefix and
answer. If the answer changes prefix token IDs, the scorer refuses the
continuation. The causal logits predicting each answer token are read from
its preceding position with padding masked. Sum and mean scores exclude
EOS. The separately reported terminated sum includes the canonical EOS
token, so it also scores whether the model would stop after the answer.
Greedy generation is unconstrained and scored separately. The parser only
accepts a complete EOS-terminated label or one exact normalized answer from
the frozen alternatives. Valid answers outside the aliases and truncations
count as generation failures under this protocol.

The roster has 48 distinct underlying items, with 12 each for practice,
selection, measurement qualification and sealed adaptation. Each split has
four word-order, four date-order and four fraction-order questions. The
runner chooses between sum and mean on practice, persists that decision, and
then evaluates the chosen rule on independent measurement qualification.
The paired bootstrap samples underlying questions within family. Individual
renderings are related and do not increase the independent sample size.
The protocol also records answer lengths, margin shifts, position and label
counts, completed generation and likelihood–generation agreement.
[Measurement protocol](../experiments/phase_iiic/measurement_protocol.json)

Headroom requires the selected likelihood rule and a completed, parsed
generation to be wrong on all four renderings of an item. An unparsed or
incomplete response alone is insufficient. A family needs at least two
persistent errors in both practice and selection plus solved cross-family
controls. Both measurement splits must pass the frozen measurement gate.
The separate sealed adaptation roster stays with the host and is opened
only after the candidate decisions on selection close.

The new update backend differentiates the target's signed answer score
through the actual answer tokens. It collects sensitivities at a permitted
Qwen projection, projects the gradient to a rank-one update on selected
columns, and tests a fixed schedule of rounded probes at two radii. The
soft objective combines logistic target loss, binary KL against the
parent's preservation scores, and an update norm cost. The guard forbids
target regressions, preservation flips and excessive preservation margin
drops, and requires a positive target gain. The compiler compares the
linear prediction with the measured objective. It rejects a probe with a
negative predicted loss or a large prediction error and can use a smaller
measured probe or abstain. The host validates the proposed patch, reloads
it from disk, and checks that its practice margins agree with the measured
rounded probe.

The single-cycle protocol fixes four arms using the same layer, probe
schedule and patch limits. A and B optimize label scores. B adds the hard
guard. C changes the guarded objective to the qualified content score.
D substitutes count-matched practice controls from a declared protected
family. All arms would face the same independent content and generation
gate. A gain requires new jointly correct items and a gain in completed
generation across all four renderings, with no target regressions or
preservation changes. This protocol is separate from five-cycle
qualification. [Stage 1 protocol](../experiments/phase_iiic/stage_one_protocol.json)

## Practice-only response diagnostics

In the main practice factorial, letter instruction with letter scoring
reached 23/48. The crossed letter instruction with mean content scoring
reached 20/48. Content instruction reached 24/48 with letter scoring and
21/48 with mean content scoring. The point estimate for the instruction
effect was +1/48 at either representation, and mean content scoring was
3/48 lower at either instruction. Mean content scoring was more stable
across option orders, with 15/24 stable pairs compared with 1/24 for
letter/letter. Only 3/12 underlying items were correct in all four
content-mean presentations. Sum scored 23/48, with 1/12 correct in all
four presentations. Both rules failed the frozen practice gate. The
practice decision locked mean as the descriptive contender.

Letter generation completed and parsed on 48/48 practice presentations
and was correct on 23. Content generation reached EOS on 38/48. None
matched the frozen parser. Of those 38 completed outputs, 26 were labels
alone, nine were a label with its matching answer text, and three were
other text. The separate format diagnosis does not change the gate.

The option-display probe held the content instruction, questions and
answers fixed and replaced only the visible option labels with bullets.
On 12 presentations of six practice questions, completed parses rose
from 0/12 to 10/12 and correct generated presentations from 0/12 to 4/12.
No item was generated correctly in both orders under either display.
Mean likelihood accuracy stayed 4/12 and summed likelihood stayed 5/12.
The probe took 64.5 wall seconds and 291.2 CPU seconds.
[Paired option-display evidence](../results/phase-iii-c/option-display-probe.json)

The practice invariance probe found 8/12 changed label choices when the
same examples switched from A/B to X/Y. Changing the printed date and
fraction forms changed 0/8 mean content choices in that probe. The raw
content choice agreed with a neutral no-question prompt on 10/12
presentations. In a post-hoc reanalysis of those six questions, subtracting
the neutral mean score raised correct presentations from 4/12 to 8/12 and
both-order correctness from 1/6 to 4/6. This reuses the measured practice
scores, so it does not provide independent confirmation or affect the
frozen rule. [Neutral-prior reanalysis](../results/phase-iii-c/neutral-prior-analysis.json)

## Independent result and headroom

**The frozen measurement failed qualification. Stage 1 is HOLD.** The
practice-locked mean score met one of six checks on practice and one of
six on qualification. The one passing qualification check was agreement
with generation among eligible parses, based on a single response.

| Frozen condition | Practice | Selection | Independent qualification |
|---|---:|---:|---:|
| Letter instruction, label score | 23/48 | 26/48 | 28/48 |
| Letter instruction, mean content score | 20/48 | 21/48 | 22/48 |
| Content instruction, label score | 24/48 | 25/48 | 22/48 |
| Content instruction, mean content score | 21/48 | 26/48 | 23/48 |
| Letter generation, completed and correct | 23/48 | 26/48 | 27/48 |
| Content generation, completed and correct | 0/48 | 0/48 | 1/48 |

On qualification, mean content scoring was 10.4 percentage points below
letter/letter. The paired question bootstrap interval was −29.2 to +10.4
points. Mean content scoring selected the same answer content in 15/24
reversed-order pairs, compared with 4/24 for letter/letter. Three of 12
questions were scored correctly in all four content-mean presentations.
The higher consistency did not bring higher accuracy. Summed content
likelihood scored 25/48. The diagnostic EOS-terminated sum scored 29/48,
but got all four presentations right on only 2/12 questions. That score
was never eligible for rule selection and is not evidence of generation
improvement. [Frozen measurement records](../results/phase-iii-c/measurement/)

Holding scoring fixed on qualification, changing the instruction moved
letter accuracy down 6/48 and mean content accuracy up 1/48. The paired
interval on their factorial interaction was −12.5 to +43.8 percentage
points. There were zero separate-versus-joint tokenization mismatches
across 1,152 scored continuations. This rules out that particular boundary
mismatch in this run. It leaves response compliance and answer preference
as distinct observed problems.

The model reached EOS on 46/48 content qualification prompts. Of the 46
completed outputs, 32 were labels alone, 11 included a label followed by
the matching text, two were other text, and one matched the requested
answer. On practice and selection there were no eligible content
generations to confirm a persistent wrong answer or a solved control.
Even on likelihood alone, the numbers of all-four-wrong items on practice
and selection were 2 and 1 for word order, 0 and 1 for date order, and 1
and 0 for fraction order. No family met the two-per-split threshold.
This protocol could not establish usable learning headroom.
[Generation failure categories](../results/phase-iii-c/generation-diagnosis.json)

The host ran the actual Stage 1 admission command and recorded zero
candidate attempts. No real weight update or independent adaptation
evaluation ran. The sealed adaptation roster remains unscored and the
archive records its hash. The measurement run took 837.2 wall seconds,
4,832.3 CPU seconds and 6.66 GB peak RSS. The two extra inference probes
used another 801.4 CPU seconds. The inference suite passed 165 tests with
the optional MCP test skipped. The toy BF16 test reproduced measured
margins after saving and reloading the patch. The generation verification
fixture also exercised exact checkpoint lineage and rollback.
[Stage 1 HOLD receipt](../results/phase-iii-c/stage-one/report.json)

For the next measurement experiment, use fresh questions to compare
unlabeled content prompts with the current labeled display, and freeze
raw and neutral-subtracted likelihood candidates before scoring. Require
agreement with completed generation and correctness under both orders.
The exploratory subtraction gain had a six-question paired interval
from −33.3 to +91.7 percentage points. That uncertainty and the absence
of any correct both-order generation in the bullet probe make a new
measurement trial the next justified step. Keep the same strict
independent weight acceptance gates once there is measurable headroom.
