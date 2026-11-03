# WeightBridge: Memory → neural-weight experiment plan

## Guiding hypothesis

A persistent external memory with exact provenance can furnish **candidate neural interventions**, but memory accumulation is not itself weight learning. Evaluated weight changes should be sparse, reversible, independently scored, and attributable to specific evidence. **Hypothesis only**: model-guided edits with purpose-built tools can find useful corrections at lower total cost than naive fine-tuning on selected defect classes.

## Architecture: independent instrument, not a mandatory PEGASUS stage

```text
RecallTree / Bug2Eval / CopyRun (immutable receipts)
       |  eligible, scoped, deduplicated, no personal data by default
       v
Failure cluster + executable failure specification
       |
       v
Weight diagnostic agent  <-- inspect / tensors / forward-probe / ablation
       |                     (all bounded, logged, no test-set answers)
       v
PatchProposal (hypothesis, exact base hash, allowed coordinates)
       |
       v
Host-owned PatchGate --> isolated new checkpoint, never in-place
       |
       v
Independent evaluation (frozen practice/holdout, controls, robustness)
       |
       v
Genome candidate/evaluation record → operator approval → adoption / rollback
```

## Work packages (incremental)

1. **v0.1, implemented here**: .npz/unsharded safetensors reader, bounded sparse patch, exact hashes, checkpoint immutability, synthetic paired test. Does not contain a model agent.
2. **v0.2, proposed**: adapter for Qwen3.5-0.8B with sharded safetensors, tokenizer/chat template pinning, inference batch tool and budget; compare the author's publicly released sparse edit against unedited checkpoint (score **independently** using identical sampling). Add pass/budget receipts; never mix benchmark/test samples into diagnosis.
3. **v0.3, proposed**: rank-1 and directional projection edit primitives; read-only activation traces, attention maps, candidate causal locators, loss-free ablation; controls: random same-budget patch, equal-compute black-box search, LoRA/gradient-based and no-edit baselines.
4. **v0.4, proposed**: MemoryBridge adapter for sanitized, receipt-verified, user-authorized Pegasus observations; cluster repeated errors, label them verified/unresolved, prevent duplicated runs from counting as independent evidence; no auto-ingestion of private notebooks.
5. **v0.5, proposed**: stricter evaluation/Genome interface: multiple independent seeds, real holdouts, related/unrelated capability tests, calibration, refusals, memorization/leakage checks, confidence intervals, held/rejected candidates and cost accounting. Add restart-persistence test and rollback rehearsal.

## Measurement

Let \(W\) denote base weights and \(\Delta\) proposed changes. For a task family, compare the same seeded tasks under base/candidate/controls:

\[
\underset{\Delta}{\operatorname{maximize}}\quad
Q_{\mathrm{heldout}}(W+\Delta)-Q_{\mathrm{heldout}}(W)
-\lambda\Vert\Delta\Vert_0
-\mu\,\mathrm{Regress}(W,\Delta)
-\nu\,\mathrm{TotalCost}(\Delta)
\]

This is an *evaluation objective*, not an implemented differentiable optimizer. \(Q\) includes correctness, repeatability, instruction compliance and other task-specific metrics; \(\mathrm{Regress}\) measures worst-case/critical capability degradation, not just mean score. Choose and freeze scoring and thresholds **before** revealing test outcomes to the editor. Repeated exploration on the same holdout contaminates the holdout; keep a final independent test set for selection-free confirmation.

## Interfaces to existing PEGASUS

- **RecallTree:** the existing host selects eligible evidence; do not give WeightBridge arbitrary project or private-agent path arguments.
- **MemoryAtlas:** display weight-proposal provenance and lineage from recorded links only; no inferred causality or approval.
- **ExperimentForge:** independent experiment runner; numerical synthetic evaluation is not proof for a production model.
- **Genome:** store proposals as unapproved records. A numeric patch cannot authorize tools or bypass an operator's Ask/Allow/Block control.
- **CopyRun:** preserve model source hash, weights hash, patch manifest hash, exact prompts, settings, seeds, seeds for stochastic sampling, inference passes, error codes, resource cost and rollback result.

## Critical negative controls and limits

- **Defect removal is not capability acquisition.** The cited authors show Qwen's repetition-loop fix but could not establish similar improvement for a Gemma model without those loops. Treat novel-capability claims as separate H2 tests.
- Separate weight-space reasoning from tool-advantaged conventional optimization; compare fair compute/token/eval budgets.
- Do not give the diagnostic editor the final holdout. Never select a winner by repeatedly checking the holdout and reporting its nominal confidence interval.
- Restrict full model output or raw private RecallTree content from logs; sanitize data and honor model/data licenses.
- Small sparse patches can still cause broad degradation, malicious backdoors, or brittle behavior; require locality tests, canary tests, stable source hashes and explicit activation authorization.
- CPU-only tool inspection does not mean benchmark inference is cheap; record actual runtime/energy and model-serving cost.

## First research success condition

Reproduce the Qwen edit's qualitative loop reduction with exact public checkpoint/patch and pinned settings; compare with an equal-cost no-edit, random-patch and a conventional repair baseline; prove the candidate persisted after independent restart and could be rolled back. Only then try AI-generated **new** patches on previously unseen defects, using a separate frozen holdout.
