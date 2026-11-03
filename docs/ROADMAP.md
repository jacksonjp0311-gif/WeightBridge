# WeightBridge research roadmap

## Current milestone disposition

| Milestone | Status | Evidence |
|---|---|---|
| 1. Preserve and reproduce v0.1 | COMPLETE | Baseline commit `321547d`; 11 original tests passed; v0.1 demo rerun |
| 2. Real-model inspection and patch reconstruction | COMPLETE for local safetensors/Qwen path | Sharded/unsharded inspection and host-policy patcher; p50c hashes match |
| 3. Bounded original-versus-published benchmark | PARTIAL / HOLD | Small CPU smoke completed but truncated; published datasets/scores not run |
| 4. AI-directed diagnosis and candidate edit | PILOT COMPLETE / TRANSFER HOLD | Deterministic worker cohort plus one Codex MCP discovery on a hidden-bias task; no Qwen proposal |
| 5. Memory-guided diagnosis | PILOT COMPLETE / GENERAL CLAIM HOLD | Four-condition synthetic ablation across ten task seeds |

## Immediate next experiment

Run the amended 20-instance, four-mechanism LLM study in
[`KNOWLEDGE_LEDGER.md`](KNOWLEDGE_LEDGER.md). The earlier v1 plan targeted the
deterministic worker and remains recorded as unrun. For v2, freeze and hash the
LLM model/version, tool prompt, MCP host policy, task generation,
practice/holdout split, eight-candidate budget, evaluator and matched random
control before exposing the agent. The agent must not see hidden labels. Keep
the original gate: at least 70% holdout repairs, at most 5% regression among
baseline-correct items, and at least a 20 percentage-point advantage over
matched random edits with a task-seed-clustered paired 95% lower bound above
zero. Archive every call, rejection, failure and candidate receipt. The MCP
tool adapter must be expanded to real Hugging Face inference only after this
synthetic transfer gate and the Qwen resource gate are satisfied.

## Qwen benchmark gate

Do not start the full Experiment 8 matrix until all of these are true:

1. The original model and p50c reconstruction are pinned and verified again in
   the intended runtime.
2. Dataset dependencies and exact frozen benchmark data versions are prepared;
   hashes, splits, grading code and license terms are recorded.
3. A representative time and memory estimate supports the planned samples,
   32,768-token cap and 10/25/50-pass experimental history without altering
   the author configuration silently.
4. Base, p50c, matched random and an appropriate conventional baseline share
   the same frozen prompts, sampling settings and paired seeds.
5. Practice problems remain separate from final holdout problems; no outcome
   from the final holdout feeds patch selection.

After the gate, first run a predeclared paired subset with the exact author
settings and enough tokens to finish. Expand only when the recorded runtime
supports it. The full paper result is not reproduced until each suite and
sampling count match the published protocol and independent metrics are
recorded.

## Further work after the gates

- Extend policy-safe patches with optional bounded low-rank edits and compare
  them with the current sparse edits on a task with a frozen evaluator.
- Add multi-writer-safe append semantics, ledger schema migration and
  artifact-content verification for every referenced path.
- Probe activation/attention summaries on Qwen only after a benchmark failure
  is observable and the probe can be tied to a testable hypothesis.
- Test memory retrieval with related and mismatched experiences, equalized
  evaluator calls, multiple defect families and a checkpoint-restart boundary.
- Study edit magnitude and parameter count only with deliberately varied
  interventions and enough independent defects to estimate a response curve.
- Keep integration with RecallTree, MemoryAtlas, Genome, ExperimentForge and
  private PEGASUS data out of scope until separately authorized and backed by
  these isolated results.
