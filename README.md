# WeightBridge v0.1 — evidence-gated neural weight editing

A **working, isolated proof-of-concept** for reading model tensors, accepting an AI-proposed **sparse numerical or targeted directional patch as data**, validating the patch against a host-controlled policy, writing an immutable candidate and comparing the result on a frozen holdout. It runs fully offline on CPU.

**Scope:** reproducible sandbox, not a method for teaching arbitrary LLMs new capabilities. The included synthetic classification task simulates a localized failure. Its successful edit is pre-specified to demonstrate patch application and evaluation; **an AI did not discover it**. No source in this package changes PEGASUS, ChatGPT, your installed models or their memories.

## Quick start

```bash
python -m pip install -e '.[test,safetensors]'
python -m weightbridge demo --dir ./demo-run
python -m weightbridge inspect ./demo-run/base.npz
python -m pytest -q
```

The demo creates `base.npz`, `patch.json`, `policy.json`, `candidate.npz`, `candidate.npz.receipt.json` and `evaluation.json`. The base checkpoint is never overwritten; a successful evaluation never deploys a model or claims learning.

## Give an AI these tools

**Read-only:**
- `inspect` — tensor names, shapes, parameter counts, sampled statistics, checkpoint hash.
- Next: activation probes, rank/spectral summaries, behavioral probes through an isolation runner.

**Candidate-only:**
- `apply` — take a JSON patch (individual parameter edits or a targeted directional projection), enforce the allowlist, parameter-change budget and exact base SHA-256; produce a new checkpoint and receipt.
- `compare-demo` — compare the original and candidate on the **same** frozen synthetic examples.
- Next: editable tensor slices, low-rank and orthogonal projection edits, sharded safetensors adapters, independent benchmark runner.

**Never grant an agent direct production checkpoint write access.** The host controls paths, policy, test data, promotion and rollback. A patch is a proposal, not authorization.

### Patch contract

```json
{
  "schema": "weightbridge.patch.v1",
  "base_sha256": "<exact SHA-256 of checkpoint>",
  "reason": "A specific causal hypothesis, not a verified fact",
  "evidence_ids": ["receipt-or-trace-id"],
  "changes": [{"tensor": "b", "index": [1], "op": "set", "value": 0.0}]
}
```

Policy (host-owned, **not chosen by the AI**):

```json
{
  "schema": "weightbridge.policy.v1",
  "allow_tensors": ["b"],
  "max_entries": 1,
  "max_fraction": 0.20,
  "max_abs_delta": 2.0
}
```

Optional projection primitive: `"projections": [{"tensor":"M", "op":"remove_column_direction", "columns":[3], "direction":[1.0, 0.0, 0.0], "strength":1.0}]`. The direction length must equal the matrix row count; this removes its normalized component only from specified columns, with every touched entry counted against the host budget. This is an experimental operation, not a guaranteed behavior repair.

Apply an externally proposed patch to a **different, new file**:

```bash
python -m weightbridge apply --base demo-run/base.npz \
  --patch demo-run/patch.json --policy demo-run/policy.json \
  --out demo-run/independent-candidate.npz
python -m weightbridge compare-demo --base demo-run/base.npz \
  --candidate demo-run/independent-candidate.npz --out demo-run/independent-eval.json
```

Supports small unsharded `.safetensors` and NumPy `.npz` files; it does **not** yet patch full sharded Hugging Face checkpoints or support Qwen/Gemma decoding. For large models, reading all weights to inspect them is currently memory-intensive; build a lazily loaded architecture adapter before real-model trials.

## What this builds toward

- RecallTree provides eligible, receipt-backed observations. Do not train on or export private memory without separate permission.
- A diagnosis agent proposes a *hypothesis* about a neural mechanism associated with a repeated failure.
- A weight-inspection agent gets scoped tensor summaries and controlled activation/forward-pass probes.
- The agent proposes edits; the **host**, not the AI, applies policy limits and writes candidates.
- ExperimentForge compares baseline, candidate and strong baselines (LoRA, MEMIT/ROME where applicable) on *fresh held-out tasks*, locality tests, adversarial cases and budgets.
- Genome stores separate candidate/evaluation/approval/release/rollback records.
- Only an explicit operator-approved promotion can alter which model the host starts. The main model is immutable, and rollback is a pointer to the known original checkpoint, not a speculative reverse edit.

Review `docs/RESEARCH_PLAN.md` for the experimental roadmap and failure modes.

## Source inspiration and independence

- Takehi, Teoh, Zhang and Khattab (Nov 2, 2026), *Sparks of Weight-Space Programming*: https://rikiyat.github.io/blog/weight-space-programming/
- Original publicly released logs, not included or copied here: https://github.com/RikiyaT/lm_understand-logs
- Related weight-editing baselines: https://github.com/zjunlp/EasyEdit

This code is original research scaffolding. It is not an independent reproduction of the paper's benchmark results.
