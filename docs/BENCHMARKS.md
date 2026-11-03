# Benchmark status

| Suite / run | Inputs and settings | Independently measured result | Status |
|---|---|---|---|
| v0.1 synthetic demo | 2,000 deterministic examples; seed 813 | Base 75.30%; pre-specified candidate 98.45%; paired 95% interval for gain [21.25, 25.05] percentage points | PASS as tooling demonstration; not agent discovery |
| Published p50c reconstruction | Pinned Qwen3.5-0.8B BF16 checkpoint and authors' p50c sparse artifact | 130,188 changed BF16 values; both target tensor hashes and full candidate file hash matched | PASS for tensor reconstruction only |
| Qwen paired CPU smoke | Two hand-authored prompts; one sample each; exact published sampling except 64-token cap | Base and p50 both 0/2 accurate, 0/2 completed answers and 2/2 token-limit exits; paired accuracy difference 0 | INCONCLUSIVE; outputs truncated |
| Earlier Qwen exploratory smoke | Two hand-authored prompts; one sample each; 256-token cap | Both variants 2/2 cap exits and 0/2 boxed answers; tail-repeat proxy 0.004132 base vs 0 p50 | INCONCLUSIVE; proxy not validated |
| Original synthetic discovery | Three isolated seeds; each has 64 practice and 64 hidden holdout examples | Diagnostic: 50% to 100% on all three; random one-entry control 50%; one output-bias parameter changed | PASS for the injected toy defect only |
| Synthetic MWC | Ten task seeds; 640 paired examples per condition | No edit 50%; memory-only 100%; weight-only 98.125%; memory+weight 100% | PASS as a narrow synthetic pilot; no general MWC claim |
| Conventional optimizer baseline | Full-batch SGD, 400 practice-only steps; five seeds | 50% baseline to 100% candidate accuracy on 320 holdout examples; all 12 toy parameters changed | PASS as toy-only gradient baseline |
| Qwen randomized control | Same 130,188 changed-value count and layers 15/19 | Candidate built; no behavioral measurement | HOLD |
| Published benchmark matrix | MATH-500, GSM8K and IFEval at full sample counts and 32,768-token cap | Not run; no benchmark downloads, no CUDA/vLLM and estimated CPU cost is prohibitive | HOLD |

## Published patch reconstruction details

Pinned model: `Qwen/Qwen3.5-0.8B`, revision
`2fc06364715b967f1860aea9cf38778875588b17`, safetensors whole-file SHA-256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.

| Tensor | Layer | Shape | Target columns | Changed BF16 values | Author edited hash |
|---|---:|---|---:|---:|---|
| `model.language_model.layers.15.mlp.down_proj.weight` | 15 | 1024 × 3584 | 64 | 65,260 | `82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53` |
| `model.language_model.layers.19.mlp.down_proj.weight` | 19 | 1024 × 3584 | 64 | 64,928 | `8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2` |

The author reports 131,072 targeted positions (64 × 1024 × 2). The sparse
artifact contains 130,188 BF16 changes; 884 targeted positions round back to
the base value. WeightBridge's independent reconstruction applied 130,188
sparse set operations through the host patch policy and obtained the same two
tensor hashes and full candidate hash. The candidate model is kept outside the
repository.

## Qwen smoke findings

The current paired smoke uses the source-compatible answer parser, which
ignores the model's reasoning text before `</think>` and rejects an empty
`\boxed{}` answer. All four outputs stopped at the 64-token cap. This is
INCONCLUSIVE, not a zero-accuracy estimate for the full task. The older
256-token exploratory smoke also truncated every output; its repeat-rate
difference is only a diagnostic proxy.

## Runnable entry points

- `python experiments/published_reproduction/rebuild_published_edit.py --base ... --out ...`
- `python benchmarks/runners/run_hf_benchmark.py --help` for the bounded JSON/JSONL runner
- `python experiments/original_weight_edits/run_toy_discovery.py --seed 842` for a new independent toy seed
- `python experiments/memory_ablation/run_mwc_ablation.py`
- `python benchmarks/runners/plot_results.py`

For published dataset preparation, install `.[benchmark]`, use explicit dataset
revisions, then run `prepare_exp8_datasets.py` and `run_published_matrix.py`.
The default matrix uses the published sample counts, temperature 0.6, top-p
0.95, top-k 20, presence penalty 1.5, thinking enabled and a 32,768-token cap.
It expects a local base and reconstructed candidate, a prepared dataset
directory, and a checkout of Google's original IFEval instruction-following
evaluation code. A matrix run has not yet been completed.

The latest ledger-derived aggregate and charts are
[`benchmark_summary.json`](../results/benchmark_summary.json) and the PNGs in
[`results/figures`](../results/figures). Figures are generated from the raw
records and do not include the paper's reported scores as measurements.

The Qwen smoke script loads one model per process and uses local files only.
Run once with the original pinned directory and once with a reconstructed p50c
directory. Each output path must be new.
