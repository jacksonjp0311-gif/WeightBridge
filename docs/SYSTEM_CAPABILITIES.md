# Captured system capability and compatibility report

Captured 2026-11-03 UTC during the WeightBridge v0.2 audit. Resource
availability values are snapshots and can change.

| Resource / runtime | Observed capability |
|---|---|
| OS | Windows 11 Pro, build `10.0.26200.0` |
| CPU | AMD Ryzen AI MAX+ 395, 16 physical cores / 32 logical processors |
| RAM | 132,766,306,304 bytes total; 83,966,746,624 bytes available at capture |
| GPU | AMD Radeon 8060S integrated graphics; adapter reports 4,293,918,720 bytes shared memory; dedicated VRAM measurement unavailable |
| CUDA | Unavailable |
| Storage | C: volume had 499,941,556,224 bytes free at capture |
| Project Python | 3.12.10, NumPy 2.5.4, pytest 9.2.0, safetensors 0.8.0, psutil 7.2.2; optional Matplotlib 3.11.2 for figures |
| Inference Python | 3.12.10, PyTorch 2.15.0+cpu, Transformers 5.22.1, safetensors 0.8.0, psutil 7.2.2 |
| Optional runtimes | vLLM absent; CUDA absent; `causal_conv1d` absent; `flash-linear-attention` absent |
| Benchmark tools at capture | `datasets` and `math-verify` absent from the project environment |
| Model inference | CPU-only Transformers fallback attention; about 10 output tokens/s in the bounded Qwen smoke |

## Pinned model accounting

The pinned model is `Qwen/Qwen3.5-0.8B`, revision
`2fc06364715b967f1860aea9cf38778875588b17`. The downloaded single safetensors
file is 1,746,942,600 bytes with SHA-256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
The checkpoint tensor map contains 873,438,784 entries including
20,452,864 MTP auxiliary entries. The Transformers inference module instantiates
852,985,920 parameters because that runtime omits the checkpoint-only MTP
auxiliary tensors. This accounts for the two parameter totals seen in reports.
The checkpoint and derived candidates were kept in the external temporary
cache and were not committed.

Machine-readable capture: [`results/system_capabilities.json`](../results/system_capabilities.json).
