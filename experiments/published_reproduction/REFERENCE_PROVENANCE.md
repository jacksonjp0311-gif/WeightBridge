# Published Experiment 8 input provenance

The local reference patch is copied from the authors' `lm_understand-logs`
repository at commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f` (Experiment 8,
run `p50c`). The files are `reference/p50c_manifest.json` and
`reference/p50c_edit.npz`; upstream classifies logs, data, and networks as
CC BY 4.0. Attribution: Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar
Khattab, “Sparks of Weight-Space Programming: LLMs can natively read & write
neural network weights,” November 2, 2026, and the associated public run logs.

The reconstruction script does not execute the authors' code. It reads their
sparse BF16 payloads, validates the pinned base tensor hashes, converts them to
bounded declarative set operations, applies them through WeightBridge's host
policy, then checks the published edited-tensor hashes. The 1.75 GB base and
reconstructed model are not included in this repository.

Model source: `Qwen/Qwen3.5-0.8B`, revision
`2fc06364715b967f1860aea9cf38778875588b17`, model card declares Apache-2.0.
This project does not redistribute the model checkpoint.

The original paper and Experiment 8 evaluation scripts, run logs and patch
manifest were inspected from the source repository above; protocol details
captured locally are in `benchmarks/configs/published_exp8_reference.json`.
EasyEdit was reviewed as a related model-editing baseline framework, but was
not installed or used as an experimental comparison in this execution:
https://github.com/zjunlp/EasyEdit

Dataset sources reviewed: [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500),
[GSM8K](https://huggingface.co/datasets/openai/gsm8k), and
[IFEval](https://huggingface.co/datasets/google/IFEval). The full dataset files
were not downloaded or redistributed. IFEval scoring uses the
[original Google evaluation source](https://github.com/google-research/google-research/tree/master/instruction_following_eval).
