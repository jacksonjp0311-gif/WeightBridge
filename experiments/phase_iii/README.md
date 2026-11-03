# Natural Qwen adaptation study

This study starts from the pinned, unmodified Qwen3.5-0.8B checkpoint. The
practice screen measures answer choices from that checkpoint on authored prompts.
The study chooses natural failure families with observed errors and mixed
predictions. The solver derives its candidate corrections from practice responses.

The pilot uses word and calendar ordering. The independent five-cycle study
adds fraction comparison and revisits the first two families with fresh inputs.
Each cycle has 12 practice targets, 12 practice sentinels, and a disjoint
evaluation set of the same size. The final suite has 8 fresh targets per
family and 12 sentinels. The answer labels are balanced in every family and
split. A sentinel measures prediction preservation, even when the original
prediction is incorrect. The prompts vary examples within a family and keep
the wording template fixed. The reported outcome is restricted to next-token
choice scoring.

## Run against frozen inputs

Use a Python environment with `pip install -e ".[inference,safetensors]"`.
Use the archived `material` directory for byte-exact replay of an existing
protocol. The base argument is a local Hugging Face model directory with the
same model assets and SHA-256 pinned in `material/protocol.json`.

```bash
python experiments/phase_iii/run_natural_qwen_study.py run --base /path/to/base --materialized /path/to/material --out /path/to/new-run
```

The runner starts separate search and verification processes. Search is given
the practice data and intent. Verification receives frozen holdout paths after
search closes. Use host-owned permissions to keep held-out files inaccessible to
untrusted agent accounts. Each cycle has two candidate slots. The controller
tries the first strength at each of the first two allowed tensors in intent
order, subject to the same-family tensor preference from previously accepted
cycles. The pilot starts at 0.03 each cycle. The five-cycle protocol alternates
0.03 and 0.07.

The host can measure the frozen first proposal, a seeded output-row permutation
and a two-step gradient baseline after all cycles finish. The gradient
comparator optimizes two rank-one factors from practice data. The random arm
preserves the requested norm and coordinates, and the exact realized change is
measured after bfloat16 rounding. These arms share the cycle's original base
and evaluation inputs. The host records them separately from the adaptation lineage.

```bash
python experiments/phase_iii/run_natural_qwen_controls.py --base /path/to/base --materialized /path/to/material --study-run /path/to/new-run --frozen-config benchmarks/configs/phase_iii_natural_qwen_v1_qualification.json --out /path/to/new-controls
```

The `natural_qwen_suites.py` generator can screen practice families or create
new frozen material. The archived material preserves the original JSON bytes
across operating systems, including line endings. A source preregistration
records the protocol digest and implementation hashes. Use `freeze-pilot` with
`--families word-order date-order`, or `freeze` with `--families word-order
date-order fraction-order word-order date-order`, and fresh output paths. Treat
the released prompts as public after publication. A new study of selection
requires fresh sealed inputs.

The portable export keeps exact patch, receipt, prediction and suite evidence.
Patch files use gzip and decompress to the source bytes. The manifest records
both hashes. A portable copy of metadata that embeds a local path has its own
hash and the original record hash. Model shards remain in local storage.
