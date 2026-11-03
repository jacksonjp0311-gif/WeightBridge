# AI-directed Qwen weight edit pilot

This pilot tests whether a local Qwen3.5-0.8B controller can choose useful
weight edits from practice feedback. The controller receives the damaged
model's practice score and the results of earlier proposals. It chooses the
layer and strength of each new proposal using its logits over allowed
options. A fixed compiler fits the rank-one direction from the damaged
model's practice activations. The controller chooses which edits to try.
The host selects the finalist from measured practice results.

The manifest fixes three fresh faults, one per layer, and six proposals
per case. Each control is matched to the corresponding AI-selected
proposal in layer, columns, rank and norm. Each arm uses the same number
of practice forward batches. The search receives practice scores only.
A separate evaluator reloads the saved finalist, verifies its receipt,
and scores the sealed holdout. The gate requires at least half of lost
target accuracy recovered, a strict advantage over the matched random
finalist, no paraphrase regression, and preservation of every pristine
sentinel prediction.

The pilot uses a CLI experiment and a constrained AI controller. The
existing [direct-proposal qualification](AGENT_TOOLS.md) tests a separate
workflow where an AI writes its own patch. The [roadmap](ROADMAP.md) tracks
the gates for exposing real checkpoints through MCP.

## Results

| Injected layer | Pristine | Damaged | AI-directed edit after reload | Matched random | Sentinel flips vs. pristine | Gate |
|---|---:|---:|---:|---:|---:|---|
| 19 | 24/24 | 19/24 | 24/24 | 19/24 | 2/16 | FAIL |
| 21 | 24/24 | 20/24 | 22/24 | 24/24 | 3/16 | FAIL |
| 23 | 24/24 | 21/24 | 24/24 | 21/24 | 0/16 | PASS |

The passing case recovered all three lost target answers and preserved
all 16 unrelated predictions relative to both pristine and damaged
checkpoints. Its paraphrase split scored 12/12 before and after the edit.
An independent run of the product evaluator reproduced the gain on the
saved candidate. The 19th-layer case also recovered every lost target
answer, but two sentinel predictions still differed from pristine.
In the 21st-layer case, random scored higher on targets and the
AI-directed edit lost one paraphrase answer. One of three cases cleared
the gate. The result establishes a working constrained edit loop for one
injected fault on this checkpoint.

[Raw scores, controller decisions and patch receipts](../results/qwen-ai-edit-pilot-v1/)
· [Product evaluation report](../results/qwen-ai-edit-pilot-v1/product-evaluation/report.json)

## Replay

Use the [pinned local Qwen checkpoint](QWEN_BLIND_REPAIR.md) and an inference
Python environment. Use new scratch and evidence directories.

```powershell
$runner = 'experiments/original_weight_edits/run_qwen_ai_edit_pilot.py'
$base = 'C:/path/to/pinned/Qwen3.5-0.8B'
$run = Join-Path $env:TEMP 'weightbridge-ai-edit-replay'
python $runner run --pristine $base --run $run
python $runner export --run $run --out results/qwen-ai-edit-pilot-replay
python experiments/original_weight_edits/verify_qwen_ai_edit_pilot.py results/qwen-ai-edit-pilot-replay
```
