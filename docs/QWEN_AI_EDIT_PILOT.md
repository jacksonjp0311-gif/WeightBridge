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
