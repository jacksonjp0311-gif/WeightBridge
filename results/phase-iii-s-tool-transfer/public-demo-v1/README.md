# Public offline illustration, prospectively frozen

**Model execution: NOT_EXECUTED. Model result: null.** Fixture and tokenizer
checks have passed; those checks are not model-generated successes.

This small public population contains five counterfactual families and forty
prompts: two factual variants, two frozen wordings and two inventory/registry
orders. Each of `search_files`, `read_file`, `run_tests`, `clarify` and `no_action`
is correct on eight presentations. The model receives the prompt, not its gold
action. Gold actions are used only by the frozen post-generation parser and
in-memory simulator. No generated action invokes a real shell, file, test runner
or network operation.

The rows were generated from the explicitly public seed recorded in
[manifest.json](manifest.json), then namespaced with `public-demo-`. That prefix
cannot occur in the formal generator's two-word numeric slugs. No private
roster, private generation seed or private output was read or copied. The
wording is the existing frozen `confirmation` wording; new identifiers in that
wording are not a new linguistic-transfer test. These are illustrative public
fixtures, never a qualification bank or training population.

The roster SHA-256 is
`9cf02001f5e202d76c2befc64f37ba7a23412ce6212d7988bdf21bc09ecdc8f9`.
The frozen format example is identical to the experiment's example. Local
tokenizer-only validation measured 336–425 native prompt tokens and at most
30 canonical-answer tokens including `<|im_end|>` (ID 248046). The unchanged
endpoint permits 512 prompt tokens and 96 generated tokens. Greedy inference
remains unrestricted; there is no repair, constrained decoding, logit bias,
answer extraction or fallback.

Run source, roster and oracle checks from the repository root:

```powershell
python results/phase-iii-s-tool-transfer/demo.py --self-test
python results/phase-iii-s-tool-transfer/demo.py `
  --verify results/phase-iii-s-tool-transfer/public-demo-v1
```

These commands do not load model weights or run inference. The source manifest
uses the same checked-out byte conventions as the main Phase S report.

## Host-controlled model execution, only after study finalization

Do not run this during candidate fitting or selection. Use only an already
materialized merged checkpoint whose exact reconstruction and locked identity
were verified by the existing study. An independently rejected candidate may
still be illustrated if that exact stored artifact exists; label its scientific
failure prominently. Never merge an ineligible adapter merely to create a demo.
If no permitted stored candidate exists, this illustration remains NOT_EXECUTED.

The helper checks the existing arm lock and its recorded merge/reconstruction
evidence. It accepts no adapter argument and does not build or execute a model.
The host supplies the actual private artifact paths after finalization:

```powershell
$Python = '<recorded local runtime Python>'
$Protocol = '<finalized arm protocol.json>'
$Lock = '<existing arm lock JSON>'
$Merged = '<already verified merged checkpoint directory>'
$Ledger = '<existing shared Phase S execution ledger>'
$DemoOutput = '<new isolated demo evidence directory>'

& $Python results/phase-iii-s-tool-transfer/demo.py `
  --prepare results/phase-iii-s-tool-transfer/public-demo-v1 `
  --protocol $Protocol --lock $Lock --checkpoint $Merged `
  --ledger $Ledger --out $DemoOutput --study-finalized

# This next command performs real inference; do not run before host finalization.
$Plan = Get-Content -LiteralPath (Join-Path $DemoOutput 'command.json') -Raw | ConvertFrom-Json
& $Plan.argv[0] $Plan.argv[1..($Plan.argv.Count - 1)]
```

The command uses the existing `experiments.phase_iiis_tool_transfer.execute`
resource monitor and the frozen evaluator at `stage=demo`, with explicit source,
roster and stored-checkpoint hashes. The proposed enclosing limits are 300 wall
seconds, 2,000 observed CPU seconds and 8 GiB conservative process-tree memory.
The shared eighteen-hour ledger remains authoritative; these limits create no
extra global allowance. No automatic retry or overwrite is allowed.

The evaluator writes the actual full generated output, native tokens,
completion reason, strict parser result and simulated outcome for each of these
public cases. The immutable fixture manifest stays unchanged. Save any eventual
model outcome in a separate versioned result; report all forty cases rather than
selecting a successful example. A successful illustration is neither a new
independent qualification nor permission to deploy or admit a checkpoint.
