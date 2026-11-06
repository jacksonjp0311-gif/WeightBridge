# Phase S command path

This is the command path implemented by the frozen Phase S code, not an
arbitrary-intent training API. Run from the repository root in the recorded
runtime. The examples use host-supplied paths and do not expose private seeds,
rosters or responses. Do not run them against an already executed study: claims
and evidence are write-once, and implicit retry/resume is refused.

## Versioned task and explicit parent

The task specification is the complete arm `protocol.json`, schema
`weightbridge.phase_iiis.protocol.v1`, together with its hash-bound task manifest,
data bundle and authorization. Its relevant fields include:

```json
{
  "task_contract": "weightbridge.tool_task.v1",
  "tool_completion": {
    "schema": "weightbridge.native_tool_completion.v1",
    "end_token_id": 248046,
    "end_token_surface": "<|im_end|>"
  },
  "checkpoint_sha256": "<the declared arm parent>",
  "task_manifest_sha256": "<the frozen population manifest hash>",
  "source_sha256": {"<source path>": "<checked-out byte hash>"}
}
```

This fragment is explanatory, not a substitute for a complete authorized
protocol. The sequential parent is verified R8
`2d9102982dc0f0fdf3397308cd397bc2d5239eeb99ef6297e5f09f6f6deba158`;
the original-joint parent is
`7d4532858cf839d05ec34da088114df47238322f86f5e6e5c6693594a21a03b8`.
The supplied checkpoint is inspected and must match its declared identity.

[`prepare_task_completion`](../../weightbridge/adaptation/task_completion.py)
dispatches only explicit `task="tool"` or `task="decimal"` rows. Tool rows supply
a visible prompt, validated fixture and gold action, and canonical JSON
`completion`. Training includes every answer token and the native end token;
prompt and padding labels are masked. Decimal rows retain the historical
leading-space answer and newline endpoint. The evaluator does not receive a
calculator or gold-answer decoding constraint.

A new task requires an authored, tested schema, visible semantics, independent
oracle/simulator, completion contract, calibration and fresh authorization.
Changing prose or adding an unknown task name is not a supported route to a
qualified new capability. There is no `--task-spec` shortcut: the actual worker
accepts `--protocol`, and the host runner resolves that protocol from `--study`.

## Freeze, then run the declared arms

`freeze_study` requires already completed disposable calibration and timing,
validated historical exclusions and a compatible inventory. It checks resource
feasibility before creating fresh host-private populations. It does not perform
those preflight model calls itself. Public redacted manifests cannot replace
the private input evidence. Use a new approved root for a new replication;
never regenerate the current sealed study to obtain different questions.

```powershell
$Python = '<recorded runtime Python executable>'
$Preflight = '<new host root with completed measured preflight>'
$Original = '<verified original checkpoint directory>'
$R8 = '<verified rounded R8 checkpoint directory>'
$Inventory = '<validated per-parent inventory JSON>'
$Study = Join-Path $Preflight 'study-v1'
$Ledger = Join-Path $Preflight 'execution'

& $Python -m experiments.phase_iiis_tool_transfer.freeze_study `
  --preflight-root $Preflight --base $Original --historical-inventory $Inventory

& $Python -m experiments.phase_iiis_tool_transfer.host baselines `
  --study $Study --original $Original --decimal-parent $R8 --ledger $Ledger

& $Python -m experiments.phase_iiis_tool_transfer.run_arm `
  --study $Study --base $R8 --ledger $Ledger --arm sequential
& $Python -m experiments.phase_iiis_tool_transfer.host assess `
  --study $Study --arm sequential

& $Python -m experiments.phase_iiis_tool_transfer.run_arm `
  --study $Study --base $Original --ledger $Ledger --arm original-joint
& $Python -m experiments.phase_iiis_tool_transfer.host assess `
  --study $Study --arm original-joint
```

Run arms sequentially. They share one cumulative resource ledger. Each runner
executes the disposable smoke, validates its fresh reload, starts one main
training attempt, and measures the two predeclared development checkpoints.
Only step 128 can become a finalist; step 64 is diagnostic. Assessment does not
authorize opening a protected bank or promote a checkpoint.

The internal training worker's real interface is below. The host runner supplies
its authorization, bundle, smoke gate and enclosing resource monitor. Invoking
the worker directly is not a replacement for those host checks:

```powershell
& $Python -m experiments.phase_iiis_tool_transfer.train `
  --base $ActualParent --protocol $ArmProtocol --authorization $TrainingAuthorization `
  --bundle $TrainingBundle --out $NewTrainingOutput --mode main --smoke-gate $SmokeGate
```

## Stored checkpoint evaluation and host stage order

The reusable evaluation path takes the versioned protocol and explicit stored
checkpoint and roster identities. The following is the existing worker syntax
inside the resource monitor; `$Stage`, authorization and limits must already be
allowed by the host. No adapter arguments are used for merged-model evidence.

```powershell
& $Python -m experiments.phase_iiis_tool_transfer.execute `
  --ledger $Ledger --name $UniqueEvaluationName `
  --module experiments.phase_iiis_tool_transfer.evaluate `
  --wall 1800 --cpu 12000 --memory 8589934592 -- `
  --base $StoredCheckpoint --protocol $ArmProtocol --rows $AuthorizedRoster `
  --out $NewWorkerReceipt --authorization $EvaluationAuthorization `
  --expected-checkpoint $StoredCheckpointSHA --expected-rows $RosterSHA `
  --stage $Stage --task tool
```

Use `--task decimal` for the unchanged legacy endpoint. Stage names are not
permissions: an authorization must bind the exact protocol, stage, roster and
representation. The worker verifies its actual `--base` before inference. The
shared ledger can refuse a launch even when these per-worker caps are satisfied.

The host then follows this order, without fitting or choosing another candidate:

1. Validate development, merge an eligible step-128 adapter into a new BF16
   directory, and recheck merged development. Reconstruct separately, compare
   exact stored hashes and native responses, and replay the actual parent.
2. Record each arm's finalist or terminal failure, then lock **both** decisions.
   Existing evidence operations use `python -m
   experiments.phase_iiis_tool_transfer.host request --file PRIVATE_REQUEST.json`
   with schema `weightbridge.phase_iiis.host_request.v1`. The host supplies
   reviewed request fields and receipt bindings; this guide does not fabricate
   a lock or merge authorization.
3. For each locked eligible finalist, use the prospectively pinned
   [supplemental protected host v3](host_tools/protected-v3/README.md), in order:
   historical retention, qualification, confirmation, finalize. Historical R
   Bank A is retention only; R Bank B remains inaccessible. Qualification and
   confirmation are separate fresh banks. Confirmation requires a completed,
   passing qualification transaction and its exact replay.

```powershell
& $Python results/phase-iii-s-tool-transfer/host_tools/protected-v3/protected_host_v3.py `
  --request $HostApprovedPrivateRequest
```

Each private request has schema
`weightbridge.phase_iiis.protected_host_request.v3`, the driver's exact SHA-256,
one explicit action and checkpoint/protocol/ledger bindings. Execute one request
at a time. The published driver retains audited local paths; another machine
requires a reviewed path adaptation and new source hash, not an assertion of
byte-identical reproduction. Interrupted or failed transactions cannot open the
next bank. Missing stages remain unmeasured, never PASS. Finalization grants no
production or lineage authority.

Public archive hashes can be checked using the commands in the
[research report](../../docs/PHASE_III_S_TOOL_TRANSFER.md#tests-evidence-and-reproduction).
For an offline illustration after study finalization, the separate
[public demo](public-demo-v1/README.md) prepares the same bounded evaluator at
`stage=demo` against an already merged, reconstructed artifact. It accepts no
adapter-only fallback and is not independent qualification.

## Publish only explicitly released host metadata

After the host closes a protected action, publish a new aggregate snapshot.
Never pass a protected roster, model worker output or private request to the
reporter. The following shows one completed qualification transaction; the host
chooses the exact arm and already authorized receipt paths:

```powershell
$Arm = 'sequential'  # Or 'original-joint'; no automatic bank discovery.
$GateRoot = "$Arm/protected-v3"
$NewPublicSnapshot = '<new versioned public output directory>'
& $Python results/phase-iii-s-tool-transfer/export_outcome.py `
  --study $Study --ledger $Ledger --out $NewPublicSnapshot `
  --gate "$Arm-history=$GateRoot/historical-retention/gate.json" `
  --gate "$Arm-history-transaction=$GateRoot/historical-retention/completed.json" `
  --gate "$Arm-qualification=$GateRoot/qualification/gate.json" `
  --gate "$Arm-qualification-replays=$GateRoot/qualification/replays.json" `
  --gate "$Arm-qualification-transaction=$GateRoot/qualification/completed.json"
```

Additional explicitly named metadata follows the same `--gate LABEL=RELATIVE`
interface. All paths below are relative to `$GateRoot`:

| Observed outcome | Metadata paths to include |
|---|---|
| Interrupted/failed host action | `<action>/failed.json`; include an already scored `gate.json` and `replays.json` only if they actually exist |
| Completed confirmation | `confirmation/gate.json`, `confirmation/replays.json`, `confirmation/completed.json` |
| Finalization | `finalize/final-verification.json`, `finalize/formal-stage-closure.json`, `finalize/completed.json` |
| Interrupted finalization | `finalize/failed.json`; no completed-final claim |

Use `failed.json` instead of `completed.json` for an interrupted transaction,
not a manufactured completion receipt. A completed transaction can also report
a scientific FAIL. A behavioral gate's `scored_gate_passed` is separate from
the enclosing transaction's `completed`, `passed` and `exact_replays_passed`.
Unopened confirmation remains null. The reporter preserves those distinctions,
redacts private paths/outputs/IDs, and does not independently rerun the scientific
verifier. For two arms, explicitly add the second arm's existing metadata with
different labels; no protected content is auto-discovered. Costs come once from
the shared execution ledger, not from adding host elapsed time to worker time.

Check the new public snapshot without reading private evidence:

```powershell
& $Python results/phase-iii-s-tool-transfer/export_outcome.py --self-test
& $Python results/phase-iii-s-tool-transfer/export_outcome.py --verify $NewPublicSnapshot
```

For a smaller historical-only supplement, the completed gate, transaction and
matching closed metadata review must already exist. This command does not read
training logs, worker responses, rosters or later banks:

```powershell
& $Python results/phase-iii-s-tool-transfer/export_outcome.py `
  --study $Study --historical-supplement sequential --out $NewPublicSnapshot
```

Use `original-joint` only after that arm's separate historical transaction and
audit close. A failed or absent binding is refused before publication.
