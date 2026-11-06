# Prospective protected-stage host driver

This separately hashed source snapshot supplements the frozen 162-file Phase S
scientific implementation. It was prepared and tested before either new formal
bank was opened. It changes no training, examples, scorer, finalist, endpoint, or
acceptance threshold. The model is executed only by the existing S evaluator and
shared resource monitor. Source publication is not a successful stage execution.

`protected_host_v1.py` has four explicit, separately requested actions:

1. `historical-retention`: validate both S arm decisions, then release the fixed
   old R rank-eight battery through the pinned R validators and the previously
   published bridge. Measure each locked candidate, its reconstruction and its
   own parent on the original ordered rows. R Bank B is excluded.
2. `qualification`: require the actual historical retention gate and all earlier
   prerequisites. Compare original, R8 and the locked stored candidate on the
   frozen new qualification bank. Perform fresh full-output reconstruction and
   own-parent replay of that same opened stage, including when its behavioral
   gate fails.
3. `confirmation`: additionally require the qualification outcome and its exact
   replay to recompute successfully. A top-level success flag is insufficient.
4. `finalize`: call the frozen S final verifier with the actual opened stages and
   recorded replays. Missing independent stages remain absent and cannot pass.

There is no action that trains, refits, redraws, picks another checkpoint, or
automatically opens the next bank. The operator must invoke each action only
after its prerequisites. A failed arm does not prevent another locked, qualified
arm from continuing under its own unchanged contract.

Parent workers may be cached only for the same bank bytes, scientific source,
task manifest and endpoint. Their original protocol remains in each WorkerSpec.
Fresh replay loads the reconstructed candidate directory or the exact actual
parent directory, checked against the recorded process command's `--base` path.
Sequential retention requires R8 decimal behavior and the original/R8 general
control union. The original-parent control retains the original behavior; it is
not required to match R8's decimal gains.

Use the pinned local environment with a private source-hash-bound request:

```text
python protected_host_v1.py --request PRIVATE_REQUEST.json
```

The request schema is `weightbridge.phase_iiis.protected_host_request.v1`, with
`driver_sha256`, `action`, `study`, `arm`, `ledger`, `original`, and
`decimal_parent`. Actual requests, rosters, private seeds, model outputs and model
weights are intentionally not published. The source retains the real directory
constants, following the relocation guidance in the parent host-tools README.
Any relocated helper is a new host-source derivative requiring its own hash.

The self-test source and receipt are included. Run it with a new output path:

```text
python protected_host_selftest_v1.py --out NEW_RECEIPT.json
```

Eight synthetic access/cache/path/replay assertions passed, with no model calls
or protected roster reads. These are negative gate fixtures, not end-to-end
positive model evidence. The published receipt reports the measured scope and
leaves memory unmeasured. Later study receipts must establish which protected
actions actually ran and their complete compute and scientific results.

All files are bound by `manifest.json`. This supplement does not replace the
earlier host-tools manifest or alter its byte commitments.
