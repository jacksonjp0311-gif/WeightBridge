# Final public checkout verification

This procedure verifies published bytes, redaction boundaries and deterministic public fixtures. It does not replay private model outputs, open protected banks, execute inference, or establish a scientific PASS. Use the exact final published commit, not a moving branch name alone.

The completed rehearsal at `3750911dd6f0e368346b98d5c3ea3c97661b269b` is recorded in `clean-checkout-3750911-v1.json`. It verified 31 aggregate artifacts, 162 scientific source files, both 151-file calibration manifests, four supplemental host packages and the explicitly public demo fixture. The clone remained clean. Historical Q/R source and evidence were unchanged.

## Fresh Windows checkout

Use a new directory and `core.autocrlf=true`. Respect `.gitattributes`; do not normalize every file to LF. The scientific manifest currently binds 139 LF-only files, 22 CRLF-only files and one mixed-newline file. `measurement.py` and all Phase S result artifacts have explicit byte-preservation attributes.

```powershell
$finalCommit = 'REPLACE_WITH_EXACT_FINAL_PUBLISHED_SHA'
$reviewDirectory = "C:/Users/jacks/AppData/Local/Temp/wb-s-final-review-$finalCommit"
git -c core.autocrlf=true clone --no-hardlinks --single-branch --branch dev/phase-iii-s-tool-transfer https://github.com/jacksonjp0311-gif/WeightBridge $reviewDirectory
git -C $reviewDirectory checkout --detach $finalCommit
git -C $reviewDirectory rev-parse HEAD
git -C $reviewDirectory status --porcelain
Set-Location $reviewDirectory
$reviewPython = 'C:/Users/jacks/AppData/Local/Temp/weightbridge-reference-inspection/venv/Scripts/python.exe'
```

The resolved HEAD must equal the pinned commit, and the checkout must be clean before and after verification. Record whether the remote branch still points to the same commit; a later branch advance must not silently change the reviewed revision.

## Existing public validators

```powershell
& $reviewPython results/phase-iii-s-tool-transfer/export_preflight.py --verify results/phase-iii-s-tool-transfer/preflight-v1
& $reviewPython results/phase-iii-s-tool-transfer/export_preregistration.py --verify results/phase-iii-s-tool-transfer/preregistration-v1
& $reviewPython results/phase-iii-s-tool-transfer/export_outcome.py --verify results/phase-iii-s-tool-transfer/sequential-development-v1
& $reviewPython results/phase-iii-s-tool-transfer/export_preflight.py --self-test
& $reviewPython results/phase-iii-s-tool-transfer/export_preregistration.py --self-test
& $reviewPython results/phase-iii-s-tool-transfer/export_outcome.py --self-test
& $reviewPython results/phase-iii-s-tool-transfer/demo.py --verify results/phase-iii-s-tool-transfer/public-demo-v1
```

Also run `export_outcome.py --verify` on each new final aggregate snapshot actually published. Do not invent a missing `outcome-v1` directory or mark an absent stage as passing. The outcome validator explicitly reports `scientific_outcomes_reverified: false`.

Verify the 162-entry `preregistration-v1/source-manifest.json` against checkout file bytes. The development and final snapshots must carry the same scientific source map. Verify each calibration protocol's `source_sha256` separately; that earlier 151-file scope is not interchangeable with the full study scope. Cache hashes per path during one audit to avoid repeated work.

Verify the `files` maps in these independent supplemental manifests, including file byte counts:

- `host_tools/manifest.json`
- `host_tools/protected-v1/manifest.json`
- `host_tools/protected-v2/manifest.json`
- `host_tools/protected-v3/manifest.json`

Their sources remain outside the frozen scientific manifest. V1 and V2 are unrun, superseded proposals. V3's existing synthetic receipt has 26 checks; verify its source and selftest hashes. Do not invoke a protected host driver during a public-only audit. Its original local path constants are retained for exact-byte provenance and are not authority to read those paths.

## Redaction and claim checks

Existing aggregate validators reject raw row, prompt, response, token, operand, private-path and private-seed payloads. Check the newly tracked file list separately for model weights, adapters, optimizer state, training bundles, host requests, bank rosters and per-row protected results. The 40 `public-demo-v1` presentations are an explicit illustrative-fixture exception, not a protected population or a model evaluation result. Supplemental source code can contain its published local path constants; those are not private data rows.

Compare Q/R historical source and evidence directories against pinned R commit `baa336e`. Keep their archives unchanged. Confirm that all unopened, interrupted or unavailable final stages remain null, UNMEASURED or an explicit failure, never zero or PASS. Verify the final report distinguishes adapter, merged checkpoint and reconstructed checkpoint identities, development gains, formal gates, natural EOS completion, legacy first-line validity, and exact native replay.

Write a new redacted receipt once. Include the actual reviewed commit, validator and manifest hashes, check counts, skipped checks, Git cleanliness, observed audit wall/CPU time, and cost scope. Do not count this archive audit as model compute or independent behavioral reproduction. Do not replace an earlier audit receipt.
