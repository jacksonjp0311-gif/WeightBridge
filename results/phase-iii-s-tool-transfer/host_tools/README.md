# Supplemental Phase S host orchestration

These are byte-identical copies of the private host helpers reviewed during the
frozen Phase S study. Their hashes are recorded separately in `manifest.json`.
They are **outside the 162-file scientific source manifest**. They do not change
the model, training recipe, dataset, endpoint, fixed checkpoint selection, or
behavioral gates implemented by the frozen experiment modules.

The source snapshot is prospective: at publication of this package, no positive
post-development or protected-bank path had run. Later outcome receipts must
identify which helpers and branches actually executed. Source availability and
software checks are not model-improvement evidence.

## Files and authority

- `post_development_host_v1.py` calls the existing bounded worker launcher after
  completed development. It handles an eligible stored merge, merged development,
  reconstruction, fresh development replay, parent restoration, and an arm lock.
  Its separate `lock-both` action fixes both arm decisions. It cannot open a
  historical or new protected bank and cannot train another adapter.
- `host_stage_bridge_v1.py` derives supplemental host receipts from existing
  evidence. Its historical release first checks the frozen S locks in a separate
  process, then uses the pinned R validators to release only the previously used
  rank-eight battery. Its historical retention check computes outcomes from the
  actual validated records; it does not accept an unverified success flag.
- `bridge-preaccess-tests.json` and `postdevelopment-preaccess-tests.json` are
  aggregate receipts for the limited checks actually performed. They contain no
  examples, labels, model outputs, private seeds, or protected roster contents.

Every request pins the selected helper's exact SHA-256 before it can operate.
Every receipt is write-once. The helpers grant no production or lineage authority.
Missing or failed stages cannot become successful verification. A merged behavior
failure may produce explicitly labelled `NOT_RUN` terminal bookkeeping; that is
not a completed reconstruction or a pass.

## Execution and replay scope

Run the helper with the pinned Phase S Python environment and an operator-owned
private request file:

```text
python post_development_host_v1.py --request PRIVATE_REQUEST.json
python host_stage_bridge_v1.py --request PRIVATE_REQUEST.json
```

The two schemas are `weightbridge.phase_iiis.postdevelopment_request.v1` and
`weightbridge.phase_iiis.host_bridge_request.v1`. The source contains the exact
required fields. The published package deliberately contains no actual requests,
raw rosters, worker outputs, adapters, or checkpoint files.

The post-development helper uses only the two existing arm protocols and the
shared Phase S resource ledger. It checks the worker command's physical `--base`
path before recording reconstructed-artifact replay. The sequential reconstruction
uses the already independently verified rounded R reconstruction as its parent,
then performs the separate S merge. Reuse of that R artifact is not a newly
executed R reconstruction or a single summed-delta merge.

Historical decimal remeasurement uses the S explicit-parent evaluator, which
calls the unchanged Q decimal measurement endpoint on byte-identical R rosters
in their original order. Historical source and authorization replay occurs in
the pinned R checkout. This is not a fresh rerun of the old R study, and its
regression battery is not fresh transfer evidence. Unopened R Bank B is excluded.

New formal qualification and confirmation remain controlled by the frozen S
`study`, `verification`, `evaluate`, and `host` modules. These helpers do not
automatically open them. Each model must replay its own recorded native output;
semantic agreement alone cannot establish exact reconstruction or rollback.

## Portability and verification

The archived sources intentionally retain their actual local path constants.
Those paths identify directories, not private data contents or credentials.
They are not portable launch defaults. To reproduce the exact archived run,
restore the declared directories and runtime, including both pinned checkouts.

For another machine, create a separately identified host-only derivative with
the path constants replaced, inspect and test it, and pin the derivative's new
hash in new private requests before access. Do not silently rewrite these source
copies, reuse their hashes, or claim the relocated derivative was the code used
in this study. The frozen scientific source hashes and all data/endpoint gates
must still verify independently.

`manifest.json` binds both sources, this note, and the published test receipts.
Check each SHA-256 against file bytes. The repository already marks Phase S result
artifacts as `-text`, preserving these exact bytes across Git newline settings.

The recorded checks comprise syntax compilation and five bounded preaccess or
physical-path fixtures. Positive merge, reconstruction, historical, and formal
execution were not tested by those fixtures. Costs reported by the fixtures are
their narrow checking scopes; import/launch overhead and memory were not fully
measured and must not be read as model resource costs.
