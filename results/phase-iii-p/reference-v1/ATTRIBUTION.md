# External-reference attribution

Source: RikiyaT/lm_understand-logs, commit
`4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.

Article: Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab,
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).

The copied source files retain their original bytes. Their source URLs,
SHA-256 hashes and sizes are recorded in `source/source-receipt.json` and
`source/metadata-source-receipt.json`. Upstream [LICENSE](source/LICENSE)
assigns scripts to MIT and the authors' other logs, data and networks to
CC BY 4.0. Quoted third-party content retains its own license. No full benchmark
dataset or Qwen checkpoint is included here.

WeightBridge independently inspected the code and reconstructed the pinned
`p50c` patch through its existing checkpoint engine. No upstream Python was
executed. The patch and manifest are unchanged; the new importer, exact-Decimal
roster, measurements and analyses are WeightBridge work. The source authors
do not endorse these measurements.

This is a short greedy decimal endpoint comparison. The authors' reported
long sampled thinking-mode scores are external research claims, not our own
reproduced benchmark results. Their final held-out-practice report includes
five losses, so it is not a zero-regression adaptation demonstration.
