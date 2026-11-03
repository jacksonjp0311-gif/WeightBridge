# AWL adaptation intent

An AWL document asks WeightBridge to explore specific edits for a stated
deficiency. It records the agent's observation and hypothesis and limits the
search to named tensors and strengths. The host owns the checkpoint, practice
examples, edit policy, candidate budget, and independent evaluation.

```json
{
  "schema": "weightbridge.awl.v1",
  "observe": "threshold-reversal",
  "diagnose": "The target answer direction has weakened on the practice prompts.",
  "target": ["model.language_model.layers.23.mlp.down_proj.weight"],
  "optimize": {
    "method": "activation_ridge",
    "strengths": [0.02, 0.03, 0.05]
  }
}
```

| Field | Accepted value |
|---|---|
| `schema` | The literal `weightbridge.awl.v1`. |
| `observe` | Nonempty deficiency ID or description, up to 256 characters. |
| `diagnose` | Nonempty hypothesis, up to 512 characters. |
| `target` | Between 1 and 32 distinct tensor names, each at most 256 characters with no surrounding whitespace. The host must authorize each name for direction edits. |
| `optimize` | An object with exactly `method` and `strengths`. The method is `activation_ridge`, and strengths are 1 to 32 distinct finite numbers greater than zero and at most one. |

These five fields are required and extra fields are rejected. The parser
accepts decoded JSON and returns a frozen `AdaptationIntent`. Observation and
diagnosis are descriptive text. They travel with the intent without controlling
the numerical fit.

`plan_candidates` checks every requested tensor against the host direction
allowlist. It visits the strengths in the given order and the requested tensors
at each strength. Host preferences can reorder those tensors within a strength.
The remaining candidate budget caps the resulting sequence.

`synthesize_patch` calls the existing Qwen3.5 MLP direction compiler for one
tensor and strength. The compiler reads the base and practice suite and fits
a rank-one update to the answer direction while penalizing the fitted response
on practice sentinels. It returns a `weightbridge.patch.v1` proposal with the base and tensor
hashes. The host applies its edit policy and measures the candidate.

Preserve, validate, retain, and revert are host actions. The host defines the
unrelated behavior it wants to preserve, verifies candidate artifacts on an
independent suite, and decides which checkpoint to retain or use for rollback.
The intent names the edit search while evaluation and checkpoint selection stay
under host control.
