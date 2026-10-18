# Export the paper leaderboard

This optional CPU-only tool reads already-scored artifacts. It does not load
models, download weights or replace the benchmark's raw diagnostic results.
The export follows Appendix F.3 under `paper-f3-equal-world-v1`.

## Input manifest

Paths are relative to the manifest. Every entry explicitly declares its track
and separates simulation from real-world artifacts. A metric reference may
point to a run's `metrics.json` or a published metric `summary.json`. Scores in
these sources are fractions, not already multiplied by 100.

```json
{
  "schema_version": "robovalue-leaderboard-input-v1",
  "baselines": [
    {
      "baseline": "ExampleModel",
      "track": "zero-shot",
      "worlds": {
        "sim": {"sa": "simulation/metrics.json"},
        "real": {"sa": "real/metrics.json"}
      }
    }
  ]
}
```

The minimal example intentionally lacks the other metrics; the output records
their raw scores as null and their normalized leaderboard contributions as zero.
A complete entry supplies `sa`, `tga_easy`, `tga_hard`, `voc`, `voc_mem`,
`cycle_voc`, `fpl`, `trr`, `vs` and `csvc`; `sia` is optional and remains separate
from the four dimensions. Missing domain coverage is not averaged away.
Supplied leaderboard metrics must cover all ID/ENV/EMB domains, except ID-only
TRR/CSVC and separately reported SIA. Incomplete domain artifacts are rejected,
not silently converted to zero-valued leaderboard components.

```bash
python -m vmbmk.cli results leaderboard manifest.json --output leaderboard.json
```

The output must not already exist. Tracks are ranked separately; the tool never
infers a track from a checkpoint name. Source hashes, protocols and valid task
coverage are retained. Protocol, score range, task-macro and Cycle-VOC component
consistency checks reject incompatible sources. These checks do not certify
all historical cohorts or reproduce model inference.

## VROC without a new inference run

New Cycle-VOC results also retain `components.forward` and `components.reverse`
under `continuous-cycle-halves-v1`, using the same trajectory selection,
continuous history, shared turn, constant-input convention and task/domain
weights as the combined score. The combined score and query plan are unchanged.
VROC comes from the reverse component, not from subtracting independently
aggregated or rounded VOC tables. Legacy combined-only results must be rescored
from their cached Cycle-VOC predictions with `score_cycle_voc_components` before
export; they are not silently treated as compatible component artifacts.

## Aggregation and units

1. Average tasks within each world/domain, then average the two world means.
   Never pool simulation and real-world tasks into one cohort.
2. Most metrics use ID/ENV/EMB weights 1/2, 1/4, 1/4. TRR and CSVC use ID only.
   VROC instead uses equal ID/ENV/EMB weights, as Appendix F.3 specifies.
3. Convert to x100. Normalize VOC, Memory-VOC, VROC and CSVC using `(x+100)/2`,
   and FPL using `100-x`. Other metrics keep their percentage scores. Missing
   components contribute zero only after normalization, without redistribution.
4. Task-State Understanding averages SA and TGA, with TGA averaging CT/CF.
   Temporal Progress Monitoring assigns 1/4 each to VOC and Memory-VOC, and
   1/2 to VROC. Failure and Recovery Reasoning averages FPL and TRR; Value
   Consistency averages VS and CSVC. Overall averages the four dimensions.

`raw_worlds` retains fractional domain means, `raw_equal_world_x100` retains
unrounded equal-world scores or null, and `normalized_components_x100`,
`dimensions_x100` and `overall_x100` retain unrounded leaderboard values.
`missing_components` explicitly lists zero-filled normalized components.
