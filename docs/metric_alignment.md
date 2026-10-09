# Metric alignment (2026-10-06)

On 2026-10-06 the author confirmed that current VOC is exactly the forward
half of Cycle-VOC, not the legacy standalone successful-episode cohort.
Both use the same normal-ST selection, annotated milestones and shared turn;
TRR branches and diverse solutions are excluded. Constant predictions score
zero. VOC uses the cycle forward Spearman component and preserves task/domain
aggregation. The old standalone implementation is removed. The voc key remains,
but forward-position query IDs and voc-cycle-forward-v2 reject legacy caches.
MEM-VOC is unchanged and remains a separate history-dependent metric.

The author confirms that the main result tables use the scoring definitions in
RoboValue__A_Fine_Grained_Sim_and_Real_Benchmark_for_Unified_Evaluation_of_Robotic_Value_Models
(6).pdf. This supersedes the old CSVC scoring described in earlier project notes.
Pages 11-12 define CSVC=(G-D)/(G+D), without clipping; G+D=0 receives -1.
The local scorer now follows this ratio, including negative and zero-denominator
task scores in the covered-task macro mean. Missing semantic coverage remains
N/A. The new protocol is semantic-gain-symmetric-ratio-v2; completed rmse-v1
caches cannot be reused or merged with it. Normal repeats still contribute one
semantic-median solution profile, and diagnostic statistics do not change scores.

SIA is already aligned: Appendix F.2 (page 58) specifies geometric aggregation
within each task followed by task-level averaging. It is not a global geometric
mean over all tasks, and this correction does not change SIA queries or scores.
The other reviewed primary metric formulas remain unchanged. This is an
implementation alignment, not proof that archived model results were reproduced.
The PDF mentions three boundary pairs for a small set of very short CSVC subtasks
but does not identify that set or its selection rule; the existing four-pair
query plan is retained pending the frozen query selection used for those rows.

TRR uses binary branch conjunctions and time-coverage-weighted all-pairs
directions. Only continued error accepts zero. Its default grid is
episode-anchored round(i*fps/10), with native-frame top-up to min(10, stage
length), earliest empty-grid seed and earliest ties. This is the verified
simulation sampling policy; historical real preparers may seed empty grids
at the midpoint. Exact archived reproduction requires its frozen selection.
The new protocol rejects old continuous-score checkpoints. Existing results
and raw predictions are not rewritten.

VS is registered as vs in the normal YAML evaluation entry point. Use base
mode and explicitly select tasks/domains. Cycle-VOC&VS is grouped under
metrics/cycle_voc_vs/ and uses one shared normal-ST cohort/query plan. As confirmed by the author on
2026-10-06, it reuses Cycle-VOC's forward predictions on the episode-anchored
5 Hz native grid; the reverse half and final Cycle-VOC score are not inputs.
VOC or CYCLE-VOC task coverage is sufficient; no separate VS declaration is
required. When both metrics are selected, Cycle-VOC runs first, regardless of
YAML order. VS-only runs compute the Cycle-VOC dependency automatically.
Cycle-VOC annotations must cover every required 5 Hz frame. Missing frames
raise an error; predictions are never interpolated or silently regenerated.
The source must use base mode and cover every VS task/domain. VS excludes unsuccessful,
TRR and diverse-solution episodes, and records exclusions. Curves need seven
points; all eligible tasks must share the same covered domains. No model
range normalization is applied. The first/last three values are replaced by
their respective means for ER only; nonflat coverage uses exact raw adjacent
equality and native timestamps. Zero processed variation means ER=1, so an
original constant curve scores VS=0. Average episode VS, not mean ER times
mean coverage. New inference does not reinterpret sample-hz as resampling
cached predictions and does not overwrite archived VS outputs. The input
protocol is now vs_v1-cycle-forward-5hz-v2; old completed VS caches are rejected.
The pure vs_v1 formula is unchanged.

metrics/cycle_voc_vs/vs_scoring.py is copied unchanged from Xiao Jinyang's
original vs_v1 implementation (`value_stability/metric.py`).
Source SHA256: b82b4f6fbe6ba7f11d6e20adda67dca7960e7bf6a1661c15c3391791753ec81b.

The 2026-10-01 review below predates this PDF comparison. Its previous claim
that CSVC already matched the paper was incorrect: the implementation used a
clipped 1-D/(G+epsilon) score and excluded zero-gain tasks. The correction above
replaces that behavior. SA/SD (strict terminal 2% comparison), TGA (strict top1
trajectory gains), VOC, MEM-VOC, Cycle-VOC, ZigZag FPL, TRR and VS retain their
reviewed formulas. Older local section files still describe the retired
contiguous FPL extractor. The separate campaign scheduler allow-list is not
expanded by this change.

Example metric block in an existing model YAML:

    metrics:
      vs:
        mode: base
        domains: [id, env, emb]

Validation: 144 focused tests and 48 subtests pass. One thousand seeded
curves with irregular timestamps match every field of the original VS
implementation exactly. Read-only simulation preflight finds 525 eligible
episodes across 15 tasks, 66,521 queries; 420 unsuccessful episodes,
126 TRR branches and 160 diverse solutions are excluded. No inference,
training, dependency installation, result overwrite, commit or push is run.
