# Developer reading and change guide

This guide is for a Python developer who knows basic machine learning but has
not participated in RoboValue's experiments. It explains the existing code;
it is not a runnable evaluation example or evidence of checkpoint reproduction.
Dataset preparation, model execution and real integration require external
assets and the appropriate model environment.

## Repository map

| Location | Responsibility | Start here when changing |
| --- | --- | --- |
| src/vmbmk/cli.py | Public commands and explicit credential loading | Command arguments |
| src/vmbmk/runner/run.py | Run configuration, selection, resume and metric dispatch | Run orchestration |
| src/vmbmk/metrics/registry.py | Metric names, aliases and lazy planners/runners | Shared metric registration |
| src/vmbmk/adapters/registry.py | Lazy adapter classes and checkpoint identity | Model registration |
| src/vmbmk/data/dataset.py | Metadata/annotation validation and dataset objects | Supported input schemas |
| src/vmbmk/inference/queries.py | Typed queries/results and their JSON representations | Inference wire contracts |
| src/vmbmk/inference/dispatch.py | Validation, worker processes, sharding and resume | Inference scheduling |
| src/vmbmk/inference/worker.py | Adapter execution and checkpoint records | Worker behavior |
| src/vmbmk/data/playback.py | Frame/time/history preparation | Temporal model input |
| src/vmbmk/adapters/ | Isolated model-specific inputs and inference | A particular model |
| src/vmbmk/metrics/ | Metric query planners, scoring and run wrappers | Scientific behavior |
| src/vmbmk/serialization.py | Shared structured-file reading/writing | Serialization |
| src/vmbmk/tools/results/publication.py | Result validation, compatible merges and transactions | Published artifact handling |
| src/vmbmk/tools/results/cli.py | Result subcommands backed by validation/status/sync | Result tools without inference |
| src/vmbmk/runner/batch.py | Generic batch scheduling, retries and incremental task selection | Batch orchestration |
| src/vmbmk/runner/resume.py | Versioned config/code/query fingerprints and task/domain completion checks | Compatible result reuse |
| src/vmbmk/runner/policy.py | Recorded model/metric constraints, separate from scheduling | Evaluation eligibility |
| src/vmbmk/tools/visualization/ | Shared configuration and protocol-specific curve workflows | Curve inference and rendering |
| configs/ | One xxxconfigs.yaml per family plus generalconfigs.yaml | Model/run settings |
| tests/ | Metric scoring and one minimal CPU test per adapter | Scoring contracts |
| src/vmbmk/tools/ | Data checks, result operations and visualization implementations | Auxiliary tools dispatched by the package CLI |
| src/vmbmk/tools/data.py | Read-only video frame checks | Dataset asset operations |
| docs/baselines/ | Per-family source, environment and checkpoint setup | First installation |
| envs/ | Separate model environment declarations | Model setup |

Do not infer a public API from a date-stamped script or local verification file.
Some historical experiments exist only in Git history. Inventory and validation
documents describe particular cleanup snapshots, not an alternative runtime.
Checkpoints, datasets and generated outputs are external; data/ is not a complete
benchmark distribution. License/provenance review is still required for release.

## Folder and naming rules

Keep files together by responsibility, not by the date or experiment that created them:

    src/vmbmk/
      cli.py                     stable public commands
      data/                      dataset and playback
      inference/                 queries, dispatch and worker
      runner/                    run, batch, policy and resume
      adapters/                  baseline implementations, registry and video_inputs
      metrics/
        registry.py              metric identities and lazy entry points
        utils/                   pure statistics, correlation, aggregation and result checks
        sia/                     evaluation, judge and scoring
        mem_voc.py               MEM-VOC planning, scoring and execution
        cycle_voc_vs/            shared planning, forward VOC, cycle and VS scoring
        sa.py, tga.py, ...        metrics with one implementation file
      tools/
        data.py                  read-only dataset asset checks
        results/                 cli, validation, status, sync and publication
        visualization/           value_curves, selected_curves, cycle_curves and svg_grid
      serialization.py           structured file I/O shared across packages
      errors.py                  shared error types

Tests are grouped under tests/metrics and tests/adapters. Shared temporary
fixture construction is tests/synthetic_data.py. Historical cleanup records
remain in Git history; old implementation paths are provenance, not current APIs.
A single-file metric does not need its own otherwise-empty folder.

### Import migration

TOPReward has one registered adapter in adapters/topreward.py. YAML
model_options.backend selects qwen or molmo; the old topreward_molmo adapter
is removed. Backend-specific loaders and processors remain lazy and preserve
their existing scoring paths. Configuration and environments documents explain
selection and the separate dependency runtimes; no separate Molmo guide is needed.

Result validation exposes validate_run(config_path, run_dir, data_root), returning
the same inspection report used by the manual CLI. Batch scheduling calls this
function directly; argument parsing and exit codes belong to tools/results/cli.py.
Publication accepts a PublicationRequest containing baseline, config path, source
run, scored results, resume contracts and execution mode, not a scheduler Job.
Both batch and manual publication use this same interface. The former validation
module command is replaced by python -m vmbmk.cli results validate.

Metric helpers formerly in metrics/common.py are split into metrics/utils/
statistics.py, correlation.py, aggregation.py and results.py. Query execution
lives in inference/execution.py, not in the pure metric helpers. MEM-VOC is
metrics/mem_voc.py; the former metrics/voc/ directory is removed. Cycle-VOC
imports the public Spearman helper rather than a private MEM-VOC function.
Scoring formulas, history policy, query IDs and aggregation order are unchanged
by this organization pass. Resume fingerprints include the new helper paths.

The package root exports (such as vmbmk.Dataset), existing vmbmk.cli commands,
YAML metric/model keys and result filenames remain unchanged. The repository-root
tools/ wrappers are removed. Use python -m vmbmk.cli data, results or visualize;
their implementations live in vmbmk.tools, with no duplicate wrappers. Internal
module paths change: use vmbmk.runner.run instead of vmbmk.evaluation,
vmbmk.inference.dispatch instead of vmbmk.inference, and
vmbmk.tools.results.publication instead of vmbmk.publication. Metric implementation
paths are listed in metrics/registry.py. No duplicate compatibility modules are
kept. Update external scripts that imported internal modules; historical scripts
remain historical rather than being silently rewritten.

Cycle-VOC&VS now shares metrics/cycle_voc_vs/planning.py for normal-ST selection
and the complete continuous cycle query plan. The former metrics/voc/cycle.py
is cycle_voc_vs/cycle.py; metrics/vs/evaluation.py and metrics/vs/scoring.py are
cycle_voc_vs/vs.py and cycle_voc_vs/vs_scoring.py. The pure VS scoring source
is unchanged. In a shared run, both scorers receive the same CyclePlan; VS
does not repeat episode eligibility selection. Public cycle_voc and vs metric
keys, query IDs, frames, scores and result filenames remain separate/unchanged.

Moving implementation files changes code fingerprints. Existing completed
outputs are not migrated or overwritten; missing mode must reject an incompatible
old fingerprint and require an explicit rerun or separate output root. Scientific
protocol IDs, query IDs, frames, scoring and aggregation do not change in this pass.

## Recommended reading path

1. cli.build_parser and cli.main distinguish validate, infer, run and sia-judge.
   The normal evaluation entry is python -m vmbmk.cli run CONFIG; the installed
   vmbmk command points to the same main. There is no python -m vmbmk entry.
2. RunConfig.load in runner/run.py validates YAML and resolves configured paths.
   _reference_settings owns one-shot/zero-shot reference parsing. RunConfig.metrics
   retains the tuple contract (metric_name, mode, domains); changing that layout
   would affect callers, not just readability.
3. run_evaluation loads metric-scoped data, selects compatible tasks/domains,
   builds the model mapping and dispatches via metrics.registry.metric_runner. It manages partial
   and completed output directories; it does not publish final tables.
4. Dataset.load in data/dataset.py builds Dataset, Task and Episode objects. Whole-dataset
   validation and metric-scoped validation are different contracts. Do not widen
   metric-scoped validation as a side effect of a refactor. The evaluation entry
   passes its loaded Dataset to metric runners. Standalone runners scope loading
   to their metric; workers scope known benchmark query prefixes to their metrics.
   Arbitrary user query IDs retain whole-dataset validation.
5. Follow a metric's build_*_queries, score_* and run_* functions. Query planning
   decides cohorts, frames and history; scoring consumes results; the run wrapper
   connects them to inference and creates the metric-specific result object.
6. SA and several other metrics use inference.execution.run_queries to serialize queries
   and call inference.run_inference. VOC uses cycle_voc_vs.forward and the
   cycle forward-segment scorer; MEM-VOC uses metrics.mem_voc.run_voc_mem,
   which also uses the shared run_queries bridge. Follow
   _run_worker_process to worker.run_worker, _evaluate_chunk and adapters.create_adapter.
   Imports in the factory are lazy.
   Native mode uses the Adapter's metric-specific method instead of the base
   operation interfaces. SIA has its own generation/judging pipeline in metrics/sia/evaluation.py.
7. Follow the scorer's aggregation, then run_evaluation's checkpoint and final
   metrics.json writes. Read result_publication.md before calling publication tools.

## Inputs and boundary contracts

### Run settings and paths

Ordinary runs require model, batch_size, python, data, output, metrics and tasks.
Adapters require gpu and checkpoint by default. A CPU/service adapter may declare
`requires_gpu = False` and/or `requires_checkpoint = False` to omit those fields;
this does not relax existing baseline requirements. See the
[model service integration guide](api.md) and
[configuration](configuration.md) for model-specific fields.
Top-level python/checkpoint/data/output paths and reference_data are resolved
relative to the YAML directory when relative; sia_responses is also config-relative.
Paths inside model_options follow the consuming Adapter's contract and are not
automatically resolved by RunConfig.load.

gpu is a nonnegative integer or a nonempty list of distinct nonnegative integers.
tasks defines the run scope; metric_tasks optionally selects a subset per metric.
candidate_tasks records TGA candidate vocabulary separately from scored target
tasks. Neither task scope is passed to model adapters.
The Adapter mapping contains adapter, python, optional checkpoint and batch_size, then
reference settings and model_options. Reserved options cannot override core fields.
Do not put credentials in this mapping or result files.

### Dataset objects

The existing layout is DATASET/TASK/metadata.json and
DATASET/TASK/episodes/EPISODE/{metadata.json,annotation.json,assets...}.
Task metadata includes task_id, instruction and supports.metrics; subtask and TGA
metadata are metric-dependent. Episode metadata identifies episode_id, world_type
(sim/real), domain (id/env/emb), success, fps, num_frames and assets. Asset paths
are relative to the episode directory and validated by _asset_path.
Metric annotations are parsed by _validate_annotation. Inspect that parser and
the metric's tests before introducing a new field; not every metric needs every
annotation. Whole-dataset validation also checks robot_data metadata/assets.

Episode does not retain world_type as a field. Do not assume a combined dataset
will produce separate sim/real aggregate views automatically. Keep those cohorts
separate through the dataset/run protocol and label exported coverage honestly.

### Query and result records

| Type | Fields beyond query_id/op | Meaning |
| --- | --- | --- |
| StateRef | task_id, episode_id, anchor_frame | Episode anchor, not a decoded image or a universal history window |
| ValueQuery | state, instruction, playback | Value operation; playback is forward/reverse/cycle |
| CompareQuery | state_a, state_b, instruction | Ordered comparison; value-difference adapters compute value(b)-value(a) |
| SubtaskQuery | state, instruction | Text generation/classification input |
| Result | score for value/compare; output for subtask | A result linked by query_id and operation |

Serialization lives in inference/queries.py. read_queries checks nonempty input and unique
IDs; result validation checks coverage and operation compatibility. Numeric wire
scores must be finite. Worker base dispatch groups operations, then restores
the input query order. Preserve IDs, frame selection, ordering and playback when
refactoring; matching the final mean alone is insufficient.

The comparison helper in adapters/base.py is not a universal conversion rule:
direct-comparison models may have their own input and output policies. Likewise,
anchor_frame does not authorize removing history from a memory-sensitive model.

### Results, resume and publication

A completed run writes output/CONFIG_STEM/config.yaml, metrics.json and
operations.jsonl. Resume uses .CONFIG_STEM.tmp, .metrics.partial.json and per-metric
operation files. SIA generate intentionally returns an intermediate directory and
sia_generation.json, not a completed scored run. It also retains response/judge
artifacts under its SIA cache. Do not equate a generated artifact with a score.

metrics.json maps metric names to metric-specific objects. Many include mean,
domains and protocol, but their detailed task/episode layouts are not identical.
Do not force them into a new universal schema or averaging rule. Native stored
units, presentation multipliers, exclusions and protocol identities must remain
distinct. See [alignment](metric_alignment.md) and [publication](result_publication.md).

Completed-cache validation is implemented by _validate_completed_run and
_read_metrics. Protocol changes require compatible cache handling, not relabeling
old predictions/results. Publication is explicit, uses a separate artifact schema,
and permits only supported incremental merges with matching protocols.

## Trace one metric: SA/SD

This source-reading exercise does not require real trajectories or inference.

1. Find sa in metrics.registry.METRICS: its lazy runner is sa.run_sa.
2. In sa.py, _plan requires both success and failure coverage in each selected
   task/domain group; incomplete groups raise an error rather than being skipped.
   It constructs terminal ValueQuery records. edge_frames in
   metrics/utils/statistics.py uses max(1, ceil(num_frames * 0.02)) terminal frames.
3. run_sa calls run_queries, then _score_sa_details. The scorer validates result
   coverage, averages each episode's queried terminal values, and compares every
   success episode against every failure episode within task/domain.
4. A strict success value > failure value is a win; ties get zero credit. Each
   task/domain score is wins divided by the number of success/failure pairs.
   run_sa averages tasks within each domain, then domain means. This is SA's
   implemented order, not a common rule to impose on all metrics.
5. Distinguish the primary score from gap and terminal_values diagnostics. The
   returned protocol is terminal-value-tail2pct-average-sa-v5.
6. tests/metrics/test_sa.py verifies terminal selection, ties, coverage and scoring.
   Orchestration is outside the current automated test scope.

## Where to make and verify a change

| Change | Implementation sites | Verification scope |
| --- | --- | --- |
| Run/reference configuration | runner.run.RunConfig.load, _reference_settings | Targeted inspection; outside the test suite |
| CLI option | cli.build_parser, _inference_model/main | Targeted inspection; outside the test suite |
| Dataset/annotation format | data validators, Dataset.load; consuming planner | Affected metric tests when scoring inputs change |
| Query/result schema | inference/queries.py, inference validators, worker dispatch | Targeted inspection; affected metric tests when scoring inputs change |
| Add an Adapter | adapters/base.py interface, new model module, adapters.registry.ADAPTERS | One CPU smoke test in tests/adapters/test_interfaces.py |
| Add/change a metric | planner/scorer/run wrapper; metrics.registry.METRICS; dataset eligibility/annotations | Relevant metric scoring tests |
| Export behavior | tools/results/publication.py and explicit tools | Targeted inspection; outside the test suite |

Adding a metric may also require cache validation in _read_metrics and publication
support; those are separate contracts, not automatic registrations. Adding native
behavior requires checking worker dispatch and the consuming metric's mode policy.
Keep scoring independent of model loading, videos and output-file writes.

For a scoring modification, start with the affected tests in [testing](testing.md).
Do not add rerun, cache, publication, configuration, or orchestration tests to this
reduced suite. For structural changes, compare against a
saved copy of the current local source, not Git HEAD when uncommitted work exists.
Compare errors/defaults and query IDs/frames, episode scores and aggregation.
Never relax tolerances or remove failing tests to make a refactor appear equivalent.

## Limits of a handoff review

Current-source evidence is recorded in testing.md: 108 CPU tests, a real
RoboDopamine-3B ID trajectory through joint VOC/Cycle-VOC/VS inference, completed
result validation and cache reuse on 2026-10-06. This is stronger than source-only
inspection but does not establish all-model, OOD, clean-install or paper-table
reproduction. ProcVLM's retained forward-fill behavior and provenance's asset
size/mtime identities remain explicit exceptions to stronger guarantees.

Source tracing and fixture/mock tests establish only their inspected/tested
contracts. They do not verify dataset availability, video decoding, checkpoint
loading, GPU throughput, live API judging or paper-result reproduction. Record
these as unverified rather than zero coverage or successful integration.
An agent following this guide is a proxy review; independent newcomer feedback
remains useful and must not be claimed without an actual participant.
