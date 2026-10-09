# Configuration and evaluation

For model services, use a model-specific adapter through the same runner as
local models. See the [integration guide](api.md) and
[mock configuration](../configs/mock_service.example.yaml). Adapters declare
whether they require a local GPU or checkpoint; a local Python worker is still
required. Service-specific options belong in that adapter's `model_options`.

A configuration identifies a baseline environment, checkpoint, dataset, output
root, selected tasks and metric settings. Paths are explicit and refer to the
machine executing inference. Replace placeholders before running examples.

Start with [baseline installation](baselines/README.md) for runtime and checkpoint
preparation, and [data preparation](data.md) for dataset roots and storage layout.

The configs/ directory contains one <family>configs.yaml per baseline family
and generalconfigs.yaml as a common-field example. GVL is excluded.
failsafeconfigs.yaml selects SIA only; other templates select a minimal SA
run. Replace metrics and tasks explicitly for your intended evaluation.
These examples replace the old task-, size- and metric-specific templates;
they do not reproduce archived experiment selections. Batch evaluation uses
explicit configuration settings and the shared evaluation policy; old campaign
status and disabled historical providers are not runtime policy.


For code navigation and modification sites, see [the developer guide](developer_guide.md).
Relative top-level python, checkpoint, data and output paths are resolved from
the configuration's directory, as are reference_data and sia_responses. Nested
model_options paths retain their Adapter-specific handling.

    model: robometer
    gpu: 0
    batch_size: 2
    python: /path/to/model-env/bin/python
    checkpoint: /path/to/checkpoints/robometer
    data: /path/to/dataset
    output: /path/to/results
    metrics:
      vs:
        mode: base
        domains: [id, env, emb]
    tasks: [organize_table]

Use an existing core Python with python -m vmbmk.cli run CONFIG, or bash vmbmk.sh run CONFIG
on Linux. The shell wrapper uses python3, or VMBMK_PYTHON when explicitly set;
it does not parse YAML. Model workers still use the configuration's python field.
Normal evaluation does not publish or overwrite final results.

## Tool entry points

Result operations share one entry point:

    python -m vmbmk.cli results validate --config CONFIG --run-dir RUN --dataset DATASET
    python -m vmbmk.cli results status --config-root CONFIGS --results-root RESULTS --dataset DATASET --output STATUS_JSON
    python -m vmbmk.cli results publish RUN --results-root RESULTS --final-results-root FINAL

The two old result validators are replaced by validate, which checks query
identity as well as counts and task/domain coverage. SIA query validation is
not supported here; it is rejected explicitly, not treated as a completed run.
Status inspects configured runs. Legacy campaign migration, specialized
Cycle-VOC-to-VOC rescoring and TGA-only coverage reporting are removed.
Published outputs and archived result files are not deleted.


Curve commands use the same RunConfig and adapter mapping as evaluation:

    python -m vmbmk.cli visualize value --config CONFIG TASK STRIDE EPISODE
    python -m vmbmk.cli visualize selection --config CONFIG --selection SELECTION_JSON
    python -m vmbmk.cli visualize cycle --config CONFIG TASK EPISODE --stride 10

All three curve commands run inference; none is an offline result-only viewer.
Selection curves default to ordinary value queries. Pass --native-metric explicitly
when the selected adapter supports that interface; metric YAML blocks no longer
implicitly switch the selection tool to native VOC.
Cycle keeps continuous forward/reverse history and its own rendering/scoring
rules. The old visualize_value_curve.sh wrapper is removed: launch these tools
with an existing Python that has the benchmark dependencies; model workers use
the configuration's python field. Sampling and protocol identities are unchanged.
Use each subcommand's --help for options.

Dataset asset checks are read-only:

    python -m vmbmk.cli data check-frames DATASET --tasks TASK_A TASK_B --views front --workers 4

This checks video length against metadata and VOC/MEM-VOC frame annotations,
not every metric annotation schema. Omitted selectors inspect all tasks/views.
The model-specific RoboDojo HDF5 reference exporter is removed; reference paths
remain external inputs to evaluation and no data assets are deleted.

Batch evaluation is a separate generic entry point:

    python -m vmbmk.cli batch --data DATASET --config-root CONFIGS --baselines all --tasks all --metrics all --dry-run

The scheduler retains GPU selection, retries, grouped execution and missing-task
completion. Its supported metrics are sa, tga_easy, tga_hard, voc, voc_mem,
cycle_voc, fpl, trr, vs and csvc. SIA remains a single-config operation and does
not support incremental publication merging. With --metrics all,
metrics come from the selected configs, and unsupported metrics raise an error.
Metric settings must exist in the templates; domains are never invented.
--data is explicit. Configs can have arbitrary YAML filenames; generalconfigs
is an example, not a scheduled run. Config/model identity drives publication.
Model eligibility and workload constraints live in runner/policy.py,
separate from runner/batch.py: FailSafe is SIA-only; RoboReward excludes VOC variants;
ProcVLM LoRA requires matching task checkpoints and retains known exclusions;
TOPReward TGA keeps its verified batch size. These restrictions are not universal
runtime certification. Historical environment blocker notes are not active policy.
Actual scheduling still uses Linux locking. --dry-run does not launch inference,
query GPUs or publish scores; it writes only the job plan. Model workers keep
isolated configured Python environments. Incremental TGA preserves the original
candidate vocabulary while querying only missing target tasks.
Cycle-VOC preserves complete history, TRR complete branch groups, and CSVC
complete within-task solution comparisons. Missing mode re-evaluates whole
tasks and merges their measured cells, not isolated history frames or branches.
VS merges its independently aggregated ER and time-coverage diagnostics while
keeping episode VS as the primary estimand. CSVC uses aggregate_csvc, retaining
matched-coverage exclusions and bootstrap diagnostics. Its paper-aligned ratio
retains negative scores and includes the defined -1 zero-denominator penalty.

Execution modes share the same batch entry point:

    python -m vmbmk.cli batch --data DATASET --config-root CONFIGS --baselines robometer --tasks all --metrics sa --mode normal
    python -m vmbmk.cli batch --data DATASET --config-root CONFIGS --baselines robometer --tasks all --metrics sa --mode missing
    python -m vmbmk.cli batch --data DATASET --config-root CONFIGS --baselines robometer --tasks all --metrics sa --mode rerun

- normal schedules every eligible selected task, retaining compatible local
  completed/partial-run reuse and the existing TGA merge behavior.
- missing skips only compatible published task/domain cells. If one requested
  domain is missing, the task is evaluated again and its incoming cells replace
  old cells when merging. --incremental is an alias for --mode missing.
- rerun starts fresh without reusing local completed or partial inference. It
  replaces the selected metric publication, not a missing-task merge; select
  the full intended task coverage when replacing a published metric.

Missing mode requires batch-resume-v2 records in published summaries. Legacy
summaries and changed model settings, planner/scoring code or query metadata
are rejected, not silently skipped. Use an explicit rerun or a separate results
root for incompatible outputs. Fingerprints cover configuration, code, planned
queries and task/domain metadata. Code fingerprints select the metric, its shared
scoring dependencies, the current adapter and shared execution code, not all
unrelated adapters. They do not hash video, checkpoint or external
reference asset bytes. Keep those assets immutable when reusing results.

Rerun preserves local runs under OUTPUT/rerun_history and previous published
summaries/task files under FINAL_RESULTS_ROOT/rerun_history. No GPU inference
is launched by selecting --dry-run. Single-config execution supports normal
and rerun, but missing-task selection is a batch operation:

    python -m vmbmk.cli run CONFIG --mode rerun

## Directory migration

Every local run records provenance.json with package-code and dataset identities.
The top-level runner loads the selected metrics' Dataset once and passes it to
each metric runner. Standalone metric runners use metric-scoped loading too.
Metadata is content-hashed; asset identities use file size and modification time,
not full video/checkpoint content hashes. Normal reuse and interrupted resumes
reject missing or changed provenance; choose a new output root or explicit rerun.
Do not copy legacy caches into a new run to bypass this check. These identities
are local safety records, not proof of archived scientific reproduction.
The 2026-10-06 RoboDopamine smoke run exercised combined VOC/Cycle-VOC/VS
configuration, result validation and normal completed-cache reuse on a remote GPU server.
Use the same task/domain selection and base mode to share the cycle predictions;
see testing.md for exact evidence and limits, rather than treating a config
template as proof of environment readiness.

Run configuration code now lives in src/vmbmk/runner/run.py; batch planning and
completion live beside it in batch.py and resume.py. Public CLI commands and
xxxconfigs.yaml names are unchanged. Implementation moves change code
fingerprints, so prior completed outputs require an explicit compatibility
review/rerun before missing mode can reuse them. No published files are moved.

## Internal task scopes

Public templates need only tasks. Generated batch configurations may add
metric_tasks for scored targets and candidate_tasks for TGA candidate vocabulary.
These are execution-plan fields, never adapter model_options. The old
model_options.tga_candidate_tasks field is rejected; move its per-metric mapping
to candidate_tasks. Query IDs and candidate selection are unchanged.

## Metric selection

### TOPReward backend selection

Use the single toprewardconfigs.yaml template and model: topreward. Set
model_options.backend to qwen (default) or molmo. Qwen uses its existing
TOPReward source-linked runtime with batch_size: 2. Molmo uses its Molmo2
checkpoint and the topreward_molmo dependency runtime; set batch_size: 1.

| Field | Qwen | Molmo |
| --- | --- | --- |
| model | topreward | topreward |
| model_options.backend | qwen | molmo |
| batch_size | 2 | 1 |
| python | Qwen TOPReward runtime | Molmo TOPReward runtime |
| checkpoint | Qwen checkpoint | Molmo2 checkpoint |

Change backend, batch_size, python and checkpoint together. Do not merely change
the checkpoint and assume that its loader or dependency environment is detected.

Only Molmo accepts model_options.max_frames (default 128) and
max_input_length (default 32768). Shared view, sample_fps, prompt and reduction
options remain in the adapter. The backend is explicit in saved configuration;
publication identities distinguish Molmo from Qwen checkpoints.
Replace the old model: topreward_molmo with model: topreward and
model_options: {backend: molmo}. The separate adapter and standalone Molmo
document are removed; loader, processor and reward implementations remain
backend-specific inside one adapter. This is local model execution, not HTTP API
support. Old code fingerprints require a new run directory or explicit rerun.

VOC is the forward half of Cycle-VOC, including its shared turn and normal-ST
cohort. The former standalone VOC implementation is removed; its entry point
now lives in metrics/cycle_voc_vs/forward.py. VOC excludes TRR branches and
diverse solutions and requires at least three annotated milestones. Its protocol
is voc-cycle-forward-v2; older VOC caches must not be reused. The voc result key
is retained, while query IDs now identify cycle forward positions. When VOC and
Cycle-VOC use the same base-mode task/domain selection, one cycle inference
provides both scores. Native contracts remain separate and are not cross-reused.
VOC-only execution infers only the forward half. MEM-VOC remains distinct.
MEM-VOC is the public name of the memory metric. The YAML/result key voc_mem,
query IDs and legacy dataset declaration VOC-MEM remain unchanged; MEM-VOC is
also accepted as a metric-selection alias. This is a naming change, not a
scoring or input-protocol change.

VS reuses the forward 5 Hz grid from Cycle-VOC, with unchanged vs_v1 scoring.
The combined implementation is named Cycle-VOC&VS under metrics/cycle_voc_vs/.
It selects normal successful trajectories once for a shared run, excluding
TRR branches and diverse solutions. Both metric keys and scores remain distinct;
do not introduce a combined scalar or rename cycle_voc/vs in existing configs.
Select both with mode base to share one inference run; YAML order does not
matter. Cycle-VOC must cover all VS tasks and domains. A VS-only run computes
its Cycle-VOC dependency automatically. VOC/CYCLE-VOC task declarations are
used for VS selection. Missing annotated 5 Hz frames are errors, not resampled
or interpolated predictions. Old VS result caches use an incompatible input
protocol and require a new output directory.

### Separate ID and OOD reporting

Cycle-VOC&VS already groups predictions and scores by the original episode
domain: id is ID, env is ENV-OOD and emb is EMB-OOD. Never relabel OOD episodes
as ID or combine ENV-OOD and EMB-OOD into an undocumented single score.
Both metrics must select the intended domains explicitly:

    metrics:
      cycle_voc:
        mode: base
        domains: [id]
      vs:
        mode: base
        domains: [id]

Use [env] or [emb] in both blocks for separate OOD runs. If separate output
directories are desired, use distinct runtime config stems, for example
baseline-idconfigs.yaml, baseline-envconfigs.yaml and baseline-embconfigs.yaml.
Outputs are OUTPUT/<config-stem>/metrics.json and operations.jsonl. These are
runtime selections of the same family template, not additional public model
templates or new configuration fields.

Alternatively, select [id, env, emb] in both blocks for one shared run. Report
metrics.json's cycle_voc.domains.<domain>.mean and vs.domains.<domain>.mean;
their tasks mappings preserve per-task scores, and VS episode records retain
each domain. The existing top-level mean averages selected domain means and
is not a substitute for separate ID/ENV-OOD/EMB-OOD results. It is unchanged
for compatibility. Domains without eligible normal successful trajectories
raise a coverage error when requested; record that coverage as N/A, not zero.

SIA judging automatically reads config.yaml beside the SIA implementation,
at src/vmbmk/metrics/sia/config.yaml in a source checkout. It is a private local
file ignored by Git, not a public evaluation configuration:

    api_key: "YOUR_PRIVATE_API_KEY"
    base_url: "https://api.deepseek.com/chat/completions"
    model: "YOUR_EXISTING_JUDGE_MODEL"

Use the actual endpoint/model supplied by your judge deployment. All three
fields are optional; null means unset. On POSIX, create/edit the file privately
and restrict access with chmod 600 src/vmbmk/metrics/sia/config.yaml. Group/other
access is rejected. On Windows, restrict access to your own user. The path is
resolved relative to config.py, not the shell's working directory.

For a different private location, set VMBMK_SIA_CONFIG to its path. An explicitly
configured missing or invalid file is an error, not a silent fallback. Only
api_key, base_url and model are accepted. Settings precedence is explicit
judge arguments, then DEEPSEEK_API_KEY/DEEPSEEK_BASE_URL/DEEPSEEK_MODEL environment
variables, then private file, then existing endpoint/model defaults. No API key
default exists. Git ignore does not protect manual archives, scp or forced
staging: exclude this file when transferring or publishing project code.
Create it separately on each machine that runs judging. The file is not copied
into run configs, response bundles,
judgement identities or result artifacts; no evaluation YAML field is added.

Then run python -m vmbmk.cli run CONFIG --sia-stage judge. Judging reuses saved
generation responses and does not reload the GPU model. Never paste keys into
task messages, shell history, logs or public configs; this file is plaintext,
not an encrypted credential store.

Supported keys: sa, sia, tga_easy, tga_hard, voc, voc_mem, cycle_voc, fpl,
trr, vs and csvc. Mode base is the shared contract; native is allowed only
where an existing metric/adapter explicitly supports it. Preserve the model's
history policy and output semantics. Select only covered tasks/domains;
unsupported combinations are not zero-score results. The same EvaluationPolicy
is used by single-config and batch entry points. A single-config request raises
on a prohibited combination; batch records the reason and skips it. Planner
preflight uses each task's available domains, then validates the complete selected
cohort against the requested domains.

## One-shot and model-specific settings

Model defaults live in their Adapters; YAML supplies required locations and
non-default overrides. Historical field migrations are not the current
configuration contract.
VLAC selects one_shot when reference_data is supplied and zero_shot otherwise;
its default reference view is observation.images.cam_high with ref_num=6.
The CLI and YAML entry points both defer this default to the adapter.
An explicit shot_mode remains an override. RoboDopamine defaults
to one_shot with front references but still consumes three evaluation views.
References must not overlap evaluated episodes. ProcVLM enables LoRA when
model_options.checkpoint_by_task maps exact task IDs to verified LoRA
checkpoints; without that mapping it uses the base model. An explicit
model_options.use_lora remains an override. Neither mode nor its publication
identity is inferred from a one-shot substring in a checkpoint path. Its default max_model_len is
32768. Its source_root option is not consumed by the current Adapter;
the evqa package must already be available in the selected environment. LIV
still requires source_root to load its upstream implementation. Missing checkpoints
must not silently fall back to another task or the base model. Existing adapter
options, such as view, context sampling and batching, retain their semantics.
Templates do not auto-discover references or rewrite their instructions.

## Credentials

Supply API credentials in the environment. The CLI reads a secrets file only
when VMBMK_SECRETS_FILE is explicitly set; existing environment values win.
An explicitly configured missing file is an error, even when credentials are
already available in the environment. Leave the variable unset to use only
environment credentials.
Files must satisfy the existing private-file permission and syntax checks.
Never put keys or OTPs in configurations, output metadata or this repository.

## Output and publication

Runs write output/<config-stem>/metrics.json and config.yaml, along with
metric-specific operation records. Metric objects preserve mean/domain/task
results and protocol metadata where defined by that metric. Auxiliary details
are not substitutes for primary scores. Never label missing domains as observed.

Use tools/results.py publish explicitly with --results-root and
--final-results-root to publish a completed run. --merge performs intentional
compatible result merges; keep model/checkpoint and protocol identities separate.
Old metric checkpoints are not evidence of reproduction under a newer protocol.

CSVC (csvc) supports domains: [id] only. Its result adds an ID domain view while
retaining scientific task_results and aggregation. SIA generation returns an
explicit generated artifact, never a score. See docs/result_publication.md for
the versioned publication index and supported incremental merges.

Website consumers should read exported artifacts and display N/A for unsupported
coverage; inference belongs in offline model workers, not website requests.
