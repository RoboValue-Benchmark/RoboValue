# Prepare evaluation data

[Back to README](../README.md) | [Baseline setup](baselines/README.md) | [Run configuration](configuration.md)

This guide describes data storage and validation for organizer-side evaluation
deployments. The RoboValue team supplies the benchmark data and keeps the
held-out test set private for centralized evaluation. Model submitters provide
an inference service, adapter and runnable synthetic example, as described in
the [service and adapter guide](api.md); they do not need to source an additional
robotics dataset.

## Choose a dataset root

Use the RoboValue dataset supplied for the evaluation deployment. Dataset assets
are distributed separately from the source code; the private test set is not publicly
available for download or local evaluation. This repository does not convert
arbitrary videos into benchmark annotations.
Recommended roots are `data/dataset_sim/` and `data/dataset_real/`; alternative
storage locations are accepted by the YAML `data` field.

Point `data` to the directory whose immediate children are task directories:

```text
data/dataset_real/
  organize_table/
    metadata.json
    episodes/
      episode_001/
        metadata.json
        annotation.json
        front.mp4
        wrist_left.mp4          when required by the baseline/data protocol
        wrist_right.mp4         when required by the baseline/data protocol
        robot.parquet          when declared by metadata
```

Filenames for assets are examples: actual videos and robot data are resolved
from episode metadata. Task metadata declares task identity, instructions,
subtasks and supported metrics. Episode metadata records identity, success,
world type, domain, FPS, frame count and relative asset paths. `annotation.json`
contains the metric-specific annotations. Preserve the supplied schemas rather
than inferring success or stages from filenames.

Asset paths are resolved relative to the episode directory. Both episode-local
files and explicit cross-directory paths such as
`../../../../raw_data/<collection>/<task>/videos/front.mp4` are supported,
including symbolic links to shared storage. Preserve that relative directory
layout when moving a dataset. Absolute asset paths and missing files are rejected;
the loader does not search alternate roots or substitute similarly named files.
Dataset metadata therefore selects files outside the dataset directory as well;
only load trusted metadata. This rule applies to videos and declared robot data.

Do not point `data` to `data/` if it contains several datasets and references.

## Configure paths and coverage

For a config stored under `configs/`, top-level paths are relative to **that
config's directory**, not the shell's current directory. For example:

```yaml
data: ../data/dataset_real
checkpoint: ../checkpoints/RoboReward-8B
python: ../.model-envs/roboreward/bin/python
output: ../output
tasks: [organize_table]
```

Replace task IDs with tasks that actually exist and support the selected metric.
Simulation/real and ID/ENV-OOD/EMB-OOD results are distinct; do not combine them
into an invented common average. See [metric contracts](metric_alignment.md).
Nested `model_options` paths have adapter-specific resolution: use absolute
paths for LIV source roots and ProcVLM task LoRA checkpoints.

## One-shot references

The RoboValue team supplies one training demonstration per task for One-Shot.
Keep these references in a separate supplied dataset, for example
`data/reference/`, and set `reference_data: ../data/reference` when appropriate.
Do not use evaluated episodes as reference demonstrations. The reference schema
is baseline-dependent: RoboDopamine requires its reference input, while adding
references selects VLAC one-shot. ProcVLM one-shot instead uses task-specific
external LoRA checkpoints. Follow the corresponding baseline guide.

## Validate before inference

From the repository root, using the launcher Python with PyYAML:

```bash
bash vmbmk.sh validate data/dataset_real
```

This checks the full dataset schema and declared asset paths, not model readiness,
decoding every video or reproducing a score. Metric execution performs its own
metric-scoped validation. A missing annotation should be fixed in the supplied
data, not hidden by changing scoring or inventing default values.
For a read-only frame-count check, see the `data check-frames` command in the
[configuration guide](configuration.md).

## Keep data out of Git

The repository ignores evaluation datasets, checkpoint directories and generated
outputs. LIV's distributed assets are an explicit exception: `config.yaml` is
a regular Git file, while `model.pt` and `RN50.pt` use Git LFS; see the
[LIV guide](baselines/liv.md). Other weights remain outside Git.
Keep secrets out of run configs. Store result artifacts in the configured output
root; do not mix generated scores into the dataset or checkpoint folders.
