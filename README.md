<div align="center">
  <img src="docs/assets/robovalue-logo.png" alt="RoboValue" width="520" />
</div>

<div align="center">
  <!-- Add the paper and dataset URLs when the public links are available. -->
  <a href="" title="Paper link to be added"><img src="https://img.shields.io/badge/Paper-D85C6A?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white" alt="Paper (coming soon)" /></a>
  <a href="https://robovalue-benchmark.github.io/"><img src="https://img.shields.io/badge/Website-6853C9?style=for-the-badge&amp;logo=googlechrome&amp;logoColor=white" alt="Project website" /></a>
  <a href="" title="Dataset link to be added"><img src="https://img.shields.io/badge/Dataset-D49A3A?style=for-the-badge&amp;logo=huggingface&amp;logoColor=white" alt="Dataset (coming soon)" /></a>
  <a href="https://robovalue-benchmark.github.io/doc/"><img src="https://img.shields.io/badge/Documentation-409C96?style=for-the-badge&amp;logo=readthedocs&amp;logoColor=white" alt="Documentation" /></a>
  <a href="https://robovalue-benchmark.github.io/leaderboard/"><img src="https://img.shields.io/badge/Leaderboard-527BC3?style=for-the-badge&amp;logo=weightsandbiases&amp;logoColor=white" alt="Leaderboard" /></a>
  <a href="https://robovalue-benchmark.github.io/community/"><img src="https://img.shields.io/badge/Community-4F9D69?style=for-the-badge&amp;logo=wechat&amp;logoColor=white" alt="Community" /></a>
</div>

<h1 align="center">
  <sub>RoboValue: A Fine-Grained Sim-and-Real Benchmark<br />for Unified Evaluation of Robotic Value Models</sub>
</h1>

<p align="center">
  <strong>Task-State Understanding · Temporal Progress Monitoring<br />Failure and Recovery Reasoning · Value Consistency</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#benchmark">Benchmark</a> ·
  <a href="#dataset">Dataset</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#baseline-guides">Baselines</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="#citation">Citation</a>
</p>

## Overview

**RoboValue** evaluates how reliably robotic value models understand task
requirements and changes throughout execution. Its fine-grained diagnostic
scenarios examine instruction grounding, progress and regression, execution
history, recovery outcomes, and subtask credit across valid solutions.

Shared **scalar, pairwise, and textual interfaces** connect heterogeneous models
to a common evaluation framework while preserving their native value semantics.
The repository provides model adapters, metric implementations, configuration
templates, isolated baseline environments, and result tools for simulation and
real-world evaluation.

<table align="center">
  <tr>
    <td align="center"><strong>15</strong><br />Simulation tasks</td>
    <td align="center"><strong>20</strong><br />Real-world tasks</td>
    <td align="center"><strong>3,500</strong><br />Training demonstrations</td>
    <td align="center"><strong>2,792</strong><br />Test trajectories</td>
  </tr>
</table>

<div align="center">
  <img src="docs/assets/overview.png" alt="RoboValue framework: sim-and-real evaluation data, shared interfaces, and four complementary capability dimensions" width="100%" />
</div>

<p align="center">
  <sub>Diagnostic trajectories and instruction variations connect value predictions to task requirements, execution history, and recovery outcomes.</sub>
</p>

## What's New

- **[2026/10]** Evaluation code and baseline setup guides are available in this repository.
- **[2026/10]** The initial [project website](https://robovalue-benchmark.github.io/), documentation, and leaderboard are online. Public paper and dataset download links will be added as they become available.

## Benchmark

RoboValue evaluates four complementary capabilities through a shared protocol:

| Capability | What it evaluates | Metrics |
| --- | --- | --- |
| **Task-State Understanding** | Successful execution, instruction grounding, and the active subtask | SA, TGA-CT, TGA-CF, SIA |
| **Temporal Progress Monitoring** | Progress and regression, including visually similar states with different histories | VOC, Cycle-VOC, Memory-VOC |
| **Failure and Recovery Reasoning** | Failure onset, unresolved errors, and effective or ineffective recovery | FPL, TRR |
| **Value Consistency** | Stable, informative feedback and comparable subtask gains across valid solutions | VS, CSVC |

Results cover **in-domain**, **cross-embodiment**, and **cross-environment**
conditions. Simulation and real-world coverage, as well as the two types of
distribution shift, are tracked separately. Missing or unsupported coverage is
reported as **N/A**. FPL measures localization error, so lower is better; higher
is better for the other primary metrics.

| Evaluation track | Task-specific demonstrations | Status |
| --- | --- | --- |
| **Zero-Shot** | Released checkpoints without task-specific adaptation or reference demonstrations | Evaluated in the current paper |
| **One-Shot** | One training demonstration per task for conditioning or adaptation, separate from the test trajectories | Evaluated in the current paper |
| **Full-Data** | Training or fine-tuning on the complete training split | Planned |

Explore the [leaderboard](https://robovalue-benchmark.github.io/leaderboard/)
for capability and metric comparisons. See the
[metric contracts](docs/metric_alignment.md) for the implemented scoring rules,
cohorts, units, and protocol versions.

## Dataset

**RoboValue-Dataset** spans 35 dual-arm manipulation tasks with separate training
and annotated test splits:

- **Training:** 3,500 expert demonstrations, with 100 per task, collected under standard in-domain conditions.
- **Testing:** 2,792 separate trajectories covering standard, cross-embodiment, and cross-environment conditions.

<div align="center">
  <img src="docs/assets/dataset.png" alt="RoboValue dataset: 15 simulation and 20 real-world tasks with failure and recovery, long-horizon temporal, and multi-solution trajectories" width="100%" />
</div>

| Domain | Tasks | Standard embodiment | Cross-embodiment platform |
| --- | --- | --- | --- |
| **Simulation** | 15 | ARX | Dual-arm UR5e |
| **Real world** | 20 | AgiBot Genie02 | Dual-arm ARX |

Beyond fluent expert execution, diagnostic trajectories include **error
continuation**, **effective and ineffective recovery**, **recurring visual states
with different histories**, and **alternative valid subtask orders**. Test
annotations support subtask identification, failure localization, recovery-stage
reasoning, and cross-solution comparison.

Dataset download links are coming soon. Obtain the benchmark data separately
and follow the **[data preparation guide](docs/data.md)** for schemas, annotations,
and asset paths. This code repository does not include the full dataset.

## Quick Start

### 1. Clone the repository

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/RoboValue-Benchmark/RoboValue.git
cd RoboValue
```

The clone command defers the LIV weight download until needed. LIV users can
follow the [LIV guide](docs/baselines/liv.md) to retrieve its Git LFS assets.

### 2. Set up one baseline

Choose a **[baseline guide](#baseline-guides)** and complete its source,
environment, and checkpoint setup. Each guide specifies compatible dependencies,
download locations, local directories, configuration fields, and verification
scope.

The benchmark launcher requires **Python 3.10+** and **PyYAML**. Model dependencies
run in separate environments; the provided lock-based installer currently targets
**Linux x86_64**. See the [shared prerequisites](docs/baselines/README.md#prerequisites)
for Python, uv, CUDA/build requirements, and checkpoint tooling.

### 3. Prepare data and run an evaluation

Copy the selected baseline's template and replace every `/path/to/...`
placeholder. Set the model interpreter, checkpoint, dataset root, output path,
available GPUs, actual task IDs, and eligible metrics. For example, after
completing the RoboReward guide:

```bash
cp configs/roborewardconfigs.yaml configs/roboreward-local.yaml
# Edit python, checkpoint, data, output, gpu, tasks, and metrics first.
bash vmbmk.sh validate data/dataset_real
bash vmbmk.sh run configs/roboreward-local.yaml
```

`data` must point to **one dataset root containing task directories**, such as
`data/dataset_real/`. Validation checks the dataset schema and assets without
loading a model. A successful evaluation writes `metrics.json`, a saved
`config.yaml`, and metric-specific operation/provenance artifacts under the
configured output root. See [configuration and commands](docs/configuration.md).

<details>
<summary><strong>Recommended directory layout</strong></summary>

```text
RoboValue/
  data/
    dataset_sim/                 simulation dataset
    dataset_real/                real-world dataset
    reference/                   separate one-shot reference data
  checkpoints/<model-name>/      weights, tokenizer, and model configuration
  .baseline-sources/<baseline>/  downloaded upstream code
  .model-envs/<runtime>/         isolated Python environments
  .cache/huggingface/            optional download cache
  configs/                      editable run configurations
  output/<config-stem>/          generated evaluation artifacts
```

These paths are conventions; external storage is supported through configuration
paths. Downloaded code, datasets, weights, environments, caches, and outputs stay
out of Git, except for the explicitly distributed LIV assets. Other model
checkpoints must be supplied separately.

</details>

<details>
<summary><strong>Environment and evaluation commands</strong></summary>

| Command | Responsibility |
| --- | --- |
| `bash envs/source.sh BASELINE` | Download required upstream code at a fixed commit |
| `bash envs/setup.sh BASELINE --plan` | Check source paths, lock inputs, and platform prerequisites |
| `bash envs/setup.sh BASELINE` | Install locked dependencies and attach source import paths |
| `bash vmbmk.sh run CONFIG` | Run the selected evaluation using the YAML `python` interpreter |

Source acquisition does not download weights. RoboMeter requires an explicit GPU
for its final installation probe; LIV also requires CLIP RN50. Follow the
model-specific guides for the complete setup sequence.

</details>

## Baseline Guides

| Baseline | Installation | Configuration |
| --- | --- | --- |
| **RoboMeter** | [Guide](docs/baselines/robometer.md) | [YAML](configs/robometerconfigs.yaml) |
| **RoboDopamine** | [Guide](docs/baselines/robodopamine.md) | [YAML](configs/robodopamineconfigs.yaml) |
| **ProcVLM** | [Guide](docs/baselines/procvlm.md) | [YAML](configs/procvlmconfigs.yaml) |
| **RoboReward** | [Guide](docs/baselines/roboreward.md) | [YAML](configs/roborewardconfigs.yaml) |
| **VLAC** | [Guide](docs/baselines/vlac.md) | [YAML](configs/vlacconfigs.yaml) |
| **TOPReward** (Qwen / Molmo) | [Guide](docs/baselines/topreward.md) | [YAML](configs/toprewardconfigs.yaml) |
| **RoboFAC** | [Guide](docs/baselines/robofac.md) | [YAML](configs/robofacconfigs.yaml) |
| **RynnValue** | [Guide](docs/baselines/rynnvalue.md) | [YAML](configs/rynnvalueconfigs.yaml) |
| **LIV** | [Guide](docs/baselines/liv.md) | [YAML](configs/livconfigs.yaml) |
| **FailSafe-labeled SIA integration** | [Provenance and setup](docs/baselines/failsafe.md) | [YAML](configs/failsafeconfigs.yaml) |

The FailSafe-labeled integration is available for SIA; equivalence to the official
implementation remains unverified. GVL and ReWiND are excluded from the provided
baseline installer. Individual guides distinguish available integrations from
tested execution.

## Documentation

| Guide | Contents |
| --- | --- |
| [Baseline setup](docs/baselines/README.md) | Installation prerequisites and model-specific instructions |
| [Data preparation](docs/data.md) | Dataset structure, metadata, annotations, and asset paths |
| [Configuration and commands](docs/configuration.md) | YAML fields, evaluation modes, and CLI usage |
| [Environment details](docs/environments.md) | Lockfiles, custom roots, and native builds |
| [Metric contracts](docs/metric_alignment.md) | Scoring definitions, eligible cohorts, and protocol changes |
| [Result interfaces](docs/result_publication.md) | Validation, publication, and SIA intermediate outputs |
| [Developer guide](docs/developer_guide.md) | Package layout, call flow, and extension points |
| [Testing and evidence](docs/testing.md) | CPU tests, bounded model checks, and known limitations |

<details>
<summary><strong>Verification scope and developer checks</strong></summary>

Source acquisition and path checks were verified for the six source-dependent
baselines. Fresh-source imports passed in existing ProcVLM, VLAC, TOPReward Qwen,
and FailSafe environments. Clean-machine installation for every baseline has not
been established; LIV and RoboMeter new-environment validation remains pending.
See the [testing guide](docs/testing.md) for prerequisites and recorded evidence.

CPU scoring and interface tests require no weights, videos, or GPU:

```bash
PYTHONPATH=src:tests python -m pytest tests -q
```

Licensing and third-party redistribution review remains required for upstream
assets; checkpoint availability does not grant redistribution rights.

</details>

## Community

Join the [RoboValue community](https://robovalue-benchmark.github.io/community/)
to discuss the benchmark, evaluation protocols, and robotic value models.
For questions or bug reports, open an
[issue](https://github.com/RoboValue-Benchmark/RoboValue/issues) with the baseline,
metric, environment versions, configuration without credentials, and relevant
traceback.

## Citation

If RoboValue supports your research, please cite our work. The public paper link
and BibTeX entry will be added when publication details are finalized.
