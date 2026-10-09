<div align="center">
  <img src="docs/assets/robovalue-logo.png" alt="RoboValue" width="520" />
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Paper-Coming_soon-D85C6A?logo=arxiv&amp;logoColor=white" alt="Paper: coming soon" />
  <a href="https://robovalue-benchmark.github.io/"><img src="https://img.shields.io/badge/Website-6853C9?logo=googlechrome&amp;logoColor=white" alt="Project website" /></a>
  <img src="https://img.shields.io/badge/Dataset-Coming_soon-D49A3A?logo=huggingface&amp;logoColor=white" alt="Dataset download: coming soon" />
  <a href="https://robovalue-benchmark.github.io/doc/"><img src="https://img.shields.io/badge/Documentation-409C96?logo=readthedocs&amp;logoColor=white" alt="Documentation" /></a>
  <a href="https://robovalue-benchmark.github.io/leaderboard/"><img src="https://img.shields.io/badge/Leaderboard-527BC3?logo=weightsandbiases&amp;logoColor=white" alt="Leaderboard" /></a>
  <a href="https://robovalue-benchmark.github.io/community/"><img src="https://img.shields.io/badge/Community-4F9D69?logo=wechat&amp;logoColor=white" alt="Community" /></a>
</div>

<div align="center">

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md) [![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](README.zh-CN.md)

</div>

<h1 align="center">
  <sub>RoboValue: A Fine-Grained Sim-and-Real Benchmark<br />for Unified Evaluation of Robotic Value Models</sub>
</h1>

**RoboValue** is an evaluation benchmark for robotic value models across simulation and the real world. It measures task-state understanding, temporal progress monitoring, failure and recovery reasoning, and value consistency through fine-grained diagnostic trajectories. Shared scalar, pairwise, and textual interfaces enable unified evaluation of heterogeneous models while preserving their native value semantics.

<div align="center">
  <img src="docs/assets/overview.png" alt="RoboValue overview: sim-and-real data, shared model interfaces, and four evaluation capabilities" width="100%" />
</div>

<p align="center">
  <a href="#leaderboard">Leaderboard</a> ·
  <a href="#whats-new">What's New</a> ·
  <a href="#benchmark-overview">Benchmark Overview</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#contributing">Contributing</a> ·
  <a href="#citation-and-acknowledgement">Citation</a>
</p>

## Leaderboard

Compare robotic value models across the four evaluation capabilities and simulation, real-world, and generalization conditions on the **[interactive leaderboard](https://robovalue-benchmark.github.io/leaderboard/)**. Zero-Shot and One-Shot configurations are ranked **within their own tracks**.

The table shows the top three configurations per track from the [2026-10-07 manuscript snapshot](https://github.com/RoboValue-Benchmark/RoboValue-Benchmark.github.io/blob/c1c54ae51dd79dd8d292b9155c128d61b68b1535/public/data/results.json). Overall is the mean of four normalized capability scores on a **0–100** scale (higher is better); **SIA is reported separately and excluded from Overall**. See the full leaderboard for metric coverage, aggregation rules, and model details.

| Track | Rank | Model | Overall ↑ |
| --- | :---: | --- | ---: |
| **Zero&#8209;Shot** | 🥇 1 | RoboMeter-4B | **60.05** |
| Zero-Shot | 🥈 2 | RynnValue-4B | 57.70 |
| Zero-Shot | 🥉 3 | RynnValue-8B | 56.25 |
| **One&#8209;Shot** | 🥇 1 | Robo-Dopamine 2.0-8B Preview | **58.27** |
| One-Shot | 🥈 2 | ProcVLM-2B | 57.82 |
| One-Shot | 🥉 3 | Robo-Dopamine 2.0-4B Preview | 57.52 |

Few-Shot evaluation is complete; its protocol details and leaderboard entries will be added. Full-Data evaluation is planned. See [evaluation protocols and status](#evaluation-protocols-and-status) below.

## What's NEW!

- **[2026/10]** 🎉 Few-Shot evaluation is complete. Protocol details and leaderboard entries will follow.
- **[2026/10]** 🔥 Zero-Shot and One-Shot results are available on the [leaderboard](https://robovalue-benchmark.github.io/leaderboard/).
- **[2026/10]** 🔥 Evaluation code, [baseline setup guides](docs/baselines/README.md), and the [project website](https://robovalue-benchmark.github.io/) are available. Public paper and dataset download links are coming soon.

## Benchmark Overview

RoboValue connects **diagnostic data**, **capability-specific metrics**, and **controlled evaluation protocols** to examine what robotic value models understand throughout task execution.

### Dataset

**RoboValue-Dataset** covers **35 dual-arm manipulation tasks**. It pairs expert training demonstrations with a separate annotated test split spanning standard execution, failures and recovery, long-horizon temporal reasoning, and alternative valid solutions.

| Simulation tasks | Real-world tasks | Training demonstrations | Test trajectories |
| :---: | :---: | :---: | :---: |
| **15** | **20** | **3,500** (100 per task) | **2,792** |

<div align="center">
  <img src="docs/assets/dataset.png" alt="RoboValue dataset: simulation and real-world tasks with failure and recovery, temporal, and multi-solution trajectories" width="100%" />
</div>

| Domain | Standard embodiment | Cross-embodiment platform |
| --- | --- | --- |
| **Simulation** | ARX | Dual-arm UR5e |
| **Real world** | AgiBot Genie02 | Dual-arm ARX |

Training demonstrations are collected under standard in-domain conditions. Test trajectories cover **in-domain**, **cross-embodiment**, and **cross-environment** conditions, with annotations for subtask identification, failure localization, recovery reasoning, and cross-solution comparison.

**Dataset downloads are coming soon.** The full dataset is distributed separately from this code repository. See the [data preparation guide](docs/data.md) for schemas, annotations, and asset paths.

### Evaluation Metrics

Four complementary capabilities are assessed through **11 metrics**, including SIA as a separate subtask-identification evaluation. Each capability tests a different aspect of value-model behavior.

| Capability | What it evaluates | Metrics |
| --- | --- | --- |
| **Task-State Understanding** | Successful execution, instruction grounding, and the active subtask | SA, TGA-CT, TGA-CF, SIA |
| **Temporal Progress Monitoring** | Progress and regression, including similar visual states with different histories | VOC, Cycle-VOC, Memory-VOC |
| **Failure and Recovery Reasoning** | Failure onset, unresolved errors, and effective or ineffective recovery | FPL, TRR |
| **Value Consistency** | Stable feedback and comparable subtask gains across valid solutions | VS, CSVC |

**FPL ↓** measures localization error; **all other primary metrics ↑** are better when higher. Metric-level results report missing or unsupported coverage as **N/A**. Simulation and real-world results, including the two distribution shifts, are reported separately; TRR and CSVC use in-domain trajectories. See the [metric definitions](https://robovalue-benchmark.github.io/doc/get-started/protocol/) for what each metric measures and the [implementation and protocol notes](docs/metric_alignment.md) for scoring alignment and version changes.

For YAML configuration, TGA-CT, TGA-CF, and Memory-VOC use `tga_easy`, `tga_hard`, and `voc_mem`, respectively. Memory-VOC is called MEM-VOC in the code documentation; see [metric selection](docs/configuration.md#metric-selection).

### Evaluation Protocols and Status

Evaluation settings specify the **task-specific demonstrations available to a model** for conditioning or adaptation. They are distinct from the test conditions above. Reference and adaptation data are kept separate from test trajectories.

| Setting | Task-specific data | Evaluation status | Results and setup |
| --- | --- | --- | --- |
| **Zero&#8209;Shot** | None | ✅ Complete | [Leaderboard](https://robovalue-benchmark.github.io/leaderboard/) · [Setup guides](docs/baselines/README.md) |
| **One&#8209;Shot** | 1 training demonstration per task | ✅ Complete | [Leaderboard](https://robovalue-benchmark.github.io/leaderboard/?track=one) · [Setup guides](docs/baselines/README.md) |
| **Few&#8209;Shot** | Several demonstrations per task; count to be documented | ✅ Complete | Protocol, configurations, and results to be added |
| **Full&#8209;Data** | Complete training split | 📋 Planned | Results to follow |

The released tools document Zero-Shot and One-Shot model settings. See [configuration and commands](docs/configuration.md) for supported model and metric combinations; baseline templates must be adapted to each experiment.

## Quick Start

The steps below assume you already have access to the benchmark data; public downloads are coming soon. Start with **one baseline**, prepare its runtime and checkpoint, then validate the data and run an evaluation. The supplied installation path targets **Linux x86_64** and requires a launcher with **Python 3.10+ and PyYAML**, plus **uv** and the baseline-specific CUDA/build dependencies. See the [shared prerequisites](docs/baselines/README.md#prerequisites) before installation.

**1. Clone the repository.**

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/RoboValue-Benchmark/RoboValue.git
cd RoboValue
```

This defers the LIV weight download until needed; LIV users should follow the [LIV guide](docs/baselines/liv.md) for its Git LFS assets.

**2. Set up a baseline and download its checkpoint.**

The example below uses **RoboReward** with an existing **Python 3.10.x** interpreter. Complete the shared prerequisites, including the `hf` CLI, and provide FFmpeg 4–8 shared libraries as described in the [RoboReward guide](docs/baselines/roboreward.md).

```bash
bash envs/setup.sh roboreward --plan
bash envs/setup.sh roboreward

hf download teetone/RoboReward-8B \
  --revision 3a185b4fce2b1253643105be1f234ae618b9732f \
  --local-dir checkpoints/RoboReward-8B
```

Model dependencies run in isolated environments. For other models, choose a [baseline guide](docs/baselines/README.md); each guide specifies source code, compatible dependencies, checkpoint assets, and verification scope.

**3. Prepare the data, configure the run, and evaluate.**

Obtain the dataset separately and follow the [data guide](docs/data.md). Copy the baseline template:

```bash
cp configs/roborewardconfigs.yaml configs/roboreward-local.yaml
```

For data placed in `data/dataset_real/`, edit `configs/roboreward-local.yaml` as below. This starter run evaluates **SA on one task in the ID condition**. Set an available GPU and replace `organize_table` if your dataset uses a different task ID.

```yaml
model: roboreward
gpu: 0
batch_size: 1
python: ../.model-envs/roboreward/bin/python
checkpoint: ../checkpoints/RoboReward-8B
data: ../data/dataset_real
output: ../output
metrics:
  sa:
    mode: base
    domains: [id]
tasks: [organize_table]
```

The YAML paths are relative to `configs/`: `../data/dataset_real` points to the same dataset as the CLI argument `data/dataset_real` when you run these commands from the repository root. Use absolute paths for data, environments, or checkpoints stored elsewhere.

```bash
bash vmbmk.sh validate data/dataset_real
bash vmbmk.sh run configs/roboreward-local.yaml
```

Validation checks schemas and assets without loading a model and prints `valid` on success. With this configuration, a successful evaluation writes **`output/roboreward-local/metrics.json`**, the saved **`config.yaml`**, and metric-specific operation/provenance artifacts in the same run directory. After the first run succeeds, copy the configuration to a new filename under `configs/` to create a separate run directory before expanding the task and eligible metric selections. See [configuration and commands](docs/configuration.md).

<details>
<summary><strong>Baseline guides and configuration templates</strong></summary>

| Baseline | Installation | Configuration |
| --- | --- | --- |
| **RoboMeter** | [Guide](docs/baselines/robometer.md) | [YAML](configs/robometerconfigs.yaml) |
| **Robo-Dopamine** | [Guide](docs/baselines/robodopamine.md) | [YAML](configs/robodopamineconfigs.yaml) |
| **ProcVLM** | [Guide](docs/baselines/procvlm.md) | [YAML](configs/procvlmconfigs.yaml) |
| **RoboReward** | [Guide](docs/baselines/roboreward.md) | [YAML](configs/roborewardconfigs.yaml) |
| **VLAC** | [Guide](docs/baselines/vlac.md) | [YAML](configs/vlacconfigs.yaml) |
| **TOPReward** (Qwen / Molmo) | [Guide](docs/baselines/topreward.md) | [YAML](configs/toprewardconfigs.yaml) |
| **RoboFAC** | [Guide](docs/baselines/robofac.md) | [YAML](configs/robofacconfigs.yaml) |
| **RynnValue** | [Guide](docs/baselines/rynnvalue.md) | [YAML](configs/rynnvalueconfigs.yaml) |
| **LIV** | [Guide](docs/baselines/liv.md) | [YAML](configs/livconfigs.yaml) |
| **FailSafe-labeled SIA integration** | [Provenance and setup](docs/baselines/failsafe.md) | [YAML](configs/failsafeconfigs.yaml) |

The FailSafe-labeled integration is available for SIA; equivalence to the official implementation remains unverified. GVL and ReWiND are excluded from the provided baseline installer. Individual guides distinguish available integrations from tested execution.

</details>

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

These paths are conventions; external storage is supported through configuration paths. Downloaded code, datasets, weights, environments, caches, and outputs stay out of Git, except for the explicitly distributed LIV assets. Other model checkpoints must be supplied separately.

</details>

<details>
<summary><strong>More documentation</strong></summary>

| Guide | Contents |
| --- | --- |
| [Baseline setup](docs/baselines/README.md) | Installation prerequisites and model-specific instructions |
| [Data preparation](docs/data.md) | Dataset structure, metadata, annotations, and asset paths |
| [Configuration and commands](docs/configuration.md) | YAML fields, evaluation modes, and CLI usage |
| [Environment details](docs/environments.md) | Lockfiles, custom roots, and native builds |
| [Metric implementation notes](docs/metric_alignment.md) | Scoring alignment and protocol changes |
| [Result interfaces](docs/result_publication.md) | Validation, publication, and SIA intermediate outputs |
| [Developer guide](docs/developer_guide.md) | Package layout, call flow, and extension points |
| [Testing and evidence](docs/testing.md) | CPU tests, bounded model checks, and known limitations |

</details>

## Contributing

Contributions are welcome! Help extend RoboValue with **model adapters**, **reproducible evaluation results**, **task and annotation improvements**, or **documentation fixes**. Start with the [developer guide](docs/developer_guide.md) and [testing guide](docs/testing.md), then open a pull request describing the change and its validation.

For questions or bug reports, open an [issue](https://github.com/RoboValue-Benchmark/RoboValue/issues) with the baseline, metric, environment versions, configuration without credentials, and relevant traceback. Join the [RoboValue community](https://robovalue-benchmark.github.io/community/) to discuss evaluation protocols and robotic value models.

## Citation and Acknowledgement

If you find **RoboValue** helpful in your research, please cite our work. The public paper link and BibTeX entry will be added when publication details are finalized.

We thank the teams behind the model implementations and checkpoints used by RoboValue, including [RoboMeter](docs/baselines/robometer.md), [Robo-Dopamine](docs/baselines/robodopamine.md), [ProcVLM](docs/baselines/procvlm.md), [RoboReward](docs/baselines/roboreward.md), [VLAC](docs/baselines/vlac.md), [TOPReward](docs/baselines/topreward.md), [RoboFAC](docs/baselines/robofac.md), [RynnValue](docs/baselines/rynnvalue.md), and [LIV](docs/baselines/liv.md). Their setup guides link to the corresponding upstream projects and model assets.
