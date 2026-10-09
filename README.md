<div align="center">
  <img src="docs/assets/robovalue-logo.png" alt="RoboValue" width="520" />
</div>

<div align="center">
  <!-- Add official arXiv and Hugging Face URLs once released. -->
  <img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv" alt="arXiv: paper" height="20" />
  <img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&amp;logoColor=white" alt="HuggingFace" height="20" />
  <a href="https://robovalue-benchmark.github.io/doc/"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&amp;logo=readthedocs" alt="Documentation" height="20" /></a>
  <!-- Add the Chinese documentation URL when available. -->
  <img src="https://img.shields.io/badge/中文文档-red?logo=readthedocs" alt="中文文档" height="20" />
  <a href="https://robovalue-benchmark.github.io/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&amp;logoColor=white" alt="Website" height="20" /></a>
  <a href="https://robovalue-benchmark.github.io/leaderboard/"><img src="https://img.shields.io/badge/Leaderboard-527BC3?logo=weightsandbiases&amp;logoColor=white" alt="Leaderboard" height="20" /></a>
  <a href="https://robovalue-benchmark.github.io/community/"><img src="https://img.shields.io/badge/Community-2E8B57" alt="Community" height="20" /></a>
</div>

<div align="center">

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md) [![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](README.zh-CN.md)

</div>

<h1 align="center">
  <sub>RoboValue: A Fine-Grained Sim-and-Real Benchmark<br />for Unified Evaluation of Robotic Value Models</sub>
</h1>

**RoboValue** is an evaluation benchmark for robotic value models across simulation and the real world. It measures task-state understanding, temporal progress monitoring, failure and recovery reasoning, and value consistency through fine-grained diagnostic trajectories.

<div align="center">
  <img src="docs/assets/overview.png" alt="RoboValue overview: sim-and-real data, shared model interfaces, and four evaluation capabilities" width="100%" />
</div>

## 🏆 Leaderboard

View full rankings and model configurations on the [**RoboValue Leaderboard**](https://robovalue-benchmark.github.io/leaderboard/), with separate Zero-Shot and One-Shot tracks.

## What's NEW!

- [2026/10] 🔥 Our paper **RoboValue: A Fine-Grained Sim-and-Real Benchmark for Unified Evaluation of Robotic Value Models** is officially released.

## ✨ Highlights

- 🎯 **Fine-grained value evaluation.** Assess task-state understanding, temporal progress monitoring, failure and recovery reasoning, and value consistency through a shared evaluation protocol.
- 🤖 **35 tasks across simulation and the real world.** RoboValue includes **15 simulation tasks** adapted from [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/README.md) and **20 real-world dual-arm manipulation tasks**, with **3,500 expert training demonstrations** and **2,792 separate test trajectories**.
- 🔍 **Diagnostic execution scenarios.** Counterfactual instructions, recurring visual states, effective and ineffective recoveries, and alternative valid subtask orders expose errors that outcome accuracy and forward-progress correlation can overlook.
- 🌍 **Controlled generalization tests.** Evaluate in-domain and under separate **cross-embodiment** and **cross-environment** shifts, without additional adaptation to the shifted conditions.

<div align="center">
  <img src="docs/assets/dataset.png" alt="RoboValue dataset: simulation and real-world tasks with failure and recovery, temporal, and multi-solution trajectories" width="100%" />
</div>

**Dataset downloads are coming soon.** See the [data preparation guide](docs/data.md) for dataset structure and usage.

<details>
<summary><strong>Capability and metric reference</strong></summary>

| Capability | What it evaluates | Metrics |
| --- | --- | --- |
| **Task-State Understanding** | Successful execution, instruction grounding, and the active subtask | SA, TGA-CT, TGA-CF, SIA |
| **Temporal Progress Monitoring** | Progress and regression, including similar visual states with different histories | VOC, Cycle-VOC, Memory-VOC |
| **Failure and Recovery Reasoning** | Failure onset, unresolved errors, and effective or ineffective recovery | FPL, TRR |
| **Value Consistency** | Stable feedback and comparable subtask gains across valid solutions | VS, CSVC |

**FPL ↓** measures localization error; **all other primary metrics ↑** are better when higher. **SIA is reported separately and excluded from the leaderboard's Overall score.** Missing or unsupported results are shown as **N/A**. See the [metric definitions](https://robovalue-benchmark.github.io/doc/get-started/protocol/) for scoring and evaluation conditions.

</details>

### Evaluation Status

Tracks specify the **task-specific demonstrations available for conditioning or adaptation**. These demonstrations are kept separate from test trajectories; generalization conditions are evaluated independently.

| Setting | Task-specific data | Status |
| --- | --- | --- |
| **Zero-Shot** | None | ✅ Results available |
| **One-Shot** | 1 training demonstration per task | ✅ Results available |
| **Few-Shot** | Multiple demonstrations per task | 📋 Planned |
| **Full-Data** | Complete training split | 📋 Planned |

## 🧩 Supported Value Models

Model-specific adapters expose shared **scalar scoring**, **pairwise comparison**, and **textual subtask** interfaces while retaining model-dependent value semantics. **✅** marks an available adapter interface, including model-specific conversions; **—** means unavailable. Model names link to setup guides; see the [interface definitions](docs/developer_guide.md#query-and-result-records) for input/output formats.

| Model family | Scalar | Pairwise | Text (SIA) | Configuration |
| --- | :---: | :---: | :---: | --- |
| [**RoboMeter**](docs/baselines/robometer.md) | ✅ | ✅ | — | [YAML](configs/robometerconfigs.yaml) |
| [**Robo-Dopamine**](docs/baselines/robodopamine.md) | ✅ | ✅ | — | [YAML](configs/robodopamineconfigs.yaml) |
| [**ProcVLM**](docs/baselines/procvlm.md) | ✅ | ✅ | ✅ | [YAML](configs/procvlmconfigs.yaml) |
| [**RoboReward**](docs/baselines/roboreward.md) | ✅ | ✅ | — | [YAML](configs/roborewardconfigs.yaml) |
| [**VLAC**](docs/baselines/vlac.md) | ✅ | ✅ | — | [YAML](configs/vlacconfigs.yaml) |
| [**TOPReward (Qwen / Molmo)**](docs/baselines/topreward.md) | ✅ | ✅ | — | [YAML](configs/toprewardconfigs.yaml) |
| [**RoboFAC**](docs/baselines/robofac.md) | ✅ | ✅ | ✅ | [YAML](configs/robofacconfigs.yaml) |
| [**RynnValue**](docs/baselines/rynnvalue.md) | ✅ | ✅ | — | [YAML](configs/rynnvalueconfigs.yaml) |
| [**LIV**](docs/baselines/liv.md) | ✅ | ✅ | — | [YAML](configs/livconfigs.yaml) |
| [**FailSafe-labeled integration**](docs/baselines/failsafe.md)† | — | — | ✅ | [YAML](configs/failsafeconfigs.yaml) |

Metric eligibility and verification scope vary by model; see the [configuration guide](docs/configuration.md#metric-selection) and each setup guide. †The SIA-only FailSafe integration requires the original local assets; equivalence to the official implementation remains unverified.

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
<summary><strong>More documentation</strong></summary>

| Guide | Contents |
| --- | --- |
| [Baseline setup](docs/baselines/README.md) | Installation prerequisites and model-specific instructions |
| [Data preparation](docs/data.md) | Dataset structure, metadata, annotations, and asset paths |
| [Configuration and commands](docs/configuration.md) | YAML fields, evaluation modes, and CLI usage |
| [Model API integration](docs/api.md) | Organizer-run remote inference, service contract, and CPU mock validation |
| [Environment details](docs/environments.md) | Lockfiles, custom roots, and native builds |
| [Metric implementation notes](docs/metric_alignment.md) | Scoring alignment and protocol changes |
| [Result interfaces](docs/result_publication.md) | Validation, publication, and SIA intermediate outputs |
| [Developer guide](docs/developer_guide.md) | Package layout, call flow, and extension points |
| [Testing and evidence](docs/testing.md) | CPU tests, bounded model checks, and known limitations |

</details>

## 🗂️ Repository Structure

```text
RoboValue/
├── configs/              # Per-model evaluation templates
├── envs/                 # Isolated runtimes, dependency locks, and setup scripts
├── src/vmbmk/
│   ├── adapters/         # Model integrations and shared interfaces
│   ├── metrics/          # Query planning and metric scoring
│   ├── data/             # Dataset validation and trajectory playback
│   ├── inference/        # Typed queries and model workers
│   ├── runner/           # Evaluation, batching, and resume
│   ├── tools/            # Data checks, result handling, and visualization
│   └── cli.py            # Command-line entry points
├── docs/                 # Setup guides and protocol documentation
├── tests/                # Metric, adapter, and data tests
└── vmbmk.sh              # Evaluation launcher
```

Datasets, downloaded model assets, environments, and generated outputs are configured separately; see [data preparation](docs/data.md) and [baseline setup](docs/baselines/README.md#source-runtime-and-checkpoint-roots) for storage conventions. LIV's distributed assets use Git LFS.

## Contributing

Contributions are welcome! Help extend RoboValue with **model adapters**, **reproducible evaluation results**, **task and annotation improvements**, or **documentation fixes**. Start with the [developer guide](docs/developer_guide.md) and [testing guide](docs/testing.md), then open a pull request describing the change and its validation.

For questions or bug reports, open an [issue](https://github.com/RoboValue-Benchmark/RoboValue/issues) with the baseline, metric, environment versions, configuration without credentials, and relevant traceback. Join the [RoboValue community](https://robovalue-benchmark.github.io/community/) to discuss evaluation protocols and robotic value models.

## Citation

If you find **RoboValue** helpful in your research, please cite our paper:

```bibtex
% BibTeX citation to be added.
```
