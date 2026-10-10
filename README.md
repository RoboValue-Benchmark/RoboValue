<div align="center">
  <img src="docs/assets/robovalue-logo.png" alt="RoboValue" width="520" />
</div>

<div align="center">
  <!-- Add official arXiv and Hugging Face URLs when available. -->
  <img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv" alt="arXiv: paper link pending" height="20" />
  <img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&amp;logoColor=white" alt="Hugging Face: link pending" height="20" />
  <a href="https://robovalue-benchmark.github.io/doc/"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&amp;logo=readthedocs" alt="Documentation" height="20" /></a>
  <!-- Add the Chinese documentation URL when available. -->
  <img src="https://img.shields.io/badge/中文文档-red?logo=readthedocs" alt="Chinese documentation: link pending" height="20" />
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

**RoboValue** is an evaluation benchmark for robotic value models across simulation and the real-world, using fine-grained diagnostic trajectories to assess task outcomes and execution processes.

<div align="center">
  <img src="docs/assets/overview.png" alt="RoboValue overview: sim-and-real data, shared model interfaces, and four evaluation capabilities" width="100%" />
</div>

## 🏆 Leaderboard

View full rankings and model configurations on the [**RoboValue Leaderboard**](https://robovalue-benchmark.github.io/leaderboard/), with separate Zero-Shot and One-Shot tracks.

## 📰 What's New <a name="whats-new"></a>

- [2026/10] 🔥 Our paper **RoboValue: A Fine-Grained Sim-and-Real Benchmark for Unified Evaluation of Robotic Value Models** is officially released.

## ✨ Highlights

- 🎯 **Fine-grained value evaluation.** Assess task-state understanding, temporal progress monitoring, failure and recovery reasoning, and value consistency through a shared evaluation protocol.

- 🤖 **Diverse sim-and-real manipulation tasks.** RoboValue covers **15 simulation tasks** adapted from [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/README.md) and **20 real-world dual-arm manipulation tasks**.

- 🔍 **Diagnostic execution scenarios.** Counterfactual instructions, recurring visual states, effective and ineffective recoveries, and alternative valid subtask orders expose errors that outcome accuracy and forward-progress correlation can overlook.

- 🌍 **Controlled generalization tests.** Evaluate in-domain and under separate **cross-embodiment** and **cross-environment** shifts, without additional adaptation to the shifted conditions.

<div align="center">
  <img src="docs/assets/dataset.png" alt="RoboValue dataset: simulation and real-world tasks with failure and recovery, temporal, and multi-solution trajectories" width="100%" />
</div>

<details>
<summary><strong>Capability and metric reference</strong></summary>

| Capability | Metric | What it evaluates |
| --- | --- | --- |
| **Task-State Understanding** | **SA ↑** — Success Accuracy | Whether successful episodes receive higher terminal values than unsuccessful ones |
| | **TGA-CT / TGA-CF ↑** — Task Grounding Accuracy | Whether the correct instruction yields a larger value gain than cross-task (CT) or counterfactual (CF) instructions for the same execution |
| | **SIA ↑** — Subtask Identification Accuracy | Identification of the active subtask from generated descriptions, scored by a separate LLM judge |
| **Temporal Progress Monitoring** | **VOC ↑** — Value-Order Correlation | Rank correlation between predicted values and temporal progress along successful trajectories |
| | **Cycle-VOC ↑** | Progress and regression during continuous forward–reverse playback, with preceding history retained |
| | **Memory-VOC ↑** | Progress ordering in long-horizon trajectories where similar visual states recur under different execution histories |
| **Failure and Recovery Reasoning** | **FPL ↓** — Failure-Point Localization | Normalized timing error between the onset inferred from the largest detected value decline and the annotated failure onset |
| | **TRR ↑** — Trajectory Recovery Reasoning | Required value trends during initial failure, continued error, recovery attempts, and their successful or failed outcomes |
| **Value Consistency** | **VS ↑** — Value Stability | Stable, informative feedback that avoids unnecessary fluctuations and prolonged flat regions |
| | **CSVC ↑** — Cross-Solution Value Consistency | Comparable local value gains for the same semantic subtask across different valid solutions |

**↑ Higher is better; ↓ lower is better.**

</details>

### Evaluation Status

| Setting | Task-specific data | Status |
| --- | --- | --- |
| **Zero-Shot** | None | ✅ Results available |
| **One-Shot** | 1 training demonstration per task | ✅ Results available |
| **Full-Shot** | Complete training split | 📋 Planned |

## 🧩 Supported Value Models

**✅** marks metrics supported by the current implementation in `base` mode; **—** means unsupported. See the [baseline guides](docs/baselines/README.md) or follow the model names for setup and scope, and [validation evidence](docs/testing.md) for tested coverage.

These models are optional evaluation baselines. Running the public CPU mock or integrating your own model does not require installing them.

**TGA** includes TGA-CT and TGA-CF; **VOC family** includes VOC, Cycle-VOC, and Memory-VOC. CSVC currently supports ID evaluation only.

| Model family | SA / TGA | SIA | VOC family | FPL / TRR | VS / CSVC | Config |
| --- | :---: | :---: | :---: | :---: | :---: | --- |
| [**RoboMeter**](docs/baselines/robometer.md) | ✅ | — | ✅ | ✅ | ✅ | [YAML](configs/robometerconfigs.yaml) |
| [**Robo-Dopamine**](docs/baselines/robodopamine.md) | ✅ | — | ✅ | ✅ | ✅ | [YAML](configs/robodopamineconfigs.yaml) |
| [**ProcVLM**](docs/baselines/procvlm.md) | ✅ | ✅ | ✅ | ✅ | ✅ | [YAML](configs/procvlmconfigs.yaml) |
| [**RoboReward**](docs/baselines/roboreward.md) | ✅ | — | — | ✅ | ✅ | [YAML](configs/roborewardconfigs.yaml) |
| [**VLAC**](docs/baselines/vlac.md) | ✅ | — | ✅ | ✅ | ✅ | [YAML](configs/vlacconfigs.yaml) |
| [**TOPReward (Qwen / Molmo)**](docs/baselines/topreward.md) | ✅ | — | ✅ | ✅ | ✅ | [YAML](configs/toprewardconfigs.yaml) |
| [**RoboFAC**](docs/baselines/robofac.md) | ✅ | ✅ | ✅ | ✅ | ✅ | [YAML](configs/robofacconfigs.yaml) |
| [**RynnValue**](docs/baselines/rynnvalue.md) | ✅ | — | ✅ | ✅ | ✅ | [YAML](configs/rynnvalueconfigs.yaml) |
| [**LIV**](docs/baselines/liv.md) | ✅ | — | ✅ | ✅ | ✅ | [YAML](configs/livconfigs.yaml) |
| [**FailSafe-labeled integration**](docs/baselines/failsafe.md) | — | ✅ | — | — | — | [YAML](configs/failsafeconfigs.yaml) |

## 🚀 Evaluate Your Model <a name="quick-start"></a>

The RoboValue team supplies the benchmark data. To submit a model, provide an **inference service (a callable model prediction API)** and a **model-specific adapter**; the **team reviews the integration, prepares queries, and computes and reports metrics**. The team retains the private test set, which is not publicly released for download or participant-side evaluation.

**1. Identify your model and evaluation setting.**

Specify the model or checkpoint version, preprocessing and prompt revisions, and evaluation setting. Keep these versions fixed during evaluation.

- **Zero-Shot:** use the model without task-specific fine-tuning or reference demonstrations.
- **One-Shot:** use one reference demonstration per task, supplied by the team on the evaluation side, through your adapter.
- **Full-Shot (planned):** fine-tune on the training split supplied by the team before providing your service, and record the training setup. The fine-tuned model uses the same inference interface as Zero-Shot, without training demonstrations attached to each query. See the [data guide](https://robovalue-benchmark.github.io/doc/get-started/data/#dataset-training) for training-data access information.

**2. Provide your inference service.**

The model can remain in your own environment. Provide a reachable endpoint and document its native input/output format and authentication requirements. The service receives the observations and instructions needed for inference; test labels and scoring remain with the organizers. If observations must stay within organizer-controlled systems, deploy the service there.

**3. Implement your model-specific adapter.**

Start from the [adapter interface](src/vmbmk/adapters/base.py) and [CPU mock adapter](src/vmbmk/adapters/mock_service.py). Prepare the camera views, history, sampled frames, padding, and preprocessing your model requires, then call the service and return its predictions. Document these input requirements and confirm the required observations are available. Implement only the supported `value`, `compare`, and/or `subtask` methods. Metrics unavailable because an operation is unsupported are reported as **N/A, not zero**.

**4. Hand over the integration for evaluation.**

[Contact the RoboValue team](#contact) with the adapter source and dependencies, an example configuration, service specification, model/preprocessing versions, evaluation setting, and a runnable synthetic integration example. Share connection details and credentials privately; read credentials from environment variables. The team reviews the integration and runs the applicable metrics under the agreed protocol.

See [Service & Adapter](https://robovalue-benchmark.github.io/doc/model-api/) for examples and the handoff checklist, and [Evaluation Workflow](https://robovalue-benchmark.github.io/doc/get-started/evaluation/) for responsibilities and reporting.

<details>
<summary><strong>Adapter methods and prediction semantics</strong></summary>

| Method | Input per query | Returns |
| --- | --- | --- |
| `value` | One observation sequence and task instruction | A finite scalar in the model's native units |
| `compare` | Two observation sequences, ordered A then B, and a task instruction | A finite comparison score with documented direction |
| `subtask` | One observation sequence and task instruction | A nonempty predicted subtask description |

Each method accepts a batch of queries and returns one prediction per query in the same order. Preserve the model's native prediction target, units, and score direction; no common min–max normalization is required. Declare whether `compare` uses the model's native comparison or `V(b) − V(a)`; the value-difference helper is appropriate only when it matches the model's comparison semantics. These are explicit choices, not automatic fallbacks.

For SIA, return the predicted subtask description; the organizers' judge computes its score. Report service failures as errors, never as zero predictions or N/A.

</details>

<details>
<summary><strong>Run the public CPU mock check</strong></summary>

Start with **Python 3.10+**. Check your version with `python3 --version`; if it is older, replace `python3` in the environment-creation command below with an installed compatible interpreter, such as `python3.12`. These commands are for macOS / Linux:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/RoboValue-Benchmark/RoboValue.git
cd RoboValue
python3 -m venv .venv
source .venv/bin/activate
python -m pip install PyYAML Pillow
PYTHONPATH=src:tests python -m unittest discover -s tests/adapters -p test_mock_service.py -v
```

The clone command skips large model-weight downloads. The tests generate temporary data and synthetic images to check the mock adapter with local toy predictions; a successful run ends with `OK`. No GPU, model checkpoint, running API, or private test data is needed. The check does not call your service or produce an official model evaluation.

For your integration, replace the mock's input preparation and prediction functions with your model's input policy and service calls, then validate them with a synthetic example. The [example configuration](configs/mock_service.example.yaml) illustrates the handoff format with placeholder paths.

</details>

<details>
<summary><strong>More documentation</strong></summary>

| Guide | Contents |
| --- | --- |
| [Baseline setup](docs/baselines/README.md) | Prerequisites and instructions for deploying selected baselines on the evaluation side |
| [Data preparation](docs/data.md) | Evaluation-side dataset structure, metadata, annotations, and asset paths |
| [Configuration and commands](docs/configuration.md) | YAML fields, evaluation modes, and CLI usage |
| [Service & Adapter](https://robovalue-benchmark.github.io/doc/model-api/) | Provider handoff, native interfaces, references, and adapter examples |
| [Environment details](docs/environments.md) | Lockfiles, custom roots, and native builds |
| [Metric implementation notes](docs/metric_alignment.md) | Scoring alignment and protocol changes |
| [Result interfaces](docs/result_publication.md) | Validation, publication, and SIA intermediate outputs |
| [Developer guide](docs/developer_guide.md) | Package layout, call flow, and extension points |
| [Testing and evidence](docs/testing.md) | CPU tests, bounded model checks, and known limitations |

</details>

## 🗂️ Repository Structure

Main code, configuration, and entry points are shown below; some files are omitted:

```text
RoboValue/
├── configs/              # General, baseline, and mock evaluation templates
├── envs/                 # Model environment definitions, locks, and setup scripts
├── src/vmbmk/
│   ├── adapters/         # Model integrations and shared interfaces
│   ├── metrics/          # Query planning and metric scoring
│   ├── data/             # Dataset loading, validation, and playback code
│   ├── inference/        # Query/result types, inference dispatch, and workers
│   ├── runner/           # Evaluation, batching, and resume
│   ├── tools/            # Data checks, result handling, and visualization
│   └── cli.py            # Command-line entry points
├── docs/                 # Setup guides and protocol documentation
├── tests/                # Metric, adapter, and data tests
└── vmbmk.sh              # Evaluation launcher
```

## 📝 Citation <a name="citation"></a>

If you find **RoboValue** helpful in your research, please cite our paper. The paper link and official BibTeX entry will be added here.

## 📬 Contact <a name="contact"></a>

For model evaluation and research enquiries, please contact:

- **Shengbang Liu**: [liushengbang0209@gmail.com](mailto:liushengbang0209@gmail.com)
- **Zhengye Du**: [duzhengye20060120@gmail.com](mailto:duzhengye20060120@gmail.com)
- **Zhilong Wan**: [zhilongwan666@gmail.com](mailto:zhilongwan666@gmail.com)
- **Chenxiang Xia**: [chenxiangxia48@gmail.com](mailto:chenxiangxia48@gmail.com)
- **Chang Ge**: [gechang0706@gmail.com](mailto:gechang0706@gmail.com)
- **Jinyang Xiao**: [xiaojy36@gmail.com](mailto:xiaojy36@gmail.com)
- **Nan Wang**: [bigcileng@gmail.com](mailto:bigcileng@gmail.com)
- **Chao Yu**: [zoeyuchao@gmail.com](mailto:zoeyuchao@gmail.com)
