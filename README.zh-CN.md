<div align="center">
  <img src="docs/assets/robovalue-logo.png" alt="RoboValue" width="520" />
</div>

<div align="center">
  <!-- Add official arXiv and Hugging Face URLs once released. -->
  <img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv" alt="arXiv 论文" height="20" />
  <img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&amp;logoColor=white" alt="HuggingFace" height="20" />
  <a href="https://robovalue-benchmark.github.io/doc/"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&amp;logo=readthedocs" alt="文档" height="20" /></a>
  <a href="https://robovalue-benchmark.github.io/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&amp;logoColor=white" alt="项目网站" height="20" /></a>
  <a href="https://robovalue-benchmark.github.io/leaderboard/"><img src="https://img.shields.io/badge/Leaderboard-527BC3?logo=weightsandbiases&amp;logoColor=white" alt="排行榜" height="20" /></a>
  <a href="https://robovalue-benchmark.github.io/community/"><img src="https://img.shields.io/badge/WeChat-green?logo=wechat&amp;logoColor=white" alt="微信群" height="20" /></a>
</div>

<div align="center">

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md) [![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](README.zh-CN.md)

</div>

<h1 align="center">
  <sub>RoboValue: A Fine-Grained Sim-and-Real Benchmark<br />for Unified Evaluation of Robotic Value Models</sub>
</h1>

**RoboValue** 是一个覆盖仿真与真实世界的机器人价值模型评测基准。它通过细粒度诊断轨迹，评测模型的任务状态理解、时序进度监测、失败与恢复推理以及价值一致性。

<div align="center">
  <img src="docs/assets/overview.png" alt="RoboValue 概览：仿真与真实世界数据、统一模型接口和四类评测能力" width="100%" />
</div>

<p align="center">
  <a href="#排行榜">排行榜</a> ·
  <a href="#最新动态">最新动态</a> ·
  <a href="#基准概览">基准概览</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#参与贡献">参与贡献</a> ·
  <a href="#引用与致谢">引用与致谢</a>
</p>

## 排行榜

下表展示 Zero-Shot 和 One-Shot 两个赛道中排名前三的模型配置，各赛道独立排名。Overall 是四类能力归一化得分的均值，范围为 **0–100**，越高越好；**SIA 单独报告，不计入 Overall**。

| 赛道 | 排名 | 模型 | Overall ↑ |
| --- | :---: | --- | ---: |
| **Zero&#8209;Shot** | 🥇 1 | RoboMeter-4B | **60.05** |
| Zero-Shot | 🥈 2 | RynnValue-4B | 57.70 |
| Zero-Shot | 🥉 3 | RynnValue-8B | 56.25 |
| **One&#8209;Shot** | 🥇 1 | Robo-Dopamine 2.0-8B Preview | **58.27** |
| One-Shot | 🥈 2 | ProcVLM-2B | 57.82 |
| One-Shot | 🥉 3 | Robo-Dopamine 2.0-4B Preview | 57.52 |

## 最新动态

- [2026/10] 🔥 我们的论文 **RoboValue: A Fine-Grained Sim-and-Real Benchmark for Unified Evaluation of Robotic Value Models** 正式发布。

## 基准概览

本节介绍 RoboValue 的数据集、评测指标、统一评测接口与当前评测进展。

### 数据集

**RoboValue-Dataset** 覆盖 **35 个双臂操作任务**。数据集包含专家训练示范和独立标注的测试集，涵盖标准执行、失败与恢复、长时序推理以及不同的有效解决方案。

| 仿真任务 | 真实世界任务 | 训练示范 | 测试轨迹 |
| :---: | :---: | :---: | :---: |
| **15** | **20** | **3,500**（每个任务 100 条） | **2,792** |

<div align="center">
  <img src="docs/assets/dataset.png" alt="RoboValue 数据集：仿真与真实世界任务，以及失败与恢复、时序和多解轨迹" width="100%" />
</div>

训练示范在域内条件下采集。测试轨迹覆盖**域内**、**跨本体**和**跨环境**条件；任务设计与泛化设置详见论文。

**数据集下载即将开放。** 数据集结构与使用方式详见[数据准备指南](docs/data.md)。

### 评测指标

四类互补能力通过 **11 项指标**进行评测，其中 SIA 作为独立的子任务识别评测报告。每类能力关注价值模型行为的不同方面。

| 能力 | 评测内容 | 指标 |
| --- | --- | --- |
| **任务状态理解** | 执行是否成功、指令与状态是否对应，以及当前所处的子任务 | SA, TGA-CT, TGA-CF, SIA |
| **时序进度监测** | 任务进展与回退，包括视觉状态相近但历史不同的情况 | VOC, Cycle-VOC, Memory-VOC |
| **失败与恢复推理** | 失败发生时刻、尚未解决的错误，以及有效或无效的恢复 | FPL, TRR |
| **价值一致性** | 稳定的反馈，以及不同有效解决方案中可比较的子任务价值增益 | VS, CSVC |

**FPL ↓** 衡量定位误差，越低越好；**其余主要指标 ↑** 均越高越好。缺失或不支持的指标结果以 **N/A** 标记。评分方式与评测条件详见[指标定义](https://robovalue-benchmark.github.io/doc/get-started/protocol/)；代码中的指标名称与选项详见[配置指南](docs/configuration.md#metric-selection)。

### 评测接口

统一的**标量打分**、**成对比较**和**文本输出**接口支持不同类型的机器人价值模型。接口定义与输入输出格式详见[项目文档](docs/developer_guide.md#query-and-result-records)。

### 评测协议与进展

评测设置规定了**模型可用于条件输入或适应的任务专属示范数据**，与上述测试条件属于不同维度。参考数据和适应数据均与测试轨迹分离。

| 设置 | 任务专属数据 | 评测进展 |
| --- | --- | --- |
| **Zero&#8209;Shot** | 无 | ✅ [结果已公布](https://robovalue-benchmark.github.io/leaderboard/) |
| **One&#8209;Shot** | 每个任务 1 条训练示范 | ✅ [结果已公布](https://robovalue-benchmark.github.io/leaderboard/?track=one) |
| **Few&#8209;Shot** | 每个任务多条示范 | 📋 计划中 |
| **Full&#8209;Data** | 完整训练集 | 📋 计划中 |

当前结果与工具文档涵盖 **Zero-Shot 和 One-Shot** 模型设置。支持的模型与指标组合详见[配置与命令](docs/configuration.md)。

## 快速开始

以下步骤假设你已获得基准数据访问权限；公开下载即将开放。从**一个基线模型**开始，准备运行环境和模型权重，然后验证数据并运行评测。提供的安装流程面向 **Linux x86_64**，需要用于启动评测的 **Python 3.10+ 和 PyYAML**，以及 **uv** 和对应基线所需的 CUDA 与构建依赖。安装前请查看[通用前置条件](docs/baselines/README.md#prerequisites)。

**1. 克隆仓库。**

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/RoboValue-Benchmark/RoboValue.git
cd RoboValue
```

此命令将 LIV 权重的下载推迟到实际需要时。使用 LIV 时，请按照 [LIV 指南](docs/baselines/liv.md)获取对应的 Git LFS 资源。

**2. 配置基线并下载模型权重。**

以下以 **RoboReward** 为例，需要已有的 **Python 3.10.x** 解释器。请先满足通用前置条件，包括安装 `hf` CLI，并按照 [RoboReward 指南](docs/baselines/roboreward.md)准备 FFmpeg 4–8 共享库。

```bash
bash envs/setup.sh roboreward --plan
bash envs/setup.sh roboreward

hf download teetone/RoboReward-8B \
  --revision 3a185b4fce2b1253643105be1f234ae618b9732f \
  --local-dir checkpoints/RoboReward-8B
```

模型依赖在隔离环境中运行。使用其他模型时，请选择相应的[基线指南](docs/baselines/README.md)；各指南说明了源码、兼容依赖、权重资源和验证范围。

**3. 准备数据、配置运行参数并开始评测。**

单独获取数据集，并按照[数据指南](docs/data.md)准备数据。复制基线模板：

```bash
cp configs/roborewardconfigs.yaml configs/roboreward-local.yaml
```

假设数据位于 `data/dataset_real/`，按下例编辑 `configs/roboreward-local.yaml`。首次运行仅评测**一个任务在 ID 条件下的 SA 指标**。请设置可用的 GPU；如果数据集中的任务 ID 不同，请替换 `organize_table`。

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

YAML 中的路径相对于 `configs/` 解析：其中的 `../data/dataset_real` 与从仓库根目录运行以下命令时的 `data/dataset_real` 指向同一数据集。若数据、环境或权重位于其他位置，也可使用绝对路径。

```bash
bash vmbmk.sh validate data/dataset_real
bash vmbmk.sh run configs/roboreward-local.yaml
```

验证过程检查数据结构和资源，不加载模型；成功时输出 `valid`。使用上述配置完成评测后，结果保存在 **`output/roboreward-local/metrics.json`**，同一运行目录内还会保存 **`config.yaml`** 以及各指标对应的操作记录和溯源文件。首次运行成功后，先在 `configs/` 下复制为新的配置文件名，以生成独立运行目录，再按照[配置与命令](docs/configuration.md)扩展任务和适用指标。

<details>
<summary><strong>基线指南与配置模板</strong></summary>

| 基线 | 安装 | 配置 |
| --- | --- | --- |
| **RoboMeter** | [指南](docs/baselines/robometer.md) | [YAML](configs/robometerconfigs.yaml) |
| **Robo-Dopamine** | [指南](docs/baselines/robodopamine.md) | [YAML](configs/robodopamineconfigs.yaml) |
| **ProcVLM** | [指南](docs/baselines/procvlm.md) | [YAML](configs/procvlmconfigs.yaml) |
| **RoboReward** | [指南](docs/baselines/roboreward.md) | [YAML](configs/roborewardconfigs.yaml) |
| **VLAC** | [指南](docs/baselines/vlac.md) | [YAML](configs/vlacconfigs.yaml) |
| **TOPReward**（Qwen / Molmo） | [指南](docs/baselines/topreward.md) | [YAML](configs/toprewardconfigs.yaml) |
| **RoboFAC** | [指南](docs/baselines/robofac.md) | [YAML](configs/robofacconfigs.yaml) |
| **RynnValue** | [指南](docs/baselines/rynnvalue.md) | [YAML](configs/rynnvalueconfigs.yaml) |
| **LIV** | [指南](docs/baselines/liv.md) | [YAML](configs/livconfigs.yaml) |
| **标记为 FailSafe 的 SIA 集成** | [来源与配置](docs/baselines/failsafe.md) | [YAML](configs/failsafeconfigs.yaml) |

标记为 FailSafe 的集成可用于 SIA；其与官方实现的等价性尚未验证。提供的基线安装器不包含 GVL 和 ReWiND。各模型指南分别说明了可用集成与已验证的运行情况。

</details>

<details>
<summary><strong>建议目录结构</strong></summary>

```text
RoboValue/
  data/
    dataset_sim/                 仿真数据集
    dataset_real/                真实世界数据集
    reference/                   独立的 One-Shot 参考数据
  checkpoints/<model-name>/      权重、分词器和模型配置
  .baseline-sources/<baseline>/  下载的上游源码
  .model-envs/<runtime>/         隔离的 Python 环境
  .cache/huggingface/            可选的下载缓存
  configs/                      可编辑的运行配置
  output/<config-stem>/          生成的评测结果文件
```

上述路径为推荐约定，也可通过配置路径使用外部存储。除明确随仓库分发的 LIV 资源外，下载的源码、数据集、权重、环境、缓存和输出均不纳入 Git。其他模型权重需单独提供。

</details>

<details>
<summary><strong>更多文档</strong></summary>

| 指南 | 内容 |
| --- | --- |
| [基线配置](docs/baselines/README.md) | 安装前置条件与各模型的使用说明 |
| [数据准备](docs/data.md) | 数据集结构、元数据、标注和资源路径 |
| [配置与命令](docs/configuration.md) | YAML 字段、评测模式和 CLI 用法 |
| [环境详情](docs/environments.md) | 依赖锁定文件、自定义根目录和原生依赖构建 |
| [指标实现说明](docs/metric_alignment.md) | 评分对齐与协议变更 |
| [结果接口](docs/result_publication.md) | 验证、发布和 SIA 中间输出 |
| [开发者指南](docs/developer_guide.md) | 包结构、调用流程和扩展接口 |
| [测试与验证记录](docs/testing.md) | CPU 测试、限定范围的模型检查和已知限制 |

</details>

## 参与贡献

欢迎参与贡献！你可以添加**模型适配器**、提交**可复现的评测结果**、**改进任务与标注**，或**修正文档**。请先阅读[开发者指南](docs/developer_guide.md)和[测试指南](docs/testing.md)，然后提交 Pull Request，说明改动内容及验证方式。

如有问题或需报告错误，请提交 [Issue](https://github.com/RoboValue-Benchmark/RoboValue/issues)，并附上基线、指标、环境版本、不含凭据的配置和相关错误堆栈。欢迎加入我们的 [WeChat 微信群](https://robovalue-benchmark.github.io/community/)，讨论评测协议与机器人价值模型。

## 引用与致谢

如果 **RoboValue** 对你的研究有所帮助，欢迎引用我们的论文：

```bibtex
% 待补充正式 BibTeX 引用。
```

感谢 RoboValue 所使用的模型实现与权重背后的团队，包括 [RoboMeter](docs/baselines/robometer.md)、[Robo-Dopamine](docs/baselines/robodopamine.md)、[ProcVLM](docs/baselines/procvlm.md)、[RoboReward](docs/baselines/roboreward.md)、[VLAC](docs/baselines/vlac.md)、[TOPReward](docs/baselines/topreward.md)、[RoboFAC](docs/baselines/robofac.md)、[RynnValue](docs/baselines/rynnvalue.md) 和 [LIV](docs/baselines/liv.md)。对应配置指南提供了上游项目与模型资源的链接。
