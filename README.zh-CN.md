<div align="center">
  <img src="docs/assets/robovalue-logo.png" alt="RoboValue" width="520" />
</div>

<div align="center">
  <!-- Add official arXiv and Hugging Face URLs when available. -->
  <img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv" alt="arXiv 论文：链接待补充" height="20" />
  <img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&amp;logoColor=white" alt="HuggingFace：链接待补充" height="20" />
  <a href="https://robovalue-benchmark.github.io/doc/"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&amp;logo=readthedocs" alt="文档" height="20" /></a>
  <!-- Add the Chinese documentation URL when available. -->
  <img src="https://img.shields.io/badge/中文文档-red?logo=readthedocs" alt="中文文档：链接待补充" height="20" />
  <a href="https://robovalue-benchmark.github.io/"><img src="https://img.shields.io/badge/Website-blue?logo=googlechrome&amp;logoColor=white" alt="项目网站" height="20" /></a>
  <a href="https://robovalue-benchmark.github.io/leaderboard/"><img src="https://img.shields.io/badge/Leaderboard-527BC3?logo=weightsandbiases&amp;logoColor=white" alt="排行榜" height="20" /></a>
  <a href="https://robovalue-benchmark.github.io/community/"><img src="https://img.shields.io/badge/Community-2E8B57" alt="社区" height="20" /></a>


</div>

<div align="center">

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md) [![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](README.zh-CN.md)

</div>

<h1 align="center">
  <sub>RoboValue: A Fine-Grained Sim-and-Real Benchmark<br />for Unified Evaluation of Robotic Value Models</sub>
</h1>

**RoboValue** 通过诊断轨迹与统一评测协议，衡量机器人价值模型在仿真与真实世界中评估任务结果和执行过程的可靠性。

<div align="center">
  <img src="docs/assets/overview.png" alt="RoboValue 概览：仿真与真实世界数据、统一模型接口和四类评测能力" width="100%" />
</div>

## 📰 最新动态 <a name="最新动态"></a>

- [2026/10] 🔥 我们的论文 **RoboValue: A Fine-Grained Sim-and-Real Benchmark for Unified Evaluation of Robotic Value Models** 正式发布。

## ✨ Highlights

- 🤖 **RoboValue-Dataset。** 包含基于 [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/README.md) 构建的 **15 个仿真任务**和 **20 个真实世界双臂操作任务**，覆盖多样的操作技能与任务要求。专家示范用于模型适应；独立测试轨迹考察成功执行、失败与恢复、依赖历史的进度判断，以及多种有效解决方案。

- 🎯 **RoboValue-Benchmark。** 通过统一协议评测**任务状态理解**、**时序进度监测**、**失败与恢复推理**和**价值一致性**。诊断轨迹与反事实指令揭示仅凭结果判别和正向进度相关性难以发现的能力缺陷。

- 🌍 **受控泛化评测。** 分别设置**跨本体**与**跨环境**评测，检验机器人平台或视觉条件变化时，价值判断能否保持可靠，且不针对变化后的条件进行额外适应。

- 🏆 **RoboValue-Leaderboard。** 在各赛道内提供总体排名与分能力表现，既呈现模型的综合水平，也指出仍需改进的具体能力。

<div align="center">
  <img src="docs/assets/dataset.png" alt="RoboValue 数据集：仿真与真实世界任务，以及失败与恢复、时序和多解轨迹" width="100%" />
</div>

<details>
<summary><strong>能力与指标速查</strong></summary>

| 能力 | 指标 | 评测内容 |
| --- | --- | --- |
| **任务状态理解** | **SA ↑** — 成功准确率 | 成功轨迹的终止价值是否高于失败轨迹 |
| | **TGA-CT / TGA-CF ↑** — 任务语义对齐准确率 | 对同一执行，正确指令的价值增益是否高于跨任务（CT）或反事实（CF）指令 |
| | **SIA ↑** — 子任务识别准确率 | 从模型生成的描述识别当前子任务，由独立 LLM 裁判评分 |
| **时序进度监测** | **VOC ↑** — 价值顺序相关性 | 成功轨迹中，预测价值与时序进度的秩相关性 |
| | **Cycle-VOC ↑** | 在保留前序历史的连续正放—倒放过程中，价值是否反映进展与回退 |
| | **Memory-VOC ↑** | 在相似视觉状态反复出现、执行历史不同的长时序轨迹中，价值是否反映进度顺序 |
| **失败与恢复推理** | **FPL ↓** — 失败点定位 | 最大检测价值下降所指示的失败起点与标注起点之间的归一化时间误差 |
| | **TRR ↑** — 轨迹恢复推理 | 初始失败、错误延续、恢复尝试及其成功或失败结果各阶段的价值变化方向 |
| **价值一致性** | **VS ↑** — 价值稳定性 | 反馈是否稳定且持续提供信息，避免多余波动和长时间不变 |
| | **CSVC ↑** — 跨解决方案价值一致性 | 不同有效解决方案中，同一语义子任务是否获得可比较的局部价值增益 |

**↑ 越高越好；↓ 越低越好。**

</details>

## 🧩 支持的价值模型

**✅** 表示当前实现的 `base` 模式支持，**—** 表示不支持。模型配置与适用范围见[基线指南](docs/baselines/README.md)（也可点击模型名称），实际验证情况见[验证记录](docs/testing.md)。

下列模型是可选评测基线。运行公开 CPU mock 或接入自己的模型无需安装这些基线。

| 模型系列 | SA / TGA | SIA | VOC 系列 | FPL / TRR | VS / CSVC | 配置 |
| --- | :---: | :---: | :---: | :---: | :---: | --- |
| [**标记为 FailSafe 的集成**](docs/baselines/failsafe.md) | — | ✅ | — | — | — | [YAML](configs/failsafeconfigs.yaml) |
| [**LIV**](docs/baselines/liv.md) | ✅ | — | ✅ | ✅ | ✅ | [YAML](configs/livconfigs.yaml) |
| [**ProcVLM**](docs/baselines/procvlm.md) | ✅ | ✅ | ✅ | ✅ | ✅ | [YAML](configs/procvlmconfigs.yaml) |
| [**Robo-Dopamine**](docs/baselines/robodopamine.md) | ✅ | — | ✅ | ✅ | ✅ | [YAML](configs/robodopamineconfigs.yaml) |
| [**RoboFAC**](docs/baselines/robofac.md) | ✅ | ✅ | ✅ | ✅ | ✅ | [YAML](configs/robofacconfigs.yaml) |
| [**RoboMeter**](docs/baselines/robometer.md) | ✅ | — | ✅ | ✅ | ✅ | [YAML](configs/robometerconfigs.yaml) |
| [**RoboReward**](docs/baselines/roboreward.md) | ✅ | — | — | ✅ | ✅ | [YAML](configs/roborewardconfigs.yaml) |
| [**RynnValue**](docs/baselines/rynnvalue.md) | ✅ | — | ✅ | ✅ | ✅ | [YAML](configs/rynnvalueconfigs.yaml) |
| [**TOPReward (Qwen / Molmo)**](docs/baselines/topreward.md) | ✅ | — | ✅ | ✅ | ✅ | [YAML](configs/toprewardconfigs.yaml) |
| [**VLAC**](docs/baselines/vlac.md) | ✅ | — | ✅ | ✅ | ✅ | [YAML](configs/vlacconfigs.yaml) |

## 🚀 如何评测你的模型 <a name="快速开始"></a>

RoboValue 数据由团队提供。提交模型时，请提供**推理服务（可调用的模型预测 API）**和**模型专用适配器（Adapter）**；**RoboValue 团队负责审核、构造查询，并计算和报告评测指标**。正式测试集由团队保管，不公开下载或供参与者本地评测。

**1. 明确模型版本与评测赛道。**

说明模型或检查点版本、预处理和提示词修订版本，以及采用的评测赛道。评测期间保持这些版本不变。

- **Zero-Shot：** 不进行任务专属训练，也不使用参考示范。
- **One-Shot：** 每个任务使用团队提供的 1 条训练示范，作为参考输入或用于任务专属微调，并说明具体使用方式。
- **Full-Shot（计划中）：** 在提供服务前，使用团队提供的训练集完成训练或微调，并记录训练设置。训练后的模型使用与 Zero-Shot 相同的推理接口，无需在每次查询中附带训练示范。训练数据获取方式以[数据说明](https://robovalue-benchmark.github.io/doc/get-started/data/#dataset-training)为准。

**2. 提供推理服务。**

模型可以继续在你的环境中运行。请提供可访问的服务端点，并说明原生输入输出格式和身份认证要求。服务会接收推理所需的观测和指令；测试标签与评分由组织方掌握。若观测数据必须保留在组织方管控的系统内，则将服务部署在该环境中。

**3. 实现模型专用适配器。**

参考 [Adapter 接口](src/vmbmk/adapters/base.py)和 [CPU mock 适配器](src/vmbmk/adapters/mock_service.py)，按模型要求准备相机视角、历史观测、采样帧、填充和预处理，再调用服务并返回模型预测。请说明这些输入要求，并确认所需观测数据可用。仅实现模型支持的 `value`、`compare` 和/或 `subtask` 方法。因不支持相应操作而无法评测的指标记为 **N/A，而非零分**。

**4. 交付接入材料并对接评测。**

向 [RoboValue 团队](#contact)提供适配器源码与依赖、示例配置、服务说明、模型和预处理版本、评测赛道，以及可运行的合成数据集成示例。连接信息和凭据通过私密渠道分享；凭据从环境变量中读取。团队审核接入后，按约定协议运行适用指标。

适配器示例与交付清单见 [Service & Adapter](https://robovalue-benchmark.github.io/doc/model-api/)；各方职责和结果报告流程见[评测流程](https://robovalue-benchmark.github.io/doc/get-started/evaluation/)。

<details>
<summary><strong>适配器方法与预测语义</strong></summary>

| 方法 | 每条查询的输入 | 返回内容 |
| --- | --- | --- |
| `value` | 一条观测序列和任务指令 | 以模型原生单位表示的有限标量 |
| `compare` | 按 A、B 顺序排列的两条观测序列和任务指令 | 有限的比较分数，需说明分数方向的含义 |
| `subtask` | 一条观测序列和任务指令 | 非空的预测子任务描述 |

每种方法接收一批查询，并按相同顺序为每条查询返回一个预测。保留模型原生的预测目标、单位和分数方向，无需统一进行最小—最大归一化。请声明 `compare` 使用模型的原生比较，还是 `V(b) − V(a)`；仅当差值符合模型的比较语义时，才使用现有差值辅助函数。二者是明确选择的实现方式，不会自动作为彼此的备用方案。

对于 SIA，返回预测的子任务描述，由组织方的评判器计算分数。服务调用失败时应报错，不得返回零值预测或 N/A。

</details>

<details>
<summary><strong>运行公开 CPU mock 示例检查</strong></summary>

先准备 **Python 3.10+**。可用 `python3 --version` 检查版本；如果默认版本过旧，请将下方创建环境命令中的 `python3` 换为已安装的兼容解释器，例如 `python3.12`。以下命令适用于 macOS / Linux：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/RoboValue-Benchmark/RoboValue.git
cd RoboValue
python3 -m venv .venv
source .venv/bin/activate
python -m pip install PyYAML Pillow
PYTHONPATH=src:tests python -m unittest discover -s tests/adapters -p test_mock_service.py -v
```

克隆命令会跳过模型权重的大文件下载。测试自动生成临时数据和合成图像，用本地演示预测验证 mock 适配器；成功时最后显示 `OK`。无需 GPU、模型权重、运行中的 API 或私有测试数据。该检查不会调用你的服务，也不构成模型的正式评测。

接入自己的模型时，请将 mock 的输入准备和预测函数替换为模型自身的输入策略与服务调用，再通过合成示例验证集成。[示例配置](configs/mock_service.example.yaml)展示交付格式，其中路径需要按部署环境填写。

</details>

<details>
<summary><strong>更多文档</strong></summary>

| 指南 | 内容 |
| --- | --- |
| [基线配置](docs/baselines/README.md) | 评测端按需部署基线模型的前置条件与使用说明 |
| [数据准备](docs/data.md) | 评测端的数据集结构、元数据、标注和资源路径 |
| [配置与命令](docs/configuration.md) | YAML 字段、评测模式和 CLI 用法 |
| [Service & Adapter](https://robovalue-benchmark.github.io/doc/model-api/) | 接入交付、原生接口、参考示范和适配器示例 |
| [环境详情](docs/environments.md) | 依赖锁定文件、自定义根目录和原生依赖构建 |
| [指标实现说明](docs/metric_alignment.md) | 评分对齐与协议变更 |
| [结果接口](docs/result_publication.md) | 验证、发布和 SIA 中间输出 |
| [开发者指南](docs/developer_guide.md) | 包结构、调用流程和扩展接口 |
| [测试与验证记录](docs/testing.md) | CPU 测试、限定范围的模型检查和已知限制 |

</details>

## 🗂️ Repository Structure

主要代码、配置与入口如下（省略部分文件）：

```text
RoboValue/
├── configs/              # 通用、基线模型与 mock 评测配置模板
├── envs/                 # 模型环境定义、依赖锁定与安装脚本
├── src/vmbmk/
│   ├── adapters/         # 模型集成与统一接口
│   ├── metrics/          # 查询规划与指标评分
│   ├── data/             # 数据集读取、验证与轨迹回放代码
│   ├── inference/        # 查询与结果类型、推理调度与工作进程
│   ├── runner/           # 评测、批处理与断点续跑
│   ├── tools/            # 数据检查、结果处理与可视化
│   └── cli.py            # 命令行入口
├── docs/                 # 配置指南与评测协议文档
├── tests/                # 指标、适配器与数据测试
└── vmbmk.sh              # 评测启动脚本
```

## 🏆 排行榜

总体排名、分能力得分和模型配置见 [**RoboValue 排行榜**](https://robovalue-benchmark.github.io/leaderboard/)。根据可使用的任务专属训练数据，基准设立三个赛道：

| 赛道 | 任务专属数据 | 评测进展 |
| --- | --- | --- |
| **Zero-Shot** | 无 | ✅ 结果已公布 |
| **One-Shot** | 每个任务 1 条训练示范 | ✅ 结果已公布 |
| **Full-Shot** | 完整训练集 | 📋 计划中 |

已开放的赛道分别排名。Full-Shot 将支持使用完整 RoboValue 训练集进行训练或微调的模型。

## 📝 引用 <a name="引用"></a>

如果 **RoboValue** 对你的研究有所帮助，欢迎引用我们的论文。论文链接和正式 BibTeX 条目待补充。

## 📬 Contact <a name="contact"></a>

模型评测与研究交流，请联系 RoboValue 团队的主要成员：

- **Shengbang Liu**: [liushengbang0209@gmail.com](mailto:liushengbang0209@gmail.com)
- **Zhengye Du**: [duzhengye20060120@gmail.com](mailto:duzhengye20060120@gmail.com)
- **Zhilong Wan**: [zhilongwan666@gmail.com](mailto:zhilongwan666@gmail.com)
- **Chenxiang Xia**: [chenxiangxia48@gmail.com](mailto:chenxiangxia48@gmail.com)
- **Chang Ge**: [gechang0706@gmail.com](mailto:gechang0706@gmail.com)
- **Jinyang Xiao**: [xiaojy36@gmail.com](mailto:xiaojy36@gmail.com)
- **Nan Wang**: [bigcileng@gmail.com](mailto:bigcileng@gmail.com)
- **Chao Yu**: [zoeyuchao@gmail.com](mailto:zoeyuchao@gmail.com)
