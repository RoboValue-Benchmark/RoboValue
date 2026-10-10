<div align="center">
  <img src="docs/assets/robovalue-logo.png" alt="RoboValue" width="520" />
</div>

<div align="center">
  <!-- Add official arXiv and Hugging Face URLs once released. -->
  <img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv" alt="arXiv 论文" height="20" />
  <img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&amp;logoColor=white" alt="HuggingFace" height="20" />
  <a href="https://robovalue-benchmark.github.io/doc/"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&amp;logo=readthedocs" alt="文档" height="20" /></a>
  <!-- Add the Chinese documentation URL when available. -->
  <img src="https://img.shields.io/badge/中文文档-red?logo=readthedocs" alt="中文文档" height="20" />
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

**RoboValue** 是一个覆盖仿真与真实世界的机器人价值模型评测基准，通过细粒度诊断轨迹评测任务结果与执行过程。

<div align="center">
  <img src="docs/assets/overview.png" alt="RoboValue 概览：仿真与真实世界数据、统一模型接口和四类评测能力" width="100%" />
</div>

## 🏆 排行榜

完整排名与模型配置见 [**RoboValue 排行榜**](https://robovalue-benchmark.github.io/leaderboard/)，Zero-Shot 和 One-Shot 赛道分别排名。

## 📰 最新动态 <a name="最新动态"></a>

- [2026/10] 🔥 我们的论文 **RoboValue: A Fine-Grained Sim-and-Real Benchmark for Unified Evaluation of Robotic Value Models** 正式发布。

## ✨ Highlights

- 🎯 **细粒度价值评测。** 通过统一评测协议，衡量模型的任务状态理解、时序进度监测、失败与恢复推理和价值一致性。

- 🤖 **丰富的仿真与真实世界操作任务。** RoboValue 包含基于 [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/README.md) 构建的 **15 个仿真任务**和 **20 个真实世界双臂操作任务**。

- 🔍 **面向诊断的执行场景。** 通过反事实指令、重复出现的视觉状态、有效与无效恢复，以及不同的有效子任务顺序，揭示结果准确率和正向进度相关性可能掩盖的错误。

- 🌍 **分别检验两类泛化能力。** 在域内条件及独立的**跨本体**、**跨环境**条件下评测，不针对变化后的条件进行额外适应。

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

### 评测进展

| 设置 | 任务专属数据 | 评测进展 |
| --- | --- | --- |
| **Zero-Shot** | 无 | ✅ 结果已公布 |
| **One-Shot** | 每个任务 1 条训练示范 | ✅ 结果已公布 |
| **Full-Data** | 完整训练集 | 📋 计划中 |

## 🧩 支持的价值模型

**✅** 表示当前适配器与运行策略在 `base` 模式下支持的指标，仍需满足任务覆盖范围和模型资源要求；**—** 表示不支持。模型名称链接到接入指南，实际验证范围见[验证记录](docs/testing.md)。

**TGA** 包含 TGA-CT 和 TGA-CF；**VOC 系列**包含 VOC、Cycle-VOC 和 Memory-VOC。CSVC 当前仅支持 ID 评测。

| 模型系列 | SA / TGA | SIA | VOC 系列 | FPL / TRR | VS / CSVC | 配置 |
| --- | :---: | :---: | :---: | :---: | :---: | --- |
| [**RoboMeter**](docs/baselines/robometer.md) | ✅ | — | ✅ | ✅ | ✅ | [YAML](configs/robometerconfigs.yaml) |
| [**Robo-Dopamine**](docs/baselines/robodopamine.md) | ✅ | — | ✅ | ✅ | ✅ | [YAML](configs/robodopamineconfigs.yaml) |
| [**ProcVLM**](docs/baselines/procvlm.md)<sup>1</sup> | ✅ | ✅ | ✅ | ✅ | ✅ | [YAML](configs/procvlmconfigs.yaml) |
| [**RoboReward**](docs/baselines/roboreward.md)<sup>2</sup> | ✅ | — | — | ✅ | ✅<sup>2</sup> | [YAML](configs/roborewardconfigs.yaml) |
| [**VLAC**](docs/baselines/vlac.md) | ✅ | — | ✅ | ✅ | ✅ | [YAML](configs/vlacconfigs.yaml) |
| [**TOPReward (Qwen / Molmo)**](docs/baselines/topreward.md) | ✅ | — | ✅ | ✅ | ✅ | [YAML](configs/toprewardconfigs.yaml) |
| [**RoboFAC**](docs/baselines/robofac.md) | ✅ | ✅ | ✅ | ✅ | ✅ | [YAML](configs/robofacconfigs.yaml) |
| [**RynnValue**](docs/baselines/rynnvalue.md) | ✅ | — | ✅ | ✅ | ✅ | [YAML](configs/rynnvalueconfigs.yaml) |
| [**LIV**](docs/baselines/liv.md) | ✅ | — | ✅ | ✅ | ✅ | [YAML](configs/livconfigs.yaml) |
| [**标记为 FailSafe 的集成**](docs/baselines/failsafe.md)<sup>3</sup> | — | ✅ | — | — | — | [YAML](configs/failsafeconfigs.yaml) |

<sup>1</sup> ProcVLM one-shot 需要任务专属 LoRA 权重，且该模式的 VOC 和 Memory-VOC 不包含 `press_by_number` 与 `swap_blocks`。

<sup>2</sup> RoboReward 当前运行策略禁用了 VOC 系列。VS 和 CSVC 保留了实现，但论文未报告 RoboReward 的这两项评测；本行表示代码支持范围，不代表论文结果覆盖范围。

<sup>3</sup> 仅支持 SIA 的 FailSafe 集成需要原始本地资源；尚未验证其与官方实现的等价性。

## 🚀 快速开始 <a name="快速开始"></a>

评测自己的模型时，只需提供**模型推理服务和专用 Adapter**。Adapter 封装模型现有的预测接口；**RoboValue 团队负责查询构造、指标计算、汇总和结果报告**，并在私有测试集上执行评测。完整流程见[模型接入指南](docs/api.md)。

**1. 准备模型服务。**

在自己的环境中运行模型，说明推理端点、原生请求与响应格式，以及固定的模型和预处理版本。若评测设置需要训练参考示范，请按约定在评测前于模型侧准备。

**2. 实现并检查 Adapter。**

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/RoboValue-Benchmark/RoboValue.git
cd RoboValue
```

参考 [Adapter 接口](src/vmbmk/adapters/base.py)和 [CPU mock Adapter](src/vmbmk/adapters/mock_service.py)，按模型原生方式完成输入准备、服务调用和输出映射。只实现模型支持的操作，并按原始查询顺序返回预测：

| 方法 | 返回内容 |
| --- | --- |
| `value` | 模型给出的标量评价 |
| `compare` | 对有序上下文 A、B 的比较分数 |
| `subtask` | 当前子任务的文本描述 |

保留模型原生的相机视角、历史输入要求和输出语义。RoboValue 会根据这些预测计算评测指标。

使用已安装 **PyYAML 和 Pillow 的 Python 3.10+**，可运行公开的 CPU mock 示例检查：

```bash
PYTHONPATH=src:tests python -m unittest discover -s tests/adapters -p test_mock_service.py -v
```

该检查使用合成图像验证示例 Adapter，无需 GPU、模型权重、运行中的 API 或私有测试数据。[示例配置](configs/mock_service.example.yaml)展示交付格式，其中路径需要按部署环境填写。

**3. 对接正式评测。**

向 [RoboValue 团队](#contact)提供 Adapter、示例配置、服务说明、模型与预处理版本，以及可运行的合成示例。服务凭据单独提供。团队审核接入后，使用私有测试集运行适用指标。

<details>
<summary><strong>更多文档</strong></summary>

| 指南 | 内容 |
| --- | --- |
| [基线配置](docs/baselines/README.md) | 安装前置条件与各模型的使用说明 |
| [数据准备](docs/data.md) | 数据集结构、元数据、标注和资源路径 |
| [配置与命令](docs/configuration.md) | YAML 字段、评测模式和 CLI 用法 |
| [模型服务与 Adapter](docs/api.md) | 接入交付、原生接口、参考示范和 CPU mock 示例 |
| [环境详情](docs/environments.md) | 依赖锁定文件、自定义根目录和原生依赖构建 |
| [指标实现说明](docs/metric_alignment.md) | 评分对齐与协议变更 |
| [结果接口](docs/result_publication.md) | 验证、发布和 SIA 中间输出 |
| [开发者指南](docs/developer_guide.md) | 包结构、调用流程和扩展接口 |
| [测试与验证记录](docs/testing.md) | CPU 测试、限定范围的模型检查和已知限制 |

</details>

## 🗂️ Repository Structure

```text
RoboValue/
├── configs/              # 各模型的评测配置模板
├── envs/                 # 隔离运行环境、依赖锁定与安装脚本
├── src/vmbmk/
│   ├── adapters/         # 模型集成与统一接口
│   ├── metrics/          # 查询规划与指标评分
│   ├── data/             # 数据集验证与轨迹回放
│   ├── inference/        # 类型化查询与模型工作进程
│   ├── runner/           # 评测、批处理与断点续跑
│   ├── tools/            # 数据检查、结果处理与可视化
│   └── cli.py            # 命令行入口
├── docs/                 # 配置指南与评测协议文档
├── tests/                # 指标、适配器与数据测试
└── vmbmk.sh              # 评测启动脚本
```

数据集、下载的模型资源、运行环境和生成结果单独配置；存储约定详见[数据准备](docs/data.md)与[基线配置](docs/baselines/README.md#source-runtime-and-checkpoint-roots)。随仓库分发的 LIV 资源使用 Git LFS。

## 📝 引用 <a name="引用"></a>

如果 **RoboValue** 对你的研究有所帮助，欢迎引用我们的论文。正式 BibTeX 条目将在此补充。

## 📬 Contact <a name="contact"></a>

模型评测与研究交流，请联系：

- **Shengbang Liu**: [liushengbang0209@gmail.com](mailto:liushengbang0209@gmail.com)
- **Zhengye Du**: [duzhengye20060120@gmail.com](mailto:duzhengye20060120@gmail.com)
- **Zhilong Wan**: [zhilongwan666@gmail.com](mailto:zhilongwan666@gmail.com)
- **Chang Ge**: [gechang0706@gmail.com](mailto:gechang0706@gmail.com)
- **Chenxiang Xia**: [chenxiangxia48@gmail.com](mailto:chenxiangxia48@gmail.com)
- **Jinyang Xiao**: [xiaojy36@gmail.com](mailto:xiaojy36@gmail.com)
- **Nan Wang**: [bigcileng@gmail.com](mailto:bigcileng@gmail.com)
- **Chao Yu**: [zoeyuchao@gmail.com](mailto:zoeyuchao@gmail.com)
