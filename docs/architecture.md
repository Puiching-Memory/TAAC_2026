---
icon: lucide/blocks
---

# 架构与概念

这页先给完整项目结构，再解释各层职责和依赖方向。当前仓库的核心设计是：实验包保持薄，框架层承接共享训练、评估、推理、打包和运行时能力。

## 项目结构

```text
TAAC_2026/
├── .agents/                  # Codex skill，给模型提供读代码方向
├── .github/workflows/         # CI 与 GitHub Pages 部署
├── docker/                    # 本地容器镜像与入口脚本
├── docs/                      # 文档源文件，Zensical 从这里构建站点
├── experiments/               # 可被加载、训练、评估、推理或打包的实验包
├── figures/                   # README / 项目级图片资产
├── src/taac2026/              # 共享框架代码
├── tests/                     # 单元测试与测试支撑代码
├── tools/                     # 仓库维护脚本
├── README.md                  # 仓库首页说明
├── run.sh                     # 本地和训练 bundle 共用的 shell 入口
├── pyproject.toml             # Python 包、依赖、脚本入口和工具配置
├── uv.lock                    # uv 锁文件
└── zensical.toml              # 文档站配置与导航
```

下面这些目录经常会出现在工作区，但不是架构源文件：

| 路径 | 性质 |
| ---- | ---- |
| `site/` | Zensical 本地构建产物 |
| `outputs/` | 训练、推理、bundle 和 benchmark 输出 |
| `.venv/` | 本地虚拟环境 |
| `.cache/`、`.benchmarks/`、`.pytest_cache/`、`.ruff_cache/` | 工具缓存 |
| `__pycache__/` | Python 字节码缓存 |

## 框架代码结构

`src/taac2026` 是共享框架。实验包不应该复制这里的训练、推理、数据读取或打包逻辑。

```text
src/taac2026/
├── api.py                     # 给实验包使用的稳定 public facade
├── domain/
│   ├── config.py              # PCVR train/model/data/cache/pipeline/optimizer/NS 配置
│   ├── experiment.py          # ExperimentSpec 插件契约
│   ├── metrics.py             # AUC / LogLoss / GAUC / 诊断指标
│   ├── requests.py            # TrainRequest / EvalRequest / InferRequest
│   ├── runtime_config.py      # AMP、compile、determinism、loss 和 optimizer 边界配置
│   ├── schema.py              # FeatureSchema 与时间桶常量
│   └── sidecar.py             # train_config sidecar 契约与版本
├── application/
│   ├── benchmarking/
│   │   ├── generate_pcvr_synthetic_dataset.py
│   │   ├── pcvr_data_pipeline_benchmark.py
│   │   ├── pcvr_optimizer_benchmark.py
│   │   └── pcvr_tilelang_ops_benchmark.py
│   ├── bootstrap/
│   │   ├── inference_bundle.py
│   │   ├── inference_bundle_entrypoint.py
│   │   └── run_sh.py
│   ├── evaluation/
│   │   ├── cli.py
│   │   ├── infer.py
│   │   ├── runtime.py
│   │   └── workflow.py
│   ├── experiments/
│   │   ├── discovery.py
│   │   ├── experiment.py
│   │   ├── factory.py
│   │   ├── registry.py
│   │   └── runtime.py
│   ├── packaging/
│   │   ├── cli.py
│   │   └── service.py
│   └── training/
│       ├── args.py
│       ├── cli.py
│       └── workflow.py
└── infrastructure/
    ├── accelerators/
    │   ├── chunking.py
    │   ├── group_reduce.py
    │   ├── prefix_scan.py
    │   ├── tensor_validation.py
    │   ├── tensor_ops.py
    │   ├── attention/
    │   │   ├── flash_attention.py
    │   │   ├── gated_delta_rule_capabilities.py
    │   │   ├── gated_delta_rule.py
    │   │   ├── kernels/
    │   │   │   ├── gated_delta_rule/
    │   │   │   └── tilelang.py
    │   │   └── mla.py
    │   ├── embedding/
    │   │   └── embedding_bag.py
    │   ├── normalization/
    │   │   ├── kernels/
    │   │   │   └── tilelang.py
    │   │   └── rms_norm.py
    │   └── tilelang_runtime.py
    ├── bundles/
    │   ├── manifest_store.py
    │   └── zip_writer.py
    ├── data/
    │   ├── batches.py
    │   ├── cache.py
    │   ├── dataset.py
    │   ├── observation.py
    │   ├── pipeline.py
    │   ├── sample_dataset.py
    │   ├── shuffle.py
    │   └── transforms.py
    ├── experiments/
    │   └── module_loader.py
    ├── io/
    │   ├── files.py
    │   ├── json.py
    │   └── streams.py
    ├── modeling/
    │   ├── embeddings.py
    │   ├── model_contract.py  # ModelInput、schema -> model、batch -> input contract
    │   ├── normalization.py
    │   ├── sequence.py
    │   ├── tensors.py
    │   └── tokenizers.py
    ├── optimization/
    │   ├── muon.py
    │   ├── registry.py
    │   ├── schedules.py
    │   └── transforms.py
    ├── platform/
    │   ├── deps.py
    │   ├── env.py
    │   └── imports.py
    ├── runtime/
    │   ├── checkpoint_io.py
    │   ├── execution.py
    │   └── trainer.py
    ├── checkpoints.py
    └── logging.py
```

`application/` 按用例组织，`infrastructure/` 按技术能力组织。看到一个新需求时，先判断它是“用例编排”还是“技术实现”；这一步比目录名本身更重要。

## 实验包结构

`experiments/` 是可加载单元。普通模型实验由 `__init__.py` 声明 `EXPERIMENT`，由 `model.py` 实现模型；复杂实验可以带私有 `layers.py`。

```text
experiments/
├── baseline/
│   ├── __init__.py            # HyFormer 干净参照的 EXPERIMENT 与 TRAIN_DEFAULTS
│   └── model.py
├── baseline_plus/
│   ├── __init__.py            # cache / augmentation / TileLang backend 默认配置
│   └── model.py
├── interformer/
│   ├── __init__.py
│   ├── layers.py
│   └── model.py
├── onetrans/
│   ├── __init__.py
│   ├── layers.py
│   └── model.py
├── symbiosis/
│   ├── __init__.py            # 自定义 hooks、额外 CLI 参数和消融默认值
│   ├── layers.py
│   └── model.py
├── host_device_info/
│   ├── __init__.py            # 维护实验：采集线上机器环境
│   └── runner.py
└── online_dataset_eda/
    ├── __init__.py            # 维护实验：线上 parquet 文本化 EDA
    └── runner.py
```

普通模型实验的稳定形状是 `__init__.py + model.py`，可选 `layers.py`。维护实验的稳定形状是 `__init__.py + runner.py`。

## 文档和测试结构

文档源在 `docs/`，站点导航在 `zensical.toml`。分区 `index.md` 负责导览，叶子页负责具体契约和命令。

```text
docs/
├── archive/                   # schema 和官方 baseline 快照
├── experiments/               # 每个实验包的结构、默认配置、修改点和测试契约
├── guide/                     # 任务型指南：打包、数据管道、测试、文档站、平台环境
├── ideas/                     # 比赛想法和技术笔记
├── papers/                    # 论文解读和技术时间线
├── assets/                    # 图片、JS、CSS
└── overrides/                 # 文档站模板覆盖
```

测试按被测层组织：

```text
tests/
├── support/                   # 测试辅助
└── unit/
    ├── application/           # CLI、workflow、bundle、experiment runtime
    ├── domain/                # schema、model contract、metrics
    ├── experiments/           # 实验包发现、契约矩阵、维护实验
    └── infrastructure/        # data、accelerators、bundles、modeling、runtime
```

## 分层职责

改代码时先判断自己在改哪一层：

| 层                | 放什么                                                                 | 不该放什么                             |
| ----------------- | ---------------------------------------------------------------------- | -------------------------------------- |
| `experiments/`    | 实验名、默认配置、模型类、实验私有 helper                              | 共享训练流程、通用数据读取、打包逻辑   |
| `domain/`         | request、config、schema、model contract、sidecar、metrics              | CLI 参数、环境变量、文件系统探测       |
| `application/`    | train/eval/infer/packaging/bootstrap 的用例流程                        | 低层 parquet 读取、具体 optimizer 算法 |
| `infrastructure/` | data、modeling、runtime、optimization、accelerators、bundles、platform | 实验选择策略和业务语义                 |

依赖方向应保持单向：

```text
experiments/ -> taac2026.api -> domain/application/infrastructure
application/ -> domain + infrastructure
infrastructure/ -> domain 或纯技术依赖
domain/ -> 标准库 / 轻量类型，不依赖 application 或 infrastructure
```

`taac2026.api` 是给实验包的稳定门面。实验包直接 import 内部模块不是绝对禁止，但应该有明确理由。

## 数据切分策略

PCVR 训练入口支持四种验证切分策略，统一由 `PCVRDataConfig.split_strategy` 和训练 CLI 的 `--split-strategy` 控制：

| 策略 | 用途 | 说明 |
| ---- | ---- | ---- |
| `row_group_tail` | 本地 smoke 和兼容默认 | 使用尾部 row group 做 valid，启动成本最低 |
| `timestamp_range` | 首选线上泛化验证 | 显式指定 `--train-timestamp-start/end` 和 `--valid-timestamp-start/end`，模拟未来时间评测 |
| `user_hash` | 用户级泛化验证 | 按首个 user sparse 特征稳定 hash 分桶，train/valid 用户尽量互斥；缺失 user key 时回退到样本位置 |
| `sample_hash` | 样本级稳定验证 | 按文件、row group、行位置稳定 hash 分桶，适合没有可靠 user key 或时间窗的场景 |

`timestamp_range`、`user_hash` 和 `sample_hash` 都是 row-level filter。为了避免随机取 batch 后被过滤为空，训练数据加载会把 `step_random` 自动降级为 `row_group_sweep`。隐藏评测出现 train/infer drift 时，优先使用 `timestamp_range`；没有可靠时间边界时，再用 `user_hash` 或 `sample_hash` 作为比随机切分更稳的验证口径。

## 边界契约与 Pydantic

Pydantic 在这个仓库里主要用于跨边界 payload，而不是替代所有内部类型。适合使用 Pydantic 的位置包括：

- 需要读写 JSON 的持久化契约，例如 checkpoint sidecar。
- 需要被平台或 bundle 消费的 manifest。
- 插件、实验包或外部入口传入的结构化 payload。

这些模型应继承 `taac2026.domain.validation.TAACBoundaryModel`，默认拒绝未知字段，避免平台或历史文件悄悄带入未定义配置。业务版本、格式号、路径范围、枚举兼容等规则应放在模型或紧邻模型的验证函数里。

内部训练上下文、张量载体、轻量不可变默认配置和热路径对象仍优先使用 dataclass 或现有专用类型。不要为了“使用 Pydantic”而整体迁移 `PCVRTrainConfig` 这类实验默认配置；更好的做法是在它们序列化到 sidecar、manifest 或平台 payload 时做边界校验。

## 实验包是什么

每个实验包通过 `EXPERIMENT` 暴露能力：

```text
experiments/baseline/
├── __init__.py   # 导出 EXPERIMENT，声明默认配置和模型类
└── model.py      # 当前实验的模型实现
```

普通 PCVR 模型实验通常用 `create_pcvr_experiment()` 创建。维护类实验可以直接导出 `ExperimentSpec`，例如 `host_device_info` 和 `online_dataset_eda`。

框架不靠目录名判断实验类型，而是看 `EXPERIMENT` 的能力和 metadata。

## 公共 API

实验包应优先从 `taac2026.api` 导入共享能力。这样实验包依赖的是稳定表面，而不是内部目录布局。

常用导入包括：

- `PCVRTrainConfig`、`PCVRModelConfig`、`PCVRNSConfig`
- `PCVRDataPipelineConfig`、cache 和 transform 配置
- `RuntimeExecutionConfig`、`PCVRLossConfig`、`PCVRLossTermConfig`
- `ModelInput`
- 建模 primitives，例如 tokenizer、embedding bank、RMSNorm
- `create_pcvr_experiment`

如果一个实验包需要深度 import `application/` 或 `infrastructure/`，先想一下这是不是共享能力应该沉到框架层。

## 关键持久化契约

这几个文件会跨训练、评估、推理和 bundle 边界，不应随手改格式：

| 文件                            | 写入方             | 读取方                 | 作用                              |
| ------------------------------- | ------------------ | ---------------------- | --------------------------------- |
| `model.safetensors`             | trainer            | evaluation / inference | 模型权重                          |
| `schema.json`                   | checkpoint sidecar | model contract         | 重建 feature schema               |
| `train_config.json`             | checkpoint sidecar | runtime hooks          | 重建模型配置、NS 分组和运行时参数 |
| `.taac_training_manifest.json`  | package train      | bundle bootstrap       | 训练 bundle 元数据                |
| `.taac_inference_manifest.json` | package infer      | inference bootstrap    | 推理 bundle 元数据                |

checkpoint sidecar 是 runtime contract，不是日志。改模型构造参数、NS 配置或 schema 解析时，要确认当前 checkpoint 的评估和推理路径。

## 一次训练怎么流动

本地入口：

```bash
bash run.sh train --experiment experiments/baseline
```

核心流程：

1. `run.sh` 分发到 `taac-train`。
2. training CLI 解析实验包和输出目录。
3. experiment registry 加载 `experiments/<name>/__init__.py` 的 `EXPERIMENT`。
4. PCVR train workflow 解析默认配置、数据、schema 和 hooks。
5. data infrastructure 读取 parquet，构造 batch，执行 cache / transforms。
6. model contract 把 schema 和 batch 转成模型构造参数与 `ModelInput`。
7. trainer 执行训练，写 checkpoint 和 sidecar。

训练产物的关键约定：

```text
global_step*/
├── model.safetensors
├── schema.json
└── train_config.json
```

评估和推理会读取 checkpoint 同目录的 sidecar 来重建模型输入契约。

相关实现文件：

| 阶段                 | 文件                                                   |
| -------------------- | ------------------------------------------------------ |
| CLI 解析             | `src/taac2026/application/training/cli.py`             |
| PCVR 参数 parser     | `src/taac2026/application/training/args.py`            |
| 实验加载             | `src/taac2026/application/experiments/registry.py`     |
| 默认训练 workflow    | `src/taac2026/application/training/workflow.py`        |
| 数据读取             | `src/taac2026/infrastructure/data/dataset.py`          |
| batch -> model input | `src/taac2026/infrastructure/modeling/model_contract.py` |
| trainer              | `src/taac2026/infrastructure/runtime/trainer.py`       |
| checkpoint IO        | `src/taac2026/infrastructure/runtime/checkpoint_io.py` |

## 本地数据和线上数据

本地 PCVR smoke 不让用户传 `--dataset-path`，样例 parquet 来自 Hugging Face `TAAC2026/data_sample_1000`。仓库不提交 parquet 数据；样例 schema 参考快照归档在 `docs/archive/files/schema/sample_1000_raw.schema.json`，本地命令应显式传 `--schema-path`。这样本地数据来源和线上平台注入路径能保持清晰边界。

线上 bundle 相反：真实数据路径来自平台变量，例如 `TRAIN_DATA_PATH`、`EVAL_DATA_PATH` 和 `TAAC_SCHEMA_PATH`。bundle 模式使用平台 Python，不依赖线上 `uv`。

这条边界很重要：不要把本地 demo 路径写死进实验包，也不要让线上 bundle 依赖仓库外的开发工具。

本地 `run.sh` 默认 runner 是 `uv`。bundle runner 默认是 `python`。这由 `src/taac2026/infrastructure/platform/env.py` 的 platform adapter 决定。

## 数据管道

数据管道由 `PCVRDataPipelineConfig` 描述，实现在 `infrastructure/data/`。

默认顺序可以理解为：

1. parquet -> base batch
2. base-batch cache
3. 结构性 transform，例如 sequence crop / multi-view
4. 随机性 transform，例如 feature mask / domain dropout
5. shuffle、rebatch、prefetch

训练可以启用随机增强；验证和推理应保持确定性。数据增强配置对象在 `src/taac2026/domain/config.py`，实际 transform 在 `src/taac2026/infrastructure/data/transforms.py`。

## 运行时和加速

训练 runtime 关注“怎么跑”：

- AMP / dtype
- `torch.compile`
- seed 和日志
- early stopping
- checkpoint IO
- optimizer 和 scheduler

加速层按算子族组织：

- attention：FlashAttention / QLA surface 和 kernels
- normalization：RMSNorm surface 和 TileLang kernels
- embedding：embedding 相关 fused operator

上层模型应依赖 operator surface，例如 `RMSNorm` 或 attention helper，而不是直接耦合 TileLang kernel 文件。

## 打包边界

线上上传物有两类：

| 类型        | 文件                          |
| ----------- | ----------------------------- |
| 训练 bundle | `run.sh + code_package.zip`   |
| 推理 bundle | `infer.py + code_package.zip` |

zip 内部只带 `src/taac2026`、当前实验包、`pyproject.toml` 和 manifest。它不会把整个仓库、测试、无关实验包或 `uv.lock` 全塞进去。

打包逻辑分层：

- `application/packaging/`：打包用例
- `infrastructure/bundles/`：manifest 和 zip primitive
- `application/bootstrap/`：线上入口解压、安装、分发

训练 bundle 的 manifest 是 `.taac_training_manifest.json`，推理 bundle 的 manifest 是 `.taac_inference_manifest.json`。`run.sh` 负责训练 bundle 解压与命令分发；`infer.py` 负责推理 bundle 解压与评测入口适配。

## 测试边界

架构层面的改动通常要按契约选测试：

| 改动                    | 重点测试                                                                                        |
| ----------------------- | ----------------------------------------------------------------------------------------------- |
| 实验包入口 / 默认配置   | `tests/contract/experiments/test_packages.py`，`tests/contract/experiments/test_runtime_contract_matrix.py` |
| model contract / schema | `tests/unit/domain/test_model_contract.py`                                                      |
| 训练 CLI / workflow     | `tests/unit/application/training/`                                                              |
| 评估 / 推理             | `tests/unit/application/evaluation/`，`tests/unit/application/experiments/test_pcvr_runtime.py` |
| bundle                  | `tests/integration/application/packaging/`，`tests/unit/application/bootstrap/`，`tests/integration/application/bootstrap/`                        |
| 数据管道                | `tests/unit/infrastructure/data/`                                                               |
| accelerator             | `tests/unit/infrastructure/accelerators/`                                                       |

## 改动时的判断

- 只换模型结构：优先改 `experiments/<name>/model.py`。
- 只换默认超参或数据增强：改 `experiments/<name>/__init__.py`。
- 多个实验都会用到：考虑放进 `src/taac2026/infrastructure/modeling/` 或 `data/`。
- 改 CLI、bundle、环境变量：看 `application/` 和 `platform/`。
- 改 checkpoint、schema、sidecar：先看 `domain/` 契约，再看 runtime。

## 常见落点例子

| 需求                                       | 合适位置                                       | 原因                           |
| ------------------------------------------ | ---------------------------------------------- | ------------------------------ |
| 新模型只被一个实验使用                     | `experiments/<name>/model.py`                  | 保持实验私有，避免框架层膨胀   |
| 多个实验共用 tokenizer / pooling primitive | `src/taac2026/infrastructure/modeling/`        | 属于可复用建模实现             |
| 新增训练 CLI 参数                          | `src/taac2026/application/training/args.py`    | CLI 是用例入口，不属于模型包   |
| 新增 checkpoint sidecar 字段               | `src/taac2026/domain/sidecar.py` 和 runtime IO | 字段会跨训练、评估、推理持久化 |
| 新增线上 bundle manifest 字段              | `src/taac2026/infrastructure/bundles/`         | manifest 是打包 primitive      |
| 新增平台环境变量解析                       | `src/taac2026/infrastructure/platform/`        | 环境差异属于平台适配           |

反过来，如果一个实验包开始复制 data loader、checkpoint writer 或 bundle bootstrap，通常说明共享能力放错了位置。
