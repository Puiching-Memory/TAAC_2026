---
icon: lucide/blocks
---

# 架构与运行契约

实验包负责模型与默认值，共享框架负责一次实验如何执行。修改时先确定责任层，再沿数据和文件的生产者、消费者核对影响范围。

## 目录与依赖

| 目录 | 职责 | 典型修改 |
| --- | --- | --- |
| `experiments/` | 实验身份、模型组合、私有层与配置扩展 | 新增模型或消融开关 |
| `src/taac2026/domain/` | 配置、schema、请求、指标与 sidecar 类型 | 改变边界格式或参数约束 |
| `src/taac2026/application/` | 训练、评估、推理、打包等用例编排 | 改变流程和产物 |
| `src/taac2026/infrastructure/` | parquet、运行时、建模组件、优化器、加速器、平台 IO | 实现具体技术能力 |
| `tests/` | unit、contract、integration、CPU benchmark 与 GPU 验证 | 验证可观察契约 |
| `docs/` | 文档正文、归档与静态资源 | 改变使用说明 |

```text
experiments → taac2026.api → domain / application / infrastructure
application → domain + infrastructure
infrastructure → domain
domain → 标准库与轻量类型依赖
```

实验共享能力通过 `taac2026.api` 导入。通用基础设施不选择实验，domain 不解析 CLI、不读文件或探测设备。配置模型中的 `model.ns` 负责特征分组；通用 parquet/schema 层保持模型无关。

## 实验加载

`application/experiments/registry.py` 接受文件系统路径或 Python module，读取模块导出的 `EXPERIMENT`，并校验 `Experiment` Protocol。协议包含 `name`、`kind`、`requires_dataset`、默认配置及 `train/evaluate/infer` 方法。

普通 PCVR 实验用 `create_pcvr_experiment()` 绑定模型类、配置类型与默认值。维护工具使用 `FunctionExperiment`，只实现需要的回调；没有对应回调的操作会失败。EDA 支持训练和推理，Host Device Info 仅支持训练。

## 训练路径

```text
run.sh / taac-train
    → TrainRequest + 实验 TRAIN_DEFAULTS + CLI 覆盖
    → schema、parquet、切分、PCVRBatch
    → model_type(schema, config)
    → trainer：forward → loss → backward → optimizer → validation
    → checkpoint + sidecar + summary + telemetry
```

`PCVRTrainConfig` 及其子配置是 frozen Pydantic 模型，CLI、默认值、sidecar 共享这份表示。实验扩展配置时必须把相应 `config_type` 传给工厂。构造模型时，workflow 将 `train_config.model` 传给模型的 `config` 参数。

数据加载输出 `PCVRBatch`，模型接收 `PCVRModelInput`。`forward()` 返回 `(B,)` logits；`predict()` 返回 logits 与表征。分类概率和文件输出由共享评估流程处理。稀疏与稠密参数按模型契约分组，不能重复加入两个 optimizer。

## Schema 的唯一来源

| 场景 | 首选 | 未显式提供时 |
| --- | --- | --- |
| 显式本地数据训练 | `--schema-path` | 数据目录中的 `schema.json`，文件数据则取其父目录 |
| 默认 demo 训练 | `--schema-path` | 仓库 `data/sample_1000_raw/schema.json` |
| 评估/推理模型重建 | `--schema-path` | 所选 checkpoint 同目录的 `schema.json` |

显式路径无效会立即失败，不会继续猜测其他来源。schema 声明字段、词表与维度；observed schema 记录实际扫描统计，二者用途不同。切分和诊断详见[数据管道](guide/pcvr-data-pipeline.md)。

## Checkpoint 契约

```text
global_step<step>.../
├── model.safetensors
├── schema.json
└── train_config.json
```

| 文件 | 生产者 | 消费者 |
| --- | --- | --- |
| `model.safetensors` | `infrastructure/checkpoints.py` | 评估与推理权重加载 |
| `schema.json` | checkpoint sidecar 写入 | 模型输入契约重建 |
| `train_config.json` | typed 配置序列化 | 对应实验的配置类型加载 |
| `.taac_training_manifest.json` | 训练打包 | shell bootstrap |
| `.taac_inference_manifest.json` | 推理打包 | Python bootstrap |

`train_config.json` 是完整配置快照。缺失字段、未知字段或错误类型会失败，不能靠新默认值“修复”旧权重。修改 schema、模型配置或 sidecar 时，需要同时更新训练、评估、推理、打包和测试。

评估未传具体 checkpoint 时，从 `--run-dir` 下按 step 选择最新权重。显式 `--checkpoint` 则只解析给定文件或直接包含权重的目录。最新 step 也不等于验证指标最优；实验报告应记录最终选中的路径。

这些文件主要用于模型重建和预测，不能仅凭它们推断支持完整恢复 optimizer、采样位置与随机数状态后续训。

## CLI 与平台边界

训练外层参数由 `training/cli.py` 解析，剩余参数由 `training/args.py` 以 `use_underscores=True` 交给 Tyro。因此 `--experiment`、`--dataset-path` 和 `--optimizer.max_steps` 属于两层配置。

评估 CLI 有 `single` 与 `infer` 子命令，使用 `--batch-size`、`--num-workers` 等扁平参数。`run.sh val` 会转换为 `taac-evaluate single`。

本地 shell 默认 runner 为 uv，bundle 为平台 Python。bundle 只带共享框架和选中的实验，不能依赖仓库的其他实验、测试、文档或 lockfile。平台变量到 CLI 的映射与已知 Python 版本缺口见 [Bundle](guide/online-training-bundle.md)。

## 从问题定位源码

| 问题 | 首查文件 |
| --- | --- |
| 命令分发、平台变量 | `application/bootstrap/run_sh.py` |
| 训练参数与编排 | `application/training/cli.py`、`args.py`、`workflow.py` |
| 实验发现与执行 | `application/experiments/registry.py`、`experiment.py` |
| 评估/推理 | `application/evaluation/cli.py`、`runtime.py`、`workflow.py` |
| schema 与模型构造 | `infrastructure/modeling/model_contract.py` |
| batch 转换与采样 | `infrastructure/data/batch_converter.py`、`step_dataset.py` |
| 训练更新与保存 | `infrastructure/runtime/trainer.py`、`checkpoint_io.py` |
| 上传包与 manifest | `application/packaging/`、`infrastructure/bundles/` |

表中源码路径相对 `src/taac2026/`。新增实验见[接入指南](guide/contributing.md)，修改后按[测试指南](guide/testing.md)验证。
