---
icon: lucide/blocks
---

# 架构与概念

## 工程结构

仓库当前仍然保持分层结构，但 PCVR 运行时已经从单文件脚手架演进为一组共享栈模块。

```text
src/taac2026/
├── domain/
│   ├── config.py           # TrainRequest / EvalRequest / InferRequest / default_run_dir
│   ├── experiment.py       # ExperimentSpec 插件契约
│   └── metrics.py          # AUC / LogLoss / GAUC / 诊断指标
├── application/
│   ├── training/cli.py     # taac-train
│   ├── evaluation/
│   │   ├── cli.py          # taac-evaluate single / infer
│   │   └── infer.py        # 推理 Bundle 入口
│   ├── maintenance/
│   │   ├── bundle_packaging.py   # taac-package-train / taac-package-infer
│   │   └── generate_pcvr_synthetic_dataset.py
│   └── benchmarking/
│       └── *_benchmark.py
└── infrastructure/
  ├── bundles/
  │   ├── common.py       # code_package.zip 组装与摘要输出
  │   └── manifest.py     # 训练/推理 Bundle manifest 契约
  ├── checkpoints.py      # checkpoint 解析与保存
  ├── experiments/
  │   ├── discovery.py    # 实验包扫描
  │   └── loader.py       # 实验包加载
  ├── io/
  │   ├── files.py        # repo_root / 文件工具
  │   └── json_utils.py   # JSON 读写
  ├── platform/
  │   ├── adapters.py     # 本地 / 线上运行适配选择
  │   ├── bundle_runtime.py
  │   ├── inference_bundle.py
  │   └── run_sh.py       # run.sh 共享入口
  ├── training/
  │   ├── muon.py         # Muon 优化器支持
  │   └── runtime.py      # 通用训练 runtime 配置
  └── pcvr/               # 共享 PCVR 运行时
    ├── config.py
    ├── config_sidecar.py
    ├── data.py
    ├── data_observation.py
    ├── data_pipeline.py
    ├── data_schema.py
    ├── experiment.py
    ├── experiment_runtime.py
    ├── factory.py
    ├── flash_qla/
    ├── modeling.py
    ├── prediction_stack.py
    ├── protocol.py
    ├── runtime_stack.py
    ├── sample_dataset.py
    ├── tensors.py
    ├── tilelang_kernels.py
    ├── tilelang_ops.py
    ├── train_stack.py
    ├── trainer.py
    ├── trainer_support.py
    └── training.py
```

仓库外层的 `experiments/` 目录是插件层，不是框架核心的一部分：

```text
experiments/
├── pcvr/
│   ├── baseline/
│   │   ├── __init__.py
│   │   └── model.py
│   ├── symbiosis/
│   │   ├── __init__.py
│   │   ├── layers.py       # 兼容 re-export，真实 primitives 在共享 modeling.py
│   │   └── model.py
│   ├── interformer/
│   ├── onetrans/
│   └── ...
└── maintenance/
  ├── host_device_info/
  └── online_dataset_eda/
```

## 分层职责

### Domain

`domain` 层只定义稳定契约：

- 请求对象：`TrainRequest`、`EvalRequest`、`InferRequest`
- 插件接口：`ExperimentSpec`
- 指标函数：AUC、LogLoss、GAUC、score diagnostics

这一层不负责 PyTorch 模型搭建，也不感知具体实验包目录。

### Application

`application` 层负责把 CLI / Bundle 入口翻译成 domain request：

- `taac-train` 只解析训练命令和额外参数，然后把请求交给实验包
- `taac-evaluate single` 和 `taac-evaluate infer` 负责本地评估与推理入口
- `bundle_packaging.py` 负责生成训练 / 推理线上上传文件
- `benchmarking` 下的命令负责 benchmark CLI

### Infrastructure

`infrastructure` 层负责真正的运行时实现：

- 实验包扫描与加载
- checkpoint、JSON、文件系统工具
- Bundle 组装
- 共享 PCVR 训练 / 评估 / 推理栈
- TileLang 与 Torch 的算子适配

## 统一入口

当前最常用的仓库入口有两组：

| 入口                                   | 用途             |
| -------------------------------------- | ---------------- |
| `bash run.sh train`                    | 训练             |
| `bash run.sh val` / `bash run.sh eval` | 本地评估         |
| `bash run.sh infer`                    | 推理             |
| `uv run taac-package-train`            | 训练 Bundle 打包 |
| `uv run taac-package-infer`            | 推理 Bundle 打包 |

`run.sh` 当前只负责 `train` / `val` / `eval` / `infer`。打包不再通过 `run.sh package` 完成。

底层 CLI 注册关系仍然很简单：

| 命令            | 模块                                       |
| --------------- | ------------------------------------------ |
| `taac-train`    | `taac2026.application.training.cli:main`   |
| `taac-evaluate` | `taac2026.application.evaluation.cli:main` |
| `taac-package-train` | `taac2026.application.maintenance.bundle_packaging:training_main` |
| `taac-package-infer` | `taac2026.application.maintenance.bundle_packaging:inference_main` |

其中 `taac-evaluate` 暴露两个子命令：

- `single`：评估有标签数据
- `infer`：写出平台要求的 `predictions.json`

## 实验包契约

### PCVR 实验包

PCVR 实验包位于 `experiments/pcvr/`。最小契约是：

| 文件          | 要求                                                                               |
| ------------- | ---------------------------------------------------------------------------------- |
| `__init__.py` | 定义 `EXPERIMENT = create_pcvr_experiment(...)`，并给出默认训练配置和模型类名      |
| `model.py`    | 实现实验包自己的模型类，类名必须与 `model_class_name` 一致                         |

包内可以按需要继续拆出局部模块；现有 `layers.py` 只保留为兼容 re-export，通用 tokenizer、embedding bank、RMSNorm 和参数分组 mixin 已经集中到 `taac2026.infrastructure.pcvr.modeling`。

`create_pcvr_experiment()` 会创建 `PCVRExperiment` 适配器，并默认接上共享训练、预测和运行时 hooks。`PCVRExperiment` 本身持有：

- `train_arg_parser`
- `train_hooks`
- `prediction_hooks`
- `runtime_hooks`

普通模型实验通常不需要手写这些 hook 对象；只有像 Symbiosis 这样需要扩展模型构造或配置校验的实验才通过工厂传入 override。运行时仍会临时切换到实验包目录，导入包内 `model` 模块，再把共享训练 / 评估 / 推理栈拼起来。

### 维护类实验包

维护类实验位于 `experiments/maintenance/`，不需要模型类，也不要求实现评估 / 推理。它们直接导出 `ExperimentSpec`，通常只实现 `train_fn`。

例如：

- `host_device_info`
- `online_dataset_eda`

## 模型输入与构建

所有 PCVR 模型都接收统一的 `ModelInput`。具体张量字段由共享协议层维护，模型本身只需要遵守这份统一输入契约，而不必重新实现数据解析。

`build_pcvr_model()` 的职责是把数据 schema 和训练配置翻译成模型构造参数，包括：

- 用户 / 物品稀疏特征规格
- 稠密特征维度
- 序列词表信息
- NS tokenizer 相关配置
- 模型默认参数与实验包特定参数

模型类需要提供的核心行为保持不变：

- `forward(inputs)` -> logits
- `predict(inputs)` -> `(logits, embeddings)`
- 稀疏 / 稠密参数分组接口

## 共享 PCVR 栈

当前共享 PCVR 栈的职责已经拆得比较清楚：

| 模块                                      | 作用                                                              |
| ----------------------------------------- | ----------------------------------------------------------------- |
| `config.py`                               | 训练、模型、优化器、数据和 NS 配置                                |
| `config_sidecar.py`                       | 训练配置 sidecar 的版本化读写                                     |
| `data.py`                                 | parquet 读取、Row Group 切分、数据集工具                          |
| `data_observation.py` / `data_schema.py`  | observed schema 和 schema 解析                                    |
| `data_pipeline.py`                        | 数据增强和流水线拼装                                              |
| `sample_dataset.py`                       | 合成 / 示例数据集生成                                             |
| `modeling.py`                             | `ModelInput`、mask/pooling/attention helpers、共享 tokenizer / embedding / RMSNorm primitives |
| `protocol.py`                             | `ModelInput`、模型构建协议和 schema -> 构造参数转换               |
| `factory.py`                              | 实验默认配置与共享栈装配入口                                      |
| `training.py`                             | 训练入口拼装                                                      |
| `train_stack.py`                          | 训练 hooks 和共享训练流程                                         |
| `trainer.py` / `trainer_support.py`       | trainer 循环和训练辅助逻辑                                        |
| `runtime_stack.py`                        | checkpoint、train_config、runtime schema、observed schema sidecar |
| `prediction_stack.py`                     | 评估 / 推理 loop 和 predictor 准备                                |
| `experiment_runtime.py`                   | `PCVRExperiment` 的运行时 mixin                                   |
| `tilelang_ops.py` / `tilelang_kernels.py` | TileLang / Torch 算子适配                                         |
| `flash_qla/`                              | Flash QLA 相关 CUDA/Triton 实验算子                               |

`run.sh` 和推理 Bundle 里的 `infer.py` 现在只做最小 bootstrap：定位 / 解压 `code_package.zip`，设置 `PYTHONPATH`，再把 manifest 读取、依赖安装、实验包默认值和 CLI 分发交给 `taac2026.infrastructure.platform`。本地仓库模式默认走 `uv`，线上 Bundle 模式默认走当前 Python / Conda 环境。

## NS Groups

非序列特征分组现在主要通过实验包 `__init__.py` 中的 `PCVRNSConfig` 明确声明，而不是依赖一个外置 JSON 文件作为唯一来源。

这层配置会进入训练时生成的 `train_config.json`，之后评估和推理也以 checkpoint 侧车文件为准。
当前 `train_config.json` 仍保留 flat 配置字段，同时额外写入 `train_config_format`、`train_config_version` 和 `framework_version` 等版本元数据；读取端继续兼容旧的无版本 flat 文件。
这类 checkpoint 侧契约由 Pydantic 2 模型校验，但对外仍以普通 JSON / dict 形态读写，避免把训练热路径和实验模型代码绑到 Pydantic 对象上。

因此现在更值得记住的是：

- 分组配置是实验包默认配置的一部分
- 训练时落盘的 `train_config.json` 是评估 / 推理重建模型的真实来源之一
- 文档和代码都不应再把 `ns_groups.json` 当成现行仓库结构里的必备文件

## 训练流程

训练入口是 `taac-train` 或 `bash run.sh train`：

1. CLI 解析 `--experiment`、`--dataset-path`、`--schema-path`、`--run-dir`
2. `load_experiment_package()` 加载对应实验包
3. `PCVRExperiment.train()` 进入共享训练栈
4. 训练时构建 observed schema sidecar、checkpoint 和日志产物
5. 返回训练摘要 JSON

训练完成后，常见产物包括：

- `model.safetensors`
- `schema.json`
- `train_config.json`
- 训练 / 验证 split 的 observed schema 报告

训练循环内部仍然支持稀疏 / 稠密参数分组、AMP、`torch.compile`、early stopping 和 checkpoint sidecar。

## 评估与推理

### 本地评估

`taac-evaluate single` 或 `bash run.sh val` 会：

1. 解析实验包、数据路径和 checkpoint / run_dir
2. 读取 checkpoint 对应的 `train_config.json` 和运行时 schema
3. 运行共享 prediction loop
4. 计算分类指标
5. 写出 `evaluation.json`、`validation_predictions.jsonl` 和 `evaluation_observed_schema.json`

### 平台推理

`taac-evaluate infer` 或推理 Bundle 中的 `infer.py` 会输出：

```json
{"predictions": {"user_id": score}}
```

推理结果目录只要求最终的 `predictions.json`，不会像本地评估那样再落一套 evaluation sidecar。

## 线上打包

当前线上上传物分为两类：

- 训练 Bundle：`run.sh` + `code_package.zip`
- 推理 Bundle：`infer.py` + `code_package.zip`

两者的 zip 内部都使用 `project/` 根目录，manifest 分别是：

- `project/.taac_training_manifest.json`
- `project/.taac_inference_manifest.json`

训练 Bundle 当前格式号是 `taac2026-training-v2`，推理 Bundle 当前格式号是 `taac2026-inference-v1`。manifest 还包含独立的 `manifest_version`、`bundle_format_version`、`framework.version`、`runtime_env` 和 `compatibility` 字段；训练和推理打包都会先通过共享 Pydantic 验证器检查这些字段再写入 zip。

详见 [线上 Bundle 上传指南](guide/online-training-bundle.md)。

## 当前边界

- 主要面向 PCVR 二分类任务
- 训练 / 评估 / 推理都基于 parquet + `schema.json` 约定
- 实验扩展依赖插件式 `experiments/` 目录，而不是改动核心框架入口
- 当前文档、测试和打包实现都已经默认实验路径位于 `experiments/...`，不再使用旧的 `config/...` 结构
