---
icon: lucide/blocks
---

# 架构与概念

## 工程结构

仓库现在采用三层主包加统一实验包的结构。PCVR 仍是当前主任务，但它不再通过 `infrastructure/pcvr` 作为横切命名空间组织；领域契约、应用编排和技术实现已经回到各自层级。

```text
src/taac2026/
├── api.py                       # experiments/ 的稳定公共导入面
├── domain/
│   ├── requests.py              # TrainRequest / EvalRequest / InferRequest
│   ├── experiment.py            # ExperimentSpec 插件契约
│   ├── metrics.py               # AUC / LogLoss / GAUC / 诊断指标
│   ├── config.py                # PCVR 训练、模型、数据、优化器、NS 配置
│   ├── model_contract.py        # ModelInput / schema -> model / batch -> input contract
│   ├── schema.py                # FeatureSchema / time bucket 常量
│   └── sidecar.py               # train_config sidecar 契约与版本
├── application/
│   ├── experiments/             # 实验加载、归一化、PCVR 实验适配器和工厂
│   ├── training/                # taac-train、参数解析、训练 workflow
│   ├── evaluation/              # taac-evaluate、本地评估、推理 workflow
│   ├── packaging/               # 训练 / 推理 bundle 打包用例
│   ├── bootstrap/               # run.sh / infer.py bundle bootstrap 分发
│   └── benchmarking/            # benchmark 与数据生成 CLI
└── infrastructure/
    ├── io/                      # repo_root、文件、JSON 工具
    ├── experiments/             # import-by-path primitive
    ├── bundles/                 # zip 写入与 manifest 校验存储
    ├── platform/                # env、依赖安装、import bootstrap
    ├── data/                    # parquet dataset、observed schema、pipeline/cache/transform/shuffle
    ├── modeling/                # tokenizer、embedding、RMSNorm、sequence/tensor primitives
    ├── runtime/                 # execution config、trainer、checkpoint IO
    ├── optimization/            # optimizer registry、schedules、Muon、gradient transforms
    └── accelerators/            # attention/normalization/embedding operator boundaries 与 kernels
```

仓库外层的 `experiments/` 是插件层。每个一级子目录都是一个可加载实验包，不再按 `pcvr/` 或 `maintenance/` 做长期分类：

```text
experiments/
├── baseline/
├── interformer/
├── onetrans/
├── symbiosis/
├── host_device_info/
└── online_dataset_eda/
```

真正稳定的实验包 contract 是：`experiments/<name>/__init__.py` 导出 `EXPERIMENT`。实验是否带模型、是否只做维护任务、是否覆盖 hooks，都由 `EXPERIMENT` 的能力和 metadata 表达，而不是由目录层级表达。

## 分层职责

### Domain

`domain` 层只定义稳定语义和持久化契约，不感知 CLI、环境变量、平台分发或具体实验包路径。

- `requests.py`：训练、评估、推理请求对象，以及默认 run 目录规则
- `experiment.py`：统一实验插件契约 `ExperimentSpec`
- `config.py`：PCVR train/model/data/cache/pipeline/optimizer/NS 配置对象
- `model_contract.py`：统一 `ModelInput`、schema 到模型参数、batch 到模型输入的 contract
- `schema.py`：FeatureSchema 和时间桶常量
- `sidecar.py`：`train_config.json` sidecar 格式、版本和校验
- `metrics.py`：分类指标和诊断指标

### Application

`application` 层负责请求进入系统后的用例编排。

- `experiments/registry.py`：定位、加载并归一化实验包
- `experiments/factory.py` / `experiment.py`：把 PCVR 默认 hooks 和实验配置装配为 `ExperimentSpec` 兼容对象
- `training/args.py`：训练参数、环境变量覆盖、默认值解析
- `training/workflow.py`：训练数据、模型、trainer 和训练摘要的默认 workflow
- `evaluation/workflow.py`：prediction loop 的数据、模型、predictor 和输出流程
- `evaluation/runtime.py`：checkpoint、schema、observed schema、评估诊断编排
- `packaging/service.py`：训练 / 推理 bundle 的 use-case 级装配
- `bootstrap/run_sh.py` 和 `bootstrap/inference_bundle.py`：本地 `run.sh` 与线上 `infer.py` 的命令分发

### Infrastructure

`infrastructure` 层只放技术实现和适配器，不反向导入 `application`。

- `data/`：Parquet 数据集、Row Group split、observed schema、cache、transforms、shuffle、sample dataset
- `modeling/`：实验模型复用的 tokenizer、embedding bank、attention helpers、RMSNorm、tensor helpers
- `runtime/`：AMP / compile / logger / seed / early stopping、trainer、checkpoint IO
- `optimization/`：Muon、optimizer registry、LR schedules、orthogonal gradient transform
- `accelerators/`：按算子族组织的 operator surface 与 TileLang / Flash QLA kernels
- `platform/`：本地 / 线上平台环境、依赖安装和 import bootstrap
- `bundles/`：manifest contract 和 `code_package.zip` 写入 primitive
- `experiments/module_loader.py`：低层动态 import-by-path primitive

## 稳定公共导入面

实验包应优先从 `taac2026.api` 导入共享能力，而不是直接依赖内部目录布局。当前公共导入面包含：

- PCVR 配置对象：`PCVRTrainConfig`、`PCVRModelConfig`、`PCVRNSConfig` 等
- 训练运行时配置：`RuntimeExecutionConfig`、`BinaryClassificationLossConfig`
- 模型 contract：`ModelInput`
- 建模 primitives：`RMSNorm`、tokenizer、embedding bank、mask/pooling/attention helpers
- 工厂函数：`create_pcvr_experiment`
- RMSNorm backend 配置：`configure_rms_norm_runtime`

这样实验包面对的是稳定 API，而不是 `application` / `infrastructure` 内部文件的历史位置。

## 实验包契约

每个实验包只有一个必需入口：

```python
EXPERIMENT = ...
```

模型类实验通常通过 `create_pcvr_experiment(...)` 创建对象，并按需提供：

- `model.py`：实现模型类，类名与 `model_class_name` 一致
- `layers.py`：包内局部模型 helper，可选
- hook overrides：训练、预测或 runtime 的可选覆盖点

维护类或分析类实验可以直接导出 `ExperimentSpec`，只实现 `train_fn` 也可以。loader 只关心 `EXPERIMENT` 能力，而不关心目录名。

## 训练流程

训练入口是 `taac-train` 或 `bash run.sh train`：

1. `application/training/cli.py` 解析 `--experiment`、`--dataset-path`、`--schema-path`、`--run-dir`。
2. `application/experiments/registry.py` 加载实验包并归一化为 `ExperimentSpec`。
3. PCVR 实验对象进入 `application/training/workflow.py` 的默认训练 workflow。
4. workflow 调用 `infrastructure/data` 读取和增强数据，调用 `domain/model_contract.py` 构建模型输入与模型参数。
5. `infrastructure/runtime/trainer.py` 执行训练循环，并写出 checkpoint、schema 和 sidecar。
6. `application/evaluation/runtime.py` 生成训练 / 验证 split 的 observed schema 报告。

常见训练产物包括：

- `model.safetensors`
- `schema.json`
- `train_config.json`
- `train_split_observed_schema.json`
- `valid_split_observed_schema.json`

训练循环仍支持稀疏 / 稠密参数分组、AMP、`torch.compile`、early stopping、checkpoint sidecar 和可配置 optimizer / schedule。

## 评估与推理

### 本地评估

`taac-evaluate single` 或 `bash run.sh val` 会：

1. 解析实验包、数据路径和 checkpoint / run_dir。
2. 读取 checkpoint 对应的 `train_config.json` 和运行时 schema。
3. 运行 `application/evaluation/workflow.py` 的 prediction loop。
4. 计算分类指标。
5. 写出 `evaluation.json`、`validation_predictions.jsonl` 和 `evaluation_observed_schema.json`。

### 平台推理

`taac-evaluate infer` 或推理 bundle 中的 `infer.py` 会输出平台要求的：

```json
{"predictions": {"user_id": score}}
```

推理结果目录只要求最终的 `predictions.json`，不会像本地评估一样再落一套 evaluation sidecar。

## 数据管道

数据子系统由 `infrastructure/data` 承接，训练 workflow 负责决定 train / valid / infer 各自启用哪些 stage。默认执行顺序是：

1. Parquet / source -> base batch
2. base-batch cache
3. 结构性 transforms，例如 sequence crop、multi-view expansion
4. 随机性 transforms，例如 feature mask、domain dropout
5. shuffle / rebatch / prefetch

cache、transform 和 shuffle 都是 pipeline stage，而不是互相替代的入口。训练数据可以启用随机增强和 shuffle，验证 / 推理保持确定性读取。

## 优化与运行时

优化器和训练 runtime 已拆开：

- 稳定配置语义在 `domain/config.py`
- optimizer 构造和名字解析在 `infrastructure/optimization/registry.py`
- LR schedule 在 `infrastructure/optimization/schedules.py`
- 更新前 transform 在 `infrastructure/optimization/transforms.py`
- Muon 实现在 `infrastructure/optimization/muon.py`
- AMP、compile、日志、seed、early stopping 在 `infrastructure/runtime/execution.py`

trainer 消费这些已经解析好的运行时能力，不再把 optimizer、scheduler 和 checkpoint sidecar 混在同一个模块里。

## 加速算子层

加速实现按算子族组织：

- `accelerators/attention/`：FlashAttention、MLA、QLA operator surface
- `accelerators/attention/kernels/`：TileLang attention kernels 和 Flash QLA 低层实现
- `accelerators/normalization/`：RMSNorm operator surface
- `accelerators/normalization/kernels/`：TileLang RMSNorm kernels
- `accelerators/embedding/`：fused embedding bag 等 embedding 相关算子

上层 modeling 代码依赖 operator surface，例如 `flash_attention`、`rms_norm`，而不是直接依赖 TileLang kernel 文件或 Flash QLA 低层目录。

## 线上打包

当前线上上传物分为两类：

- 训练 bundle：`run.sh` + `code_package.zip`
- 推理 bundle：`infer.py` + `code_package.zip`

两者的 zip 内部都使用 `project/` 根目录，manifest 分别是：

- `project/.taac_training_manifest.json`
- `project/.taac_inference_manifest.json`

打包入口现在位于 `application/packaging/cli.py`：

| 命令 | 模块 |
| --- | --- |
| `taac-package-train` | `taac2026.application.packaging.cli:training_main` |
| `taac-package-infer` | `taac2026.application.packaging.cli:inference_main` |

`application/packaging/service.py` 负责 bundle use case，`infrastructure/bundles/zip_writer.py` 只负责 zip 写入 primitive，`infrastructure/bundles/manifest_store.py` 负责 manifest contract 和校验。

详见 [线上 Bundle 上传指南](guide/online-training-bundle.md)。

## 当前边界

- 当前主任务仍是 PCVR 二分类。
- 训练 / 评估 / 推理都基于 Parquet + `schema.json`。
- 实验扩展依赖统一 `experiments/<name>/` 插件目录，而不是改动核心框架入口。
- 旧的二级实验目录和 PCVR/training 基础设施入口不再作为兼容入口保留。
