---
icon: lucide/blocks
---

# 架构与概念

## 工程结构

项目采用三层领域驱动设计：

```
src/taac2026/
├── domain/                 # 纯业务逻辑（无 PyTorch 依赖）
│   ├── config.py           # TrainRequest / EvalRequest / InferRequest
│   ├── experiment.py       # ExperimentSpec / TrainFn / EvalFn / InferFn
│   └── metrics.py          # AUC / LogLoss / GAUC / Bootstrap CI
├── application/            # CLI 入口
│   ├── training/cli.py     # taac-train
│   ├── evaluation/cli.py   # taac-evaluate
│   ├── maintenance/        # taac-package-* / taac-generate-*
│   └── reporting/          # taac-plot-* / taac-tech-timeline / taac-bench-report
└── infrastructure/         # 数据加载、训练、模型构建
    ├── checkpoints.py      # Checkpoint 解析与保存
    ├── experiments/        # 实验包发现与加载
    ├── io/                 # 文件工具
    └── pcvr/               # PCVR 核心框架
        ├── config.py       # PCVRTrainConfig 层级配置
        ├── data.py         # PCVRParquetDataset
        ├── data_pipeline.py# 增强管道
        ├── experiment.py   # PCVRExperiment 适配器
        ├── modeling.py     # 可复用构建块
        ├── protocol.py     # ModelInput / build_pcvr_model
        ├── trainer.py      # PCVRPointwiseTrainer
        └── training.py     # train_pcvr_model / parse_pcvr_train_args
```

实验包独立于框架之外：

```
config/
├── baseline/
│   ├── __init__.py         # EXPERIMENT = PCVRExperiment(...)
│   ├── model.py            # 模型实现
│   └── ns_groups.json      # NS 特征分组
├── symbiosis/
├── ctr_baseline/
├── deepcontextnet/
├── hyformer/
├── interformer/
├── onetrans/
├── unirec/
└── uniscaleformer/
```

## 运行入口

核心 CLI 命令通过 `pyproject.toml` 注册：

| 命令            | 模块                                       |
| --------------- | ------------------------------------------ |
| `taac-train`    | `taac2026.application.training.cli:main`   |
| `taac-evaluate` | `taac2026.application.evaluation.cli:main` |

`taac-evaluate` 有两个子命令：

- `single` -- 评估有标签数据，输出 AUC 等指标
- `infer` -- 推理无标签数据，输出 `predictions.json`

`run.sh` 是 shell 层包装，映射 `train` / `val` / `infer` / `package` 到对应 CLI 命令，并支持 Bundle 模式（检测 `code_package.zip` 自动解压安装）。

## 实验包契约

每个实验包目录必须包含：

| 文件             | 要求                                                                                    |
| ---------------- | --------------------------------------------------------------------------------------- |
| `__init__.py`    | 定义 `EXPERIMENT = PCVRExperiment(name, package_dir, model_class_name, train_defaults)` |
| `model.py`       | 实现模型类，必须暴露与 `model_class_name` 同名的类                                      |
| `ns_groups.json` | JSON 格式的 NS 特征分组                                                                 |

`PCVRExperiment` 提供三个方法：

- `train(TrainRequest)` -- 启动训练
- `evaluate(EvalRequest)` -- 运行评估
- `infer(InferRequest)` -- 运行推理

加载机制：`load_experiment_package(value)` 通过 `sys.path` 临时修改加载实验目录中的 `model.py` 模块。

## 模型输入与构建

所有 PCVR 模型接收统一的 `ModelInput` NamedTuple：

```python
class ModelInput(NamedTuple):
    user_int_feats:       Tensor   # (B, num_user_int)
    item_int_feats:       Tensor   # (B, num_item_int)
    user_dense_feats:     Tensor   # (B, user_dense_dim)
    item_dense_feats:     Tensor   # (B, item_dense_dim)
    seq_data:             dict[str, Tensor]  # domain -> (B, num_features, max_seq_len)
    seq_lens:             dict[str, Tensor]  # domain -> (B,)
    seq_time_buckets:     dict[str, Tensor]  # domain -> (B, max_seq_len)
```

`build_pcvr_model()` 根据 schema 和配置自动构建模型，负责：

1. 解析 `schema.json` 生成 `user_int_feature_specs` / `item_int_feature_specs`
2. 加载 `ns_groups.json` 生成 `user_ns_groups` / `item_ns_groups`
3. 实例化实验包中的模型类，传入所有规格参数

模型必须实现的方法：

| 方法                             | 签名                                                      |
| -------------------------------- | --------------------------------------------------------- |
| `forward`                        | `(ModelInput) -> logits`                                  |
| `predict`                        | `(ModelInput) -> (logits, embeddings)`                    |
| `get_sparse_params`              | `() -> list[Parameter]`（来自 `EmbeddingParameterMixin`） |
| `get_dense_params`               | `() -> list[Parameter]`（来自 `EmbeddingParameterMixin`） |
| `reinit_high_cardinality_params` | `() -> None`                                              |

## NS Groups

NS（Non-Sequential）Groups 将非序列特征 ID 映射到语义分组，供 NS Tokenizer 使用。

`ns_groups.json` 格式：

```json
{
  "user_ns_groups": {
    "U1": [1, 15],
    "U2": [48, 49, 89, 90, 91]
  },
  "item_ns_groups": {
    "I1": [11, 13],
    "I2": [5, 6, 7, 8, 12]
  }
}
```

特征 ID 是列名的数字后缀（`user_int_feats_1` -> fid 1）。

两种 NS Tokenizer：

- **`group`**（`GroupNSTokenizer`）-- 每组一个 token，取组内 Embedding 均值
- **`rankmixer`**（`RankMixerNSTokenizer`）-- 全部 Embedding 拼接后分组投影，参数更多但表达力更强

## 训练流程

`PCVRPointwiseTrainer` 驱动训练循环：

1. **双优化器** -- Adagrad 处理稀疏（Embedding）参数，AdamW 处理稠密参数
2. **AMP** -- 可选混合精度训练，支持 `bfloat16` / `float16`
3. **`torch.compile`** -- 可选 JIT 编译加速
4. **Early Stopping** -- 基于验证集 AUC
5. **Checkpoint** -- 保存 `model.safetensors` + 侧车文件（`schema.json`、`ns_groups.json`、`train_config.json`）
6. **高基数 Embedding 重初始化** -- 每个 epoch 后重初始化低频特征的 Embedding

损失函数（在 `infrastructure/training/runtime.py`）：

- `compute_binary_classification_loss()` -- 支持 BCE、Focal Loss、Pairwise AUC Loss
- `sigmoid_focal_loss()` -- 处理类别不平衡
- `binary_pairwise_auc_loss()` -- 直接优化 AUC 排序

## 评估与推理

**评估**（`taac-evaluate single`）：

- 加载 Checkpoint，遍历验证集，计算 AUC / LogLoss / Brier / GAUC / Bootstrap CI
- 输出指标 JSON 和可选的逐样本预测

**推理**（`taac-evaluate infer`）：

- 加载 Checkpoint，遍历无标签数据
- 输出 `predictions.json`：`{"predictions": {user_id: score}}`

指标计算（`domain/metrics.py`）：

- `binary_auc` / `binary_logloss` / `binary_score_diagnostics`
- `binary_auc_bootstrap_ci` -- Bootstrap 置信区间
- `group_auc` -- 按用户分组的 GAUC

## 线上打包

**训练 Bundle**（`taac-package-train`）：

```
run.sh                    # 自动检测 Bundle / 本地模式
code_package.zip
├── .taac_training_manifest.json
├── pyproject.toml
├── src/taac2026/
├── config/<experiment>/
└── tools/
```

**推理 Bundle**（`taac-package-infer`）：

```
infer.py                  # 自解压 + 安装 + 推理脚本
code_package.zip
├── .taac_inference_manifest.json
├── pyproject.toml
├── src/taac2026/
├── config/<experiment>/
└── <checkpoint>/
```

环境变量：`EVAL_DATA_PATH`、`EVAL_RESULT_PATH`、`MODEL_OUTPUT_PATH`、`TAAC_SCHEMA_PATH`。

详见 [线上 Bundle 上传指南](guide/online-training-bundle.md)。

## 当前边界

- 仅支持 PCVR 二分类任务（AUC 优化）
- Parquet 列式数据格式，120 列固定 Schema
- 序列域固定为 4 个（a, b, c, d）
- 模型接口为 `ModelInput -> logits`，不支持多任务
- NS Groups 为静态分组，不支持动态特征选择
