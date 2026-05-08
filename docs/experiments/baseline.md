---
icon: lucide/activity
---

# Baseline

Baseline 是当前 PCVR 模型实验的干净参照点。它使用 `PCVRHyFormer`，保留共享运行时的默认训练流程，不额外打开数据增强或 TileLang backend。

## 快速运行

```bash
bash run.sh train \
  --experiment experiments/baseline \
  --run-dir outputs/baseline_smoke
```

评估同一个目录：

```bash
bash run.sh val \
  --experiment experiments/baseline \
  --run-dir outputs/baseline_smoke
```

推理需要传 checkpoint 和结果目录：

```bash
bash run.sh infer \
  --experiment experiments/baseline \
  --checkpoint outputs/baseline_smoke \
  --result-dir outputs/baseline_infer
```

本地 PCVR 训练默认使用仓库约定的 demo 数据；线上 bundle 才通过平台环境变量接收真实数据路径。

## 实验入口

入口文件是 `experiments/baseline/__init__.py`。它声明：

- 实验名：`pcvr_hyformer`
- 模型类：`PCVRHyFormer`
- NS tokenizer：`rankmixer`
- 非序列 token 配置：`user_tokens=5`，`item_tokens=2`
- 数据管道：不启用增强，cache 关闭
- 加速 backend：保持 torch 默认实现

核心默认值在 `TRAIN_DEFAULTS`：

| 配置块           | 当前默认                                                                   |
| ---------------- | -------------------------------------------------------------------------- |
| data             | `batch_size=256`，`num_workers=8`，`train_ratio=1.0`，`valid_ratio=0.1`    |
| seq length       | `seq_a:256,seq_b:256,seq_c:512,seq_d:512`                                  |
| optimizer        | `lr=1e-4`，`dense_optimizer_type="adamw"`，scheduler 关闭                  |
| runtime          | AMP 关闭，`torch.compile` 关闭                                             |
| sparse optimizer | Adagrad sparse lr `0.05`                                                   |
| model            | `d_model=64`，`emb_dim=64`，`num_queries=2`，`num_blocks=2`，`num_heads=4` |
| loss             | BCE，focal 参数保留但 `loss_type="bce"`                                    |

NS 分组直接写在 `PCVRNSConfig` 的 `user_groups` 和 `item_groups` 中。不要再为当前实验补一个独立 `ns_groups.json`。

## 模型结构

模型实现是 `experiments/baseline/model.py` 的 `PCVRHyFormer`。输入由共享 runtime 从 batch 转成 `ModelInput`：

```python
ModelInput(
    user_int_feats,
    item_int_feats,
    user_dense_feats,
    item_dense_feats,
    seq_data,
    seq_lens,
    seq_time_buckets,
)
```

前向过程可以按这几步理解：

1. 用户和物品离散特征经 `RankMixerNSTokenizer` 或 `GroupNSTokenizer` 生成非序列 token。
2. 用户 dense 和物品 dense 各自投影为一个 dense token。
3. 每个序列域单独 embedding，拼接 side info 后投影到 `d_model`。
4. `MultiSeqQueryGenerator` 为每个序列域生成 query token。
5. 多层 `MultiSeqHyFormerBlock` 交替更新 query、非序列 token 和序列 token。
6. 所有序列域的 query token 拼接后进入 `output_proj` 和 classifier。

`forward()` 返回 logits；`predict()` 返回 `(logits, embeddings)`，其中 embeddings 是 classifier 前的输出向量。

## 参数分组和 checkpoint 契约

`get_sparse_params()` 会收集所有 `nn.Embedding` 权重，交给 sparse optimizer。其他参数由 `get_dense_params()` 返回。

`reinit_high_cardinality_params()` 只重初始化词表超过阈值的 embedding，并保留低基数和 time embedding。这个行为会影响 sparse optimizer 重建，改 embedding 结构时要看 `tests/contract/experiments/test_packages.py` 和 runtime contract matrix。

训练 checkpoint 目录需要包含：

```text
global_step*.best_model/
├── model.safetensors
├── schema.json
└── train_config.json
```

评估和推理会依赖这些 sidecar 重建模型，不只依赖权重文件。

## 修改建议

- 改默认 batch、优化器、NS tokenizer 或数据管道：改 `experiments/baseline/__init__.py`。
- 改 HyFormer block、query generator、序列 embedding 或 classifier：改 `experiments/baseline/model.py`。
- 想评估数据增强或 TileLang backend：不要直接改 Baseline，优先看 [Baseline+](baseline-plus.md)。
- 改模型构造参数后，必须确认 `model_class_name` 和 checkpoint sidecar 仍能被评估 / 推理加载。

## 打包

训练 bundle：

```bash
uv run taac-package-train \
  --experiment experiments/baseline \
  --output-dir outputs/bundles/baseline_training
```

推理 bundle：

```bash
uv run taac-package-infer \
  --experiment experiments/baseline \
  --output-dir outputs/bundles/baseline_inference
```

训练 bundle 生成 `run.sh + code_package.zip`，推理 bundle 生成 `infer.py + code_package.zip`。zip 内会包含当前实验包、`src/taac2026/**`、`pyproject.toml` 和对应 manifest。

## 最小复核

```bash
uv run pytest tests/contract/experiments/test_packages.py -q
uv run pytest tests/contract/experiments/test_runtime_contract_matrix.py -q
```

模型结构对应 HyFormer 路线；论文解读页是 [HyFormer](../papers/hyformer.md)。
