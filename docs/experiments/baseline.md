---
icon: lucide/flask-conical
---

# Baseline

**官方 Day0 参考实现 / Official PCVR HyFormer**

## 概述

Baseline 已切换为官方页面提供的 Day0 baseline。官方参考实现分为训练端自包含 baseline 与最终评分端自包含 baseline；二者共用相同的数据处理和模型定义，差异只在训练脚本、checkpoint 保存逻辑和评分入口。当前仓库只保留可复用契约，并将长期实现收敛到 `config/baseline/` 与共享 PCVR runtime。

框架入口读取官方 `schema.json`，构造共享的 PCVR Parquet dataset，并把 batch dict 转为 `ModelInput` 后调用 `PCVRHyFormer`。`config/baseline/model.py` 是官方 `PCVRHyFormer` 的仓库化版本：主体结构来自官方参考实现，但已适配 `PCVRExperiment` 的共享构造契约，例如使用 `num_blocks` 替代原脚本里的 `num_hyformer_blocks`。

## 官方参考契约

| 比赛角色          | 官方参考内容                                                               | 当前仓库对应实现                                                  |
| ----------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 训练模型 baseline | `run.sh`、训练入口、训练器、工具函数、数据处理、模型定义、`ns_groups.json` | `run.sh`、`src/taac2026/infrastructure/pcvr/`、`config/baseline/` |
| 最终评分 baseline | `infer.py`、数据处理、模型定义                                             | `taac-package-infer` 生成的 `infer.py` 与共享推理入口             |

原始训练入口直接读取官方环境变量：`TRAIN_DATA_PATH`、`TRAIN_CKPT_PATH`、`TRAIN_LOG_PATH`、`TRAIN_TF_EVENTS_PATH`。原始评分入口读取 `MODEL_OUTPUT_PATH`、`EVAL_DATA_PATH`、`EVAL_RESULT_PATH`，要求在结果目录写出 `predictions.json`。

官方训练入口的启用配置是 RankMixer NS tokenizer：`user_ns_tokens=5`、`item_ns_tokens=2`、`num_queries=2`、`ns_groups_json=""`、`emb_skip_threshold=1000000`、`num_workers=8`。参考实现还给了一个 `GroupNSTokenizer` 备选配置：使用 `ns_groups.json`、`num_queries=1`，以满足 `rank_mixer_mode=full` 下 `d_model % T == 0` 的约束。当前 `config/baseline/ns_groups.json` 已从“示例文件”整理成实验包内资产，checkpoint 会复制它，推理时复用同一份分组。

## 模型架构

官方 baseline 使用 `PCVRHyFormer`：

- 4 个序列域：`seq_a`、`seq_b`、`seq_c`、`seq_d`
- RankMixer tokenizer：`user_ns_tokens=5`、`item_ns_tokens=2`
- `num_queries=2`，默认 hidden / embedding 维度为 64
- 稀疏参数使用官方模型暴露的 `get_sparse_params()`，默认 optimizer 会用 Adagrad + AdamW 组合
- 可选 RoPE 位置编码；注意力路径使用 `torch.nn.functional.scaled_dot_product_attention`
- 序列编码器支持 `swiglu`、`transformer`、`longer`，其中 `longer` 会把长序列压缩到最近 `top_k` 个有效 token
- 非序列特征支持 `group` 与 `rankmixer` 两类 tokenizer；`rankmixer` 会把分组后的 embedding 串接、等分，再投影成可配置数量的 NS token

官方管道的 batch 是 Python `dict`，核心键包括：`user_int_feats`、`item_int_feats`、`user_dense_feats`、`item_dense_feats`、`label`、`timestamp`、`user_id`、`_seq_domains`，以及每个序列域的 `seq_*`、`seq_*_len`、`seq_*_time_bucket`。Baseline 不再消费旧的 `BatchTensors.sparse_features` / `sequence_features`。

## 数据契约

官方 baseline 只支持原始多列 Parquet 与官方 `schema.json`。`schema.json` 中的 `user_int`、`item_int`、`user_dense` 与 `seq` 配置会被转换为 `FeatureSchema`，每个特征保留 `(fid, offset, length)`，模型构造时再转换为 `(vocab_size, offset, length)`。

数据管道的关键规则：

- 训练标签来自 `label_type == 2`；评分数据没有标签时返回全零占位 label。
- `user_id` 原样贯穿 batch，评分阶段用于构造 `predictions` 的 key。
- int、multi-hot 和序列 id 中 `<=0` 的值都当作 padding 置零。
- vocab 越界值默认裁剪为 0，并记录 out-of-bound 统计；关闭裁剪时直接报错。
- 序列域按名称排序，默认截断长度为 `seq_a:256,seq_b:256,seq_c:512,seq_d:512`。
- 时间差 bucket 由 `BUCKET_BOUNDARIES` 唯一决定，共 65 个 embedding slot，其中 0 是 padding。

## 训练与 Checkpoint

原始训练器名为 `PCVRHyFormerRankingTrainer`，但实际是 pointwise 二分类训练：损失为 BCEWithLogits 或 sigmoid focal loss，验证指标为 Binary AUC 与 binary logloss。训练循环支持 epoch 末验证，也支持 `eval_every_n_steps` 的 step 级验证。

checkpoint 目录名以 `global_step` 开头，并可带 `layer`、`head`、`hidden` 等参数片段。每个有效 checkpoint 至少需要：

- `model.pt`：严格加载的 `state_dict`。
- `schema.json`：训练时使用的特征布局。
- `train_config.json`：模型结构与数据加载超参数的单一来源。
- `ns_groups.json`：如果训练启用了 NS 分组，则复制到 checkpoint 目录内。

评分 baseline 会使用 checkpoint 内的 `schema.json` 和 `train_config.json` 重建模型；如果缺少 `train_config.json` 或其中缺少必要字段，当前共享 runtime 会直接失败，避免用隐式默认值重建出与训练不一致的模型。

## 评分输出

官方评分入口的流程是：读取 `MODEL_OUTPUT_PATH` 下的 checkpoint，按 `train_config.json` 重建 `PCVRHyFormer`，严格加载 `model.pt`，在 `EVAL_DATA_PATH` 上逐 batch 推理，最后写出：

```json
{
  "predictions": {
    "<user_id>": 0.1234
  }
}
```

预测值是 `sigmoid(logits)` 后的概率，key 必须来自测试集 `user_id`。当前推理 bundle 的生成脚本保留这个输出契约，只是把上传形态从多文件脚本改成 `infer.py` 加 `code_package.zip`。

## 默认配置

| 参数              | 值   |
| ----------------- | ---- |
| `embedding_dim`   | 64   |
| `num_layers`      | 2    |
| `num_heads`       | 4    |
| `num_queries`     | 2    |
| `epochs`          | 999  |
| `batch_size`      | 256  |
| `learning_rate`   | 1e-4 |
| `sparse_lr`       | 5e-2 |
| `pairwise_weight` | 0.0  |

## 快速运行

```bash
bash run.sh train --experiment config/baseline \
  --dataset-path /path/to/official_parquet_dir \
  --schema-path /path/to/official_parquet_dir/schema.json

bash run.sh val --experiment config/baseline \
  --dataset-path /path/to/official_parquet_dir \
  --schema-path /path/to/official_parquet_dir/schema.json
```

如果 `schema.json` 与 parquet 文件位于同一目录，可以省略 `--schema-path`。当前官方 baseline 管道要求本地 parquet 文件或 parquet 目录；缺少 schema 时会直接报错，避免静默落回旧格式。

## 输出目录

```
outputs/config/baseline/
```

## 来源

官方 Day0 baseline。仓库内长期实现位于 `config/baseline/` 和共享 PCVR runtime。
