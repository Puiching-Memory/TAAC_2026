---
icon: lucide/activity
---

# Baseline

## 摘要

Baseline 是当前仓库所有 PCVR 模型实验的“零刻度线”。它使用 `PCVRHyFormer`，保留官方 HyFormer 思路和共享 runtime 默认训练路径，不额外打开数据增强、cache、TileLang backend 或实验私有 hook。

它的价值不在于结构最新，而在于可解释：当一个新实验声称 AUC、吞吐、稳定性或线上泛化更好时，Baseline 提供最干净的参照。任何新模型都应该先回答一个简单问题：它相对这个参照多做了什么，以及收益是否来自结构本身。

## 一、问题设定

TAAC PCVR 数据同时包含三类信息：

- 用户侧静态稀疏与稠密特征。
- 候选物品侧静态稀疏与稠密特征。
- 多个行为序列域，每个域又包含 ID、时间和 side-info。

Baseline 的核心假设是：静态特征适合先压缩成少量非序列 token，行为序列适合由 query token 提取上下文，然后在 HyFormer block 中逐层交互。它不是“完全统一 token 流”，而是保留了非序列 token、序列 token 和 query token 的角色差异。

这种设计的优点是稳定、可控、易做回归。缺点也很明确：如果静态字段和长序列之间存在更细粒度的跨域依赖，早期压缩可能会损失信息。

## 二、实验入口

入口位于 `experiments/baseline/__init__.py`。

| 项目              | 默认值                                    |
| ----------------- | ----------------------------------------- |
| 实验名            | `pcvr_hyformer`                           |
| 模型类            | `PCVRHyFormer`                            |
| NS tokenizer      | `rankmixer`                               |
| user / item token | `5 / 2`                                   |
| batch size        | `256`                                     |
| 序列上限          | `seq_a:256,seq_b:256,seq_c:512,seq_d:512` |
| optimizer         | dense `adamw`，sparse Adagrad             |
| loss              | BCE                                       |
| AMP / compile     | 关闭 / 关闭                               |
| 数据增强          | 无                                        |
| cache             | `none`                                    |

NS 分组直接写在 `PCVRNSConfig.user_groups` 和 `PCVRNSConfig.item_groups` 中。当前实验不需要独立 `ns_groups.json`；sidecar 会保存这些默认值，供评估和推理重建。

## 三、输入与 tokenization

共享 runtime 会把 batch 转成 `ModelInput`：

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

Baseline 的输入流可以拆成三段。

**非序列稀疏字段。** `RankMixerNSTokenizer` 先将用户/物品离散字段 embedding，再按 `user_tokens` 和 `item_tokens` 切分成固定数量 token。相比按语义 group 直接平均，RankMixer 可以让 token 数独立于 group 数，适合用同一模型宽度比较不同 schema。

**稠密字段。** 用户 dense 与物品 dense 分别投影为 dense token，加入非序列 token 集合。Baseline 不对 dense 做额外 robust transform，因此它也是观察 dense scale drift 风险的参照。

**序列字段。** 每个序列域独立 embedding，并融合 time bucket。序列 token 不直接与所有静态 token 拼成一条长流，而是由 query token 提取域内和跨域上下文。

## 四、核心架构

`PCVRHyFormer` 的 forward 主线是：

```text
sparse/dense fields -> NS tokens
sequence domains    -> sequence tokens
queries             -> MultiSeqHyFormerBlock
updated queries     -> output projection -> classifier
```

关键模块如下：

| 模块                     | 作用                                                       |
| ------------------------ | ---------------------------------------------------------- |
| `RankMixerNSTokenizer`   | 将高维静态稀疏字段压成固定数量 NS token。                  |
| `SequenceEmbedding`      | 将每个序列域的多列 side-info 投影到统一维度。              |
| `MultiSeqQueryGenerator` | 为不同序列域生成 query token，让模型从行为历史中提取摘要。 |
| `MultiSeqHyFormerBlock`  | 交替更新 query、NS token 和 sequence token。               |
| classifier               | 使用 query context 输出 PCVR logits。                      |

这条路线的直觉是：序列建模不需要让每个历史事件都和每个静态字段全量交互；query token 可以作为信息瓶颈，把行为动态提炼成可预测的表示。

## 五、它适合观察什么

Baseline 最适合回答这些问题：

- 新模型是否真的强于官方 HyFormer 路线。
- 数据增强、cache、optimizer 或 backend 是否改变了结论。
- 模型输出是否正常分布，而不是塌缩到常数概率。
- checkpoint sidecar 能否稳定支持训练后评估和推理。

不要用 Baseline 单独证明“统一 token 架构不好”。它是参照，不是该方向的最强实现。

## 六、修改建议

- 改 batch、optimizer、NS tokenizer 或数据配置：改 `experiments/baseline/__init__.py`。
- 改 HyFormer block、query generator、序列 embedding 或 classifier：改 `experiments/baseline/model.py`。
- 想评估增强、Muon 或 TileLang：优先改 [Baseline+](baseline-plus.md)，保留 Baseline 干净。
- 改模型构造参数后，确认评估/推理能通过 `train_config.json` 和 `schema.json` 重建。

## 七、运行与验收

训练：

```bash
bash run.sh train \
  --experiment experiments/baseline \
  --run-dir outputs/baseline_smoke
```

评估：

```bash
bash run.sh val \
  --experiment experiments/baseline \
  --run-dir outputs/baseline_smoke
```

推理：

```bash
bash run.sh infer \
  --experiment experiments/baseline \
  --checkpoint outputs/baseline_smoke \
  --result-dir outputs/baseline_infer
```

打包：

```bash
uv run taac-package-train --experiment experiments/baseline --output-dir outputs/bundles/baseline_training
uv run taac-package-infer --experiment experiments/baseline --output-dir outputs/bundles/baseline_inference
```

最小复核：

```bash
uv run pytest tests/contract/experiments/test_packages.py -q
uv run pytest tests/contract/experiments/test_runtime_contract_matrix.py -q
```

论文背景见 [HyFormer](../papers/hyformer.md)。
