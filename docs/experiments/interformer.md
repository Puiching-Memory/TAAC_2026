---
icon: lucide/arrows-up-from-line
---

# InterFormer

## 摘要

InterFormer 是用户-物品交互路线的实验包。它不把所有输入一开始就揉成同一条 token 流，而是保留“非序列上下文”和“行为序列上下文”的结构差异，在 `InterFormerBlock` 内让两者反复交换信息。

它适合验证一个问题：在 PCVR 场景中，用户静态画像、候选物品和历史行为是否应该以双分支方式交互，而不是直接进入统一 Transformer。

## 一、问题设定

统一 token 流的优势是表达形式简单，但它也可能把低秩静态特征噪声传播到序列表示中。InterFormer 采取更保守的路线：用户/物品特征先形成非序列 token，行为序列仍保留时间结构，二者通过交互块逐层融合。

这种设计包含两个假设：

1. 候选物品和用户画像应尽早参与序列理解，否则模型只是在做普通行为摘要。
2. 序列上下文仍应保持独立通路，避免所有历史事件都被静态字段全量污染。

## 二、实验入口

入口位于 `experiments/interformer/__init__.py`。

| 项目                | 默认值                                    |
| ------------------- | ----------------------------------------- |
| 实验名              | `pcvr_interformer`                        |
| 模型类              | `PCVRInterFormer`                         |
| NS tokenizer        | `group`                                   |
| user / item token   | 按显式 group 数生成                       |
| batch size          | `256`                                     |
| 序列上限            | `seq_a:256,seq_b:256,seq_c:512,seq_d:512` |
| `d_model / emb_dim` | `64 / 64`                                 |
| block / head        | `2 / 4`                                   |
| `seq_top_k`         | `50`                                      |
| optimizer           | dense `adamw`，sparse Adagrad             |
| AMP / compile       | 关闭 / 关闭                               |

InterFormer 默认不启用数据增强和 cache，方便和 Baseline 做结构对照。

## 三、输入结构

InterFormer 的输入组织分成两侧。

**非序列侧。** 用户稀疏字段、物品稀疏字段和 dense token 被编码为 NS token。这里默认使用 `group` tokenizer，因此 token 更贴近 `PCVRNSConfig` 中的语义分组。

**序列侧。** 每个序列域通过 `SequenceTokenizer` 编码，并加入 sinusoidal position。模型保留每个 domain 最近 `seq_top_k` 个事件，避免长序列直接放大计算。

双侧输入在 block 内交互，而不是先拼成一条不可区分的长流。

## 四、核心架构

`PCVRInterFormer` 的 forward 主线是：

```text
user/item fields + dense -> NS tokens
sequence domains         -> sequence tokens
NS tokens <-> sequence tokens via InterFormerBlock
sequence summary + NS summary -> final gate -> classifier
```

关键模块：

| 模块               | 作用                                               |
| ------------------ | -------------------------------------------------- |
| `InterFormerBlock` | 让 NS token 与 sequence token 在同一层中交互更新。 |
| `CrossSummary`     | 从序列侧生成可与 NS summary 对齐的上下文摘要。     |
| `final_gate`       | 在静态/候选信息和序列信息之间做门控融合。          |
| classifier         | 输出 PCVR logits。                                 |

门控融合是 InterFormer 的重要保护机制。它让模型可以在候选物品强、序列弱时依赖 NS summary，也可以在历史行为更可靠时提高 sequence summary 权重。

## 五、适合观察什么

InterFormer 适合观察：

- group tokenizer 是否比 RankMixer 更利于用户-物品交互解释。
- 非序列和序列双分支是否比完全统一 token 流更稳。
- final gate 是否能缓解 item shortcut 或序列噪声。
- 与 OneTrans/TokenFormer 对比时，结构收益是否来自“交互方式”而不是训练 recipe。

它不适合直接当作长序列效率模型。序列仍需要先经过固定 top-k 裁剪，超长历史的 memory 选择不是 InterFormer 的主目标。

## 六、消融建议

建议按这个顺序做：

1. InterFormer vs Baseline：只看双分支交互是否优于 HyFormer query 结构。
2. InterFormer vs OneTrans：比较双分支交互与统一 causal stream。
3. 调整 `seq_top_k`：观察序列窗口对 AUC、耗时和预测分布的影响。
4. 替换 `group` 与 `rankmixer`：检查语义 group 和自动切分 token 的差异。
5. 关闭或弱化 final gate：验证门控融合是否真的参与决策。

## 七、运行与验收

训练：

```bash
bash run.sh train \
  --experiment experiments/interformer \
  --run-dir outputs/interformer_smoke
```

评估：

```bash
bash run.sh val \
  --experiment experiments/interformer \
  --run-dir outputs/interformer_smoke
```

打包：

```bash
uv run taac-package-train --experiment experiments/interformer --output-dir outputs/bundles/interformer_training
uv run taac-package-infer --experiment experiments/interformer --output-dir outputs/bundles/interformer_inference
```

最小复核：

```bash
uv run pytest tests/contract/experiments/test_packages.py -q
uv run pytest tests/contract/experiments/test_runtime_contract_matrix.py -q
```

论文背景见 [InterFormer](../papers/interformer.md)。
