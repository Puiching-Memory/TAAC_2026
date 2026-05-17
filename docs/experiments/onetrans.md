---
icon: lucide/arrow-right-left
---

# OneTrans

## 摘要

OneTrans 是统一 Transformer 路线的实验包。它把序列事件、用户静态特征、候选物品特征和 dense token 放进同一套编码路径，让单个主干同时承担序列建模和特征交互。

与 InterFormer 的双分支交互不同，OneTrans 的核心是“统一输入 + causal 主干 + 序列 token 逐层收缩”。它要验证的是：在保持统一架构的同时，是否可以通过 pyramid keep 机制控制长序列计算和噪声传播。

## 一、问题设定

传统推荐模型常把多域特征交互和行为序列建模分开处理，再在后期融合。OneTrans 的判断相反：如果用户画像、候选物品和历史行为本质上都可以表示为 token，那么模型应尽早在同一主干中学习它们的依赖。

这个方向的风险是计算量和序列噪声。长序列直接进入全量 attention 会带来两个问题：

- 旧事件数量多，容易稀释最近行为。
- 非序列 token 数少，可能被长序列注意力淹没。

OneTrans 因此在每层后逐步裁掉旧序列 token，让深层主干更聚焦当前有效上下文。

## 二、实验入口

入口位于 `experiments/onetrans/__init__.py`。

| 项目                | 默认值                                    |
| ------------------- | ----------------------------------------- |
| 实验名              | `pcvr_onetrans`                           |
| 模型类              | `PCVROneTrans`                            |
| NS tokenizer        | `rankmixer`                               |
| user / item token   | `5 / 2`                                   |
| batch size          | `256`                                     |
| 序列上限            | `seq_a:256,seq_b:256,seq_c:512,seq_d:512` |
| `d_model / emb_dim` | `64 / 64`                                 |
| block / head        | `2 / 4`                                   |
| `seq_top_k`         | `50`                                      |
| optimizer           | dense `adamw`，sparse Adagrad             |
| AMP / compile       | 关闭 / 关闭                               |

OneTrans 默认关闭增强和 cache，便于和 Baseline、InterFormer 做结构对照。

## 三、统一输入流

OneTrans 先构造两类 token，然后拼成一条流。

**Sequence stream。** 每个序列域使用 `SequenceTokenizer` 编码，保留最近 `seq_top_k` 个事件。域与域之间插入可学习 separator token，避免不同来源的事件在位置上直接粘连。

**NS tokens。** 用户/物品稀疏字段使用 `NonSequentialTokenizer`，dense 特征使用 `DenseTokenProjector`。如果存在序列统计信息，模型还会加入 sequence stats token，用来提供长度、空序列率或时间覆盖等粗粒度先验。

拼接顺序是：

```text
[sequence stream | user/item/dense/stats NS tokens]
```

这让 causal attention 可以把历史序列作为上下文，再由后部 NS token 汇总并参与预测。

## 四、核心架构

`PCVROneTrans` 的 forward 主线是：

```text
sequence domains -> sequence stream
user/item fields -> NS tokens
sequence + NS    -> OneTransBlock
each layer       -> shrink sequence prefix
seq summary + NS summary -> classifier
```

关键模块：

| 模块                    | 作用                                           |
| ----------------------- | ---------------------------------------------- |
| `MixedCausalAttention`  | 共享 QKV，同时给 NS token 提供位置特定投影。   |
| `MixedFeedForward`      | 普通 token 走共享 FFN，NS token 可走专属 FFN。 |
| `OneTransBlock`         | 组合 mixed attention 和 mixed FFN。            |
| `_pyramid_keep_count()` | 根据层数逐步减少保留的 sequence token。        |

Pyramid keep 是 OneTrans 的关键机制。浅层保留更多历史事件，用于建立全局上下文；深层逐渐收缩，使最终表示更接近候选相关的局部行为和静态特征。

## 五、适合观察什么

OneTrans 适合观察：

- 统一 token stream 是否优于双分支交互。
- separator token 是否能稳定区分不同序列域。
- 序列逐层收缩是否降低计算和噪声。
- NS-specific projection 是否让静态字段在统一主干中保留辨识度。

它不直接解决线上 train/infer 分布漂移。缺失模式、风险 token 和来源 metadata 不是 OneTrans 的主功能；这些问题更适合看 Symbiosis。

## 六、消融建议

1. OneTrans vs InterFormer：比较统一主干与双分支交互。
2. OneTrans vs TokenFormer：比较 pyramid shrink 与 BFTS sliding window。
3. 调整 `seq_top_k`：检查 token 数、耗时和 AUC 的折中。
4. 改 `_pyramid_keep_count()`：观察深层保留更多/更少序列 token 的影响。
5. 移除 sequence stats token：确认粗粒度序列统计是否贡献预测。

## 七、运行与验收

训练：

```bash
bash run.sh train \
  --experiment experiments/onetrans \
  --run-dir outputs/onetrans_smoke
```

评估：

```bash
bash run.sh val \
  --experiment experiments/onetrans \
  --run-dir outputs/onetrans_smoke
```

打包：

```bash
uv run taac-package-train --experiment experiments/onetrans --output-dir outputs/bundles/onetrans_training
uv run taac-package-infer --experiment experiments/onetrans --output-dir outputs/bundles/onetrans_inference
```

最小复核：

```bash
uv run pytest tests/contract/experiments/test_packages.py -q
uv run pytest tests/contract/experiments/test_runtime_contract_matrix.py -q
```

论文背景见 [OneTrans](../papers/onetrans.md)。
