---
icon: lucide/spline
---

# UniRec

## 摘要

UniRec 是一个统一推荐建模实验包，用来把若干适合 PCVR 场景的组件组合到共享 runtime 中：field-level token、dense packet、Mixture of Transducers、target-aware interest、Hybrid SiLU attention 和 Block Attention Residuals。

它不是外部 UniRec 仓库的搬运版。训练、评估、推理、checkpoint 和 bundle 仍由 `src/taac2026` 承担；实验包只保留模型结构和默认配置。它的核心目标是验证：在统一 token 框架中，是否可以通过目标感知兴趣建模和跨层残差，让候选物品与用户历史之间的匹配关系更直接地进入预测。

## 一、问题设定

许多统一模型把所有 token 交给同一个 backbone 后，只在最后做 masked mean 或 CLS readout。这种做法简单，但对 PCVR 可能不够：点击/转化通常不是“用户整体兴趣”的平均值，而是“当前候选物品是否命中用户近期与长期兴趣”的条件判断。

UniRec 因此把目标候选物品放回中心：

1. 先保留 field-level 和 dense packet 的细粒度输入。
2. 再用 MoT 从不同序列域提取分支摘要。
3. 用 item summary 查询全部序列 token，得到 target-aware interest。
4. 最后用 Hybrid attention 和 block residual 做统一融合。

## 二、实验入口

入口位于 `experiments/unirec/__init__.py`。

| 项目                | 默认值                                               |
| ------------------- | ---------------------------------------------------- |
| 实验名              | `pcvr_unirec`                                        |
| 模型类              | `PCVRUniRec`                                         |
| NS sidecar          | `rankmixer` 分组保存；模型内部使用 field-level token |
| batch size          | `256`                                                |
| 序列上限            | `seq_a:256,seq_b:256,seq_c:512,seq_d:512`            |
| `d_model / emb_dim` | `64 / 64`                                            |
| block / head        | `2 / 4`                                              |
| `seq_top_k`         | `64`                                                 |
| optimizer           | dense `muon`，sparse Adagrad                         |
| loss                | BCE + `0.02` pairwise AUC regularization             |
| AMP / compile       | BF16 AMP 开启，compile 关闭                          |

默认数据管道启用 tail crop、feature mask 和 domain dropout，适合观察结构在轻量增强下的表现。

## 三、输入与候选中心化

UniRec 的 tokenization 分三部分。

**Feature tokens。** 用户和物品稀疏字段逐字段投影，不先按 NS group 平均。这样候选物品侧强字段和用户侧稀疏字段都能在 backbone 中保留独立位置。

**Dense packets。** 用户 dense 和物品 dense 切成 packet token。相比单个 dense token，packet 更适合让不同数值簇参与局部交互。

**Sequence tokens。** 每个序列域独立编码，保留最近 `seq_top_k` 个事件，并带位置编码。

在这些基础 token 之外，UniRec 显式构造三个功能 token：

- `mot` token：来自多个序列域分支摘要的门控融合。
- `interest` token：由候选 item summary 查询所有 sequence token 得到。
- `target` token：由 user summary、item summary 和逐元素交互构造。

## 四、核心架构

`PCVRUniRec` 的 forward 主线是：

```text
field/dense tokens -> feature cross
sequence domains   -> sequence tokens
sequence tokens    -> MoT + target-aware interest
[feature | sequence | mot | interest | target]
                  -> UniRecBlock stack
                  -> classifier
```

关键模块：

| 模块                       | 作用                                               |
| -------------------------- | -------------------------------------------------- |
| `FeatureCrossLayer`        | 对非序列字段做显式字段交互。                       |
| `MixtureOfTransducers`     | 每个序列域独立摘要，再按候选/用户上下文门控融合。  |
| `TargetAwareInterest`      | 用 item summary 查询行为序列，生成候选感知兴趣。   |
| `HybridSiLUGatedAttention` | 将注意力输出与 SiLU 门控结合，提高非线性交互能力。 |
| `BlockAttentionResidual`   | 在层间注入 block summary，缓解深层信息衰减。       |
| `UniRecBlock`              | 组合 Hybrid attention、残差调节和 FFN。            |

这套结构的直觉是：统一 backbone 负责最终融合，但候选物品相关的兴趣信号要先被显式拉出来，否则 readout 容易退化为粗粒度用户画像。

## 五、适合观察什么

UniRec 适合观察：

- target-aware interest 是否提升候选匹配能力。
- MoT 是否能从不同序列域提取互补信号。
- Block Attention Residuals 是否让深层表示更稳定。
- pairwise AUC regularization 是否改善排序而不破坏 BCE calibration。
- Muon 在该结构上的收益是否大于 AdamW。

它不适合单独回答“纯统一 token 流是否足够”，因为它引入了多个候选中心化功能模块。

## 六、消融建议

1. 去掉 pairwise AUC term：检查排序正则的贡献。
2. 去掉 `TargetAwareInterest`：观察候选查询序列是否必要。
3. 去掉 MoT，只保留 masked mean：检查多域分支摘要价值。
4. 关闭 Block Attention Residuals：观察深层稳定性。
5. 将 dense optimizer 从 `muon` 改为 `adamw`：隔离 optimizer 效应。
6. 与 TokenFormer 对比：观察“纯统一结构”与“候选中心化结构”的差异。

## 七、运行与验收

训练：

```bash
bash run.sh train \
  --experiment experiments/unirec \
  --run-dir outputs/unirec_smoke
```

评估：

```bash
bash run.sh val \
  --experiment experiments/unirec \
  --run-dir outputs/unirec_smoke
```

打包：

```bash
uv run taac-package-train --experiment experiments/unirec --output-dir outputs/bundles/unirec_training
uv run taac-package-infer --experiment experiments/unirec --output-dir outputs/bundles/unirec_inference
```

最小复核：

```bash
uv run pytest tests/contract/experiments/test_packages.py -q
uv run pytest tests/contract/experiments/test_runtime_contract_matrix.py -q
```
