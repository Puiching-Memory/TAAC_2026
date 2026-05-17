---
icon: lucide/bar-chart-3
---

# RankUp

## 摘要

RankUp 是一个面向“高有效秩表征”的 PCVR 实验包。它参考 *RankUp: Towards High-rank Representations for Large Scale Advertising Recommender Systems* 的问题意识，但实现保持本仓库口径：不改共享 runtime，不引入额外数据依赖，先跑通可训练、可评估、可打包的实验闭环。

它关注的不是“再设计一种注意力”，而是一个更基础的问题：深层推荐模型参数变多以后，token 表征是否真的更丰富。如果深层表示逐渐低秩化，那么更深的 backbone 可能只是在重复少数主方向，AUC 和泛化都会受限。

## 一、为什么独立

TokenFormer 和 Symbiosis 负责统一 token-stream 方向，强调多域特征与序列行为如何在同一架构内交互。RankUp 的切入点不同：它从输入组织和表征诊断开始，试图减少 token 间冗余，让 backbone 在加深后仍保留更高维、更分散的表达。

边界可以这样理解：

- TokenFormer：BFTS + NLIR，解决统一 token 流中的注意力范围和非线性交互。
- Symbiosis：metadata、缺失、风险和 memory selector，解决线上分布漂移与长序列预算。
- RankUp：随机稀疏重组、多 embedding、global token、cross dense token 和 effective-rank 诊断，解决深层表示容量。

## 二、实验入口

入口位于 `experiments/rankup/__init__.py`。

| 项目                | 默认值                                                |
| ------------------- | ----------------------------------------------------- |
| 实验名              | `pcvr_rankup`                                         |
| 模型类              | `PCVRRankUp`                                          |
| NS sidecar          | `rankmixer` 分组保存；模型内部随机重组 sparse feature |
| user / item token   | `8 / 4`                                               |
| batch size          | `256`                                                 |
| 序列上限            | `seq_a:256,seq_b:256,seq_c:512,seq_d:512`             |
| `d_model / emb_dim` | `128 / 64`                                            |
| block / head        | `2 / 4`                                               |
| `seq_top_k`         | `96`                                                  |
| optimizer           | dense `adamw`，sparse Adagrad                         |
| AMP / compile       | 关闭 / 关闭                                           |
| loss                | BCE                                                   |

默认数据管道启用 tail crop、feature mask 和 domain dropout。运行时保持保守，便于先观察结构和 effective-rank 指标。

## 三、输入重组

RankUp 的输入处理故意不沿用语义 group 平均。它认为语义 group 可能把强相关字段聚在一起，导致 token 表示在进入 backbone 之前就变窄。

核心组件包括：

| 组件                             | 当前实现                                                          | 目的                               |
| -------------------------------- | ----------------------------------------------------------------- | ---------------------------------- |
| Randomized Permutation Splitting | 初始化时按固定 seed 打乱 sparse feature index，再均匀分配到 token | 降低同类高相关字段聚集。           |
| Multi-embedding                  | 每个 sparse feature 使用两套独立 `FeatureEmbeddingBank`           | 给同一离散信号多个几何视角。       |
| Dense packet                     | user/item dense 分包投影                                          | 避免 dense 全部压成单个向量。      |
| Cross dense token                | user/item dense 重叠维度做 element-wise product 后投影            | 显式建模用户-候选连续特征交互。    |
| Global token                     | 非序列 token 聚合后投影为第一个 token                             | 给每层 token mixing 一个全局锚点。 |
| Task token                       | 单目标下一个 readout token，多目标时可随 `action_num` 扩展        | 为多目标解耦预留接口。             |

## 四、核心架构

`PCVRRankUp` 的 forward 主线是：

```text
random sparse groups + dense packets + sequence tokens
      -> global/candidate/task/body token batch
      -> RankUpBlock stack
      -> pool(global, candidate, task, body)
      -> classifier
```

`RankUpSelfAttention` 负责 token mixing，`RankUpBlock` 组合 attention 和 FFN。它没有复杂的 metadata mask，也没有候选中心化 attention；这让 effective-rank 诊断更容易归因到输入重组和多 embedding 视角本身。

## 五、有效秩诊断

RankUp 的一个关键产物是 TensorBoard scalar：

```text
RankUp/effective_rank/input/<phase>
RankUp/effective_rank/block*_tm/<phase>
RankUp/effective_rank/block*_ffn/<phase>
```

这些指标用于观察：

- 输入 token 是否比语义 group token 更分散。
- Attention 后表示是否立即低秩化。
- FFN 是否恢复或继续压缩秩。
- 不同层之间是否出现阻尼振荡或持续下降。

因此 RankUp 的验收不应只看 AUC。更合理的判断是：AUC 不明显退化时，effective rank 是否更高、更稳定；如果 AUC 提升但秩明显坍塌，说明模型可能仍依赖 shortcut。

## 六、适合观察什么

RankUp 适合观察：

- 随机稀疏重组是否优于语义 group。
- 多 embedding 是否提升离散 token 的表示多样性。
- global token 是否帮助深层混合。
- cross dense token 是否给 user/item dense 提供有效交互。
- effective rank 与 AUC、logloss、预测方差之间是否一致。

它不适合作为统一建模最终形态；它更像一个表示诊断和高秩假设验证平台。

## 七、消融建议

当前包还没有暴露自定义 CLI 参数。后续扩展建议按这个顺序增加配置面：

1. `rankup_num_embedding_tables`：比较单 embedding 与多 embedding。
2. `rankup_random_split_seed`：检查随机重组稳定性。
3. `rankup_use_cross_dense_token`：验证 dense 交互 token 贡献。
4. `rankup_use_global_token`：验证全局 token 对深层 effective rank 的影响。
5. `rankup_log_effective_rank`：控制诊断频率，避免长训日志过密。

比较对象优先选择 TokenFormer 和 Symbiosis：前者代表纯结构统一流，后者代表分布感知统一流。

## 八、运行与验收

训练：

```bash
bash run.sh train \
  --experiment experiments/rankup \
  --run-dir outputs/rankup_smoke
```

CPU smoke：

```bash
bash run.sh train \
  --experiment experiments/rankup \
  --run-dir outputs/rankup_smoke \
  --device cpu \
  --num_workers 0 \
  --batch_size 8 \
  --max_steps 1 \
  --schema-path docs/archive/files/schema/sample_1000_raw.schema.json
```

最小复核：

```bash
uv run pytest tests/contract/experiments/test_packages.py -q
uv run pytest tests/contract/experiments/test_runtime_contract_matrix.py -q
```

若参与线上 bundle，额外检查 zip 只包含当前实验包和共享 `src/taac2026`。
