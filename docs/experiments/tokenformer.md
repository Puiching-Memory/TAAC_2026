---
icon: lucide/layers-3
---

# TokenFormer

## 摘要

TokenFormer 是一个统一多域特征与行为序列的 PCVR 实验包。它把用户静态 field token、dense packet、历史行为 token、候选物品 token 和 target readout token 放进同一条 token 流，通过 BFTS 分层注意力与 NLIR 非线性门控，尝试缓解统一建模中的序列坍塌传播问题。

它继承统一 token 架构的优点：输入形式同构、主干可堆叠、field-field、sequence-sequence 和 sequence-field 交互在同一计算范式内完成。同时它也显式承认统一架构的风险：低秩静态特征过早污染序列表示、长序列注意力成本过高、深层网络可能继续聚合已经不相关的远距 token。

## 一、问题设定

在推荐系统里，用户静态画像和用户行为序列常被分成两条技术路线。前者偏多域特征交叉，后者偏序列建模。简单拼接这两类表示会带来一个结构性问题：静态字段往往更稀疏、更低维、更容易形成 shortcut；当它们与行为序列粗暴融合时，序列 token 的高维表达可能被压扁，形成类似“序列坍塌传播”的现象。

TokenFormer 的假设是：

1. 输入应完全统一成 token stream，而不是晚期融合多个分支。
2. 浅层网络需要 full attention 建立跨域全局关系。
3. 深层网络应逐层收缩注意力感受野，专注局部时序结构。
4. Attention 输出需要非线性门控，避免线性注意力继续放大低秩噪声。

## 二、实验入口

入口位于 `experiments/tokenformer/__init__.py`。

| 项目                | 默认值                                                         |
| ------------------- | -------------------------------------------------------------- |
| 实验名              | `pcvr_tokenformer`                                             |
| 模型类              | `PCVRTokenFormer`                                              |
| NS sidecar          | `rankmixer` 分组仍写入 sidecar，模型内部使用 field-level token |
| batch size          | `256`                                                          |
| 序列上限            | `seq_a:256,seq_b:256,seq_c:512,seq_d:512`                      |
| `d_model / emb_dim` | `64 / 64`                                                      |
| block / head        | `4 / 4`                                                        |
| `seq_top_k`         | `64`                                                           |
| attention           | 底层 full causal，顶层 sliding window                          |
| position            | RoPE 默认开启                                                  |
| optimizer           | dense `adamw`，sparse Adagrad                                  |
| AMP / compile       | BF16 AMP 开启，compile 关闭                                    |
| loss                | BCE                                                            |

默认数据管道启用 tail crop、feature mask 和 domain dropout，用于提升统一 token 流对缺失和序列扰动的鲁棒性。

## 三、统一输入流

TokenFormer 的输入被组织为：

```text
[user fields | user dense | sep | sequence events | sep | item fields | item dense | target]
```

各部分含义如下：

| Token 类型           | 当前实现                                                    | 目的                                 |
| -------------------- | ----------------------------------------------------------- | ------------------------------------ |
| user field token     | 每个用户稀疏字段经 `FeatureEmbeddingBank` 单独投影          | 避免过早把异构字段平均成少量 group。 |
| dense packet token   | user/item dense 按 packet 切分后投影                        | 让连续特征参与同一 token mixing。    |
| sequence event token | 每个序列域经 `SequenceTokenizer` 编码并保留 tail window     | 保留行为时序信号。                   |
| separator token      | 可学习边界 token                                            | 区分用户、序列和候选物品片段。       |
| item field token     | 每个物品稀疏字段单独投影                                    | 给候选物品保留细粒度字段表达。       |
| target token         | user summary、item summary 和逐元素交互投影后加可学习 token | 作为最终 readout 锚点。              |

这里没有显式 type embedding。TokenFormer 让 separator token 与 RoPE 位置几何承担边界表达，减少人为把 unified stream 再拆回多个类型分支的倾向。

## 四、BFTS 分层注意力

BFTS 是 “Bottom-Full-Top-Sliding” 的实现口径。TokenFormer 将网络层分为两个阶段：

**底层 full causal attention。** 浅层需要看到更完整的跨域上下文。用户字段、dense、行为序列和候选物品可以在统一空间里建立全局依赖。

**顶层 sliding window attention。** 深层不再继续全局聚合所有 token，而是限制每个 query 只关注附近窗口。窗口会随层数收缩，让深层逐步聚焦更近、更直接的时序和候选相关信号。

在代码中，这个策略由 `_bfts_attention_mask()` 生成。它同时处理三类约束：

- causal 顺序约束。
- padding / 全空序列安全 mask。
- target token 全局读取权限。

直觉上，BFTS 把“全局融合”和“局部提纯”拆到不同层级中：浅层负责建立统一上下文，深层负责过滤远距噪声。

## 五、NLIR 非线性门控

TokenFormer block 的核心更新为：

```text
attention_update = Attention(Norm(tokens), BFTS_mask)
nlir_update      = sigmoid(G(tokens)) * attention_update
tokens           = tokens + nlir_update
tokens           = tokens + SwiGLU(Norm(tokens))
```

NLIR 把 attention 输出视作需要被输入状态调制的交互表示，而不是直接残差相加。这个逐元素乘法门控有两个作用：

- 增强非线性表达能力，让模型能选择性放大高价值交互。
- 抑制低秩静态噪声在深层持续传播。

Feed-forward 使用 SwiGLU，是为了让前馈阶段也保持乘法交互的表达风格，避免 attention 分支的 NLIR 与 FFN 分支语义割裂。

## 六、适合观察什么

TokenFormer 不应该只看最终 AUC。更有价值的观察包括：

- BFTS 是否在相同 `seq_top_k` 下比 OneTrans 的 pyramid shrink 更稳。
- NLIR 门控是否改善预测分布，减少输出塌缩。
- field-level token 是否比 RankMixer 压缩 token 保留更多静态字段差异。
- target token 是否捕获候选物品与用户历史之间的匹配信号。
- 训练耗时是否可接受，尤其是 full attention 底层的 token 数成本。

## 七、消融建议

建议按下面顺序拆解：

1. TokenFormer vs OneTrans：比较 BFTS sliding window 与 pyramid token shrink。
2. TokenFormer vs Symbiosis：比较无 metadata 的纯结构统一流与分布感知统一流。
3. 关闭 RoPE：观察 separator + sinusoidal fallback 是否足够。
4. 固定所有层 full attention：检查深层 sliding window 的计算和精度贡献。
5. 将 NLIR gate 替换为直接残差：检查非线性门控是否缓解坍塌。
6. 降低/提高 `seq_top_k`：观察窗口长度和 full attention 成本。

## 八、运行与验收

训练：

```bash
bash run.sh train \
  --experiment experiments/tokenformer \
  --run-dir outputs/tokenformer_smoke
```

CPU smoke：

```bash
bash run.sh train \
  --experiment experiments/tokenformer \
  --run-dir outputs/tokenformer_smoke \
  --device cpu \
  --num_workers 0 \
  --batch_size 8 \
  --max_steps 1 \
  --schema-path docs/archive/files/schema/sample_1000_raw.schema.json
```

打包：

```bash
uv run taac-package-train --experiment experiments/tokenformer --output-dir outputs/bundles/tokenformer_training
uv run taac-package-infer --experiment experiments/tokenformer --output-dir outputs/bundles/tokenformer_inference
```

最小复核：

```bash
uv run pytest tests/contract/experiments/test_packages.py -q
uv run pytest tests/contract/experiments/test_runtime_contract_matrix.py -q
```
