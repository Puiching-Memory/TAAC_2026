---
icon: lucide/zap
---

# Symbiosis

Symbiosis 是仓库里用于探索“统一序列建模与特征交互”的 PCVR 实验包。2026-05-12 起，当前实现正式切换为 **Symbiosis V2**：非序列字段、候选物品、行为序列、缺失状态、时间信息和统计信息被翻译成统一 token stream，再交给同构、可堆叠的 backbone。

当前新训练默认启用 **Symbiosis V3** 的确定性 memory event selector。V3 不引入可学习路由器，也不增加 loss；它只改变长序列进入 unified token stream 前的固定预算分配和 memory 事件选择规则。旧 checkpoint sidecar 如果没有 V3 字段，会按 V2 路径回退，避免 V2 权重在评估/推理时突然切换选择逻辑。

V2 移除了 V1 内部兼容层。旧的 `sequence_memory` 公开属性、`_sparse_tokens` / `_encode_tokens` 私有调试接口、candidate / cross / global / context / item / sequence 六段 summary readout，以及 `symbiosis_use_*` 旧消融开关不再作为当前契约维护。外部训练、评估、推理和 checkpoint sidecar 仍使用同一个实验包入口：`pcvr_symbiosis` / `PCVRSymbiosis`。

## 快速运行

```bash
bash run.sh train \
  --experiment experiments/symbiosis \
  --run-dir outputs/symbiosis_smoke
```

关闭部分 V2 机制时使用新的 `symbiosis_v2_*` 参数，例如：

```bash
bash run.sh train \
  --experiment experiments/symbiosis \
  --run-dir outputs/symbiosis_v2_no_bias \
  --no-symbiosis-v2-use-metadata-attention-bias
```

Symbiosis 的额外参数来自 `experiments/symbiosis/config.py` 中的 `SymbiosisModelDefaults`。布尔参数由 argparse 提供 `--foo` / `--no-foo` 两种形式。

## V1 记录

V1 是比赛早期快速迭代出的复杂融合模型。它的目标是在不改共享运行时契约的前提下，把强 item 信号、长序列记忆、dense packet、候选 token 和显式交叉 token 尽快并入一个可训练模型。

V1 的主要结构是：

1. user / item 稀疏特征先按 `ns.user_groups` / `ns.item_groups` 变成语义分组 token。
2. dense 特征做 `sign(x) * log1p(abs(x))` 后拆成 dense packet token。
3. missing mask 作为额外 missing token 输入融合层。
4. user context 和 item summary 先生成 sequence query。
5. 每个序列域输出 recent token、memory block token 和 global token。
6. candidate token、cross token、global token 显式建模粗粒度交互。
7. 所有 token 进入 self-attention backbone，但 attention mask 仍按 span 名称手工约束。
8. 最终池化 candidate、cross、global、context、item、sequence 六段 summary，再拼接分类。

V1 的价值在于快速验证了 dense packet、固定预算序列压缩、结构化 attention mask、训练诊断和额外 CLI 参数。但它本质上仍是“多分支摘要后统一融合”，不是从输入 tokenization 开始统一。

## 为什么改成 V2

线上 EDA 显示：

- train 和 infer schema 一致，但分布不一致。
- `item_int_feats_11`、`item_int_feats_16`、`item_int_feats_8` 等 item 字段 cardinality 在 infer 侧明显收缩。
- `user_int_feats_15`、`user_int_feats_80`、`user_int_feats_92` 和 `item_int_feats_11` 等字段存在 null-rate drift。
- `user_dense_feats_62-66` 存在均值和方差漂移，dense 尺度不能被模型当成稳定常量。
- `seq_c`、`seq_d` 在 infer 侧更长，序列长度本身就是分布信号。
- token overlap / OOV sketch 应成为字段风险先验，而不是只在 EDA 报告里出现。

这些现象说明，模型必须在统一架构里显式表达 token 的来源、可靠性、时间和字段语义，让同一个 backbone 同时学习序列动态和多字段交互，同时避免 item shortcut、dense scale shortcut 和随机验证切分带来的虚高线下分数。

V2 因此不再继续维护 V1 的分支融合结构，而是把所有输入收敛到统一 token-stream。

## V2 当前结构

V2 的 forward 主线收敛为：

```python
def forward(inputs):
    batch = self.tokenizer(inputs)
    attention_mask = self.attention_mask(batch)
    tokens = self.backbone(batch.tokens, batch.padding_mask, attention_mask)
    embedding = self.pooler(tokens, batch)
    return self.classifier(embedding)
```

当前文件边界：

```text
experiments/symbiosis/
├── __init__.py       # 实验入口、训练默认值和 hooks
├── config.py         # V2 defaults and CLI config keys
├── model.py          # PCVRSymbiosis 组装和外部契约
├── tokenization.py   # sparse/dense/missing/event/stats unified tokenizer
├── attention.py      # metadata-driven attention mask
├── backbone.py       # 同构 unified interaction blocks
└── pooling.py        # candidate / CLS readout
```

### Unified Tokenization

V2 的 tokenizer 输出 `UnifiedTokenBatch`：

- `tokens`: `(B, T, d_model)` 统一 token stream。
- `padding_mask`: `(B, T)` token padding / dropout mask。
- `role_ids`: `(T,)`，标记 CLS、candidate、user、item、dense、missing、sequence、stats。
- `domain_ids`: `(T,)`，标记序列域。
- `risk_ids`: `(T,)`，标记高漂移或高 shortcut 风险 token。
- `cls_index` / `candidate_index`: readout token 位置。

Token 表示由 value、role、domain、risk 等元信息组成：

```text
token = value embedding/projection
      + role embedding
      + domain embedding
      + risk embedding
      + time/position embedding for sequence events
```

Sparse 字段使用字段级 missing embedding。`user_int_missing_mask` / `item_int_missing_mask` 会直接改变 sparse field token，而不是只生成额外 missing token。

Dense 字段使用 robust log transform，并拼入 missing indicator。Missing pattern 也会作为独立 token 进入统一流。

### Event-Budget Sequence Tokens

长序列不再作为外置 `sequence_memory` 子系统暴露。每个序列域先被压缩成固定预算事件 token：

```text
seq domain = memory event tokens + recent event tokens
```

V2 fallback 每域保留 `8` 个 memory event token 和 `16` 个 recent event token。V3 新训练默认改成来源复杂度感知预算：

| 来源    | recent | memory | 依据                                    |
| ------- | ------ | ------ | --------------------------------------- |
| `seq_a` | 8      | 4      | 短序列域，保留短期信号和少量历史覆盖    |
| `seq_b` | 8      | 4      | 短序列域，预算与 `seq_a` 对齐           |
| `seq_c` | 20     | 10     | infer 侧更长，给更多 recent 和 memory   |
| `seq_d` | 24     | 12     | raw 上限最高，长期和短期 token 预算最大 |

`seq_c` / `seq_d` 可以使用更大的 raw input 上限，但进入 backbone 的 token 数固定，因此 infer 侧序列变长不会线性放大计算。

序列 tokenization 在 AMP 下保持 fp32，之后的统一 backbone 仍可走 AMP。

V3 的 memory 选择不再做纯均匀抽样。默认 `quality_stratified` 先把 recent 之前的 prefix 历史切成固定数量 bucket，再在每个 bucket 里用确定性质量分数选一个事件：

```text
score = density_weight * nonzero_feature_ratio
      + time_weight * has_valid_time_bucket
      + recency_weight * bucket_internal_recency
      - duplicate_penalty * consecutive_duplicate
```

这样仍然保持时间覆盖，但优先避免把 memory token 浪费在全零、弱信息或重复事件上。

### Metadata Attention

V2 不使用 V1 的 span-aware mask。结构先验由 token metadata 生成：

- CLS / candidate readout 可以读取全局 token。
- field token 可以读取同类字段、candidate、missing 和 stats。
- sequence event token 可以读取 candidate、item、sequence、missing 和 stats。
- stats token 可以读取全局。
- high-risk item / dense / missing token 仍参与交互，但会被 risk metadata 和训练期 dropout 约束。

这样 block 仍是同构的，区别只来自 metadata 生成的 attention mask。

### Candidate / CLS Readout

V2 不再拼接六段 V1 summary。Readout 由 CLS token、candidate token 和全局 masked mean 组成，再进入分类器。候选物品仍是中心 token，但不再绕过统一 backbone 直接进入分类器。

### Diagnostics

训练诊断改为 V2 token-stream 口径：

- `SymbiosisV2/tokens/active_ratio/<phase>`
- `SymbiosisV2/tokens/count/<phase>`
- `SymbiosisV2/tokens/high_risk_ratio/<phase>`
- `SymbiosisV2/embedding/norm_mean/<phase>`

旧的 V1 token health、latent diversity、span usage 指标不再维护。

## V2 参数

| 参数                                         | 控制                                                                   |
| -------------------------------------------- | ---------------------------------------------------------------------- |
| `symbiosis_v2_use_dense_tokens`              | 是否加入 dense token                                                   |
| `symbiosis_v2_use_missing_tokens`            | 是否加入独立 missing pattern token                                     |
| `symbiosis_v2_use_sequence_stats_tokens`     | 是否加入 sequence stats token                                          |
| `symbiosis_v2_use_metadata_attention_bias`   | 是否使用 metadata-driven attention mask                                |
| `symbiosis_v2_use_candidate_readout`         | readout 是否使用 candidate token                                       |
| `symbiosis_v2_tokenization_mode`             | `group` / `group_compressed` / `random_chunk` sparse tokenization 模式 |
| `symbiosis_v2_sparse_seed`                   | sparse tokenization 种子，保留给随机分块模式                           |
| `symbiosis_v2_recent_event_tokens`           | 每个序列域保留的 recent event token 数                                 |
| `symbiosis_v2_memory_event_tokens`           | 每个序列域保留的 memory event token 数                                 |
| `symbiosis_v2_user_dense_tokens`             | user dense packet 数；新训练默认 `3`                                   |
| `symbiosis_v2_item_dense_tokens`             | item dense packet 数                                                   |
| `symbiosis_v2_user_missing_tokens`           | user missing pattern token 数                                          |
| `symbiosis_v2_item_missing_tokens`           | item missing pattern token 数                                          |
| `symbiosis_v2_high_risk_token_dropout_rate`  | 训练期 high-risk token dropout 比例                                    |
| `symbiosis_v2_compress_large_ids`            | 是否对高基数 sparse / sequence id 做 hash compression                  |
| `symbiosis_v2_compile_backbone`              | 开启 `--compile` 时是否只编译 V2 backbone                              |
| `symbiosis_v3_enabled`                       | 新训练默认开启；旧 sidecar 缺失时回退关闭                              |
| `symbiosis_v3_memory_selection_mode`         | `uniform` / `stratified` / `quality_stratified`                        |
| `symbiosis_v3_recent_event_tokens_by_domain` | 按 domain 配置 recent token 预算                                       |
| `symbiosis_v3_memory_event_tokens_by_domain` | 按 domain 配置 memory token 预算                                       |
| `symbiosis_v3_memory_density_weight`         | memory 事件非零字段密度权重                                            |
| `symbiosis_v3_memory_time_weight`            | memory 事件有效时间桶权重                                              |
| `symbiosis_v3_memory_recency_weight`         | memory bucket 内相对新近权重                                           |
| `symbiosis_v3_memory_duplicate_penalty`      | 连续重复事件惩罚                                                       |

推荐消融顺序：

1. `--no-symbiosis-v3-enabled`：回到 V2 均匀 memory，检查 V3 selector 总贡献。
2. `--symbiosis-v3-memory-selection-mode stratified`：只保留分桶覆盖，去掉质量打分。
3. `--symbiosis-v3-memory-selection-mode uniform`：检查旧均匀 memory 行为。
4. `--symbiosis-v3-memory-event-tokens-by-domain seq_a:0,seq_b:0,seq_c:0,seq_d:0`：只保留 recent events，检查 memory budget。
5. `--no-symbiosis-v2-use-metadata-attention-bias`：检查 metadata mask 的贡献。
6. `--no-symbiosis-v2-use-candidate-readout`：检查 candidate-centered readout 的贡献。
7. `--no-symbiosis-v2-use-missing-tokens`：检查独立 missing pattern token 的贡献。
8. `--no-symbiosis-v2-use-sequence-stats-tokens`：检查 sequence stats token 的贡献。
9. `--symbiosis-v2-tokenization-mode group_compressed`：检查压缩稀疏 tokenization。
10. 调整 `--symbiosis-v2-high-risk-token-dropout-rate`：检查 item / dense shortcut 约束强度。

## 训练默认值

Symbiosis V2 默认仍使用自定义 hooks，但模型内部不再依赖 V1 私有接口。

- 实验名：`pcvr_symbiosis`
- 模型类：`PCVRSymbiosis`
- 默认层数：`num_blocks=4`
- 默认维度：`d_model=256`，`emb_dim=256`，`num_heads=8`
- 默认 sequence raw 上限：`seq_a:256,seq_b:256,seq_c:1024,seq_d:2048`
- 默认验证切分：`timestamp_auto`，按 `valid_ratio=0.1` 自动使用最新 timestamp 样本做 valid
- 默认 V3 token 预算：`seq_a=8+4`，`seq_b=8+4`，`seq_c=20+10`，`seq_d=24+12`
- 默认 dense packet：user dense `3`，item dense `1`
- 默认 optimizer：`muon`
- 默认 scheduler：关闭
- 默认验证 probe：`none`，早停监控 `auc`
- 默认运行时：AMP bf16 开启，`torch.compile` 开启；默认编译 V2 backbone
- 默认 loss：单个 BCE loss term

训练数据管道仍保留鲁棒性增强：

```python
PCVRFeatureMaskConfig(probability=0.03)
PCVRNonSequentialSparseDropoutConfig(probability=0.10)
PCVRDomainDropoutConfig(probability=0.02)
```

## 打包

```bash
uv run taac-package-train \
  --experiment experiments/symbiosis \
  --output-dir outputs/bundles/symbiosis_training
```

```bash
uv run taac-package-infer \
  --experiment experiments/symbiosis \
  --output-dir outputs/bundles/symbiosis_inference
```

Symbiosis 有自定义训练、预测和运行时 hooks；改动后建议同时验证训练和推理 bundle。

## 改动前先看

- 实验入口和额外 CLI 参数：`experiments/symbiosis/__init__.py`
- V2 参数：`experiments/symbiosis/config.py`
- 模型组装：`experiments/symbiosis/model.py`
- Tokenization：`experiments/symbiosis/tokenization.py`
- Attention mask：`experiments/symbiosis/attention.py`
- Backbone：`experiments/symbiosis/backbone.py`
- Readout：`experiments/symbiosis/pooling.py`
- 实验包契约测试：`tests/contract/experiments/test_packages.py`
- 运行时契约矩阵：`tests/contract/experiments/test_runtime_contract_matrix.py`

如果只是新增普通模型，不要从 Symbiosis 复制；它的 hook 和额外参数会让新实验复杂很多。

## 最小复核

```bash
uv run pytest tests/contract/experiments/test_packages.py -q
uv run pytest tests/contract/experiments/test_runtime_contract_matrix.py -q
uv run pytest tests/unit/application/training/test_cli.py -q
```
