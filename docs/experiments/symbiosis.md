---
icon: lucide/zap
---

# Symbiosis

## 摘要

Symbiosis 是仓库里最完整的“分布感知统一 token-stream”实验包。它把非序列字段、候选物品、行为序列、缺失状态、时间信息、序列统计和风险先验统一翻译成 token，再交给同构 backbone。

当前实现以 Symbiosis V2 为主体，并在新训练默认启用 V3 的确定性 memory event selector。V3 不引入可学习路由器，也不增加 loss；它只改变长序列进入 unified token stream 前的预算分配和 memory 事件选择规则。旧 checkpoint sidecar 如果没有 V3 字段，会按 V2 路径回退，避免老权重评估/推理时行为漂移。

## 一、为什么需要 Symbiosis

线上 EDA 暴露了一个核心问题：schema 一致不代表分布一致。训练集和推理集可能在 item 基数、缺失率、dense 尺度、序列长度和 token overlap 上同时漂移。普通随机 valid 很容易高估依赖 item memorization 或 dense shortcut 的模型。

Symbiosis 的假设是：统一架构不能只统一数值表示，还必须统一表达 token 的来源、可靠性和风险。也就是说，模型需要知道一个 token 是用户字段、候选物品、缺失模式、序列事件还是统计摘要，并在 attention 中使用这些 metadata。

## 二、实验入口

入口位于 `experiments/symbiosis/__init__.py`。

| 项目                | 默认值                                                  |
| ------------------- | ------------------------------------------------------- |
| 实验名              | `pcvr_symbiosis`                                        |
| 模型类              | `PCVRSymbiosis`                                         |
| 自定义 hook         | train/prediction build model，runtime load train config |
| batch size          | `128`                                                   |
| split / sampling    | `timestamp_auto` / `row_group_sweep`                    |
| 序列上限            | `seq_a:256,seq_b:256,seq_c:1024,seq_d:2048`             |
| `d_model / emb_dim` | `256 / 256`                                             |
| block / head        | `4 / 8`                                                 |
| dense optimizer     | `muon`                                                  |
| sparse lr           | `0.01`                                                  |
| AMP / compile       | BF16 AMP 开启，compile 开启                             |
| loss                | BCE                                                     |
| validation          | `probe_mode="none"`，early stopping 监控 AUC            |

默认数据增强：

```python
PCVRFeatureMaskConfig(probability=0.03)
PCVRNonSequentialSparseDropoutConfig(probability=0.10)
PCVRDomainDropoutConfig(probability=0.02)
```

## 三、V1 到 V2 的变化

V1 是早期快速融合模型：user/item 稀疏 group、dense packet、missing token、sequence memory、candidate token、cross token 和 global token 最终进入一个复杂 readout。它验证了很多组件，但本质仍是“多分支摘要后融合”。

V2 做了更彻底的改变：

```text
all inputs -> UnifiedTokenBatch
UnifiedTokenBatch -> metadata attention mask
tokens -> homogeneous backbone
CLS/candidate/global readout -> classifier
```

旧的 `sequence_memory` 公开属性、`_sparse_tokens` 调试接口、六段 V1 summary readout，以及 `symbiosis_use_*` 旧开关不再作为当前契约维护。

## 四、统一 tokenization

V2 tokenizer 输出 `UnifiedTokenBatch`：

| 字段                            | 形状 / 含义                                                         |
| ------------------------------- | ------------------------------------------------------------------- |
| `tokens`                        | `(B, T, d_model)` 统一 token stream                                 |
| `padding_mask`                  | `(B, T)` padding 与训练期 dropout mask                              |
| `role_ids`                      | `(T,)`，CLS、candidate、user、item、dense、missing、sequence、stats |
| `domain_ids`                    | `(T,)`，序列域来源                                                  |
| `risk_ids`                      | `(T,)`，高漂移或 shortcut 风险标记                                  |
| `cls_index` / `candidate_index` | readout token 位置                                                  |

Token 表示由四类信息相加：

```text
token = value embedding/projection
      + role embedding
      + domain embedding
      + risk embedding
      + time/position embedding for sequence events
```

Sparse 字段使用字段级 missing embedding。`user_int_missing_mask` 与 `item_int_missing_mask` 会直接改变 sparse token，而不是只在旁边放一个缺失标志。Dense 字段先做 robust log transform，再拼入 missing indicator；missing pattern 也会作为独立 token 进入统一流。

## 五、V3 memory event selector

长序列不会全量进入 backbone。每个序列域被压缩成固定预算：

```text
seq domain = memory event tokens + recent event tokens
```

V2 fallback 每域使用 `8` 个 memory event 和 `16` 个 recent event。V3 新训练默认采用来源感知预算：

| 来源    | recent | memory | 直觉                                      |
| ------- | ------ | ------ | ----------------------------------------- |
| `seq_a` | 8      | 4      | 短序列域，少量历史覆盖即可。              |
| `seq_b` | 8      | 4      | 与 `seq_a` 对齐。                         |
| `seq_c` | 20     | 10     | infer 侧更长，保留更多 recent 与 memory。 |
| `seq_d` | 24     | 12     | raw 上限最高，预算最大。                  |

默认 `quality_stratified` 先把 recent 之前的 prefix 历史切成固定数量 bucket，再按确定性质量分数选事件：

```text
score = density_weight * nonzero_feature_ratio
      + time_weight * has_valid_time_bucket
      + recency_weight * bucket_internal_recency
      - duplicate_penalty * consecutive_duplicate
```

这样既保留时间覆盖，又避免 memory token 被全零、弱信息或重复事件占掉。

## 六、Metadata Attention

Symbiosis 不把所有 token 都默认互相可见，而是用 role/domain/risk 生成结构先验：

- CLS 与 candidate readout 可以读取全局 token。
- field token 可以读取同类字段、candidate、missing 和 stats。
- sequence token 可以读取 candidate、item、sequence、missing 和 stats。
- stats token 可以读取全局。
- high-risk token 仍参与交互，但受 risk embedding 和训练期 dropout 约束。

这让 backbone 仍保持同构，差异来自 mask，而不是写多个独立专家模块。

## 七、Readout 与诊断

V2 不再拼接 V1 的 candidate、cross、global、context、item、sequence 六段 summary。当前 readout 由 CLS token、candidate token 和全局 masked mean 组成，再进入 classifier。

训练诊断使用统一 token-stream 口径：

```text
SymbiosisV2/tokens/active_ratio/<phase>
SymbiosisV2/tokens/count/<phase>
SymbiosisV2/tokens/high_risk_ratio/<phase>
SymbiosisV2/embedding/norm_mean/<phase>
```

这些指标用于判断 token 是否被过度 dropout、风险 token 比例是否异常、embedding norm 是否漂移。

## 八、参数面

主要 V2/V3 参数如下：

| 参数                                         | 控制                                              |
| -------------------------------------------- | ------------------------------------------------- |
| `symbiosis_v2_use_dense_tokens`              | 是否加入 dense token                              |
| `symbiosis_v2_use_missing_tokens`            | 是否加入 missing pattern token                    |
| `symbiosis_v2_use_sequence_stats_tokens`     | 是否加入 sequence stats token                     |
| `symbiosis_v2_use_metadata_attention_bias`   | 是否使用 metadata-driven attention mask           |
| `symbiosis_v2_use_candidate_readout`         | readout 是否使用 candidate token                  |
| `symbiosis_v2_tokenization_mode`             | `group` / `group_compressed` / `random_chunk`     |
| `symbiosis_v2_high_risk_token_dropout_rate`  | 训练期 high-risk token dropout                    |
| `symbiosis_v2_compress_large_ids`            | 高基数 sparse / sequence id 是否 hash compression |
| `symbiosis_v2_compile_backbone`              | 开启 `--compile` 时是否只编译 V2 backbone         |
| `symbiosis_v3_enabled`                       | 是否启用 V3 selector                              |
| `symbiosis_v3_memory_selection_mode`         | `uniform` / `stratified` / `quality_stratified`   |
| `symbiosis_v3_recent_event_tokens_by_domain` | 各 domain recent 预算                             |
| `symbiosis_v3_memory_event_tokens_by_domain` | 各 domain memory 预算                             |
| `symbiosis_v3_memory_*_weight`               | memory 事件质量打分权重                           |

布尔参数由 argparse 提供 `--foo` 和 `--no-foo` 两种形式。

## 九、消融建议

推荐顺序：

1. `--no-symbiosis-v3-enabled`：回到 V2 均匀 memory，检查 V3 selector 总贡献。
2. `--symbiosis-v3-memory-selection-mode stratified`：只保留分桶覆盖，去掉质量打分。
3. `--symbiosis-v3-memory-selection-mode uniform`：检查纯均匀 memory 行为。
4. `--symbiosis-v3-memory-event-tokens-by-domain seq_a:0,seq_b:0,seq_c:0,seq_d:0`：只保留 recent events。
5. `--no-symbiosis-v2-use-metadata-attention-bias`：检查 metadata mask 贡献。
6. `--no-symbiosis-v2-use-candidate-readout`：检查 candidate-centered readout。
7. `--no-symbiosis-v2-use-missing-tokens`：检查 missing pattern token。
8. `--no-symbiosis-v2-use-sequence-stats-tokens`：检查 sequence stats token。
9. `--symbiosis-v2-tokenization-mode group_compressed`：检查压缩稀疏 tokenization。
10. 调整 `--symbiosis-v2-high-risk-token-dropout-rate`：检查 shortcut 约束强度。

## 十、运行与验收

训练：

```bash
bash run.sh train \
  --experiment experiments/symbiosis \
  --run-dir outputs/symbiosis_smoke
```

关闭 metadata attention 的示例：

```bash
bash run.sh train \
  --experiment experiments/symbiosis \
  --run-dir outputs/symbiosis_no_bias \
  --no-symbiosis-v2-use-metadata-attention-bias
```

打包：

```bash
uv run taac-package-train --experiment experiments/symbiosis --output-dir outputs/bundles/symbiosis_training
uv run taac-package-infer --experiment experiments/symbiosis --output-dir outputs/bundles/symbiosis_inference
```

最小复核：

```bash
uv run pytest tests/contract/experiments/test_packages.py -q
uv run pytest tests/contract/experiments/test_runtime_contract_matrix.py -q
uv run pytest tests/unit/application/training/test_cli.py -q
```

源码入口：

- `experiments/symbiosis/config.py`
- `experiments/symbiosis/tokenization.py`
- `experiments/symbiosis/attention.py`
- `experiments/symbiosis/backbone.py`
- `experiments/symbiosis/pooling.py`
- `experiments/symbiosis/model.py`
