---
icon: lucide/zap
---

# Symbiosis

最先进的实验包，集成了多项架构创新和训练优化。

## 包结构

```
experiments/symbiosis/
├── __init__.py   # 包含显式 NS 分组配置
└── model.py
```

## 模型思路

Symbiosis 在 HyFormer 基础上引入共生学习机制，通过 11 个独立特性开关控制不同组件的启用。核心创新：

- **Context Exchange** -- 用户和物品上下文之间的信息交换（消融实验中影响最大的组件）
- **Fourier Time** -- 傅里叶时间编码，捕获时间周期性模式
- **Multi-Scale** -- 多尺度注意力，同时关注局部和全局模式
- **Domain Gate** -- 域门控，动态加权不同序列域的贡献
- **Compressed Memory** -- 压缩记忆机制（block_size=16, top_k=8, recent_tokens=64）
- **RoPE** -- 旋转位置编码（rope_base=1,000,000）

## 训练

| 参数                   | 值                                         |
| ---------------------- | ------------------------------------------ |
| 模型类                 | `PCVRSymbiosis`                            |
| NS Tokenizer           | `rankmixer` (user_tokens=5, item_tokens=2) |
| `num_blocks`           | 3                                          |
| `num_heads`            | 4                                          |
| `dropout_rate`         | 0.02                                       |
| `dense_optimizer_type` | `orthogonal_adamw`                         |
| `amp`                  | True, `amp_dtype=bfloat16`                 |
| `compile`              | True                                       |
| `batch_size`           | 128                                        |
| `pairwise_auc_weight`  | 0.05                                       |

```bash
uv run taac-train \
  --experiment experiments/symbiosis
```

## 消融开关

11 个特性开关（`PCVRSymbiosisConfig`）：

| 开关                      | 说明            |
| ------------------------- | --------------- |
| `use_user_item_graph`     | 用户-物品交互图 |
| `use_fourier_time`        | 傅里叶时间编码  |
| `use_context_exchange`    | 上下文交换机制  |
| `use_multi_scale`         | 多尺度注意力    |
| `use_domain_gate`         | 域门控          |
| `use_candidate_decoder`   | 候选解码器      |
| `use_action_conditioning` | 动作条件化      |
| `use_compressed_memory`   | 压缩记忆        |
| `use_attention_sink`      | 注意力 Sink     |
| `use_lane_mixing`         | 车道混合        |
| `use_semantic_id`         | 语义 ID         |

消融结果（`outputs/ablations/symbiosis_smoke/`，1000 样本 Bootstrap，seed=42）：

| 配置                | AUC 变化                  |
| ------------------- | ------------------------- |
| base                | 基准                      |
| no_context_exchange | **-0.005791**（最大下降） |
| no_fourier_time     | 较小下降                  |
| no_multi_scale      | 较小下降                  |
| no_user_item_graph  | 较小下降                  |

## 评估输出

Checkpoint 保存在 `outputs/pcvr_symbiosis-<slug>/`，格式与 Baseline 一致。

评估命令：

```bash
uv run taac-evaluate single \
  --experiment experiments/symbiosis
```

## 线上打包

```bash
# 训练 Bundle
uv run taac-package-train --experiment experiments/symbiosis --output-dir outputs/bundle

# 推理 Bundle
uv run taac-package-infer --experiment experiments/symbiosis --output-dir outputs/bundle
```
