---
icon: lucide/zap
---

# Symbiosis

Symbiosis 是基于线上 EDA 结果重做的统一 PCVR 消融实验。它不再把用户-物品图、上下文交换、多尺度汇总、attention sink、lane mixing、伪 semantic id 等旧模块并联在一起，而是围绕 UniTok 思想做统一 token-stream 建模：字段 token、dense packet、候选 token、长序列 memory token 和 item 侧强信号在同一个 backbone 里交互。当前序列 memory 吸收了 DeepSeek-V4 CSA/HCA 的轻量化思路：保留最近窗口，用可学习 block 权重做压缩，并用低秩多头 indexer 选择候选相关 block。

## 快速运行

```bash
bash run.sh train \
  --experiment experiments/symbiosis \
  --run-dir outputs/symbiosis_smoke
```

消融时关闭单个结构，例如：

```bash
bash run.sh train \
  --experiment experiments/symbiosis \
  --run-dir outputs/symbiosis_no_sequence_memory \
  --no-symbiosis-use-sequence-memory
```

Symbiosis 的额外参数来自 `experiments/symbiosis/__init__.py` 中的 `SymbiosisModelDefaults`。布尔参数由 argparse 提供 `--foo` / `--no-foo` 两种形式。

## 设计依据

线上 EDA 显示：

- item 侧离散特征信号明显强于 user 侧，模型容量应优先保护 item 表示。
- 四个序列域都很长，且 `seq_b` / `seq_c` / `seq_d` 重复率很高，直接全长 attention 不现实。
- 高基数序列 side-info 不适合全量 vocabulary 化，需要截断、压缩或聚合。
- dense 特征存在大尺度列，进入模型前应做稳健变换。

因此当前 Symbiosis 删除了旧版中不利于比赛的设计：

- 删除伪 `semantic_id`：它只是 item token mean 后再投影，不是外部 semantic id。
- 删除 `attention_sink`：当前任务不是自回归长上下文生成，sink token 只会引入额外噪声。
- 删除单动作 `action_conditioning`：`action_num=1` 时它退化为可学习常量。
- 删除 `lane_mixing` / `multi_scale` / `context_exchange`：这些路径和序列 reader 重叠，增加延迟和归因成本。
- 默认关闭 RoPE、pairwise AUC、scheduler 和训练 batch cache；默认启用数据增强、Muon、AMP bf16 与 `torch.compile`。线上训练中 `torch.compile` 的 Inductor 首次编译会占用额外主机内存，Symbiosis 默认不再叠加 OPT/shared cache。

## 实验入口

- 实验名：`pcvr_symbiosis`
- 模型类：`PCVRSymbiosis`
- 默认 backbone：统一 token-stream self-attention
- 默认层数：`num_blocks=2`
- 默认序列 memory：recent window + learned compressed block top-k + global token
- 默认 optimizer：`muon`
- 默认 scheduler：关闭
- 默认数据增强：`sequence_crop` + `feature_mask` + `domain_dropout`，训练 batch cache 默认关闭
- 默认运行时：AMP bf16 开启，`torch.compile` 开启
- 默认 loss：单个 BCE loss term，`loss_terms=[{"name": "bce", "kind": "bce", "weight": 1.0}]`

Symbiosis 仍覆盖默认 hooks：

```python
train_arg_parser=parse_symbiosis_train_args
train_hook_overrides={"build_model": build_symbiosis_train_model}
prediction_hook_overrides={"build_model": build_symbiosis_prediction_model}
runtime_hook_overrides={"load_train_config": load_symbiosis_train_config}
```

这些 hook 的作用是把 `SymbiosisModelDefaults` 中的额外消融参数写进 CLI / train_config，并在评估推理时校验 checkpoint sidecar 里存在这些参数。

## 模型结构

模型实现是 `experiments/symbiosis/model.py` 的 `PCVRSymbiosis`。

前向大致分为：

1. user / item 稀疏特征默认按字段生成 token；关闭后退回 RankMixer NS grouped token。
2. dense 特征默认做 `log1p(abs(x)) * sign(x)` 后拆成少量 packet token；关闭后退回单 dense token projector。
3. user 与 item summary 生成 sequence query。
4. 每个序列域生成 bounded memory：recent tokens、可选 learned compressed block top-k、global mean token。
5. 可选 candidate token 以 item summary 初始化，保护 item 侧强信号。
6. candidate、user、dense、sequence memory、item token 进入同一个 self-attention backbone。
7. 输出拼接 candidate summary、context summary 和 item summary 后分类。

`predict()` 返回 `(logits, embeddings)`，其中 embeddings 是三段 summary 的拼接向量，形状为 `(B, d_model * 3)`。

## 消融参数

| 开关                              | 控制                                                                    |
| --------------------------------- | ----------------------------------------------------------------------- |
| `symbiosis_use_field_tokens`      | 稀疏特征是否按字段 token 化；关闭后使用 NS grouped token                |
| `symbiosis_use_dense_packets`     | dense 特征是否拆成 packet token 并做 log 缩放；关闭后使用单 dense token |
| `symbiosis_use_sequence_memory`   | 是否加入序列 memory token                                               |
| `symbiosis_use_compressed_memory` | sequence memory 中是否加入 learned compressed block top-k               |
| `symbiosis_use_candidate_token`   | 是否加入候选 token                                                      |
| `symbiosis_use_item_prior`        | 候选 token 和最终 summary 是否保留 item prior                           |
| `symbiosis_use_domain_type`       | 是否给 user/item/dense/sequence memory 加 type embedding                |

memory 参数只在 `symbiosis_use_sequence_memory=True` 时有意义：

- `symbiosis_memory_block_size`：压缩 block 大小，默认 `32`。
- `symbiosis_memory_top_k`：每个序列域保留的 query-relevant block 数，默认 `8`。
- `symbiosis_recent_tokens`：每个序列域保留的最近 token 数，默认 `32`。

推荐消融顺序：

1. `--no-compile` / `--no-amp`：需要快速调试或定位编译问题时先关闭工程优化。
2. `--no-symbiosis-use-sequence-memory`：确认序列 memory 是否贡献排序信号。
3. `--no-symbiosis-use-compressed-memory`：只保留 recent + global，检查压缩块是否值得推理成本。
4. `--no-symbiosis-use-field-tokens`：对比字段级 token 与 NS grouped token。
5. `--no-symbiosis-use-dense-packets`：检查 dense packet 和 log 缩放是否稳定。
6. `--no-symbiosis-use-item-prior`：验证 item 侧强信号是否需要显式保护。

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
- 模型实现：`experiments/symbiosis/model.py`
- 实验包契约测试：`tests/contract/experiments/test_packages.py`
- 运行时契约矩阵：`tests/contract/experiments/test_runtime_contract_matrix.py`

如果只是新增普通模型，不要从 Symbiosis 复制；它的 hook 和额外参数会让新实验复杂很多。

## 最小复核

```bash
uv run pytest tests/contract/experiments/test_packages.py -q
uv run pytest tests/contract/experiments/test_runtime_contract_matrix.py -q
uv run pytest tests/unit/application/experiments/test_pcvr_runtime.py -q
```
