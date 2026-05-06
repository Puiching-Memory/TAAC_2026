---
icon: lucide/zap
---

# Symbiosis

Symbiosis 是当前实验集中最复杂的 PCVR 模型包。它不是“最小可复制模板”，而是一个带额外模型开关、自定义 hooks 和消融空间的融合实验。

## 快速运行

```bash
bash run.sh train \
  --experiment experiments/symbiosis \
  --run-dir outputs/symbiosis_smoke
```

消融时可以关闭单个组件，例如：

```bash
bash run.sh train \
  --experiment experiments/symbiosis \
  --run-dir outputs/symbiosis_no_context_exchange \
  --no-symbiosis-use-context-exchange
```

Symbiosis 的参数名来自 `experiments/symbiosis/__init__.py` 中的 `SymbiosisModelDefaults`。布尔参数由 argparse 提供 `--foo` / `--no-foo` 两种形式。

## 实验入口

- 实验名：`pcvr_symbiosis`
- 模型类：`PCVRSymbiosis`
- NS tokenizer：`rankmixer`
- 查询数：`num_queries=1`
- 模型层数：`num_blocks=3`
- 默认运行时：AMP 开启，`torch.compile` 开启
- 优化器：`orthogonal_adamw` + cosine schedule
- 额外 loss：`pairwise_auc_weight=0.05`

模型侧默认打开的融合组件包括用户-物品图、Fourier 时间编码、上下文交换、多尺度、domain gate、候选解码器、action conditioning、压缩记忆、attention sink、lane mixing 和 semantic id。

Symbiosis 是当前少数覆盖默认 hooks 的实验包：

```python
train_arg_parser=parse_symbiosis_train_args
train_hook_overrides={"build_model": build_symbiosis_train_model}
prediction_hook_overrides={"build_model": build_symbiosis_prediction_model}
runtime_hook_overrides={"load_train_config": load_symbiosis_train_config}
```

这些 hook 的作用是把 `SymbiosisModelDefaults` 中的额外开关写进 CLI / train_config，并在评估推理时校验这些开关存在。

## 模型结构

模型实现是 `experiments/symbiosis/model.py` 的 `PCVRSymbiosis`。

前向的 `_embed()` 大致分为：

1. user / item tokenization，并可追加 semantic item token。
2. `UserItemGraphBlock` 更新 user / item token。
3. 每个序列域通过 `SequenceTokenizer` 编码，可叠加 Fourier time encoding。
4. action token 生成 action context。
5. candidate query 由 user、item、graph、action context 拼接投影得到。
6. candidate decoder 从序列中抽取候选相关上下文，可使用 compressed memory 和 attention sink。
7. prompt token + NS token 进入 unified blocks，和序列上下文交互。
8. context exchange blocks 让全局 context 再读序列。
9. multi-scale context 汇总 mean / recent / last 三种序列视角。
10. lane mixer / fusion gate 融合 unified、context、scale、graph、candidate 五条 lane。
11. action-conditioned classifier 输出 logits。

`predict()` 返回 `(logits, embeddings)`，embeddings 是融合后的候选表示。

## 消融参数

`SymbiosisModelDefaults` 定义的布尔开关会变成 CLI 参数：

| 开关 | 控制 |
| ---- | ---- |
| `symbiosis_use_user_item_graph` | user/item graph blocks |
| `symbiosis_use_fourier_time` | Fourier time encoding |
| `symbiosis_use_context_exchange` | context exchange blocks |
| `symbiosis_use_multi_scale` | mean / recent / last 多尺度汇总 |
| `symbiosis_use_domain_gate` | unified block 内的 domain gate |
| `symbiosis_use_candidate_decoder` | candidate-conditioned sequence decoder |
| `symbiosis_use_action_conditioning` | action context 和 prompt conditioning |
| `symbiosis_use_compressed_memory` | decoder compressed memory |
| `symbiosis_use_attention_sink` | decoder attention sink |
| `symbiosis_use_lane_mixing` | five-lane mixer |
| `symbiosis_use_semantic_id` | semantic item token |

布尔参数支持 `--symbiosis-use-...` 和 `--no-symbiosis-use-...`。

压缩记忆相关整数参数：

- `symbiosis_memory_block_size`
- `symbiosis_memory_top_k`
- `symbiosis_recent_tokens`

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
- 实验包契约测试：`tests/unit/experiments/test_packages.py`
- 运行时契约矩阵：`tests/unit/experiments/test_runtime_contract_matrix.py`

如果只是新增普通模型，不要从 Symbiosis 复制；它的 hook 和额外参数会让新实验复杂很多。

## 最小复核

```bash
uv run pytest tests/unit/experiments/test_packages.py -q
uv run pytest tests/unit/experiments/test_runtime_contract_matrix.py -q
uv run pytest tests/unit/application/experiments/test_pcvr_runtime.py -q
```
