---
icon: lucide/arrow-right-left
---

# OneTrans

OneTrans 是统一 Transformer 路线的实验包。它使用 `PCVROneTrans`，把用户、物品和序列特征放进更统一的编码路径，同时继续使用 RankMixer NS tokenizer。

## 快速运行

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

## 实验入口

- 实验名：`pcvr_onetrans`
- 模型类：`PCVROneTrans`
- NS tokenizer：`rankmixer`
- 非序列 token 配置：`user_tokens=5`，`item_tokens=2`
- 查询数：`num_queries=1`
- 默认 dropout：`0.02`
- 数据管道：不启用增强，cache 关闭

默认配置位于 `experiments/onetrans/__init__.py`。

## 模型结构

模型实现是 `experiments/onetrans/model.py` 的 `PCVROneTrans`。

前向结构：

1. 各序列域通过 `SequenceTokenizer` 编码。
2. 多个序列域被拼成一个 sequence stream，域之间插入可学习 separator token。
3. user / item 非序列特征和 dense token 拼成 NS token。
4. sequence stream 与 NS token 拼接后进入 `OneTransBlock`。
5. 每层后通过 `_pyramid_keep_count()` 逐步裁掉旧的序列 token，使序列 token 数向 `num_ns` 收缩。
6. 最终分别汇总 sequence 和 NS 部分，再拼接进入 classifier。

OneTrans 的关键差异是“统一 token 流 + pyramid token 保留”，而不是用户/物品双分支交互。

## 修改点

- 改统一编码 block：看 `OneTransBlock`。
- 改序列裁剪节奏：看 `_pyramid_keep_count()`。
- 改域间 separator：看 `separator_tokens` 和 `_encode_sequence_stream()`。
- 改默认 tokenizer 或超参：看 `experiments/onetrans/__init__.py`。

## 打包

```bash
uv run taac-package-train \
  --experiment experiments/onetrans \
  --output-dir outputs/bundles/onetrans_training
```

```bash
uv run taac-package-infer \
  --experiment experiments/onetrans \
  --output-dir outputs/bundles/onetrans_inference
```

## 源码入口

- 实验默认配置：`experiments/onetrans/__init__.py`
- 模型实现：`experiments/onetrans/model.py`
- 论文背景：[OneTrans](../papers/onetrans.md)

## 最小复核

```bash
uv run pytest tests/unit/experiments/test_packages.py -q
uv run pytest tests/unit/experiments/test_runtime_contract_matrix.py -q
```
