---
icon: lucide/sparkles
---

# UniTok

UniTok 是基于“理解生成统一”启发新增的统一 token-stream PCVR 实验包。它不新增训练运行时，而是在现有 PCVR 契约下，把用户字段、物品字段、dense packet、跨域行为事件和候选汇聚 token 尽早放进同一个 Transformer backbone。

## 快速运行

```bash
bash run.sh train \
  --experiment experiments/unitok \
  --run-dir outputs/unitok_smoke
```

评估：

```bash
bash run.sh val \
  --experiment experiments/unitok \
  --run-dir outputs/unitok_smoke
```

## 实验入口

- 实验名：`pcvr_unitok`
- 模型类：`PCVRUniTok`
- NS tokenizer 配置：保留默认 RankMixer 字段配置写入当前 sidecar，但模型内部使用 field-level sparse tokens
- 默认序列窗口：每个 domain 保留最近 `seq_top_k=64` 个真实事件
- 默认 dropout：`0.02`
- 数据管道：tail crop、轻量 feature mask、轻量 domain dropout

默认配置位于 `experiments/unitok/__init__.py`。

## 模型结构

模型实现是 `experiments/unitok/model.py` 的 `PCVRUniTok`。

前向结构：

1. 用户稀疏字段通过 `FeatureEmbeddingBank` 变成 field-level token，不先按 NS group 做均值摘要。
2. 用户 dense 特征被切成 1-2 个 dense packet token，保留来源 type embedding。
3. 每个序列域通过 `SequenceTokenizer` 编码，并按每行真实长度 gather 最近事件窗口。
4. 物品稀疏字段保留 field-level token，物品 dense 特征保留 packet token。
5. 一个候选汇聚 token 放在序列开头，注入 item summary，作为最终候选上下文读出位。
6. 所有 token 加 type embedding 后进入同一组 `UniTokBlock`。
7. 最终拼接 candidate summary、context summary 和 item summary，送入 classifier。

UniTok 的关键差异是减少早期压缩：它把“序列语言”和“特征交互语言”更早混到同一个 token 空间，而不是先分别编码再晚期融合。

## 修改点

- 改 field-level 稀疏 token：看 `FieldTokenProjector`。
- 改 dense packet 数量：看 `DensePacketTokenizer`。
- 改最近事件窗口：看 `seq_top_k` 和 `_encode_sequence_events()`。
- 改统一 backbone：看 `UniTokBlock` 和 `UnifiedSelfAttention`。
- 改默认增强或模型宽度：看 `experiments/unitok/__init__.py`。

## 打包

```bash
uv run taac-package-train \
  --experiment experiments/unitok \
  --output-dir outputs/bundles/unitok_training
```

```bash
uv run taac-package-infer \
  --experiment experiments/unitok \
  --output-dir outputs/bundles/unitok_inference
```

## 最小复核

```bash
uv run pytest tests/contract/experiments/test_packages.py -q
uv run pytest tests/contract/experiments/test_runtime_contract_matrix.py -q
```
