---
icon: lucide/spline
---

# UniRec

UniRec 是一个统一推荐建模实验包，用来承接外部 UniRec 思路中更适合本仓库 runtime 的部分：field-level token、dense packet、跨域序列 token、MoT 分支摘要、target-aware interest、Hybrid SiLU attention 和 Block Attention Residuals。

实现没有复制外部仓库的训练、数据读取、checkpoint 或推理入口；这些能力继续由 `src/taac2026` 的共享 PCVR runtime 提供。

## 快速运行

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

## 实验入口

- 实验名：`pcvr_unirec`
- 模型类：`PCVRUniRec`
- NS tokenizer 配置：sidecar 中保留 RankMixer 分组配置，模型内部使用 field-level sparse token
- 默认序列窗口：每个 domain 保留最近 `seq_top_k=64` 个事件
- 默认优化器：dense 参数使用 Muon，sparse embedding 使用共享 sparse optimizer
- 默认 loss：BCE 加轻量 pairwise AUC regularization
- 默认 runtime：BF16 AMP 开启，`torch.compile` 默认关闭

默认配置位于 `experiments/unirec/__init__.py`。

## 模型结构

模型实现是 `experiments/unirec/model.py` 的 `PCVRUniRec`。

前向结构：

1. 用户和物品稀疏字段通过 `FeatureEmbeddingBank` 转成 field-level token。
2. 用户和物品 dense 特征切成 dense packet token。
3. 非序列 token 先经过 feature cross layer 做显式字段交互。
4. 每个序列域通过 `SequenceTokenizer` 编码，并保留最近事件窗口。
5. MoT 分支分别汇总各序列域，再用门控融合成 `mot` token。
6. target-aware interest 用 item summary 查询所有序列 token，生成 `interest` token。
7. target token 由 user summary、item summary 和逐元素交互生成。
8. `[feature | sequence | mot | interest | target]` 进入 Hybrid SiLU attention backbone。
9. 每层后用 Block Attention Residuals 引入跨层 block summary。
10. 最后拼接 target summary、interest summary 和 user/item fallback summary 进入 classifier。

## 修改点

- 改 field-level 稀疏 token：看 `FieldTokenProjector`。
- 改 dense packet 数量：看 `DensePacketTokenizer`。
- 改 MoT 分支：看 `MixtureOfTransducers`。
- 改 target-aware interest：看 `TargetAwareInterest`。
- 改 Hybrid attention mask：看 `_build_attention_mask()`。
- 改默认增强或模型宽度：看 `experiments/unirec/__init__.py`。

## 打包

```bash
uv run taac-package-train \
  --experiment experiments/unirec \
  --output-dir outputs/bundles/unirec_training
```

```bash
uv run taac-package-infer \
  --experiment experiments/unirec \
  --output-dir outputs/bundles/unirec_inference
```

## 最小复核

```bash
uv run pytest tests/contract/experiments/test_packages.py -q
uv run pytest tests/contract/experiments/test_runtime_contract_matrix.py -q
```