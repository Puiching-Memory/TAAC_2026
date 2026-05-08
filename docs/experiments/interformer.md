---
icon: lucide/arrows-up-from-line
---

# InterFormer

InterFormer 是用户-物品交互结构的实验包。它保留共享 PCVR 运行时，把主要差异放在模型类 `PCVRInterFormer` 和 Group NS tokenizer 上。

## 快速运行

```bash
bash run.sh train \
  --experiment experiments/interformer \
  --run-dir outputs/interformer_smoke
```

评估：

```bash
bash run.sh val \
  --experiment experiments/interformer \
  --run-dir outputs/interformer_smoke
```

## 实验入口

- 实验名：`pcvr_interformer`
- 模型类：`PCVRInterFormer`
- NS tokenizer：`group`
- 查询数：`num_queries=1`
- 默认 dropout：`0.02`
- 数据管道：不启用增强，cache 关闭

更细的模型宽度、优化器和序列长度默认值见 `experiments/interformer/__init__.py`。

## 模型结构

模型实现是 `experiments/interformer/model.py` 的 `PCVRInterFormer`。

前向结构：

1. user / item 非序列特征分别 tokenization，dense 特征各投影成 token。
2. 每个序列域通过 `SequenceTokenizer` 编码，并加 sinusoidal position。
3. 多层 `InterFormerBlock` 同时更新 NS token 和序列 token。
4. `CrossSummary` 汇总序列侧上下文。
5. `final_gate` 在 NS summary 和 sequence summary 之间做门控融合。
6. classifier 输出 logits。

InterFormer 的重点是让非序列上下文和序列上下文在 block 内交互，而不是像统一 Transformer 那样把所有 token 直接拼成一条长流。

## 修改点

- 改交互结构：看 `InterFormerBlock`。
- 改最终融合：看 `CrossSummary` 和 `final_gate`。
- 改 tokenizer 或默认超参：看 `experiments/interformer/__init__.py`。
- 默认不启用数据增强，适合做结构对照。

## 打包

```bash
uv run taac-package-train \
  --experiment experiments/interformer \
  --output-dir outputs/bundles/interformer_training
```

```bash
uv run taac-package-infer \
  --experiment experiments/interformer \
  --output-dir outputs/bundles/interformer_inference
```

## 源码入口

- 实验默认配置：`experiments/interformer/__init__.py`
- 模型实现：`experiments/interformer/model.py`
- 论文背景：[InterFormer](../papers/interformer.md)

## 最小复核

```bash
uv run pytest tests/contract/experiments/test_packages.py -q
uv run pytest tests/contract/experiments/test_runtime_contract_matrix.py -q
```
