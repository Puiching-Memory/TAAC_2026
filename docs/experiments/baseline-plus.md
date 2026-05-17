---
icon: lucide/gauge
---

# Baseline+

## 摘要

Baseline+ 是 Baseline 的工程增强版。它不试图提出新的推荐结构，而是把更接近实战训练的配置集中到一个实验包里：OPT cache、轻量数据增强、Muon dense optimizer，以及 RMSNorm accelerator backend。

因此它回答的问题不是“HyFormer 架构是否最优”，而是“同一类 HyFormer 路线在更强训练配置下能走到哪里”。做模型结构消融时，Baseline+ 可以作为性能上界参照；做论文式结构对比时，要先把 cache、增强、optimizer 和 backend 的影响拆开。

## 一、为什么需要它

干净 Baseline 的好处是变量少，但它也偏保守：没有随机序列视图、没有 domain dropout、没有 cache，也没有更激进的 dense optimizer。真实比赛迭代中，训练稳定性、吞吐、长序列裁剪和 dense 参数更新方式都会影响结果。

Baseline+ 把这些工程变量放到一个明确实验包中，避免把它们悄悄混进 Baseline。这样后续模型可以有两种公平比较：

- 对照 Baseline：比较纯结构收益。
- 对照 Baseline+：比较在更强训练 recipe 下的综合收益。

## 二、实验入口

入口位于 `experiments/baseline_plus/__init__.py`。

| 项目              | 默认值                                      |
| ----------------- | ------------------------------------------- |
| 实验名            | `pcvr_baseline_plus`                        |
| 模型类            | `PCVRBaselinePlus`                          |
| NS tokenizer      | `rankmixer`                                 |
| user / item token | `5 / 2`                                     |
| batch size        | `256`                                       |
| 序列上限          | `seq_a:256,seq_b:256,seq_c:512,seq_d:512`   |
| optimizer         | dense `muon`，sparse Adagrad                |
| loss              | BCE                                         |
| AMP / compile     | 关闭 / 关闭                                 |
| cache             | `opt`，`max_batches=512`                    |
| backend           | flash attention `torch`，RMSNorm `tilelang` |

默认数据增强包括：

```python
PCVRSequenceCropConfig(views_per_row=2, seq_window_mode="random_tail", seq_window_min_len=8)
PCVRFeatureMaskConfig(probability=0.03)
PCVRDomainDropoutConfig(probability=0.03)
```

这意味着 Baseline+ 的指标变化可能来自四类因素：数据视图、cache 命中、optimizer 更新、backend 数值行为。做消融时一次只关一个开关。

## 三、架构与 Baseline 的关系

`PCVRBaselinePlus` 保持 HyFormer 主线：

```text
NS tokens + dense tokens + sequence tokens
        -> HyFormer-style interaction blocks
        -> query/context embedding
        -> classifier
```

它和 Baseline 的核心差别不在输入语义，而在工程实现面：

- 使用共享 `RMSNorm` 封装，允许运行时切换 torch / TileLang / Triton。
- 使用 `scaled_dot_product_attention` 入口，便于 attention backend 对齐共享 accelerator 层。
- 保留 `configure_flash_attention_runtime()` 和 `configure_rms_norm_runtime()` 两个包内函数，让 runtime 在构造模型前注入 backend 配置。

这种设计使 Baseline+ 成为 accelerator 与训练 recipe 的试验场，而不是把 kernel 代码散落在实验包中。

## 四、训练行为的影响

**OPT cache。** `mode="opt"` 会减少重复数据转换开销，适合中长训练；短 smoke 中收益可能不稳定，不应拿极小步数直接判断。

**随机尾窗。** `views_per_row=2` 和 `random_tail` 会让同一条样本暴露不同历史片段，增加序列鲁棒性，但也会让指标不再只反映模型结构。

**Feature mask 与 domain dropout。** 这两个增强帮助模型减少对单个字段或单个序列域的依赖，适合应对线上缺失率和覆盖漂移。

**Muon。** Dense 矩阵参数走 Muon，bias/标量等非矩阵参数回退到 AdamW 风格更新；sparse embedding 仍由共享 Adagrad 处理。

## 五、适合观察什么

Baseline+ 适合观察：

- 数据增强是否改善线上泛化风险。
- Muon 是否让同类结构更快收敛。
- RMSNorm TileLang backend 是否影响吞吐或数值稳定性。
- 新模型在“强 recipe”下是否仍有增益。

它不适合充当最小复现样例。如果本地 GPU/TileLang 环境不完整，先用 Baseline 证明训练链路，再回到 Baseline+ 排查 backend。

## 六、修改建议

- 调增强概率或 cache：改 `TRAIN_DEFAULTS.data_pipeline`。
- 切 backend：改 `PCVRModelConfig.flash_attention_backend`、`rms_norm_backend` 和 `rms_norm_block_rows`。
- 改 kernel：进入 `src/taac2026/infrastructure/accelerators/`，不要在实验包复制算子。
- 比较 Baseline 与 Baseline+：记录数据 pipeline、optimizer、backend、seed 和 CUDA 环境。

## 七、运行与验收

训练：

```bash
bash run.sh train \
  --experiment experiments/baseline_plus \
  --run-dir outputs/baseline_plus_smoke
```

打包：

```bash
uv run taac-package-train --experiment experiments/baseline_plus --output-dir outputs/bundles/baseline_plus_training
uv run taac-package-infer --experiment experiments/baseline_plus --output-dir outputs/bundles/baseline_plus_inference
```

最小复核：

```bash
uv run pytest tests/contract/experiments/test_packages.py -q
uv run pytest tests/contract/experiments/test_runtime_contract_matrix.py -q
```

涉及 TileLang 或 accelerator 行为时，再补：

```bash
uv run pytest tests/unit/infrastructure/accelerators -q
```

数据增强背景见 [PCVR 数据管道](../guide/pcvr-data-pipeline.md)，优化器背景见 [优化器](../guide/optimizers.md)。
