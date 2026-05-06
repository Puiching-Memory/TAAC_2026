---
icon: lucide/gauge
---

# Baseline+

Baseline+ 是在 Baseline 上加了“更接近实战训练”的默认组合：OPT cache、轻量数据增强，以及 TileLang attention / RMSNorm backend。

它适合拿来做性能和鲁棒性方向的起点，但不适合当最小参照。做消融时，先明确你要比较的是模型结构、数据管道，还是 backend。

## 快速运行

```bash
bash run.sh train \
  --experiment experiments/baseline_plus \
  --run-dir outputs/baseline_plus_smoke
```

如果机器没有合适的 GPU / TileLang 环境，先用 Baseline 做链路验证，再回到 Baseline+ 排查 backend。

## 与 Baseline 的差异

相对 [Baseline](baseline.md)，Baseline+ 保持同类模型接口，但默认打开这些差异：

- 实验名：`pcvr_baseline_plus`
- 模型类：`PCVRBaselinePlus`
- 数据 cache：`mode="opt"`，`max_batches=512`
- 数据增强：序列随机尾窗、特征 mask、domain dropout
- 加速 backend：`flash_attention_backend="tilelang"`，`rms_norm_backend="tilelang"`
- RMSNorm TileLang 行块：`rms_norm_block_rows=8`

这些默认值都在 `experiments/baseline_plus/__init__.py`。

## 加速 backend 契约

模型实现是 `experiments/baseline_plus/model.py` 的 `PCVRBaselinePlus`。它从 `taac2026.api` 使用：

- `RMSNorm`
- `configure_rms_norm_runtime`
- `scaled_dot_product_attention`

包内还有两个运行时配置函数：

```python
configure_flash_attention_runtime(flash_attention_backend="torch" | "tilelang")
configure_rms_norm_runtime(rms_norm_backend="torch" | "tilelang", rms_norm_block_rows=...)
```

默认配置把 attention 和 RMSNorm 都切到 TileLang：

```python
PCVRModelConfig(
    flash_attention_backend="tilelang",
    rms_norm_backend="tilelang",
    rms_norm_block_rows=8,
)
```

如果本地没有 GPU 或 TileLang 运行环境，相关 benchmark / 单测可能会报告 unsupported。这个状态说明 backend 不可用，不等价于模型包发现失败。

## 数据管道契约

Baseline+ 默认使用：

```python
PCVRDataPipelineConfig(
    cache=PCVRDataCacheConfig(mode="opt", max_batches=512),
    transforms=(
        PCVRSequenceCropConfig(
            views_per_row=2,
            seq_window_mode="random_tail",
            seq_window_min_len=8,
        ),
        PCVRFeatureMaskConfig(probability=0.03),
        PCVRDomainDropoutConfig(probability=0.03),
    ),
    seed=42,
    strict_time_filter=True,
)
```

这意味着 Baseline+ 的吞吐和指标变化可能来自三类因素：模型实现、数据增强/cache、TileLang backend。做消融时应一次只关掉一个维度。

## 模型结构

`PCVRBaselinePlus` 基本沿用 HyFormer 路线：

1. 非序列特征生成 user / item NS token。
2. dense 特征投影为 dense token。
3. 序列域独立 embedding，并带 time bucket。
4. query generator 和多层 HyFormer block 更新 token。
5. 输出 query context 进入 classifier。

相对 Baseline，主要工程差异是算子 surface 切换到共享 accelerator 层，并用 `RMSNorm` 替代部分 LayerNorm 风格模块。

## 修改建议

- 只想调增强概率：改 `experiments/baseline_plus/__init__.py` 的 `data_pipeline`。
- 只想切回 torch backend：改 `PCVRModelConfig.flash_attention_backend` 和 `rms_norm_backend`。
- 要改 TileLang kernel：改 `src/taac2026/infrastructure/accelerators/`，不要在实验包里直接写 kernel。
- 要比较 Baseline vs Baseline+：同时记录数据管道配置和 backend 配置。

## 打包

```bash
uv run taac-package-train \
  --experiment experiments/baseline_plus \
  --output-dir outputs/bundles/baseline_plus_training
```

```bash
uv run taac-package-infer \
  --experiment experiments/baseline_plus \
  --output-dir outputs/bundles/baseline_plus_inference
```

线上 bundle 会打进当前实验包和 `src/taac2026`，不会把其他实验一起带上。

## 常见误判

- Baseline+ 的指标变化不一定来自模型本身，也可能来自数据增强或 backend。
- TileLang 相关问题先看 accelerator 单测和本地 GPU 环境，不要直接归因到实验包发现机制。
- 如果只是要新增一个模型变体，先复制更接近你目标的实验包，不必从 Baseline+ 全量继承默认增强。

## 源码入口

- 实验默认配置：`experiments/baseline_plus/__init__.py`
- 模型实现：`experiments/baseline_plus/model.py`
- 数据增强说明：[PCVR 数据管道](../guide/pcvr-data-pipeline.md)

## 最小复核

```bash
uv run pytest tests/unit/experiments/test_packages.py -q
uv run pytest tests/unit/experiments/test_runtime_contract_matrix.py -q
uv run pytest tests/unit/infrastructure/accelerators/test_tilelang_ops.py -q
```
