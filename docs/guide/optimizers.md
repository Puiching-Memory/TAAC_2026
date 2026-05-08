---
icon: lucide/settings-2
---

# 优化器

当前项目已经支持 Muon 优化器。它作为 dense optimizer 接入共享 PCVR 训练器，可以通过实验默认配置或训练 CLI 选择；sparse embedding 参数仍然走单独的 Adagrad optimizer。

## 当前支持状态

| 类型                  | 配置值             | 实现和行为                                                                        |
| --------------------- | ------------------ | --------------------------------------------------------------------------------- |
| AdamW                 | `adamw`            | 默认 dense optimizer，使用 `torch.optim.AdamW(betas=(0.9, 0.98))`                 |
| Fused AdamW           | `fused_adamw`      | 使用 PyTorch fused AdamW；当前 runtime 或参数集不支持时会报错                     |
| Orthogonal AdamW      | `orthogonal_adamw` | optimizer 仍是 AdamW，但 trainer 会在 step 前对二维及以上 dense gradient 做正交化 |
| Muon                  | `muon`             | 使用仓库内 `Muon` 实现；矩阵参数走 Muon 更新，非矩阵参数回退到 AdamW 风格更新     |
| Sparse embedding 优化 | 不由此字段控制     | `get_sparse_params()` 返回的 embedding 参数走 `torch.optim.Adagrad`               |

目前各实验包默认值如下：`baseline_plus` 默认使用 `muon`；`baseline`、`interformer`、`onetrans`、`unitok` 默认使用 `adamw`；`symbiosis` 默认使用 `orthogonal_adamw`。

## 快速启用 Muon

本地训练时可以直接覆盖 dense optimizer：

```bash
bash run.sh train \
  --experiment experiments/baseline_plus \
  --run-dir outputs/baseline_plus_muon \
  --dense-optimizer-type muon \
  --max_steps 10000
```

训练 CLI 同时接受连字符和下划线参数名，所以 `--dense-optimizer-type` 和 `--dense_optimizer_type` 等价。

也可以把默认值写进实验包的 `TRAIN_DEFAULTS`：

```python
from taac2026.api import PCVROptimizerConfig, PCVRTrainConfig


TRAIN_DEFAULTS = PCVRTrainConfig(
    optimizer=PCVROptimizerConfig(
        lr=1e-4,
        max_steps=10000,
        dense_optimizer_type="muon",
    ),
)
```

CLI 参数会覆盖实验默认值。线上训练 bundle 也会走同一套训练 CLI，因此可以在 bundle 的 `run.sh` 参数里追加 `--dense-optimizer-type muon`。

## 参数分组

共享 trainer 会先看模型是否实现了 `get_sparse_params()` 和 `get_dense_params()`。

- 如果模型暴露 sparse 参数，`get_sparse_params()` 返回的 embedding 参数交给 Adagrad。
- `get_dense_params()` 返回的非 embedding 参数交给 `dense_optimizer_type` 指定的 dense optimizer。
- 如果模型没有暴露 sparse 参数，所有模型参数都会被视为 dense 参数。

因此，切到 `dense_optimizer_type="muon"` 只会改变 dense 参数的 optimizer；它不会改变 sparse embedding 的 Adagrad 学习率、weight decay 或重建策略。

## Muon 实现细节

仓库内的 `Muon` 实现面向普通 PyTorch optimizer 接口，支持 checkpoint 保存和恢复 optimizer state。

它的默认超参数由 registry 固定传入：

| 参数           | 默认值            | 说明                                           |
| -------------- | ----------------- | ---------------------------------------------- |
| `lr`           | 训练配置里的 `lr` | 和其他 dense optimizer 共用 base learning rate |
| `momentum`     | `0.95`            | Muon 矩阵参数动量                              |
| `nesterov`     | `True`            | Muon 矩阵参数使用 Nesterov update              |
| `ns_steps`     | `5`               | Newton-Schulz 正交化迭代次数                   |
| `weight_decay` | `0.01`            | 对 Muon 和 AdamW fallback 参数都生效           |
| `adamw_betas`  | `(0.9, 0.98)`     | 非矩阵参数的 AdamW fallback beta               |
| `adamw_eps`    | `1e-8`            | 非矩阵参数的 AdamW fallback epsilon            |

行为边界：

- `parameter.ndim >= 2` 的 dense 参数走 Muon 矩阵更新。
- bias、标量等非矩阵 dense 参数走 AdamW 风格更新。
- sparse gradient 不受 Muon 支持；当前 PCVR 模型通过 sparse / dense 参数分组避免把 embedding sparse gradient 交给 Muon。
- 学习率 warmup 和 scheduler 仍由 trainer 统一设置，只作用在 dense optimizer 参数组上。

## Benchmark

可以用 dense optimizer benchmark 比较系统开销：

```bash
uv run taac-benchmark-pcvr-optimizer \
  --device cuda \
  --batch-size 512 \
  --feature-dim 128 \
  --hidden-dim 512 \
  --depth 4 \
  --steps 50 \
  --warmup-steps 10 \
  --repeats 5 \
  --optimizers adamw,orthogonal_adamw,muon \
  > outputs/benchmarks/optimizer_cuda.json
```

CPU 可以用来做通路检查，但不要用 CPU 数字推断线上 GPU 训练性能。更完整的 benchmark 口径见 [性能 Benchmark](performance-benchmarks.md)。

## 源码入口

- 支持值列表：`src/taac2026/infrastructure/runtime/execution.py`
- 配置对象校验：`src/taac2026/domain/config.py`
- 训练 CLI 参数：`src/taac2026/application/training/args.py`
- dense optimizer 构造：`src/taac2026/infrastructure/optimization/registry.py`
- Muon 实现：`src/taac2026/infrastructure/optimization/muon.py`
- Orthogonal AdamW gradient transform：`src/taac2026/infrastructure/optimization/transforms.py`
- trainer 参数分组和 step：`src/taac2026/infrastructure/runtime/trainer.py`
- benchmark CLI：`src/taac2026/application/benchmarking/pcvr_optimizer_benchmark.py`
- runtime 覆盖测试：`tests/unit/infrastructure/runtime/test_trainer.py`
- CLI 覆盖测试：`tests/unit/application/training/test_cli.py`
- benchmark 覆盖测试：`tests/benchmarks/cpu/application/benchmarking/test_pcvr_optimizer_benchmark.py`

## 常见问题

| 现象                                  | 原因和处理                                                                          |
| ------------------------------------- | ----------------------------------------------------------------------------------- |
| 传了 `muon` 但 sparse 学习率没变化    | `dense_optimizer_type` 只控制 dense 参数；调整 embedding 要用 sparse optimizer 配置 |
| `fused_adamw` 在某台机器上失败        | PyTorch fused AdamW 和设备、dtype、参数集有关；换 `adamw` 或在目标机器重测          |
| `orthogonal_adamw` 和 `muon` 含义混淆 | 前者是 AdamW step 前处理 gradient；后者是独立 optimizer，对矩阵参数执行 Muon update |
| 想让 Muon 默认生效                    | 在目标实验包 `TRAIN_DEFAULTS.optimizer.dense_optimizer_type` 改成 `"muon"`          |
| 想确认训练实际用了哪个 optimizer      | 看 trainer 初始化日志里的 `dense_optimizer_type` 和 `Dense params ...` 日志         |
