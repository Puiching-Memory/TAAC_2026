---
icon: lucide/cpu
---

# Triton Kernel 说明

## 当前状态

当前工作树已经收敛到 PCVR 实验包和共享 PCVR runtime，原先的共享 `infrastructure/nn` Triton/Transformer Engine 模块与 GPU 专项测试树不再是当前可执行回归的一部分。因此本页不再提供可直接运行的 kernel 测试命令。

现在与模型实现直接相关的共享代码主要在：

- `src/taac2026/infrastructure/pcvr/modeling.py`
- `src/taac2026/infrastructure/pcvr/protocol.py`
- `src/taac2026/infrastructure/pcvr/training.py`
- `src/taac2026/infrastructure/pcvr/trainer.py`

当前可验证入口是单元测试：

```bash
uv run pytest tests/unit -q
```

## 恢复 GPU Kernel 工作时的要求

如果后续重新引入 Triton kernel 或 Transformer Engine 后端，建议把恢复工作拆成一组明确的代码和测试提交：

1. 在共享 runtime 中放置稳定接口，避免让具体实验包直接依赖 kernel 细节。
2. 为每个 kernel 保留纯 PyTorch 参考路径。
3. 用小尺寸输入覆盖 mask、padding、dtype 和 fallback 语义。
4. 在 CPU 可收集环境下保留不依赖 CUDA 的接口测试。
5. 单独提供 GPU 环境验证说明，避免默认 CI 或普通文档读者运行不存在的专项测试。

## 文档边界

本页保留的是后续恢复 GPU kernel 能力时的工程约束。当前比赛训练、线上打包和实验包接入，请以 [架构](../architecture.md)、[新增实验包](contributing.md) 和 [线上训练包](online-training-bundle.md) 为准。