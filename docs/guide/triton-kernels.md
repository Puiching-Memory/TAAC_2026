---
icon: lucide/cpu
---

# Triton Kernel 说明

## 当前状态

当前实验包未使用自定义 Triton Kernel。所有 GPU 操作通过 PyTorch 原生算子和 `torch.compile` 实现。

## 恢复 GPU Kernel 工作时的要求

如需开发自定义 Triton Kernel：

1. **环境要求**：CUDA 12.x、Triton 2.x、PyTorch 2.7+
2. **开发流程**：
   - 在 `src/taac2026/infrastructure/pcvr/kernels/` 下编写 Kernel
   - 通过 `torch.compile` 或 `triton.jit` 注册
   - 在模型中通过条件导入使用
3. **测试**：需要 GPU 环境，使用 `@pytest.mark.gpu` 标记
4. **兼容性**：Kernel 需兼容 A100/A800（比赛平台 GPU）

## 文档边界

本文档仅记录 Triton Kernel 的状态和恢复要求。实际 Kernel 开发文档待实现后补充。
