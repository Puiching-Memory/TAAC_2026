---
icon: lucide/server
---

# 比赛平台运行环境（历史观测）

本页保留仓库对 **2026-04-25 至 2026-04-27** 任务启动日志的整理。它不是当前平台 SLA，也不是本次实机检测；镜像、网络和资源配额都需要新任务重新确认。

## 观测摘要

| 项目 | 历史值 |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS |
| Python | `/opt/conda/envs/competition/bin/python3`，3.10.20 |
| Torch / CUDA | 2.7.1+cu126 / 12.6 |
| TorchRec / FBGEMM | 1.2.0+cu126 |
| 可见 GPU | 约 0.2× NVIDIA H20，19574 MiB 显存 |
| Driver / nvcc | 535.247.01 / 12.6.77 |
| gcc / g++ / make | 11.4.0 / 11.4.0 / 4.3 |
| 缺失构建工具 | cmake、ninja、pkg-config |

日志中的主机总内存不能等同于任务配额；官方声明的切片内存与 CPU 见[平台指南](official-competition-docs.md)。不要按整卡 H20 估算吞吐和 batch。

## 与当前代码的兼容性

当前项目要求 Python 3.12–3.14，本地 CUDA profile 为 `cuda132`。历史平台 Python 3.10 与当前包元数据不兼容，旧 Torch 栈也不是当前锁定环境。打包器没有自动转译 Python 或重写依赖版本的步骤。

在当前目标平台运行前，应先确认兼容解释器、Torch 与依赖。仅设置 `TAAC_RUNNER=python` 或跳过 pip 安装不能解决版本不兼容。当前 Bundle 格式和实际变量映射见[上传指南](online-training-bundle.md)。

## 网络与构建边界

历史任务通过平台代理访问 Tencent PyPI/Anaconda 镜像；公共 PyPI、GitHub、Astral、PyTorch 索引和 conda-forge 不能作为可靠启动依赖。这些是当时的探测结果，不应写成永久封禁结论。

保留平台注入的代理设置，不硬编码旧代理地址。URL 探测成功不表示 wheel、源码构建或模型下载都成功；gcc/nvcc 存在也不表示缺少 ninja/cmake 的完整 JIT 链路可用。

## 如何刷新这份记录

生成 Host Device Info 训练包，在兼容的目标任务运行：

```bash
uv run --with torch==2.13.0 taac-package-train \
  --experiment experiments/host_device_info \
  --output-dir outputs/bundles/host_device_info
```

保存任务日期与 ID、解释器、核心包、nvidia-smi、构建工具和依赖源探测。公开报告前脱敏代理凭据、私有路径和标识符。检测配置见[维护实验](../experiments/host-device-info.md)。
