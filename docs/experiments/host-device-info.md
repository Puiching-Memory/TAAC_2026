---
icon: lucide/server
---

# Host Device Info：采集运行环境

这是仅支持训练入口的维护实验，不读取数据、不训练模型、不生成 checkpoint。它把设备、解释器、包版本、代理和依赖源可达性打印到 stdout，便于对比平台任务环境。

## 本地与平台运行

```bash
uv run --with torch==2.13.0 taac-train --experiment experiments/host_device_info
uv run --with torch==2.13.0 taac-package-train \
  --experiment experiments/host_device_info \
  --output-dir outputs/bundles/host_device_info
```

平台上传生成的 `run.sh` 与 `code_package.zip`。它不接受额外训练配置参数，也不支持评估或推理包。bundle 安装环境要求仍适用，见 [Bundle](../guide/online-training-bundle.md)。

## 报告涵盖什么

| 主题 | 内容 |
| --- | --- |
| 系统 | OS、CPU、内存、磁盘、命令与构建工具 |
| GPU | 可见设备、nvidia-smi、nvcc、驱动与 CUDA |
| Python | 可执行文件、版本、路径、包摘要 |
| 网络 | URL、pip download、conda search，继承代理与 no-proxy 对照 |

探测会真实发起网络请求，部分步骤可能超时。账号形式的代理 URL 会脱敏；主机、目录和包清单仍可能出现在日志中，分享前检查报告内容。

## 改采集项

`experiments/host_device_info/__init__.py` 中的 `HOST_DEVICE_INFO_CONFIG` 提供配置，`runner.py` 实现探测。默认超时 10 秒，错误详情截断 240 字符，启用代理矩阵、pip 下载和 conda 查询；配置未暴露为 CLI。

检查 Python 版本能否满足项目要求，再核对 Torch/CUDA 和资源配额。`nvidia-smi` 可用不等于 `nvcc` 或 JIT 工具链可用，URL 可达也不等于包下载成功。

```bash
uv run --with torch==2.13.0 pytest tests/contract/experiments/test_maintenance_experiments.py -q
```

历史观测见[平台环境快照](../guide/competition-online-server.md)；更新时保存日期、任务 ID 和原始日志。
