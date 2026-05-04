---
icon: lucide/server
---

# Host Device Info

主机与设备诊断信息收集工具，以实验包形式集成，可在本地或线上环境一键采集运行环境快照。

## 概述

`host_device_info` 是一个维护类实验包（`kind: maintenance`），不涉及模型训练。它收集主机硬件、GPU、网络连通性、包管理器状态等诊断信息，输出结构化日志，用于排查线上训练环境问题。

该包不需要数据集（`requires_dataset: False`）。

## 收集内容

| 类别           | 采集项                                                         |
| -------------- | -------------------------------------------------------------- |
| 系统信息       | OS release、hostname、uptime、kernel、CPU、内存、磁盘、网络接口 |
| GPU            | nvidia-smi、nvcc 版本、/dev/nvidia* 设备节点、/dev/dri          |
| 代理环境       | HTTP_PROXY、HTTPS_PROXY 等 8 个代理变量                        |
| uv 引导状态    | uv 是否安装、版本、下载源可达性                                |
| 依赖索引连通性 | PyPI、腾讯 PyPI、Conda main/forge、腾讯 Conda、PyTorch CPU/CUDA126 |
| 网络探测矩阵   | 各站点 × inherited/no_proxy 两种代理模式                       |
| pip 下载探测   | pip download 实际下载测试（继承代理 / 无代理 / 腾讯源）        |
| conda 搜索探测 | conda search 实际搜索测试                                      |
| 构建工具       | gcc、g++、make、cmake、ninja、pkg-config、cc、c++              |
| Python 环境    | 可执行文件、版本、平台、CUDA_VISIBLE_DEVICES、已安装包列表     |

## 配置

配置项在 `experiments/maintenance/host_device_info/__init__.py` 的 `HostDeviceInfoConfig` 中定义。主要可调参数：

| 参数                       | 默认值                              | 说明                         |
| -------------------------- | ----------------------------------- | ---------------------------- |
| `probe_timeout_seconds`    | 10                                  | 网络探测超时（秒）           |
| `probe_detail_limit`       | 240                                 | 日志详情截断长度             |
| `enable_proxy_matrix`      | True                                | 是否执行连接矩阵探测         |
| `enable_pip_download_probe`| True                                | 是否执行 pip download 探测   |
| `enable_conda_search_probe`| True                                | 是否执行 conda search 探测   |
| `site_probe_targets`       | example/github/python               | 额外站点探测目标             |

## 运行

```bash
uv run taac-train --experiment experiments/maintenance/host_device_info
```

不需要 `--dataset-path` 或 `--schema-path`。

## 线上打包

```bash
uv run taac-package-train --experiment experiments/maintenance/host_device_info --output-dir outputs/bundle
```

该维护类实验支持训练 bundle 打包，生成 `run.sh` 与 `code_package.zip`，可在线上环境直接执行诊断任务。

由于 `host_device_info` 不实现模型推理接口，因此不适用 `taac-package-infer`。

## 输出

运行后直接将诊断日志打印到 stdout，同时返回结构化摘要：

```python
{
    "experiment_name": "host_device_info",
    "run_dir": "...",
    "repo_root": "...",
    "requested_profile": None,
    "requested_python": None,
}
```

诊断日志按节组织，每节以 `---- <title> ----` 分隔，便于 grep 和人工阅读。

## 与 competition-online-server 的关系

[线上运行环境速查](../guide/competition-online-server.md) 是基于多次真实任务日志整理的结论性文档；本工具是采集原始探测数据的自动化入口。两者互补：本工具生成原始数据，速查页整理稳定结论。

## 来源

- 运行器源码：`experiments/maintenance/host_device_info/runner.py`
- 包入口：`experiments/maintenance/host_device_info/__init__.py`
