---
icon: lucide/server
---

# Host Device Info

Host Device Info 是一个维护类实验包，用来在本地或线上采集运行环境快照。它不训练模型，也不需要数据集。

## 快速运行

```bash
bash run.sh train --experiment experiments/host_device_info
```

不需要传 `--dataset-path` 或 `--schema-path`。

## 实验契约

入口文件 `experiments/host_device_info/__init__.py` 导出 `ExperimentSpec`：

```python
EXPERIMENT = ExperimentSpec(
    name="host_device_info",
    kind="maintenance",
    requires_dataset=False,
)
```

`train_fn` 会拒绝 `extra_args`。如果要调采集项，需要改 `HOST_DEVICE_INFO_CONFIG` 或 `runner.py`，不是在 CLI 后面追加参数。

## 它会看什么

报告会打印到 stdout，按节输出，便于复制、grep 和贴到 issue 里。重点包括：

- 操作系统、CPU、内存、磁盘和基础命令。
- GPU、CUDA、`nvidia-smi`、`nvcc` 和设备节点。
- Python 可执行文件、版本和已安装包。
- 代理环境、常见依赖源连通性、pip / conda 探测。
- 构建工具是否存在。

具体采集项以 `experiments/host_device_info/runner.py` 为准。

## Runner 配置

`HostDeviceInfoConfig` 主要字段：

| 字段 | 默认 | 作用 |
| ---- | ---- | ---- |
| `probe_timeout_seconds` | 10 | URL / pip / conda 探测超时 |
| `probe_detail_limit` | 240 | 失败详情截断长度 |
| `enable_proxy_matrix` | True | 对站点同时测试 inherited / no_proxy |
| `enable_pip_download_probe` | True | 执行真实 `pip download` 探测 |
| `pip_download_package` | `sampleproject==4.0.0` | pip 下载探测包 |
| `enable_conda_search_probe` | True | 执行真实 `conda search` 探测 |
| `conda_probe_spec` | `python=3.10` | conda search 查询对象 |
| `site_probe_targets` | example / github / python | 额外 URL 探测目标 |

默认会脱敏带账号的代理 URL，但不会隐藏普通主机和端口。公开日志前仍应人工看一眼。

## 输出格式

采集输出以日志节分隔，例如：

```text
---- system ----
---- gpu ----
---- proxy env ----
---- dependency index probes ----
---- pip download probes ----
---- conda search probes ----
```

实验返回的 summary 至少包含：

```python
{
    "experiment_name": "host_device_info",
    "run_dir": "...",
    "repo_root": "...",
}
```

这个 summary 只适合自动化检查；真正排障要看 stdout 日志。

## 打包到线上

```bash
uv run taac-package-train \
  --experiment experiments/host_device_info \
  --output-dir outputs/bundles/host_device_info
```

上传后仍执行训练 bundle 的 `run.sh`。这个实验不支持 `taac-package-infer`。

## 源码入口

- 实验入口：`experiments/host_device_info/__init__.py`
- 采集逻辑：`experiments/host_device_info/runner.py`
- 平台结论页：[线上运行环境速查](../guide/competition-online-server.md)

## 最小复核

```bash
uv run pytest tests/contract/experiments/test_maintenance_experiments.py -q
```
