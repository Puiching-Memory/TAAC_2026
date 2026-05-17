---
icon: lucide/server
---

# Host Device Info

## 摘要

Host Device Info 是一个维护类实验包，用来在本地或线上训练容器中采集运行环境快照。它不训练模型、不读取数据集，也不生成 checkpoint；它的任务是把“这台机器到底是什么状态”变成可复制、可 grep、可贴进 issue 的文本报告。

在比赛线上环境中，很多失败并不是模型错误，而是 CUDA、驱动、Python、代理、依赖源或文件系统状态不一致。这个实验包就是为这些问题准备的第一诊断工具。

## 一、它解决什么问题

当线上训练失败时，常见信息缺口包括：

- GPU 型号、显存、驱动和 CUDA runtime 是否符合预期。
- `nvidia-smi`、`nvcc`、设备节点和构建工具是否存在。
- Python 可执行文件、版本、site-packages 和关键包是否正常。
- pip / conda / GitHub / Python index 是否能访问。
- 代理变量是否被平台注入，no-proxy 与 inherited-proxy 行为是否不同。

Host Device Info 把这些信息集中打印，避免在训练脚本里临时插入环境探针。

## 二、实验契约

入口文件 `experiments/host_device_info/__init__.py` 导出 `ExperimentSpec`：

```python
EXPERIMENT = ExperimentSpec(
    name="host_device_info",
    kind="maintenance",
    requires_dataset=False,
)
```

关键契约：

- 只支持训练入口。
- 不需要 `--dataset-path` 或 `--schema-path`。
- `train_fn` 会拒绝 extra args。
- 要改采集项，修改 `HOST_DEVICE_INFO_CONFIG` 或 `experiments/host_device_info/runner.py`。

## 三、报告内容

报告按节输出到 stdout，便于直接从平台日志复制。核心内容包括：

| 节 | 内容 |
| -- | ---- |
| system | OS、内核、CPU、内存、磁盘、基础命令。 |
| gpu | GPU、CUDA、`nvidia-smi`、`nvcc`、设备节点。 |
| python | Python 可执行文件、版本、路径、已安装包摘要。 |
| proxy env | 代理变量、脱敏后的代理 URL、no-proxy 行为。 |
| dependency probes | 常见依赖源连通性。 |
| pip / conda probes | 真实 `pip download` 或 `conda search` 探测。 |
| build tools | 编译相关命令是否存在。 |

默认会脱敏带账号的代理 URL，但普通主机和端口仍会出现。公开日志前应人工看一眼。

## 四、Runner 配置

`HostDeviceInfoConfig` 的主要字段：

| 字段 | 默认 | 作用 |
| ---- | ---- | ---- |
| `probe_timeout_seconds` | `10` | URL / pip / conda 探测超时 |
| `probe_detail_limit` | `240` | 失败详情截断长度 |
| `enable_proxy_matrix` | `True` | 同时测试 inherited / no_proxy |
| `enable_pip_download_probe` | `True` | 执行真实 pip 下载探测 |
| `pip_download_package` | `sampleproject==4.0.0` | pip 探测包 |
| `enable_conda_search_probe` | `True` | 执行真实 conda search |
| `conda_probe_spec` | `python=3.10` | conda 查询对象 |
| `site_probe_targets` | example / github / python | 额外 URL 探测目标 |

这个实验包故意不把这些配置暴露成 CLI 参数，目的是保持线上 bundle 命令简单、可复现。

## 五、运行方式

本地或线上训练入口：

```bash
bash run.sh train --experiment experiments/host_device_info
```

训练 bundle：

```bash
uv run taac-package-train \
  --experiment experiments/host_device_info \
  --output-dir outputs/bundles/host_device_info
```

上传后仍执行训练 bundle 的 `run.sh`。该实验不支持 `taac-package-infer`。

## 六、怎么解读

优先检查这些异常：

- `torch.cuda.is_available()` 为 false，但平台宣称提供 GPU。
- `nvidia-smi` 存在而 `nvcc` 不存在：可能只能运行 CUDA，不适合本地编译 kernel。
- pip probe 失败但 URL probe 成功：依赖源或证书可能有问题。
- inherited proxy 成功、no_proxy 失败：平台依赖代理访问外网。
- Python 版本与 bundle 预期不一致：优先看 `TAAC_RUNNER` / `TAAC_PYTHON` 和平台镜像。

## 七、验收

最小复核：

```bash
uv run pytest tests/contract/experiments/test_maintenance_experiments.py -q
```

源码入口：

- `experiments/host_device_info/__init__.py`
- `experiments/host_device_info/runner.py`
- [线上运行环境速查](../guide/competition-online-server.md)
