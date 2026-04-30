---
icon: lucide/server
---

# 线上运行环境速查

本页整理线上任务启动日志中已经验证过、会直接影响训练和评测 bundle 的环境事实。它不是平台硬件 SLA，也不记录每一行原始探测输出；后续如果平台镜像、网络策略或入口契约变化，只需要更新对应结论。

采样来自三次真实任务启动日志：2026-04-25 21:47:29 +0800、2026-04-26 00:27:26 +0800、2026-04-27 18:29:50 +0800。越靠后的日志包含更完整的网络、镜像源、Conda 和 Python 包探测。

## 关键结论

- 线上运行应默认使用平台 Conda Python：`/opt/conda/envs/competition/bin/python3`，Python 版本为 `3.10.20`。
- 平台已预装 CUDA 12.6 相关 PyTorch 栈，包括 `torch==2.7.1+cu126`、`torchrec==1.2.0+cu126`、`fbgemm_gpu==1.2.0+cu126`。
- 任务启动时没有 `uv`，公共 PyPI、Astral、GitHub、PyTorch 官方索引和 conda-forge 都不能作为可靠在线依赖源。
- 平台注入了小写 `http_proxy` / `https_proxy`。公共 HTTPS 隧道不可用，但继承平台代理时 Tencent PyPI 和 Tencent Anaconda 镜像可达。
- 线上可见 GPU 是切片资源，约为 `0.2x NVIDIA H20`，可用显存约 `19574 MiB`。不要按整卡 H20 设计 batch size。
- 基础编译工具 `gcc`、`g++`、`make`、`nvcc` 可用，但 `cmake`、`ninja`、`pkg-config` 缺失。不要把复杂原生构建留到任务启动阶段。
- 训练 bundle 应保持 `TAAC_RUNNER=python`，复用平台环境；不要在线执行 `uv sync` 或尝试重建 CUDA/PyTorch 栈。

## 环境概览

| 项目          | 观测值                                                                 |
| ------------- | ---------------------------------------------------------------------- |
| 操作系统      | Ubuntu 22.04.5 LTS                                                     |
| 内核          | 5.4.241-1-tlinux4-0017.7                                               |
| 任务入口      | 训练任务日志中平台最终执行 `bash /home/taiji/dl/runtime/script/run.sh` |
| 运行时 Python | `/opt/conda/envs/competition/bin/python3`                              |
| Python 版本   | 3.10.20                                                                |
| CUDA Runtime  | 12.6                                                                   |
| NVIDIA Driver | 535.247.01                                                             |
| nvcc          | 12.6.77                                                                |
| CPU           | 双路 AMD EPYC 9K84 96-Core Processor                                   |
| 逻辑核数      | 384                                                                    |
| 内存          | 2.2 TiB，Swap 为 0 B                                                   |
| GPU           | `0.2x NVIDIA H20`                                                      |
| GPU 显存      | 约 19574 MiB                                                           |
| 根文件系统    | overlay，日志中约 12T 容量                                             |

CPU、总内存和本地磁盘容量都比较宽裕，但节点负载、可用内存和具体 hostname 会随作业变化。没有 swap，内存超限后更可能被直接杀进程。

## Python 与依赖

当前线上 Python 环境已包含 217 个 Python distribution metadata 条目。与本仓库最相关的预装包包括：

| 包             | 版本           |
| -------------- | -------------- |
| `torch`        | `2.7.1+cu126`  |
| `torchvision`  | `0.22.1+cu126` |
| `torchaudio`   | `2.7.1+cu126`  |
| `torchrec`     | `1.2.0+cu126`  |
| `fbgemm_gpu`   | `1.2.0+cu126`  |
| `numpy`        | `2.2.5`        |
| `pyarrow`      | `23.0.1`       |
| `pandas`       | `2.3.3`        |
| `scikit-learn` | `1.7.2`        |
| `tensorboard`  | `2.19.0`       |
| `optuna`       | `4.8.0`        |
| `rich`         | `14.3.3`       |
| `tqdm`         | `4.67.3`       |

这意味着线上 bundle 的默认策略应是复用平台环境，而不是在任务启动时安装基础栈。若确实缺少纯 Python 包，可以优先在当前 Conda Python 中补装，并优先使用平台可达的镜像源。

```bash
python -m pip install -i https://mirrors.cloud.tencent.com/pypi/simple/ <package>
```

训练 `run.sh` 和推理 `infer.py` 也提供同样的安装钩子：bundle 的 Python 运行模式会在导入仓库代码前，从解压后的 `project/pyproject.toml` 调用当前 Python 的 pip 安装项目依赖，并默认使用腾讯 PyPI 镜像。设置 `TAAC_SKIP_PIP_INSTALL=1` 可以跳过这一步。

核心 GPU 依赖不建议在任务启动阶段覆盖安装。Torch、CUDA、FBGEMM 或 TorchRec 版本不匹配时，应通过平台镜像或基础环境解决。

## 网络与镜像源

平台只注入小写代理变量：

```text
http_proxy=http://21.100.120.217:3128
https_proxy=http://21.100.120.217:3128
```

大写 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY`、`NO_PROXY` 在日志中均未设置。

公网目标探测结果：

| 目标类别                                       | 继承代理               | 禁用代理      | 结论                              |
| ---------------------------------------------- | ---------------------- | ------------- | --------------------------------- |
| GitHub、Python、PyPI、Astral、PyTorch 官方索引 | `proxy_tunnel_failure` | `dns_failure` | 不可靠，不应作为线上任务启动依赖  |
| Tencent PyPI                                   | reachable，HTTP 200    | `dns_failure` | 继承平台代理时可作为 pip 备选源   |
| Tencent Anaconda main/free                     | reachable，HTTP 200    | `dns_failure` | 继承平台代理时可作为 Conda 备选源 |

实践建议：

- 不要无条件 `unset http_proxy https_proxy`，否则会同时关闭当前已验证可达的腾讯镜像通路。
- 不要依赖在线安装 `uv`，`https://astral.sh/uv/install.sh` 在继承代理下不可用。
- 不要依赖公共 PyPI 或 PyTorch 官方 wheel 索引。
- 如果要验证依赖源，分别测试继承代理和禁用代理两种模式；二者失败原因不同。

## GPU 与训练配置

线上可见 GPU 是切片资源：

| 项目                     | 观测值                   |
| ------------------------ | ------------------------ |
| `nvidia-smi -L`          | `GPU 0: 0.2x NVIDIA H20` |
| 可见显存                 | 约 19.6 GiB              |
| `CUDA_VISIBLE_DEVICES`   | 日志中未设置             |
| `NVIDIA_VISIBLE_DEVICES` | 平台注入 GPU UUID        |
| MIG                      | Disabled                 |

推荐按小显存 GPU 设计训练默认值：

- 优先启用 AMP，谨慎设置 batch size。
- 控制 embedding 维度、序列长度、`num_workers`、prefetch 和验证频率。
- OOM 时先降 batch size、序列长度和模型宽度，再考虑更复杂的优化。
- 本仓库 CUDA profile 与线上 CUDA 12.6 对齐；不要传旧的 `cpu` profile 作为 CUDA 依赖配置。

日志中历史 `requested_profile=cpu` 与容器内能看到 GPU 同时出现过，因此不能仅凭 profile 字段判断是否没有 GPU。

## 编译与系统工具

基础工具状态如下：

| 工具                         | 状态          |
| ---------------------------- | ------------- |
| `gcc` / `g++` / `cc` / `c++` | 可用          |
| `make`                       | 可用          |
| `nvcc`                       | 可用，12.6.77 |
| `cmake`                      | 缺失          |
| `ninja`                      | 缺失          |
| `pkg-config`                 | 缺失          |
| `ip`                         | 缺失          |

因此轻量 setuptools 源码构建理论上可能成功，但依赖 `cmake`、`ninja`、`pkg-config` 或复杂 CUDA/C++ 构建链的包，不应假设可以在线编译。需要这些依赖时，优先通过平台镜像、预构建 wheel 或离线产物解决。

## 对 Bundle 的影响

训练和推理上传包应遵守 [线上 Bundle 上传指南](online-training-bundle.md) 中的双文件结构。

训练运行建议：

```bash
export TAAC_RUNNER=python
export TAAC_PYTHON=/opt/conda/envs/competition/bin/python3
export TAAC_DATASET_PATH=/path/to/train.parquet_or_dataset_dir
export TAAC_SCHEMA_PATH=/path/to/schema.json
export TAAC_OUTPUT_DIR=/path/to/output

bash run.sh --device cuda
```

设计约束：

- `code_package.zip` 应包含运行代码和选中的实验包，不依赖线上 `uv sync`。
- `uv.lock` 只保留在仓库本地用于开发追溯，线上 bundle 不再包含它。
- 缺少纯 Python 包时，先确认平台环境，再考虑用腾讯镜像补装。
- 不要在任务启动时下载或覆盖核心 CUDA/PyTorch 栈。
- 不要把旧训练栈里的 runtime optimization 参数复制到当前 PCVR CLI，当前共享 parser 不支持这些历史参数。

## 排障速查

| 现象                    | 优先检查                                                                                  |
| ----------------------- | ----------------------------------------------------------------------------------------- |
| `uv: command not found` | 线上 bundle 是否仍在用 `TAAC_RUNNER=uv`；应切回 `TAAC_RUNNER=python`                      |
| 依赖安装失败            | 是否在访问公共 PyPI/Astral/PyTorch 索引；改用平台环境或腾讯镜像                           |
| pip 禁用代理后失败      | 直连 DNS 不可用是已观测现象；恢复平台小写代理变量后再试腾讯镜像                           |
| Torch/CUDA 版本冲突     | 不要 pip 覆盖核心 GPU 包；改平台镜像或基础环境                                            |
| 训练 OOM                | 按 19.6 GiB 显存调小 batch、序列长度和模型宽度                                            |
| 找不到 `ip`             | 容器内未提供该命令；网络排查使用 `curl`、`python` 或现有日志脚本                          |
| 平台前置 init 告警      | 2026-04-27 日志中出现过 init 前置告警后仍继续执行用户脚本；以用户脚本最终退出码和日志为准 |

## 稳定性说明

可作为当前设计默认前提的事实：

- 平台 Python/Conda 环境存在且预装 CUDA 12.6 PyTorch 栈。
- 公共公网依赖源不可靠。
- Tencent 镜像在继承平台代理时可达。
- GPU 是约 20 GiB 显存的切片资源。
- `uv` 不是线上运行前提。

需要后续继续观察的事实：

- 具体 GPU 切片比例、hostname、瞬时负载和剩余内存。
- 平台是否更换镜像、Python 环境或代理策略。
- 平台是否新增官方内网依赖源或修复公共 HTTPS 隧道。