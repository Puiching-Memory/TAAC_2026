---
icon: lucide/server
---

# 比赛线上服务器设备信息

本文根据两次真实线上任务启动日志整理，采样时间分别为 2026-04-25 21:47:29 +0800 与 2026-04-26 00:27:26 +0800。
第二次日志使用了增强版网络探测脚本，因此本文现在更侧重那些在重复观测中都成立、并且对训练 bundle 设计有直接影响的环境约束。
其中 CPU、内存、磁盘和 GPU 配额仍然属于“高概率稳定事实”，而节点负载、瞬时剩余内存和具体 hostname 仍然可能随作业而变化。

## 观测摘要

| 项目                     | 观测结果                                                                                              |
| ------------------------ | ----------------------------------------------------------------------------------------------------- |
| 任务入口                 | 平台最终执行 bash /home/taiji/dl/runtime/script/run.sh                                                |
| 探测脚本记录的 repo_root | /home/taiji/dl/runtime                                                                                |
| 请求 profile             | cpu                                                                                                   |
| 请求 Python              | 3.13                                                                                                  |
| 操作系统                 | Ubuntu 22.04.5 LTS                                                                                    |
| 内核                     | 5.4.241-1-tlinux4-0017.7                                                                              |
| CPU                      | 双路 AMD EPYC 9K84 96-Core Processor                                                                  |
| 逻辑核数                 | 384                                                                                                   |
| NUMA                     | 2 个 NUMA 节点                                                                                        |
| 内存                     | 2.2 TiB，Swap 为 0 B                                                                                  |
| GPU 可见性               | 1 个切片 GPU，可见为 0.2x NVIDIA H20                                                                  |
| GPU 显存                 | 19574 MiB                                                                                             |
| 驱动版本                 | NVIDIA Driver 535.247.01                                                                              |
| CUDA 运行时              | CUDA 12.6                                                                                             |
| nvcc                     | 12.6.77                                                                                               |
| 编译工具链               | gcc、g++、cc、c++、make 可用；cmake、ninja、pkg-config 缺失                                           |
| 网络工具                 | 容器内未提供 ip 命令                                                                                  |
| 代理环境                 | 仅设置小写 http_proxy 与 https_proxy，值为 http://21.100.120.217:3128                                 |
| 继承代理的公网 HTTPS     | 对 example、GitHub、Python、PyPI、Astral、PyTorch 索引的探测全部失败，失败类型为 proxy_tunnel_failure |
| 禁用代理后的公网访问     | 对同一批域名的探测全部失败，失败类型为 dns_failure                                                    |
| uv 与 pip                | 任务启动时 uv 不存在；在线 uv bootstrap 与在线 pip 下载都不能作为可靠前提                             |

## 计算资源

### CPU

- 架构：`x86_64`
- 处理器型号：AMD EPYC 9K84 96-Core Processor
- Socket 数：2
- 每核线程数：2
- 总逻辑核数：384
- NUMA 分布：
  - node0: `0-95,192-287`
  - node1: `96-191,288-383`
- 指令集能力中包含 AVX2、AVX-512、BF16 等扩展

这说明线上容器的 CPU 侧资源非常宽裕，数据预处理、DataLoader 并发和较重的 host 侧特征整理不会太容易先撞到 CPU 上限。

### 内存

两次日志中的 free -h 摘要略有波动，但量级一致：

| 指标       | 观测范围           |
| ---------- | ------------------ |
| 总内存     | 2.2 TiB            |
| 已使用     | 708 GiB 到 977 GiB |
| 空闲       | 908 GiB 到 1.3 TiB |
| Buff/Cache | 255 GiB 到 438 GiB |
| 可用       | 1.2 TiB 到 1.5 TiB |
| Swap       | 0 B                |

可见宿主机总内存很大，但当前节点在采样时的已使用内存波动也很明显，因此仍然不应把线上环境当成“无限内存”。尤其是没有 swap，超内存后更容易被直接杀进程。

## 存储与文件系统

### 容器内可见磁盘

| 设备      | 容量   | 备注      |
| --------- | ------ | --------- |
| `sda`     | 447.1G | 系统盘    |
| `nvme0n1` | 5.8T   | 本地 NVMe |
| `nvme1n1` | 5.8T   | 本地 NVMe |

### 当前容器挂载

日志显示根文件系统为 overlay，容量约 12T，两次采样时分别使用约 380G 与 487G，占用约 4% 到 5%。

这意味着：

- 线上运行目录本身不是一个很紧的小容器层
- 训练中间产物、解压后的 bundle 和 checkpoint 在磁盘层面通常不是首要瓶颈
- 但是否允许长期持久化、是否有作业级清理策略，仍应以平台规则为准

## 代理与出网约束

第二次日志对代理环境和公网目标做了显式探测，这部分信息比“能否看到 GPU”更直接影响线上训练成败。

### 代理环境

探测脚本记录到的代理变量为：

- HTTP_PROXY, HTTPS_PROXY, ALL_PROXY, NO_PROXY 均未设置
- http_proxy 与 https_proxy 被平台注入为 http://21.100.120.217:3128
- all_proxy 与 no_proxy 未设置

这意味着依赖 wget、curl、pip、requests 之类遵循小写代理变量的工具，默认都会走平台代理。

### 站点连通性矩阵

增强版探测脚本分别在 inherited 与 no_proxy 两种模式下测试了以下公网目标：

- https://example.com
- https://github.com
- https://www.python.org
- https://pypi.org/simple
- https://astral.sh/uv/install.sh
- https://download.pytorch.org/whl/cpu

结果如下：

| 模式      | 结果                                      | 解释                                                                                                            |
| --------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| inherited | 全部失败，失败类型为 proxy_tunnel_failure | 容器能连到 21.100.120.217:3128，但代理返回 Proxy tunneling failed: Service Unavailable，无法建立公网 HTTPS 隧道 |
| no_proxy  | 全部失败，失败类型为 dns_failure          | 去掉代理后，容器本身连最基本的公网 DNS 解析都过不去，报 Temporary failure in name resolution                    |

这比“是否有外网”更精确的结论是：

- 线上容器不能依赖平台代理访问公网 HTTPS
- 线上容器也不能依赖禁用代理后的直连公网 DNS/HTTPS
- 对当前 bundle 设计而言，应当把“在线安装 uv、在线拉 PyPI、在线拉 PyTorch 轮子”视为不可用路径

### uv 与 pip 下载探测

日志同时验证了两条常见依赖安装路径：

1. uv bootstrap
  结果是 uv 启动时不存在，而访问 https://astral.sh/uv/install.sh 的 inherited 模式探测失败类型为 proxy_tunnel_failure，因此不能把“缺少 uv 时自动补装”当成线上可靠方案。

2. pip download
  探测脚本使用 pip download 访问 https://pypi.org/simple，并测试 sampleproject==4.0.0。
  inherited 与 no_proxy 两种模式都返回 from versions: none / No matching distribution found。
  单看这一行像“包不存在”，但结合同一批日志里对 pypi.org/simple 的站点探测已经失败，更合理的解释是：当前环境根本拿不到可用的索引内容，因此不能依赖在线 pip 安装新库。

## 编译工具链

第二次日志还枚举了容器内可见的基础编译工具。结果如下：

| 工具       | 状态 | 版本或说明                   |
| ---------- | ---- | ---------------------------- |
| gcc        | 可用 | Ubuntu 11.4.0-1ubuntu1~22.04 |
| g++        | 可用 | Ubuntu 11.4.0-1ubuntu1~22.04 |
| cc         | 可用 | Ubuntu 11.4.0-1ubuntu1~22.04 |
| c++        | 可用 | Ubuntu 11.4.0-1ubuntu1~22.04 |
| make       | 可用 | GNU Make 4.3                 |
| cmake      | 缺失 | 日志中记录为 missing         |
| ninja      | 缺失 | 日志中记录为 missing         |
| pkg-config | 缺失 | 日志中记录为 missing         |

这说明线上容器并不是“完全没有本地编译能力”，但它只具备最基础的 C/C++ 编译链，缺少很多现代 Python 原生扩展或 CUDA/C++ 工程常见的构建辅助工具。

对实际任务的影响主要有三点：

- 依赖 setuptools + gcc/make 即可完成的轻量源码构建，理论上还有机会成功，但仍然会受到前面提到的代理与出网问题限制。
- 依赖 cmake、ninja 或 pkg-config 的 Python 包、CUDA 扩展、C++ 原生算子和部分第三方库，不能假设可以在任务启动时现编现装。
- 即使 gcc 与 nvcc 同时存在，也不代表线上可以临时编译完整自定义算子链路，因为缺少构建工具和外网依赖之后，常见源码安装流程通常还是会失败。

## GPU 与 CUDA 环境

### GPU 设备

| 项目                     | 观测结果                                   |
| ------------------------ | ------------------------------------------ |
| `NVIDIA_VISIBLE_DEVICES` | `GPU-27b235bd-2915-4f24-33db-bfb4bc9d2c41` |
| `CUDA_VISIBLE_DEVICES`   | 未设置                                     |
| `nvidia-smi -L`          | `GPU 0: 0.2x NVIDIA H20`                   |
| 显存                     | `0 MiB / 19574 MiB`                        |
| 功耗上限                 | 500 W                                      |
| MIG 模式                 | Disabled                                   |

虽然设备名字里带有 H20，但从 `0.2x NVIDIA H20` 和约 19.6 GiB 可见显存判断，这更像是平台切分后的 GPU 配额，而不是整卡独占。

对训练配置的直接影响是：

- 不要按完整 H20 显存去估 batch size
- 大模型实验应优先启用 AMP，并谨慎设置 embedding 维度、序列长度和负样本规模
- 即使物理机 GPU 很强，当前作业实际上拿到的是受限切片资源

### CUDA 工具链

| 项目          | 版本       |
| ------------- | ---------- |
| NVIDIA Driver | 535.247.01 |
| CUDA Runtime  | 12.6       |
| `nvcc`        | 12.6.77    |

这与仓库现有唯一的 cuda126 profile 是对齐的；如果平台仍传入历史 `cpu` profile 参数，当前入口应直接拒绝并要求切回 `cuda126`。

## 运行时行为观察

综合两次日志，可以确认几条对线上训练最关键的环境事实：

1. 平台传入的 requested_profile=cpu 与“容器内确实能看到 GPU”可以同时成立，因此 profile 选择不能仅靠是否检测到 GPU 来推断。
2. 启动阶段 uv not found 是真实现象，但在当前平台网络策略下，自动在线补装 uv 的路径并不可靠。
3. 日志里 ip not found，表示排查网络问题时不能依赖 ip addr 这类工具必然存在。
4. 平台只设置小写 http_proxy 与 https_proxy，而不是大写代理变量；排查联网失败时必须把大小写都看一遍。
5. 继承代理时公网 HTTPS 隧道全部失败，禁用代理时公网 DNS 全部失败，因此线上 bundle 必须按“无公网依赖”设计。
6. 基础编译器和 make 存在，但 cmake、ninja 与 pkg-config 缺失，因此任何需要完整原生构建链的依赖都不应指望在线现编。
7. 采样时机器负载很高，load average 大约在 47 到 66 之间，说明共享节点上的瞬时抖动是可能存在的。

## 对当前仓库的启示

结合本仓库现有线上 bundle 入口，可以得到几条比较稳妥的结论：

- run.sh 保留环境探测和失败诊断逻辑仍然有价值，但不能再把“缺少 uv 时自动在线安装”当成线上主路径。
- 线上 bundle 应尽量做到自包含：要么把 uv 和依赖预打包进去，要么切到平台内网镜像源，而不是依赖公网 PyPI、Astral 或 PyTorch 索引。
- 若实验或依赖链需要 cmake、ninja、pkg-config 或更复杂的原生构建流程，应在线下完成构建并把产物随 bundle 一起带上，而不是把源码安装留到线上。
- 文档和脚本里不应默认线上一定拿到完整 GPU，而应按小显存切片场景设计保守默认值。
- 若线上平台仍保留历史 `cpu` profile 参数，应改为显式传 `TAAC_CUDA_PROFILE=cuda126`，或直接移除该旧参数；当前仓库不再接受其他 profile。
- 日志采集脚本继续打印 CPU、内存、磁盘、GPU、nvcc、代理环境、站点矩阵和 pip 下载探测是有价值的，因为这些信息足以快速判断线上失败是环境错配、代理隧道失败，还是公网 DNS 不可用。

## 原始日志中的关键片段

下面这些字段最能概括当前线上网络约束：

```text
http_proxy=http://21.100.120.217:3128
https_proxy=http://21.100.120.217:3128
gcc=present
make=present
cmake=missing
ninja=missing
pkg-config=missing
uv_download_failure_class=proxy_tunnel_failure
site_pypi_inherited_failure_class=proxy_tunnel_failure
site_pypi_no_proxy_failure_class=dns_failure
site_astral_no_proxy_probe_detail=... Temporary failure in name resolution ...
pip_download_inherited_probe_detail=ERROR: Could not find a version that satisfies the requirement sampleproject==4.0.0 (from versions: none)
pip_download_no_proxy_probe_detail=ERROR: Could not find a version that satisfies the requirement sampleproject==4.0.0 (from versions: none)
```

## 附录：线上容器全量 pip 已安装库清单

下面保留 2026-04-26 00:27:30 那次日志里的原始输出。
这份清单来自探测脚本对当前 Python 环境已安装 distribution metadata 的枚举结果，因此会保留日志里的重复项，例如 `numpy`、`pillow` 和 `pyparsing` 同时出现了两个版本条目。

```text
installed_python_packages=217
absl-py==2.4.0
accelerate==0.32.1
aiohappyeyeballs==2.6.1
aiohttp==3.13.5
aiosignal==1.4.0
albucore==0.0.24
albumentations==2.0.8
alembic==1.18.4
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.13.0
astunparse==1.6.3
async-timeout==5.0.1
attrs==26.1.0
audioread==3.1.0
beautifulsoup4==4.14.3
bitsandbytes==0.49.2
blis==1.3.3
Bottleneck==1.4.2
catalogue==2.0.10
catboost==1.2.10
certifi==2026.2.25
cffi==2.0.0
charset-normalizer==3.4.7
click==8.3.1
cloudpathlib==0.23.0
cloudpickle==3.1.2
colorlog==6.10.1
confection==1.3.3
contourpy==1.3.1
cycler==0.12.1
cymem==2.0.13
datasets==2.14.7
decorator==5.2.1
dill==0.3.7
duckdb==1.4.4
einops==0.8.2
en_core_web_sm==3.8.0
exceptiongroup==1.3.1
fbgemm_gpu==1.2.0+cu126
filelock==3.25.2
flatbuffers==25.12.19
fonttools==4.62.1
frozenlist==1.8.0
fsspec==2023.10.0
gast==0.7.0
gensim==4.4.0
google-pasta==0.2.0
graphviz==0.21
greenlet==3.3.2
grpcio==1.80.0
h11==0.16.0
h5py==3.16.0
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==0.17.3
idna==3.11
ImageIO==2.37.3
importlib_metadata==9.0.0
iopath==0.1.10
Jinja2==3.1.6
joblib==1.5.3
jsonlines==4.0.0
keras==3.12.1
kiwisolver==1.4.9
lazy-loader==0.5
libclang==18.1.1
librosa==0.11.0
lightgbm==4.6.0
lightning-utilities==0.15.3
llvmlite==0.47.0
Mako==1.3.10
Markdown==3.10.2
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matplotlib==3.10.8
mdurl==0.1.2
mkl-service==2.5.2
mkl_fft==2.1.1
mkl_random==1.3.0
ml_dtypes==0.5.4
mpmath==1.3.0
msgpack==1.1.2
multidict==6.7.1
multiprocess==0.70.15
murmurhash==1.0.15
mypy_extensions==1.1.0
namex==0.1.0
narwhals==2.18.1
networkx==3.4.2
nltk==3.9.4
numba==0.65.0
numexpr==2.14.1
numpy==2.2.5
numpy==2.2.6
nvidia-cublas-cu12==12.6.4.1
nvidia-cuda-cupti-cu12==12.6.80
nvidia-cuda-nvrtc-cu12==12.6.77
nvidia-cuda-runtime-cu12==12.6.77
nvidia-cudnn-cu12==9.5.1.17
nvidia-cufft-cu12==11.3.0.4
nvidia-cufile-cu12==1.11.1.6
nvidia-curand-cu12==10.3.7.77
nvidia-cusolver-cu12==11.7.1.2
nvidia-cusparse-cu12==12.5.4.2
nvidia-cusparselt-cu12==0.6.3
nvidia-nccl-cu12==2.26.2
nvidia-nvjitlink-cu12==12.6.85
nvidia-nvtx-cu12==12.6.77
opencv-python==4.13.0.92
opencv-python-headless==4.13.0.92
opt_einsum==3.4.0
optree==0.19.0
optuna==4.8.0
orjson==3.11.8
packaging==26.0
pandas==2.3.3
patsy==1.0.2
peft==0.13.2
pillow==12.1.1
pillow==12.2.0
pip==26.0.1
platformdirs==4.9.4
plotly==6.6.0
polars==1.39.3
polars-runtime-32==1.39.3
pooch==1.9.0
portalocker==3.2.0
preshed==3.0.13
propcache==0.4.1
protobuf==5.29.6
psutil==7.2.2
pyarrow==23.0.1
pyarrow-hotfix==0.7
pycparser==3.0
pydantic==2.12.5
pydantic_core==2.41.5
Pygments==2.20.0
pyparsing==3.2.5
pyparsing==3.3.2
PyQt6==6.10.2
PyQt6_sip==13.11.0
pyre-extensions==0.0.32
python-dateutil==2.9.0.post0
pytorch-lightning==2.6.1
pytz==2026.1.post1
pyvers==0.2.2
PyYAML==6.0.3
regex==2026.3.32
requests==2.33.1
rich==14.3.3
sacremoses==0.1.1
safetensors==0.7.0
scikit-image==0.25.2
scikit-learn==1.7.2
scipy==1.15.3
seaborn==0.13.2
sentencepiece==0.2.1
setuptools==82.0.1
shellingham==1.5.4
simsimd==6.5.16
sip==6.15.1
six==1.17.0
smart_open==7.5.1
soundfile==0.13.1
soupsieve==2.8.3
soxr==1.0.0
spacy==3.8.14
spacy-legacy==3.0.12
spacy-loggers==1.0.5
SQLAlchemy==2.0.48
srsly==2.5.3
statsmodels==0.14.6
stringzilla==4.6.0
sympy==1.14.0
tensorboard==2.19.0
tensorboard-data-server==0.7.2
tensordict==0.11.0
tensorflow==2.19.0
tensorflow-io-gcs-filesystem==0.37.1
termcolor==3.3.0
textblob==0.20.0
thinc==8.3.13
threadpoolctl==3.5.0
tifffile==2025.5.10
timm==1.0.26
tokenizers==0.14.1
tomli==2.4.0
torch==2.7.1+cu126
torch-geometric==2.7.0
torch_cluster==1.6.3+pt27cu126
torch_scatter==2.1.2+pt27cu126
torch_sparse==0.6.18+pt27cu126
torchaudio==2.7.1+cu126
torchmetrics==1.0.3
torchrec==1.2.0+cu126
torchtext==0.18.0
torchvision==0.22.1+cu126
tornado==6.5.5
tqdm==4.67.3
transformers==4.35.0
triton==3.3.1
typer==0.24.1
typing-inspect==0.9.0
typing-inspection==0.4.2
typing_extensions==4.15.0
tzdata==2025.3
urllib3==2.6.3
wasabi==1.1.3
weasel==1.0.0
Werkzeug==3.1.8
wheel==0.46.3
wrapt==2.1.2
xgboost==3.2.0
xxhash==3.6.0
yarl==1.23.0
zipp==3.23.0
```

如果后续平台修复了代理、补齐了直连 DNS，或者提供了内网包源，建议继续追加到本文，区分“稳定事实”和“平台策略变更”。