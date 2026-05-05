---
icon: lucide/rocket
---

# 快速开始

## 前置要求

- Linux
- Python 3.10 - 3.13
- CUDA 12.6（本地 GPU 训练需要）
- `uv`
- Git

## 安装仓库环境

推荐直接使用锁定的 `uv` 环境：

```bash
git clone https://github.com/Puiching-Memory/TAAC_2026.git
cd TAAC_2026

uv python install 3.10.20

# 本地训练 / 评估
uv sync --locked --extra cuda126

# 需要测试、lint 或本地文档站时
uv sync --locked --extra dev --extra cuda126
```

说明：

- `cuda126` 是当前仓库唯一支持的本地 CUDA profile。
- `dev` extra 包含 Ruff、Pytest、Coverage 和 Zensical。
- 线上 Bundle 运行模式不依赖 `uv`；那部分见 [线上 Bundle 上传指南](guide/online-training-bundle.md)。

## 准备示例数据

本地 smoke 默认会通过 `datasets` 下载并缓存 Hugging Face 上的 1000 行样例 parquet；仓库保留一份同步的 schema 快照供默认 smoke 使用。仓库内对应的快照布局如下：

```text
data/sample_1000_raw/
├── demo_1000.parquet
└── schema.json
```

正式比赛数据只需要保持相同的 parquet + `schema.json` 约定。
本地 PCVR 训练、评估和推理入口会固定使用默认 HF 样例 parquet，不再支持显式 `--dataset-path`；`schema.json` 默认复用仓库内快照，只有切换 schema 快照时才需要显式传 `--schema-path`。线上 Bundle 仍按平台约定接收外部数据路径。

## 训练第一个实验包

顶层 `run.sh` 现在只封装训练、评估和推理入口；最简单的本地训练方式是：

```bash
bash run.sh train \
  --experiment experiments/baseline \
  --run-dir outputs/quickstart_baseline
```

如果你更喜欢直接调用 CLI，上面的命令等价于：

```bash
uv run taac-train \
  --experiment experiments/baseline \
  --run-dir outputs/quickstart_baseline
```

训练完成后，checkpoint、`train_config.json`、`schema.json` 和其他侧车文件都会写到 `outputs/quickstart_baseline/`。

## 评估一个训练结果

本地验证入口是 `taac-evaluate single`。最稳妥的做法是显式传入训练目录：

```bash
bash run.sh val \
  --experiment experiments/baseline \
  --run-dir outputs/quickstart_baseline
```

如果需要指定某个 checkpoint，也可以直接传 `--checkpoint`：

```bash
uv run taac-evaluate single \
  --experiment experiments/baseline \
  --checkpoint outputs/quickstart_baseline/best_model/model.safetensors
```

## 运行一次推理

推理入口是 `taac-evaluate infer`，它会在结果目录下写出 `predictions.json`：

```bash
bash run.sh infer \
  --experiment experiments/baseline \
  --checkpoint outputs/quickstart_baseline/best_model/model.safetensors \
  --result-dir outputs/quickstart_infer
```

## 切换到其他实验包

所有实验包统一位于 `experiments/<name>/`。PCVR 模型实验和维护 / 分析类实验使用同一个插件目录，最常见的切换方式就是替换 `--experiment`：

```bash
bash run.sh train \
  --experiment experiments/symbiosis \
  --run-dir outputs/quickstart_symbiosis
```

维护类实验同样走训练入口，例如：

```bash
bash run.sh train --experiment experiments/host_device_info
```

## 生成线上 Bundle

打包不走 `run.sh`，而是使用专门的 CLI：

```bash
# 训练 Bundle
uv run taac-package-train \
  --experiment experiments/baseline \
  --output-dir outputs/bundles/baseline_training

# 推理 Bundle
uv run taac-package-infer \
  --experiment experiments/baseline \
  --output-dir outputs/bundles/baseline_inference
```

当前训练 Bundle 输出 `run.sh + code_package.zip`，推理 Bundle 输出 `infer.py + code_package.zip`。详见 [线上 Bundle 上传指南](guide/online-training-bundle.md)。

## 测试与文档

```bash
# 单元测试
uv run pytest tests/unit -v

# 只看实验包相关测试
uv run pytest tests/unit/experiments/test_packages.py -v

# 本地文档站
uv run zensical serve
```

默认文档地址是 `http://127.0.0.1:8000`。

## 入口速查

| 入口                                   | 当前用途                |
| -------------------------------------- | ----------------------- |
| `bash run.sh train`                    | 训练实验                |
| `bash run.sh val` / `bash run.sh eval` | 本地评估一个实验        |
| `bash run.sh infer`                    | 生成 `predictions.json` |
| `uv run taac-package-train`            | 打包训练 Bundle         |
| `uv run taac-package-infer`            | 打包推理 Bundle         |
| `uv run pytest tests/unit -v`          | 运行单元测试            |
| `uv run zensical serve`                | 启动本地文档站          |

注意：`run.sh` 当前不支持 `package`、`package-infer` 或 `test` 子命令；这些能力分别由独立 CLI 和 `pytest` 负责。
