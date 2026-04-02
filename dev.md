# 开发文档

## 线下测试环境
基于 uv 管理环境。依赖的唯一事实来源是 `pyproject.toml` 与 `uv.lock`，新增/删除依赖请使用 `uv add` / `uv remove`，不要再手写 `uv pip install -r requirements.txt`。
CUDA 12.8 H800 * 8

```bash
uv python install 3.14
uv sync --locked
```

## 常用命令

```bash
# 训练 baseline
uv run taac-train --config configs/baseline.yaml
uv run taac-train --config configs/grok_din_readout.yaml

# 训练其它公开方案复现版本
uv run taac-train --config configs/creatorwyx_din_adapter.yaml
uv run taac-train --config configs/tencent_sasrec_adapter.yaml
uv run taac-train --config configs/omnigenrec_adapter.yaml
uv run taac-train --config configs/deep_context_net.yaml
uv run taac-train --config configs/unirec.yaml
uv run taac-train --config configs/unirec_din_readout.yaml
uv run taac-train --config configs/uniscaleformer.yaml

# 对已有 best.pt 直接回填多指标与分桶评估
uv run taac-evaluate --config configs/creatorwyx_din_adapter.yaml
uv run taac-batch-evaluate

# 基于现有 JSON 产物生成 matplotlib 可视化
uv run taac-visualize evaluation --input outputs/creatorwyx_din_adapter/evaluation.json
uv run taac-visualize batch-report
uv run taac-visualize summary --input outputs/grok_din_readout/summary.json
uv run taac-visualize truncation-sweep
uv run taac-visualize dataset-profile

# 分析 parquet 的 schema、特征统计与数据分布
uv run taac-feature-schema
uv run taac-feature-profile

# history truncation 多 seed 消融
uv run taac-truncation-sweep --config configs/grok_din_readout.yaml --seq-lens 128 256 384 --seeds 42 43 44
```

## 常用命令

```bash
# 训练 baseline
uv run taac-train --config configs/baseline.yaml
uv run taac-train --config configs/grok_din_readout.yaml

# 训练其它公开方案复现版本
uv run taac-train --config configs/creatorwyx_din_adapter.yaml
uv run taac-train --config configs/tencent_sasrec_adapter.yaml
uv run taac-train --config configs/omnigenrec_adapter.yaml
uv run taac-train --config configs/deep_context_net.yaml
uv run taac-train --config configs/unirec.yaml
uv run taac-train --config configs/unirec_din_readout.yaml
uv run taac-train --config configs/uniscaleformer.yaml

# 对已有 best.pt 直接回填多指标与分桶评估
uv run taac-evaluate --config configs/creatorwyx_din_adapter.yaml
uv run taac-batch-evaluate

# 基于现有 JSON 产物生成 matplotlib 可视化
uv run taac-visualize evaluation --input outputs/creatorwyx_din_adapter/evaluation.json
uv run taac-visualize batch-report
uv run taac-visualize summary --input outputs/grok_din_readout/summary.json
uv run taac-visualize truncation-sweep
uv run taac-visualize dataset-profile

# 分析 parquet 的 schema、特征统计与数据分布
uv run taac-feature-schema
uv run taac-feature-profile

# history truncation 多 seed 消融
uv run taac-truncation-sweep --config configs/grok_din_readout.yaml --seq-lens 128 256 384 --seeds 42 43 44
```

## 常用命令

```bash
# 训练 baseline
uv run taac-train --config configs/baseline.yaml
uv run taac-train --config configs/grok_din_readout.yaml

# 训练其它公开方案复现版本
uv run taac-train --config configs/creatorwyx_din_adapter.yaml
uv run taac-train --config configs/tencent_sasrec_adapter.yaml
uv run taac-train --config configs/omnigenrec_adapter.yaml
uv run taac-train --config configs/deep_context_net.yaml
uv run taac-train --config configs/unirec.yaml
uv run taac-train --config configs/unirec_din_readout.yaml
uv run taac-train --config configs/uniscaleformer.yaml

# 对已有 best.pt 直接回填多指标与分桶评估
uv run taac-evaluate --config configs/creatorwyx_din_adapter.yaml
uv run taac-batch-evaluate

# 基于现有 JSON 产物生成 matplotlib 可视化
uv run taac-visualize evaluation --input outputs/creatorwyx_din_adapter/evaluation.json
uv run taac-visualize batch-report
uv run taac-visualize summary --input outputs/grok_din_readout/summary.json
uv run taac-visualize truncation-sweep
uv run taac-visualize dataset-profile

# 分析 parquet 的 schema、特征统计与数据分布
uv run taac-feature-schema
uv run taac-feature-profile

# history truncation 多 seed 消融
uv run taac-truncation-sweep --config configs/grok_din_readout.yaml --seq-lens 128 256 384 --seeds 42 43 44
```

## 线下测试数据
```bash
uv run hf download TAAC2026/data_sample_1000 --cache-dir ./data --type dataset
```

## 线上运行环境
TODO

## 线上训练数据
TODO