# Grok

`config/gen/grok` 是从旧 `config/gen/baseline` 中正式拆分出来的本地研究方案。

## 当前定位

它不再承担“给新用户当 starter 包”的职责，而是明确表示：

1. 这是当前仓库里一条更激进的本地骨干路线。
2. 它可以继续作为研究对象演化。
3. 它保留自己的 `data.py`，让原始数据到最终模型输入的路径完整留在包内。
4. 它的训练装配目前仍复用 `baseline.utils`，避免无意义的 loss/optimizer 重复维护。

## 结构概览

1. `__init__.py`
   负责默认配置与 `EXPERIMENT` 装配，默认输出改为 `outputs/gen/grok`。
2. `data.py`
   保留 grok 自己的数据处理管线。
3. `model.py`
   保留原本的 grok 风格 unified backbone。
4. `utils.py`
   直接复用 `baseline.utils`。

## 为什么单独拆包

此前目录名叫 `baseline`，但模型名和实际结构已经明显偏向 grok 方案，这会带来两个问题：

1. baseline 的语义被污染，不再像一个“大家都可以拿来改”的参考起点。
2. 模型演化方向和 starter 文档职责被绑在一起，后续很难继续清晰维护。

拆出来之后，边界更清楚：

1. `baseline` 负责参考实现与教学价值。
2. `grok` 负责本地研究方案本身。

## 运行方式

```bash
uv run taac-train --experiment config/gen/grok
uv run taac-evaluate single --experiment config/gen/grok
uv run taac-search --experiment config/gen/grok --trials 20
```

## 当前验证状态

当前工作区里的可复核历史产物，仍主要位于旧的 legacy baseline 路径，例如：

1. `outputs/gen/baseline_optuna/trial_0019/summary.json`

这些记录代表的是 grok 这条方案的历史 lineage，而不是新的 starter baseline。等后续重新跑出 `outputs/gen/grok` 下的新产物后，再把验证记录完全迁移到新路径。
