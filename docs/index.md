---
title: 首页
hide:
  - toc
---

# TAAC 2026 文档站

<div class="hero-copy" markdown>

这套文档不再让 `README.md` 独自承担全部信息，而是按“第一次上手、命令怎么用、实验包有哪些、开发边界是什么”四条线重新组织。目标很直接：让新同学第一次进仓库时，能在几分钟内找到正确入口。

</div>

!!! note "定位"
    这是参赛队伍的工程文档，不是 TAAC 官方文档。赛题描述、时间线和数据说明仍应以 [大赛主页](https://algo.qq.com/#intro) 与 [官方样例数据页](https://huggingface.co/datasets/TAAC2026/data_sample_1000) 为准。

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **快速开始**

    ---

    从环境同步、第一次训练、第一次评估，到如何切换数据集的本地包装实验。

    [进入页面](getting-started.md){ .md-button .md-button--primary }

-   :material-console-line: **CLI 指南**

    ---

    只覆盖当前仓库真实存在的 `taac-train`、`taac-evaluate`、`taac-search`。

    [查看命令](cli.md){ .md-button }

-   :material-file-tree-outline: **项目结构**

    ---

    把 `src/taac2026`、`config/gen`、`tests`、`tools` 的职责边界讲清楚。

    [查看结构](project-layout.md){ .md-button }

-   :material-flask-outline: **实验包索引**

    ---

    汇总当前可执行实验包、可复核 smoke 结果，以及当前工作区的结论口径。

    [查看实验](experiments.md){ .md-button }

-   :material-wrench-cog-outline: **开发维护**

    ---

    包括回归入口、图表刷新、文档站本地预览，以及文档更新规则。

    [查看维护约定](dev.md){ .md-button }

-   :material-map-marker-path: **路线图**

    ---

    记录文档站下一步要补哪些页，项目本身还有哪些能力尚未落地。

    [查看路线图](TODO.md){ .md-button }

</div>

## 推荐阅读顺序

1. 第一次进入仓库，先看[快速开始](getting-started.md)。
2. 想确认命令和参数范围，再看[CLI 指南](cli.md)。
3. 想知道当前有哪些真实可跑的实验包，直接看[实验包与验证记录](experiments.md)。
4. 要改代码、补实验、补文档时，再看[开发文档](dev.md)。

## 两条最快路径

=== "第一次跑通训练链路"

    ```bash
    uv python install 3.14
    uv sync --locked

    uv run taac-train --experiment config/gen/baseline
    uv run taac-evaluate single --experiment config/gen/baseline
    uv run pytest tests -q
    ```

    如果机器支持，也可以在训练和评估时显式打开运行时优化：

    ```bash
    uv run taac-train --experiment config/gen/baseline --compile --amp --amp-dtype bfloat16
    uv run taac-evaluate single --experiment config/gen/baseline --compile --amp --amp-dtype bfloat16
    ```

=== "先看当前实验版图"

    先打开[实验包与验证记录](experiments.md)。

    这页会直接告诉你：

    - 当前分支里哪些实验包是真实存在且可执行的。
    - 各自默认输出目录和主要来源。
    - 当前工作区里哪些 `summary.json` 可以直接复核。
    - 哪些结论只是 sample parquet 上的 smoke 结果，不能外推成正式赛题结论。

## 这次重构的原则

1. 文档站按任务而不是按“历史文件名”组织。
2. 只写当前仓库真实存在的能力，不沿用旧分支里已经消失的命令。
3. 仓库文件路径统一写成内联代码，例如 `src/taac2026/train.py`，避免在站点里出现失效链接。
4. 实验结果只保留能在当前工作区直接打开产物复核的记录。
