---
title: 开发文档
icon: material/wrench-cog-outline
---

# 开发文档

这页面向维护者和贡献者，重点不是“怎么第一次跑通”，而是“改完之后怎么不把仓库搞乱”。

## 开发时的基本约束

1. 文档只写当前仓库真实存在的能力。
2. 新增实验包时，同时更新代码、测试、实验记录和文档，而不是只补一半。
3. 仓库路径统一写成内联代码，例如 `config/gen/unirec`，不要在站点里保留会失效的仓库相对链接。
4. 结果页只记录当前工作区可直接打开文件复核的产物。
5. 目录分层优先按“是否可复用”判断：共享逻辑进 `src/taac2026`，repo 专用薄脚本留在 `tools`。

## 最重要的回归入口

```bash
uv run pytest tests -q
```

为了把“主流程 smoke 回归”和“边界/故障/纯逻辑回归”拆开，当前测试约定额外使用这些 marker：

```bash
uv run pytest -m unit -q
uv run pytest -m integration -q
uv run pytest -m "smoke or fault" -q
```

其中：

1. `unit`：纯逻辑、小输入、无重型训练闭环的快速回归。
2. `integration`：跨模块协作、真实 experiment/data pipeline/search flow 的回归。
3. `smoke`：最小可运行端到端验证，重点回答“当前主流程还能不能跑”。
4. `fault`：损坏输入、缺字段、worker 失败、budget 不满足等异常路径验证。

如果需要看核心模块覆盖率，而不把 `pytest-cov` 固化成项目常驻依赖，直接使用一次性环境：

```bash
uv run --with pytest-cov pytest \
  --cov=src/taac2026/domain \
  --cov=src/taac2026/application/search \
  --cov=src/taac2026/application/training \
  --cov-report=term-missing
```

## 仓库维护脚本

repo 专用工具脚本放在 `tools/`，不要混进 `src/taac2026` 的运行时代码里。

清理项目里的 `__pycache__`：

```bash
uv run taac-clean-pycache
uv run taac-clean-pycache --dry-run
```

这个脚本默认扫描仓库根目录，并默认跳过 `.venv`、`venv`、`env`、`.tox` 和 `node_modules` 等环境目录；如果确实需要连环境目录一起清理，再显式加 `--include-env-dirs`。

当前回归覆盖：

1. 目录式实验包加载。
2. 流式 parquet 数据管线。
3. `baseline`、`grok`、`ctr_baseline`、`deepcontextnet`、`interformer`、`onetrans`、`hyformer`、`unirec`、`uniscaleformer`、`oo` 的前向构建。
4. `train` / `evaluate` 基本闭环。
5. checkpoint 兼容性校验。
6. `metrics`、`profiling`、experiment payload/loader、search worker/trial/service helper 的边界与故障路径。

## 当前测试策略

当前默认策略是“全量回归仍保持轻量，但高风险基础模块必须有直接单元覆盖”：

1. `src/taac2026/domain` 重点补边界值和空输入。
2. `src/taac2026/application/search` 重点补 budget、worker 生命周期和失败收敛。
3. `src/taac2026/application/training` 重点补 profiling、latency 和样本数/FLOPs 估算。
4. 数据管线除了固定 happy-path sample 外，也要覆盖空序列、缺特征、截断边界和 group 分布。
5. 改动上述模块时，优先补对应 `unit` 或 `fault` 测试；只有跨模块行为改变时才优先补 `integration`。

## 新增实验包时至少要补什么

以 `config/gen/<name>` 为单位，最少补齐下面几项：

1. `__init__.py`，并导出 `EXPERIMENT`。
2. `data.py`、`model.py`、`utils.py`。
3. `docs/packages/<name>.md`，说明来源、适配方式、运行命令和当前验证状态；如果额外整理了论文长文，再补 `docs/papers/<name>.md`。
4. 对应的测试覆盖或至少 forward regression。
5. `docs/experiments.md` 里的实验清单与验证记录。

如果只是概念草案而不是可执行实验包，可以像 `config/gen/symbiosis` 一样保留在目录里，但不要把它写进“当前可执行实验包”列表。

## 图表更新

当前图表逻辑分两层：

1. `src/taac2026/reporting/model_performance_plot.py` 负责读取 `summary.json`、回退 `docs/experiments.md`、合并 optuna trial，并执行实际渲染。
2. `src/taac2026/application/reporting/cli.py` 提供正式 CLI，负责仓库默认路径和命令行参数。

重画仓库根目录 `figures/` 下的两张图：

```bash
uv run taac-plot-model-performance --x-metric size
uv run taac-plot-model-performance --x-metric compute
```

## 文档站本地预览

这个仓库现在改为基于 Zensical 组织文档；为了降低迁移风险，当前先继续复用现有的 `mkdocs.yml` 兼容配置。最小预览命令是：

```bash
uv run --no-project --isolated --with zensical zensical serve
```

做构建校验时使用：

```bash
uv run --no-project --isolated --with zensical zensical build --clean
```

这里显式加 `--no-project --isolated`，目的是让文档命令只安装 Zensical 相关依赖，不去解析或同步训练环境里的 `torch` 等项目依赖。CI 同样沿用这套隔离式 docs-only 执行方式，并通过环境变量注入正式部署地址。

## GitHub Pages 自动发布

仓库现在带有 `.github/workflows/deploy-docs.yml`：

1. 每次 push 到 `main`，都会先在隔离的 docs-only 环境里运行 `zensical build --clean`。
2. 构建成功后，CI 会把 `site/` 目录上传成 GitHub Pages artifact，再交给官方 Pages deploy action 发布。
3. 也可以在 Actions 页手动触发一次 `Deploy Docs`。

这条 workflow 同样使用隔离的 docs-only 环境，不依赖项目主虚拟环境是否已经同步完成。
另外，论文页里的图片资源当前由 Git LFS 管理，所以部署 workflow 的 `actions/checkout` 必须开启 `lfs: true`，否则 Pages 会把 LFS 指针文本当成图片发布出去。

第一次启用时，还需要在 GitHub 仓库设置里启用 Pages，并把 Build and deployment 的 Source 设为 `GitHub Actions`。

## 当前 CLI 的日志与终端行为

当前 CLI 统一走 `rich + loguru`：

1. 日志走 `loguru`，输出到 Rich Console。
2. 训练、搜索和 batch 评估进度条走 `rich.progress`。
3. `--json` 这类机器可读输出仍保持纯文本 JSON，不混入彩色日志。

## 当前不该继续写进文档的内容

下面这些内容不在当前分支实现范围内：

1. 正式比赛线上提交流程。
2. 官方评测环境封装。
3. 可视化、EDA、聚类分析 CLI。
4. truncation sweep / feature engineering 专用脚本入口。

如果后续这些能力重新回到主分支，再基于实际代码补齐。
