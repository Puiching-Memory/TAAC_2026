---
icon: lucide/hard-drive-download
---

# 本地文档站与 Pages 发布

正文在 `docs/`，导航和主题在 `zensical.toml`，Zensical 生成的 `site/` 可重新构建。编辑源文件，不编辑生成 HTML。

## 本地预览和严格构建

在项目支持的 Linux 环境：

```bash
uv sync --locked --extra dev
uv run zensical serve
uv run zensical build --strict --clean
```

预览默认地址为 `http://127.0.0.1:8000`。需要外部访问时显式设置 `--dev-addr`，并自行控制网络暴露范围。

文档工具版本在 `pyproject.toml`/`uv.lock` 固定。仅编辑文档时也可以在隔离环境安装相同版本的 Zensical 构建；这种验证不等同于完成整个 Linux 项目的锁定依赖同步。

## 写作和链接约定

分区 `index.md` 提供选择入口，叶子页说明一个任务的前提、操作、产物和失败边界。源码默认值、论文结果与历史观测应注明来源，避免同一张表混用证据。

页面链接使用相对 `.md` 路径，静态图片使用相对路径。框架源码链接指向仓库 blob，避免站点构建把 `src/` 当作待发布文件。新增页面需要加入 nav；改标题或 heading 后检查旧链接中的锚点。

| 修改 | 文件位置 |
| --- | --- |
| 正文/导航 | `docs/**/*.md` / `zensical.toml` |
| 图片 | `docs/assets/` 或现有论文 `img/` |
| 样式/脚本 | `docs/assets/stylesheets/`、`docs/assets/javascripts/` |
| 模板覆盖 | `docs/overrides/partials/` |

图片使用 Git LFS，克隆或 CI checkout 需取得实际资源；只有 LFS pointer 会导致页面有链接但没有图片。

## Pages 事件路径

以 `.github/workflows/deploy-docs.yml` 为准：

| 事件 | 行为 |
| --- | --- |
| PR 修改 docs、配置、依赖或部署 workflow | Python 3.13 严格构建，只验证 |
| main 推送文档/站点配置，且无 CI 相关路径变更 | 直接构建并部署 |
| 同一推送包含 src、experiments、tests、依赖或 CI workflow | 直接部署跳过，等待 CI |
| main 的 CI 成功完成 | 用 CI 的 `head_sha` 构建并部署 |
| workflow_dispatch | 构建并部署所选 ref |

纯文档路径不触发 Python CI；混合改动应检查后续 workflow_run，不能把直接部署 skip 当成失败。线上是否更新需核对部署 commit 与 Actions 结果，本地 `site/` 不是发布证据。

## 验证范围

严格构建检查渲染、配置和部分链接，但不能代替所有站内锚点、外链可达性、命令执行或浏览器视觉检查。改工作流时另核对 path filters、job conditions、checkout ref；环境有 actionlint 时运行：

```bash
actionlint .github/workflows/ci.yml .github/workflows/deploy-docs.yml
```

缺图先查 LFS，缺页先查 nav，错误锚点先查 heading。成功构建后只提交源文件，不提交 `site/` 或缓存。
