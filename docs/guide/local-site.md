---
icon: lucide/hard-drive-download
---

# 本地文档站

文档源文件在 `docs/`，导航在 `zensical.toml`。`site/` 是构建产物，通常不要手改，也不要把它当成文档源。

## 文件职责

| 文件 / 目录 | 作用 |
| ----------- | ---- |
| `docs/**/*.md` | 文档正文，页面路径相对 `docs/` |
| `docs/assets/` | 图片、脚本和额外样式 |
| `docs/overrides/` | Zensical / Material 模板覆盖 |
| `zensical.toml` | 站点元信息、导航、主题、额外 JS / CSS |
| `site/` | 本地构建输出，可删除重建 |

新增页面时需要同时满足两件事：Markdown 文件存在，并且 `zensical.toml` 的 `nav` 中能到达它。没有进 `nav` 的页面即使能被构建，也不一定出现在站点导航里。

## 本地预览

```bash
uv sync --locked --extra dev
uv run zensical serve
```

默认地址是 `http://127.0.0.1:8000`。

指定端口：

```bash
uv run zensical serve --dev-addr 0.0.0.0:8080
```

## 构建检查

```bash
uv run zensical build --strict
```

这会生成 `site/`，并在严格模式下检查导航、站内链接和构建配置。如果只是验证文档，不需要提交 `site/`。

清理构建产物：

```bash
rm -rf site/
```

## GitHub Pages 怎么部署

以 `.github/workflows/deploy-docs.yml` 为准：

| 事件 | 条件 | 行为 |
| ---- | ---- | ---- |
| `push` 到 `main` | 只改 `docs/**`、`zensical.toml` 或 `deploy-docs.yml` | 直接构建并部署 Pages |
| `push` 到 `main` | 同时改了 `src/`、`experiments/`、`tests/`、`pyproject.toml`、`uv.lock` 或 `ci.yml` | 先等 CI 成功，再由 `workflow_run` 部署 |
| `workflow_run` | CI 在 `main` 成功完成 | 检出 CI 对应 commit，构建并部署 |
| `workflow_dispatch` | 手动触发 | 构建并部署当前选择的 ref |

部署 job 使用 Python 3.13，先轻量同步依赖：

```bash
uv sync --locked --extra cuda126 \
  --no-install-package torch \
  --no-install-package torchrec \
  --no-install-package fbgemm-gpu
```

真正构建站点时使用隔离的 Zensical：

```bash
uv run --no-project --isolated --with zensical zensical build --clean
```

所以判断“文档会不会部署”时，看 workflow 事件路径，不要只看本地 `site/` 有没有变化。

## 改页面的落点

- 改正文：只动对应 `docs/...md`。
- 改导航标题或页面顺序：改 `zensical.toml` 的 `nav`。
- 改全站脚本：放到 `docs/assets/javascripts/`，再加入 `extra_javascript`。
- 改全站样式：放到 `docs/assets/stylesheets/extra.css` 或新增 CSS 后加入 `extra_css`。
- 改模板片段：放在 `docs/overrides/partials/`。

文档链接优先写相对路径，例如 `../experiments/baseline.md` 或 `performance-benchmarks.md`。图片也放在 `docs/assets/` 下，用相对路径引用。

## 常见问题

**导航里没有新页面**

更新 `zensical.toml` 的 `nav`。

**页面能构建但链接体验差**

把导览内容收敛到对应分区的 `index.md`。普通页面应直接写任务细节，例如命令、输入输出、环境变量、源码文件和失败模式。

**本地缺 Zensical**

重新同步 dev 依赖：

```bash
uv sync --locked --extra dev
```

**图表或图片不显示**

确认资源在 `docs/assets/` 下，并使用相对路径引用。

**构建后工作区多了 `site/`**

这是本地构建产物。只验证文档时可以删除：

```bash
rm -rf site/
```
