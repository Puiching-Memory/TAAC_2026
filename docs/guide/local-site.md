# 本地生成文档站点

本项目使用 [zensical](https://pypi.org/project/zensical/) 构建文档站点，配置文件为仓库根目录的 `zensical.toml`。

## 前置条件

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) 已安装

## 构建静态站点

部分图表数据不纳入版本控制，构建前需先生成：

```bash
uv run taac-dataset-eda
uv run taac-tech-timeline
uv run --no-project --isolated --with zensical zensical build --clean
```

构建产物输出到 `site/` 目录。`--clean` 会在构建前清除旧的产物。

## 本地预览（开发服务器）

```bash
uv run --no-project --isolated --with zensical zensical serve
```

启动后在浏览器打开 `http://127.0.0.1:8000`，编辑 `docs/` 下的 Markdown 文件会自动热重载。

## 常见操作

| 操作 | 说明 |
| --- | --- |
| 新增页面 | 在 `docs/` 下创建 `.md` 文件，然后在 `zensical.toml` 的 `nav` 中添加条目 |
| 修改导航 | 编辑 `zensical.toml` 中的 `nav` 数组 |
| 自定义样式 | 编辑 `docs/assets/stylesheets/extra.css` |
| 数学公式 | 使用 `$...$`（行内）或 `$$...$$`（块级），由 MathJax 渲染 |

## 故障排查

- **端口被占用**：`uv run --no-project --isolated --with zensical zensical serve --dev-addr 127.0.0.1:8001` 换一个端口。
- **样式或脚本未生效**：加 `--clean` 重新构建，或清除浏览器缓存。
- **导航项不显示**：检查 `zensical.toml` 中 `nav` 的路径是否与 `docs/` 下的文件路径一致。
