---
icon: lucide/hard-drive-download
---

# 本地生成文档站点

## 前置条件

安装 Zensical（MkDocs Material 封装）：

```bash
uv sync --extra dev
```

## 构建静态站点

```bash
uv run zensical build
```

输出到 `site/` 目录。

## 本地预览（开发服务器）

```bash
uv run zensical serve
```

默认在 `http://127.0.0.1:8000` 启动，支持热重载。

## 常见操作

```bash
# 指定端口
uv run zensical serve --dev-addr 0.0.0.0:8080

# 仅构建不预览
uv run zensical build --strict

# 清理构建产物
rm -rf site/
```

## 故障排查

| 问题            | 解决方案                                                |
| --------------- | ------------------------------------------------------- |
| 找不到 zensical | `uv sync --extra dev` 重新安装                          |
| 页面空白        | 检查 `zensical.toml` 中的 `nav` 配置                    |
| 样式丢失        | 检查 `extra_css` 路径                                   |
| 图表不显示      | 检查 ECharts JSON 文件是否存在于 `docs/assets/figures/` |
| 热重载不工作    | 重启 `zensical serve`                                   |
