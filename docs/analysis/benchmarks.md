---
icon: lucide/gauge
---

# 性能基准

## 当前状态

当前工作树没有保留可执行的 benchmark 测试切片，也没有生成中的 ECharts benchmark 资产。`taac-bench-report` 入口仍存在，但它现在只是一个最小占位报告命令，用于把输入文件列表记录到 JSON 中。

可用命令：

```bash
uv run taac-bench-report \
		--input outputs/performance/example.json \
		--output outputs/reports/benchmark_report.json
```

输出示例：

```json
{
	"report": "benchmark",
	"input": ["outputs/performance/example.json"],
	"status": "placeholder"
}
```

## 恢复基准套件时

如果后续重新恢复性能基准，需要同步补齐三部分：

1. 可执行的测试或脚本入口。
2. 报告命令对输入 JSON 的真实解析逻辑。
3. 文档站可消费的图表资产和验收摘要。

在这三部分落地前，本页只记录当前占位入口，避免把历史 benchmark 命令误写成当前可运行流程。