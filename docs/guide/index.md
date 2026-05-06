---
icon: lucide/map
---

# 指南总览

这里是任务型文档的入口。这个页面负责帮你选文档；具体页面负责讲命令、契约、源码位置和排障细节。

## 任务入口

| 任务 | 文档 |
| ---- | ---- |
| 读官方平台规则 | [官方平台用户指南](official-competition-docs.md) |
| 上传训练或推理 bundle | [线上 Bundle 上传](online-training-bundle.md) |
| 理解线上机器、Python、CUDA、代理和镜像源 | [线上运行环境速查](competition-online-server.md) |
| 修改数据增强、cache 或数据读取 | [PCVR 数据管道](pcvr-data-pipeline.md) |
| 重测数据、optimizer 或整体 benchmark 口径 | [性能 Benchmark](performance-benchmarks.md) |
| 重测 TileLang 算子，查看每个算子的支持记录 | [Benchmark 总览](../benchmark/index.md) |
| 新增或修改实验包 | [新增实验包](contributing.md) |
| 选择最小测试集 | [测试](testing.md) |
| 本地预览文档站或理解 Pages 部署 | [本地文档站](local-site.md) |
| 清理本地缓存 | [仓库缓存清理](cache-cleanup.md) |
| 清理 GitHub Actions / Pages 远端记录 | [仓库日志管理](repo-log-management.md) |

## 写文档时的分工

- 分区 `index.md` 只做导览和选择。
- 叶子页面要写清楚可执行命令、输入输出、环境变量、源码入口和常见失败模式。
- 不把源码整段搬进文档，但要解释读者为什么要看那些源码。
- 过时的 benchmark 数字、实验结果和平台观察不要当长期事实；保留复现方法和证据来源。
