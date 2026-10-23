---
icon: lucide/git-branch-plus
---

# 新增实验包

新模型应接入共享 PCVR 流程，而不是复制 trainer、数据读取、checkpoint 或打包代码。实现模型差异，复用训练用例。

## 最小目录和声明

```text
experiments/my_experiment/
├── __init__.py
└── model.py
```

从最接近的模型开始；私有层可放 `layers.py`，扩展配置放 `config.py`。下面示例复用 Baseline 的完整配方并替换模型类型，避免省略实际 schema 所需的分组：

```python
from pathlib import Path
from taac2026.api import create_pcvr_experiment
from .model import MyModel

# 将 Baseline 的 TRAIN_DEFAULTS 定义复制到当前文件，
# 再显式修改模型和训练变量；不要跨实验 import 默认值。
EXPERIMENT = create_pcvr_experiment(
    name="pcvr_my_experiment",
    package_dir=Path(__file__).resolve().parent,
    model_type=MyModel,
    train_defaults=TRAIN_DEFAULTS,
)
```

这是声明片段，`MyModel` 和完整 `TRAIN_DEFAULTS` 必须先定义。NS 配置位于 `PCVRTrainConfig(model=PCVRModelConfig(ns=PCVRNSConfig(...)))`，不在 TrainConfig 顶层。fid 来自 schema，不是字段位置。

## 模型必须满足什么

统一构造入口是 `model_type(schema, config)`，传入的是 `PCVRModelConfig` 或实验专用模型子配置，并非完整 `PCVRTrainConfig`。模型接收 `PCVRModelInput`，`forward()` 返回 `(B,)` logits，`predict()` 返回 `(logits, embeddings)`。稀疏与稠密参数通过契约方法分组，确保没有遗漏、重复或把 COO sparse gradient 交给 dense optimizer。

共享 tokenizer、embedding 和重计算 helper 从 `taac2026.api` 导入。模型特有选择与字段解释留在实验包。依赖方向见[架构](../architecture.md)。

## 扩展配置

继承 `PCVRModelConfig` 与 `PCVRTrainConfig`，使 train config 的 `model` 字段指向专用类型，并将 `config_type=MyTrainConfig` 传给工厂。参考 `experiments/queryformer/config.py`。

CLI、默认值和 sidecar 必须消费同一类型。开关使用 `--model.<field>` 或 `--model.no_<field>`；不要再添加一套 argparse parser 或松散 metadata 来绕过字段校验。

## 发现和打包

CLI 接受 `--experiment experiments/my_experiment`，也支持 module 名。registry 校验导出的 `EXPERIMENT`。打包只带选中实验与框架，因此不得依赖另一个未打包实验目录中的代码。

维护工具使用 `FunctionExperiment`，仅声明实际实现的回调：Host Device Info 不支持 infer；EDA 的 infer 只做统计。不要为未支持的功能返回占位成功。

## 接入后的验证顺序

1. 确认 registry 能发现包，新模型进入实验契约测试矩阵。
2. 验证输出形状、空序列、padding、梯度和 sparse/dense 参数分组。
3. 完成 CPU 训练、保存三件套、评估和推理 roundtrip。
4. 扩展共享契约时执行完整 CPU 门禁；CUDA backend 另做目标设备验证。
5. 构建训练与推理包，检查 zip 并脱离仓库 import 上下文执行。

命令和范围在[测试指南](testing.md)。新增实验页并加入 `zensical.toml` 导航，说明默认配方、来源差异和已验证范围，最后严格构建文档。
