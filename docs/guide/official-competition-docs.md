---
icon: lucide/book-open-text
---

# 官方平台操作指南（历史快照）

本页整理归档的 Tencent Angel Machine Learning Platform User Guide。原文同步于 **2026-04-25**，页面当时最近保存时间为 2026-04-23 19:01。操作入口、资源和限制可能变化，实际提交以当前平台为准。

来源：[企业微信官方文档](https://doc.weixin.qq.com/doc/w3_ALQATQZyACgCNqA7rGAELTHOn2rc5?scode=AJEAIQdfAAo65iNyGaALQATQZyACg)，可能需要登录。这里的图片和文字是已有归档，不表示本次重新登录确认。

## 训练、发布、评测

1. 创建训练任务，填写名称和说明，上传脚本与辅助文件，提交后点击 Run。
2. 在 Instances 查看 Output 与 Logs；需要中止时使用 Stop。
3. 训练完成后在 Output 中选择完整 checkpoint，点击 Publish，填写模型信息，确认状态为 Released。
4. 从 Model Management 的 Model Evaluation，或评测页 Create Evaluation 创建任务。

本仓库训练上传 `run.sh + code_package.zip`，评测上传 `infer.py + code_package.zip`。生成和检查方法见[Bundle 指南](online-training-bundle.md)。

![归档训练流程](../assets/figures/guide/official-platform-guide/official-guide-02.png)

![归档任务表单](../assets/figures/guide/official-platform-guide/official-guide-03.png)

![归档 Run 入口](../assets/figures/guide/official-platform-guide/official-guide-05.png)

![归档实例产物与日志入口](../assets/figures/guide/official-platform-guide/official-guide-08.png)

## 归档入口约定

| 项目 | 归档要求 |
| --- | --- |
| 推理文件与函数 | `infer.py`，提供无参数 `main()` |
| 预测输出 | `EVAL_RESULT_PATH/predictions.json` |
| 上传大小 | 脚本总大小不超过 100 MB |
| 准备脚本 | 可额外上传 `prepare.sh`，在预激活 Conda 环境中执行 |
| checkpoint 目录 | 前缀 `global_step`，不超过 300 字符，限定字符集 |

当前仓库在 checkpoint 同目录保存权重与两份 sidecar，发布时保留整套产物。输出结构为 `{"predictions": {"user_id": probability}}`，概率范围 0–1，key 按平台数据约定。

## 变量与资源快照

训练常用 `TRAIN_DATA_PATH`、`TRAIN_CKPT_PATH`、`TRAIN_TF_EVENTS_PATH`；评测常用 `MODEL_OUTPUT_PATH`、`EVAL_DATA_PATH`、`EVAL_RESULT_PATH`、`EVAL_INFER_PATH`。`USER_CACHE_PATH` 为平台缓存目录。框架实际消费的变量及路径边界以 [Bundle](online-training-bundle.md) 为准。

归档文档给出的切片为单卡计算份额 20%、19 GiB 显存、9 CPU 核与 55 GiB 内存；软件为 Ubuntu 22.04、Python 3.10.20、CUDA 12.6、Torch 2.7.1+cu126。**该 Python 版本不满足当前项目 `>=3.12` 要求**，不能将旧操作截图视为当前代码兼容性证明。

真实日志观测与平台声明的资源口径也不完全相同，见[环境快照](competition-online-server.md)。

## 其他归档图示

![平台首页](../assets/figures/guide/official-platform-guide/official-guide-01.png)

![任务区域](../assets/figures/guide/official-platform-guide/official-guide-04.png)

![运行状态](../assets/figures/guide/official-platform-guide/official-guide-06.png)

![Instances 入口](../assets/figures/guide/official-platform-guide/official-guide-07.png)

![任务菜单](../assets/figures/guide/official-platform-guide/official-guide-09.png)

![脚本编辑器](../assets/figures/guide/official-platform-guide/official-guide-10.png)

更新这页时记录新的来源时间与环境，保持文字、图示和兼容性说明一致，再严格构建站点。
