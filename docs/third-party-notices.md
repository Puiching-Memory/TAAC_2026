---
icon: lucide/scroll-text
---

# 第三方来源与许可说明

本项目包含来源于第三方或受其设计启发的局部实现。下表说明已有源码记录的范围；它不替代上游 LICENSE、源文件头或分发义务。论文阅读笔记的图片与研究结论属于各自作者，不能因本仓库的 MIT 许可推定其可任意再许可。

## NVIDIA cuEmbed

- 来源：[NVIDIA/cuEmbed](https://github.com/NVIDIA/cuEmbed)，参考 commit `3bb39fd4ccaca831cf55d9ff4fea2998dc65359f`。
- 许可：[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)。
- 本地范围：`src/taac2026/infrastructure/accelerators/embedding/kernels/cuembed_embedding_bag_mean.cu`，加载器为同级上层的 `cuembed_runtime.py`。

该实现按 fixed-hotness lookup 思路重写，覆盖单 CUDA table、忽略 padding id 0 的非加权 mean pooling forward。它不是完整 cuEmbed 的 vendored 副本，不支持全部上游接口或 backward。

## QwenLM FlashQLA

- 来源：[QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA)，参考 commit `6ef4858`。
- 许可：MIT，具体条款以来源仓库 LICENSE 为准。
- 本地范围：`src/taac2026/infrastructure/accelerators/attention/kernels/gated_delta_rule/`。

该 kernel 族按 FlashQLA 设计启发完成 TAAC 特定的 TileLang 改写，涉及 chunked prefill forward/backward、KKT solve、recurrent state preparation 与 context parallel。surface 与 capability 声明位于 `attention/gated_delta_rule.py` 和 `gated_delta_rule_capabilities.py`。它不是完整 FlashQLA 副本。

## DualQ 与 FM Highway 设计来源

DualQ 源码注明移植自 [zzhlkw-ai/TAAC2026](https://github.com/zzhlkw-ai/TAAC2026) 的 `round2best`，MIT License；FM Highway 借鉴 [linrongc/kdd26_unidot](https://github.com/linrongc/kdd26_unidot) 的显式二阶直达思路，来源标注为 MIT。当前移植和新增部分的边界见[DualQ 文档](experiments/dualq.md)，相应源文件保留来源说明。

## 研究材料和后续维护

HyFormer、InterFormer、OneTrans 等页面链接原论文；QueryFormer 根据公开文章独立实现，没有声称使用冠军私有源码。归档官方 baseline 与 schema 保留原内容，属于独立参考材料。

引入或修改第三方代码时记录精确版本、来源、许可与本地改动范围，保留必要的版权声明。此页仅覆盖已识别的上述来源，不构成全仓库法律审计。
