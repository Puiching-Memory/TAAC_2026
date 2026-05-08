---
icon: lucide/scroll-text
---

# Third-Party Notices

This repository contains project-local code derived from or inspired by third-party open-source projects where noted in source files.

## NVIDIA cuEmbed

- Source: https://github.com/NVIDIA/cuEmbed
- Reference commit: `3bb39fd4ccaca831cf55d9ff4fea2998dc65359f`
- License: Apache License 2.0
- Local scope: `src/taac2026/infrastructure/accelerators/embedding/kernels/cuembed_embedding_bag_mean.cu` contains a TAAC-specific rewrite of cuEmbed's fixed-hotness embedding lookup pattern for `embedding_bag_mean` forward benchmarking. `src/taac2026/infrastructure/accelerators/embedding/cuembed_runtime.py` loads that CUDA source through PyTorch extension JIT.

The local implementation is not a vendored copy of the full cuEmbed project. It currently covers one CUDA table, fixed-hotness lookup, unweighted mean reduction, and padding id `0` ignored.

Apache License 2.0 text: https://www.apache.org/licenses/LICENSE-2.0
