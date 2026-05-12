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

## QwenLM FlashQLA

- Source: https://github.com/QwenLM/FlashQLA
- Reference commit: `6ef4858`
- License: MIT License
- Local scope: `src/taac2026/infrastructure/accelerators/attention/kernels/gated_delta_rule/` contains TAAC-specific TileLang kernels for the chunked gated-delta-rule (GDN Chunked Prefill) attention operator, inspired by FlashQLA's kernel design. Key files include `chunk.py` (high-level forward/backward orchestration and autograd function), `fused_fwd.py` and `fused_bwd.py` (warp-specialized fused TileLang kernels for forward and backward passes), `kkt_solve.py` (chunk-local KKT triangular solve kernel), `prepare_h.py` (recurrent state preparation kernel), and `context_parallel.py` / `cp_fwd.py` (intra-card context parallelism support). The operator surface layer lives at `src/taac2026/infrastructure/accelerators/attention/gated_delta_rule.py` and `gated_delta_rule_capabilities.py`.

The local implementation is not a vendored copy of the FlashQLA project. It is a TAAC-specific rewrite covering gated-delta-rule chunked prefill forward and backward, KKT solve, recurrent state preparation, hardware-friendly algebraic reformulation, and gate-driven intra-card context parallelism, built on TileLang.

MIT License text: https://opensource.org/licenses/MIT
