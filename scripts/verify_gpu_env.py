#!/usr/bin/env python3
"""Verify the CUDA and TorchRec toolchain inside the TAAC development environment."""

from __future__ import annotations

import argparse
import json
import logging
import sys

logging.getLogger().setLevel(logging.ERROR)

import fbgemm_gpu
import torch
import torchrec
import triton

from taac2026.domain.features import FeatureSchema, FeatureTableSpec
from taac2026.infrastructure.io.sparse_collate import keyed_jagged_from_masked_batches
from taac2026.infrastructure.nn.embedding import TorchRecEmbeddingBagAdapter


def _run_embedding_probe(device: torch.device) -> dict[str, object]:
    schema = FeatureSchema(
        tables=(
            FeatureTableSpec(
                name="probe_tokens",
                family="probe",
                num_embeddings=64,
                embedding_dim=8,
            ),
        ),
        dense_dim=0,
    )
    features = keyed_jagged_from_masked_batches(
        {
            "probe_tokens": (
                torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.long),
                torch.tensor([[True, True, False], [True, False, False]], dtype=torch.bool),
            ),
        }
    ).to(device)

    adapter = TorchRecEmbeddingBagAdapter(schema, table_names=("probe_tokens",), device=device)
    output = adapter(features)
    if output.shape != (2, 8):
        raise RuntimeError(f"Unexpected TorchRec probe output shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all().item():
        raise RuntimeError("TorchRec probe produced non-finite values")

    return {
        "shape": list(output.shape),
        "device": str(output.device),
        "dtype": str(output.dtype),
    }


def build_report(allow_missing_gpu: bool) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    if not cuda_available and not allow_missing_gpu:
        raise RuntimeError("CUDA device not available")

    device = torch.device("cuda:0" if cuda_available else "cpu")
    probe = _run_embedding_probe(device)

    if cuda_available:
        sample = torch.randn(32, 32, device=device)
        _ = sample @ sample.transpose(0, 1)
        torch.cuda.synchronize(device)

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchrec": getattr(torchrec, "__version__", "unknown"),
        "fbgemm_gpu": getattr(fbgemm_gpu, "__version__", "unknown"),
        "triton": getattr(triton, "__version__", "unknown"),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "device": str(device),
        "embedding_probe": probe,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify CUDA + torch + torchrec + fbgemm-gpu + triton")
    parser.add_argument("--allow-missing-gpu", action="store_true", help="Succeed on CPU-only hosts")
    parser.add_argument("--json", action="store_true", help="Print a JSON report")
    args = parser.parse_args()

    report = build_report(allow_missing_gpu=args.allow_missing_gpu)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"python        : {report['python']}")
        print(f"torch         : {report['torch']}")
        print(f"torchrec      : {report['torchrec']}")
        print(f"fbgemm_gpu    : {report['fbgemm_gpu']}")
        print(f"triton        : {report['triton']}")
        print(f"cuda_available: {report['cuda_available']}")
        print(f"device        : {report['device']}")
        print(f"embedding     : {report['embedding_probe']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())