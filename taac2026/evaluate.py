from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import load_config
from .data import load_dataloaders
from .evaluation import benchmark_latency, evaluate_model
from .losses import build_criterion
from .models import build_model
from .utils import ensure_dir, resolve_device, set_seed, write_json


def evaluate_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    config = load_config(Path(config_path))
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)

    _, val_loader, data_stats = load_dataloaders(
        config=config.data,
        vocab_size=config.model.vocab_size,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
    )

    model = build_model(
        config=config.model,
        dense_dim=int(data_stats["dense_dim"]),
        max_seq_len=config.data.max_seq_len,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    resolved_checkpoint_path = Path(checkpoint_path) if checkpoint_path else Path(config.train.output_dir) / "best.pt"
    checkpoint = torch.load(resolved_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    pos_weight = torch.tensor([data_stats["pos_weight"]], dtype=torch.float32, device=device)
    criterion = build_criterion(
        loss_name=config.train.loss_name,
        pos_weight=pos_weight,
        pairwise_weight=config.train.pairwise_weight,
    )

    metrics = evaluate_model(model, val_loader, criterion, device, include_bucket_metrics=True)
    latency = benchmark_latency(model, val_loader, device)

    if output_path:
        resolved_output_path = Path(output_path)
    else:
        default_name = (
            "evaluation.json"
            if resolved_checkpoint_path.name == "best.pt"
            else f"{resolved_checkpoint_path.stem}_evaluation.json"
        )
        resolved_output_path = resolved_checkpoint_path.with_name(default_name)
    ensure_dir(resolved_output_path.parent)

    payload = {
        "config_path": str(config_path),
        "checkpoint_path": str(resolved_checkpoint_path),
        "output_path": str(resolved_output_path),
        "model_name": config.model.name,
        "parameter_count": parameter_count,
        "metrics": metrics,
        "latency": latency,
        "data_stats": data_stats,
        "optimizer_name": config.train.optimizer_name,
        "loss_name": config.train.loss_name,
    }
    write_json(resolved_output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 TAAC 2026 已训练 checkpoint，并导出多指标与分桶结果。")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="训练时使用的 YAML 配置路径。",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="checkpoint 路径，默认读取对应 output_dir 下的 best.pt。",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        help="评估结果 JSON 路径，默认写到 checkpoint 同目录下的 evaluation.json。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint or None,
        output_path=args.output_path or None,
    )

    metrics = payload["metrics"]
    latency = payload["latency"]
    print(f"checkpoint={payload['checkpoint_path']}")
    print(
        f"auc={metrics['auc']:.5f} pr_auc={metrics['pr_auc']:.5f} brier={metrics['brier']:.5f} logloss={metrics['logloss']:.5f}"
    )
    print(f"gauc={metrics['gauc']['value']:.5f}")
    print(
        f"mean_latency_ms_per_sample={latency['mean_ms_per_sample']:.4f} p95_latency_ms_per_sample={latency['p95_ms_per_sample']:.4f}"
    )
    print(f"evaluation_written_to={payload['output_path']}")


if __name__ == "__main__":
    main()