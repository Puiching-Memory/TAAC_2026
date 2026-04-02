from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig, load_config
from .data import load_dataloaders
from .evaluation import benchmark_latency, evaluate_model, move_batch_to_device
from .losses import build_criterion
from .models import build_model
from .optim import build_optimizer
from .utils import ensure_dir, resolve_device, set_seed, write_json


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch["labels"])
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)

def run_training(config: ExperimentConfig) -> dict[str, Any]:
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)
    output_dir = ensure_dir(config.train.output_dir)

    train_loader, val_loader, data_stats = load_dataloaders(
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

    pos_weight = torch.tensor([data_stats["pos_weight"]], dtype=torch.float32, device=device)
    criterion = build_criterion(
        loss_name=config.train.loss_name,
        pos_weight=pos_weight,
        pairwise_weight=config.train.pairwise_weight,
    )
    optimizer = build_optimizer(model, config.train)

    best_auc = -1.0
    best_epoch = 0
    history: list[dict[str, float]] = []

    print(f"device={device}")
    print(f"model={config.model.name} parameters={parameter_count}")
    print(
        f"train_size={int(data_stats['train_size'])} val_size={int(data_stats['val_size'])} train_positive_rate={data_stats['train_positive_rate']:.4f}"
    )

    for epoch in range(1, config.train.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip_norm=config.train.grad_clip_norm,
        )
        metrics = evaluate_model(model, val_loader, criterion, device, include_bucket_metrics=False)
        epoch_result = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": metrics["loss"],
            "val_auc": metrics["auc"],
            "val_pr_auc": metrics["pr_auc"],
            "val_brier": metrics["brier"],
            "val_logloss": metrics["logloss"],
        }
        history.append(epoch_result)
        print(
            f"epoch={epoch} train_loss={train_loss:.5f} val_loss={metrics['loss']:.5f} val_auc={metrics['auc']:.5f} val_pr_auc={metrics['pr_auc']:.5f} val_brier={metrics['brier']:.5f}"
        )

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "data_stats": data_stats,
                    "best_val_auc": best_auc,
                    "best_epoch": best_epoch,
                },
                output_dir / "best.pt",
            )

    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_metrics = evaluate_model(model, val_loader, criterion, device, include_bucket_metrics=True)
    latency = benchmark_latency(model, val_loader, device)
    summary = {
        "model_name": config.model.name,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_val_auc": best_metrics["auc"],
        "best_val_pr_auc": best_metrics["pr_auc"],
        "best_val_gauc": best_metrics["gauc"]["value"],
        "best_val_brier": best_metrics["brier"],
        "best_val_logloss": best_metrics["logloss"],
        "best_metrics": best_metrics,
        "final_epoch_metrics": history[-1] if history else {},
        "latency": latency,
        "data_stats": data_stats,
        "optimizer_name": config.train.optimizer_name,
        "loss_name": config.train.loss_name,
        "history": history,
    }
    write_json(output_dir / "summary.json", summary)
    print(
        f"best_epoch={best_epoch} best_val_auc={best_metrics['auc']:.5f} best_val_pr_auc={best_metrics['pr_auc']:.5f} best_val_brier={best_metrics['brier']:.5f} mean_latency_ms_per_sample={latency['mean_ms_per_sample']:.4f} p95_latency_ms_per_sample={latency['p95_ms_per_sample']:.4f}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TAAC 2026 Grok-style baseline model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    run_training(config)


if __name__ == "__main__":
    main()
