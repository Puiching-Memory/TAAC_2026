from __future__ import annotations

import json
from pathlib import Path

from ...infrastructure.io.files import create_temporary_path, ensure_dir


def render_training_curves_plot(
    output_path: Path,
    train_losses: list[float],
    val_losses: list[float],
    val_aucs: list[float],
    best_epoch: int,
) -> None:
    if not train_losses:
        return

    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    epochs = list(range(1, len(train_losses) + 1))
    figure, (loss_axis, auc_axis) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    loss_axis.plot(epochs, train_losses, marker="o", linewidth=2.0, label="train_loss")
    loss_axis.plot(epochs, val_losses, marker="s", linewidth=2.0, label="val_loss")
    loss_axis.set_ylabel("Loss")
    loss_axis.grid(True, alpha=0.3)
    loss_axis.legend()

    auc_axis.plot(epochs, val_aucs, marker="o", linewidth=2.0, color="#2ca02c", label="val_auc")
    if 0 < best_epoch <= len(val_aucs):
        best_auc = val_aucs[best_epoch - 1]
        auc_axis.axvline(best_epoch, color="#7f7f7f", linestyle="--", linewidth=1.5, label=f"best_epoch={best_epoch}")
        auc_axis.scatter([best_epoch], [best_auc], color="#d62728", s=60, zorder=3, label=f"best_auc={best_auc:.4f}")
    auc_axis.set_xlabel("Epoch")
    auc_axis.set_ylabel("AUC")
    auc_axis.grid(True, alpha=0.3)
    auc_axis.legend()

    figure.suptitle("Training Curves")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _write_training_curves_json(path: Path, curves: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(curves, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _write_validation_predictions_jsonl(path: Path, predictions: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_training_curve_artifacts(
    output_dir: Path,
    train_losses: list[float],
    val_losses: list[float],
    val_aucs: list[float],
    best_epoch: int,
) -> None:
    output_dir = ensure_dir(output_dir)
    curves = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_auc": val_aucs,
        "best_epoch": best_epoch,
    }

    json_target = output_dir / "training_curves.json"
    plot_target = output_dir / "training_curves.png"
    staged_json = create_temporary_path(json_target)
    staged_plot = create_temporary_path(plot_target)
    try:
        _write_training_curves_json(staged_json, curves)
        render_training_curves_plot(
            output_path=staged_plot,
            train_losses=train_losses,
            val_losses=val_losses,
            val_aucs=val_aucs,
            best_epoch=best_epoch,
        )
        staged_json.replace(json_target)
        staged_plot.replace(plot_target)
    except Exception:
        staged_json.unlink(missing_ok=True)
        staged_plot.unlink(missing_ok=True)
        raise


def write_validation_predictions(path: Path, predictions: list[dict[str, object]]) -> None:
    target = Path(path)
    staged_target = create_temporary_path(target, suffix=".tmp.jsonl")
    try:
        _write_validation_predictions_jsonl(staged_target, predictions)
        staged_target.replace(target)
    except Exception:
        staged_target.unlink(missing_ok=True)
        raise


__all__ = [
    "render_training_curves_plot",
    "write_training_curve_artifacts",
    "write_validation_predictions",
]
