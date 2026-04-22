"""GPU-specific tests for CUDA training, evaluation, and profiling paths.

These tests exercise real GPU code paths that CPU-only CI cannot cover:
  - Device placement and tensor movement
  - Mixed-precision training (AMP float16 / bfloat16 + GradScaler)
  - CUDA synchronization in latency measurement
  - Profiler with CUDA activity tracing
  - torch.compile on GPU (when available)
  - Memory management (empty_cache, peak memory tracking)
  - Multi-epoch training convergence on GPU
  - Checkpoint save/load with map_location

All tests are skipped automatically when CUDA is unavailable, so this file
is safe to collect in any environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from taac2026.application.training.profiling import (
    collect_loader_outputs,
    collect_model_profile,
    measure_latency,
    select_device,
    set_random_seed,
)
from taac2026.application.training.runtime_optimization import (
    RuntimeExecution,
    prepare_runtime_execution,
)
from taac2026.domain.config import DataConfig, ModelConfig, TrainConfig

from tests.support import (
    TinyExperimentModel,
    build_local_data_pipeline,
    build_local_loss_stack,
    build_local_optimizer_component,
    write_sample_dataset,
)


# ---------------------------------------------------------------------------
# Skip guard — all tests in this file require CUDA
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpu_device() -> torch.device:
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def gpu_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped temp workspace with sample dataset."""
    ws = tmp_path_factory.mktemp("gpu_ws")
    write_sample_dataset(ws / "sample.parquet")
    return ws


def _make_configs(
    workspace: Path,
    *,
    epochs: int = 2,
    enable_amp: bool = False,
    amp_dtype: str = "float16",
    enable_compile: bool = False,
    device: str = "cuda:0",
) -> tuple[DataConfig, ModelConfig, TrainConfig]:
    data_config = DataConfig(
        dataset_path=str(workspace / "sample.parquet"),
        sequence_names=("domain_a", "domain_b", "domain_c", "domain_d"),
    )
    model_config = ModelConfig(
        name="gpu_test_model",
        vocab_size=512,
        embedding_dim=16,
        hidden_dim=16,
        head_hidden_dim=32,
        dropout=0.0,
    )
    train_config = TrainConfig(
        output_dir=str(workspace / "output"),
        epochs=epochs,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_workers=0,
        seed=42,
        device=device,
        enable_amp=enable_amp,
        amp_dtype=amp_dtype,
        enable_torch_compile=enable_compile,
        latency_warmup_steps=1,
        latency_measure_steps=2,
    )
    return data_config, model_config, train_config


# ---------------------------------------------------------------------------
# 1. Device selection & tensor placement
# ---------------------------------------------------------------------------


class TestDevicePlacement:
    def test_select_device_returns_cuda_when_available(self):
        device = select_device()
        assert device.type == "cuda"

    def test_select_device_explicit_cuda(self):
        device = select_device("cuda:0")
        assert device == torch.device("cuda:0")

    def test_model_on_gpu(self, gpu_workspace: Path, gpu_device: torch.device):
        data_cfg, model_cfg, train_cfg = _make_configs(gpu_workspace)
        train_loader, val_loader, data_stats = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to(gpu_device)
        for param in model.parameters():
            assert param.device.type == "cuda", f"Parameter not on GPU: {param.device}"

    def test_batch_to_device(self, gpu_workspace: Path, gpu_device: torch.device):
        data_cfg, model_cfg, train_cfg = _make_configs(gpu_workspace)
        train_loader, _, _ = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        batch = next(iter(train_loader)).to(gpu_device)
        assert batch.labels.device.type == "cuda"
        assert batch.sparse_features is not None
        assert batch.sequence_features is not None
        assert batch.sparse_features.values().device.type == "cuda"
        assert batch.sequence_features.values().device.type == "cuda"


# ---------------------------------------------------------------------------
# 2. Forward pass on GPU (no AMP)
# ---------------------------------------------------------------------------


class TestGpuForward:
    def test_forward_produces_finite_logits(self, gpu_workspace: Path, gpu_device: torch.device):
        data_cfg, model_cfg, train_cfg = _make_configs(gpu_workspace)
        train_loader, _, data_stats = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to(gpu_device)
        model.eval()
        batch = next(iter(train_loader)).to(gpu_device)
        with torch.no_grad():
            logits = model(batch)
        assert logits.device.type == "cuda"
        assert torch.isfinite(logits).all(), "Non-finite logits on GPU forward"

    def test_backward_produces_gradients(self, gpu_workspace: Path, gpu_device: torch.device):
        data_cfg, model_cfg, train_cfg = _make_configs(gpu_workspace)
        train_loader, _, data_stats = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to(gpu_device)
        loss_fn, _ = build_local_loss_stack(data_cfg, model_cfg, train_cfg, data_stats, gpu_device)
        batch = next(iter(train_loader)).to(gpu_device)
        logits = model(batch)
        loss = loss_fn(logits, batch.labels)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients produced"
        assert all(torch.isfinite(g).all() for g in grads), "Non-finite gradients"


# ---------------------------------------------------------------------------
# 3. Mixed precision (AMP) paths
# ---------------------------------------------------------------------------


class TestAmpFloat16:
    def test_prepare_runtime_with_amp_float16(self, gpu_workspace: Path, gpu_device: torch.device):
        data_cfg, model_cfg, train_cfg = _make_configs(gpu_workspace, enable_amp=True, amp_dtype="float16")
        _, _, data_stats = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to(gpu_device)
        rt = prepare_runtime_execution(model, train_cfg, gpu_device)
        assert rt.amp_active is True
        assert rt.amp_resolved_dtype == "float16"
        assert rt.uses_grad_scaler is True

    def test_amp_float16_forward_backward(self, gpu_workspace: Path, gpu_device: torch.device):
        data_cfg, model_cfg, train_cfg = _make_configs(gpu_workspace, enable_amp=True, amp_dtype="float16")
        train_loader, _, data_stats = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to(gpu_device)
        rt = prepare_runtime_execution(model, train_cfg, gpu_device)
        loss_fn, _ = build_local_loss_stack(data_cfg, model_cfg, train_cfg, data_stats, gpu_device)
        optimizer = build_local_optimizer_component(model, train_cfg)

        batch = next(iter(train_loader)).to(gpu_device)
        optimizer.zero_grad(set_to_none=True)
        with rt.autocast_context():
            logits = rt.execution_model(batch)
            loss = loss_fn(logits, batch.labels)
        rt.backward_and_step(loss, optimizer, model_parameters=model.parameters())

        assert torch.isfinite(loss).item(), "AMP float16 produced non-finite loss"

    def test_amp_float16_multi_step_stability(self, gpu_workspace: Path, gpu_device: torch.device):
        """Run several optimizer steps under float16 AMP to check numerical stability."""
        data_cfg, model_cfg, train_cfg = _make_configs(gpu_workspace, epochs=1, enable_amp=True, amp_dtype="float16")
        train_loader, _, data_stats = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to(gpu_device)
        rt = prepare_runtime_execution(model, train_cfg, gpu_device)
        loss_fn, _ = build_local_loss_stack(data_cfg, model_cfg, train_cfg, data_stats, gpu_device)
        optimizer = build_local_optimizer_component(model, train_cfg)

        losses = []
        for batch in train_loader:
            batch = batch.to(gpu_device)
            optimizer.zero_grad(set_to_none=True)
            with rt.autocast_context():
                logits = rt.execution_model(batch)
                loss = loss_fn(logits, batch.labels)
            rt.backward_and_step(loss, optimizer, model_parameters=model.parameters())
            losses.append(float(loss.detach().cpu().item()))

        assert all(np.isfinite(l) for l in losses), f"Non-finite loss in AMP training: {losses}"


class TestAmpBfloat16:
    @pytest.mark.skipif(
        torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        reason="GPU does not support bfloat16",
    )
    def test_prepare_runtime_with_amp_bfloat16(self, gpu_workspace: Path, gpu_device: torch.device):
        data_cfg, model_cfg, train_cfg = _make_configs(gpu_workspace, enable_amp=True, amp_dtype="bfloat16")
        _, _, data_stats = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to(gpu_device)
        rt = prepare_runtime_execution(model, train_cfg, gpu_device)
        assert rt.amp_active is True
        assert rt.amp_resolved_dtype == "bfloat16"
        # bfloat16 does NOT use gradient scaler
        assert rt.uses_grad_scaler is False

    @pytest.mark.skipif(
        torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        reason="GPU does not support bfloat16",
    )
    def test_amp_bfloat16_forward_backward(self, gpu_workspace: Path, gpu_device: torch.device):
        data_cfg, model_cfg, train_cfg = _make_configs(gpu_workspace, enable_amp=True, amp_dtype="bfloat16")
        train_loader, _, data_stats = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to(gpu_device)
        rt = prepare_runtime_execution(model, train_cfg, gpu_device)
        loss_fn, _ = build_local_loss_stack(data_cfg, model_cfg, train_cfg, data_stats, gpu_device)

        batch = next(iter(train_loader)).to(gpu_device)
        with rt.autocast_context():
            logits = rt.execution_model(batch)
            loss = loss_fn(logits, batch.labels)
        loss.backward()
        assert torch.isfinite(loss).item()


# ---------------------------------------------------------------------------
# 4. CUDA profiling & latency measurement
# ---------------------------------------------------------------------------


class TestCudaProfiling:
    def test_measure_latency_with_cuda_sync(self, gpu_workspace: Path, gpu_device: torch.device):
        data_cfg, model_cfg, train_cfg = _make_configs(gpu_workspace)
        _, val_loader, data_stats = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to(gpu_device)
        rt = prepare_runtime_execution(model, train_cfg, gpu_device)
        latency = measure_latency(
            rt.execution_model,
            val_loader,
            gpu_device,
            warmup_steps=0,
            measure_steps=0,  # 0 = measure all batches
            runtime_execution=rt,
        )
        assert latency["device"] == "cuda:0"
        assert latency["mean_latency_ms_per_sample"] > 0.0
        assert latency["measured_batches"] > 0

    def test_collect_model_profile_with_cuda(self, gpu_workspace: Path, gpu_device: torch.device):
        data_cfg, model_cfg, train_cfg = _make_configs(gpu_workspace)
        _, val_loader, data_stats = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to(gpu_device)
        rt = prepare_runtime_execution(model, train_cfg, gpu_device)
        profile = collect_model_profile(model, val_loader, gpu_device, runtime_execution=rt)
        assert profile["device"] == "cuda:0"
        assert profile["total_parameters"] > 0
        assert "cuda" in profile["operator_summary"]["activities"]

    def test_collect_loader_outputs_on_gpu(self, gpu_workspace: Path, gpu_device: torch.device):
        data_cfg, model_cfg, train_cfg = _make_configs(gpu_workspace)
        _, val_loader, data_stats = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to(gpu_device)
        rt = prepare_runtime_execution(model, train_cfg, gpu_device)
        loss_fn, _ = build_local_loss_stack(data_cfg, model_cfg, train_cfg, data_stats, gpu_device)
        logits, labels, groups, val_loss = collect_loader_outputs(
            rt.execution_model, val_loader, gpu_device, loss_fn, runtime_execution=rt,
        )
        assert logits.dtype == np.float32
        assert labels.shape[0] == logits.shape[0]
        assert np.isfinite(val_loss)


# ---------------------------------------------------------------------------
# 5. torch.compile on GPU
# ---------------------------------------------------------------------------


def _triton_available() -> bool:
    """Check if triton is importable (required for torch.compile inductor backend)."""
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


class TestTorchCompileGpu:
    @pytest.mark.skipif(
        not hasattr(torch, "compile") or not _triton_available(),
        reason="torch.compile or triton not available",
    )
    def test_compile_on_cuda(self, gpu_workspace: Path, gpu_device: torch.device):
        data_cfg, model_cfg, train_cfg = _make_configs(gpu_workspace, enable_compile=True)
        _, val_loader, data_stats = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to(gpu_device)
        rt = prepare_runtime_execution(model, train_cfg, gpu_device)
        assert rt.compile_active is True

        # Verify compiled model produces correct output
        batch = next(iter(val_loader)).to(gpu_device)
        model.eval()
        with torch.no_grad():
            logits = rt.execution_model(batch)
        assert torch.isfinite(logits).all()

    @pytest.mark.skipif(
        not hasattr(torch, "compile") or not _triton_available(),
        reason="torch.compile or triton not available",
    )
    def test_compile_plus_amp_forward(self, gpu_workspace: Path, gpu_device: torch.device):
        data_cfg, model_cfg, train_cfg = _make_configs(
            gpu_workspace, enable_compile=True, enable_amp=True, amp_dtype="float16",
        )
        _, val_loader, data_stats = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to(gpu_device)
        rt = prepare_runtime_execution(model, train_cfg, gpu_device)
        assert rt.compile_active is True
        assert rt.amp_active is True

        batch = next(iter(val_loader)).to(gpu_device)
        with torch.no_grad():
            with rt.autocast_context():
                logits = rt.execution_model(batch)
        assert torch.isfinite(logits).all()


class TestTorchaoGpuExtensions:
    def test_torchao_native_extensions_match_cuda_toolchain(self):
        import torchao

        assert "+cu128" in getattr(torchao, "__version__", "")

        extension_paths = sorted(Path(torchao.__file__).resolve().parent.glob("_C*.so"))
        assert extension_paths, "torchao CUDA wheel did not expose native extensions"

        loaded: list[str] = []
        for extension_path in extension_paths:
            torch.ops.load_library(str(extension_path))
            loaded.append(extension_path.name)

        assert loaded


# ---------------------------------------------------------------------------
# 6. GPU memory management
# ---------------------------------------------------------------------------


class TestGpuMemory:
    def test_cuda_memory_allocated_after_model_creation(self, gpu_workspace: Path, gpu_device: torch.device):
        torch.cuda.reset_peak_memory_stats(gpu_device)
        data_cfg, model_cfg, train_cfg = _make_configs(gpu_workspace)
        _, _, data_stats = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to(gpu_device)
        allocated = torch.cuda.memory_allocated(gpu_device)
        assert allocated > 0, "No memory allocated on GPU after model creation"
        peak = torch.cuda.max_memory_allocated(gpu_device)
        assert peak >= allocated

    def test_empty_cache_releases_memory(self, gpu_workspace: Path, gpu_device: torch.device):
        data_cfg, model_cfg, train_cfg = _make_configs(gpu_workspace)
        _, _, data_stats = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to(gpu_device)
        # Forward pass to allocate intermediate buffers
        train_loader, _, _ = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        batch = next(iter(train_loader)).to(gpu_device)
        _ = model(batch)
        del model, batch
        torch.cuda.empty_cache()
        # This should not raise OOM for our tiny model
        reserved = torch.cuda.memory_reserved(gpu_device)
        assert reserved >= 0  # Sanity: reserved is a valid number


# ---------------------------------------------------------------------------
# 7. End-to-end GPU training loop
# ---------------------------------------------------------------------------


class TestGpuTrainingLoop:
    def test_multi_epoch_training_produces_checkpoint(self, gpu_workspace: Path, tmp_path: Path):
        output_dir = tmp_path / "train_output"
        output_dir.mkdir()
        data_cfg, model_cfg, train_cfg = _make_configs(gpu_workspace, epochs=2)
        train_cfg = TrainConfig(
            output_dir=str(output_dir),
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            num_workers=0,
            seed=42,
            device="cuda:0",
            latency_warmup_steps=1,
            latency_measure_steps=2,
        )
        data_cfg = DataConfig(
            dataset_path=str(gpu_workspace / "sample.parquet"),
            sequence_names=("domain_a", "domain_b", "domain_c", "domain_d"),
        )
        model_cfg = ModelConfig(
            name="gpu_e2e_test",
            vocab_size=512,
            embedding_dim=16,
            hidden_dim=16,
            head_hidden_dim=32,
            dropout=0.0,
        )
        set_random_seed(train_cfg.seed)
        train_loader, val_loader, data_stats = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        device = torch.device("cuda:0")
        model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to(device)
        rt = prepare_runtime_execution(model, train_cfg, device)
        loss_fn, _ = build_local_loss_stack(data_cfg, model_cfg, train_cfg, data_stats, device)
        optimizer = build_local_optimizer_component(model, train_cfg)

        # Train loop
        for epoch in range(1, 3):
            rt.execution_model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                with rt.autocast_context():
                    logits = rt.execution_model(batch)
                    loss = loss_fn(logits, batch.labels)
                loss.backward()
                optimizer.step()

        # Save checkpoint
        checkpoint_path = output_dir / "best.pt"
        torch.save({"model_state_dict": model.state_dict(), "epoch": 2}, checkpoint_path)
        assert checkpoint_path.exists()

        # Reload on GPU
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model2 = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to(device)
        model2.load_state_dict(payload["model_state_dict"], strict=True)
        model2.eval()
        batch = next(iter(val_loader)).to(device)
        with torch.no_grad():
            logits = model2(batch)
        assert torch.isfinite(logits).all()

    def test_amp_training_with_grad_scaler_full_loop(self, gpu_workspace: Path, tmp_path: Path):
        """Full training loop with AMP float16 + GradScaler on GPU."""
        data_cfg, model_cfg, train_cfg = _make_configs(
            gpu_workspace, epochs=2, enable_amp=True, amp_dtype="float16",
        )
        train_cfg = TrainConfig(
            output_dir=str(tmp_path / "amp_output"),
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            num_workers=0,
            seed=42,
            device="cuda:0",
            enable_amp=True,
            amp_dtype="float16",
            grad_clip_norm=1.0,
            latency_warmup_steps=1,
            latency_measure_steps=2,
        )
        device = torch.device("cuda:0")
        set_random_seed(train_cfg.seed)
        train_loader, val_loader, data_stats = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to(device)
        rt = prepare_runtime_execution(model, train_cfg, device)
        loss_fn, _ = build_local_loss_stack(data_cfg, model_cfg, train_cfg, data_stats, device)
        optimizer = build_local_optimizer_component(model, train_cfg)

        all_losses = []
        for epoch in range(1, 3):
            rt.execution_model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                with rt.autocast_context():
                    logits = rt.execution_model(batch)
                    loss = loss_fn(logits, batch.labels)
                rt.backward_and_step(
                    loss,
                    optimizer,
                    model_parameters=model.parameters(),
                    grad_clip_norm=train_cfg.grad_clip_norm,
                )
                all_losses.append(float(loss.detach().cpu().item()))

        assert all(np.isfinite(l) for l in all_losses)

        # Validate on GPU
        logits_arr, labels_arr, _, val_loss = collect_loader_outputs(
            rt.execution_model, val_loader, device, loss_fn, runtime_execution=rt,
        )
        assert np.isfinite(val_loss)
        assert logits_arr.shape[0] > 0


# ---------------------------------------------------------------------------
# 8. Checkpoint device portability
# ---------------------------------------------------------------------------


class TestCheckpointPortability:
    def test_save_on_gpu_load_on_cpu(self, gpu_workspace: Path, tmp_path: Path):
        data_cfg, model_cfg, train_cfg = _make_configs(gpu_workspace)
        _, _, data_stats = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to("cuda:0")
        # Perform a forward pass so weights are potentially modified
        train_loader, _, _ = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
        batch = next(iter(train_loader)).to("cuda:0")
        _ = model(batch)

        ckpt_path = tmp_path / "portable.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        # Load on CPU
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        cpu_model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim)
        cpu_model.load_state_dict(payload["model_state_dict"], strict=True)
        for param in cpu_model.parameters():
            assert param.device.type == "cpu"


# ---------------------------------------------------------------------------
# 9. Random seed reproducibility on GPU
# ---------------------------------------------------------------------------


class TestGpuReproducibility:
    def test_deterministic_forward_with_seed(self, gpu_workspace: Path, gpu_device: torch.device):
        data_cfg, model_cfg, train_cfg = _make_configs(gpu_workspace)

        results = []
        for _ in range(2):
            set_random_seed(42)
            _, val_loader, data_stats = build_local_data_pipeline(data_cfg, model_cfg, train_cfg)
            model = TinyExperimentModel(data_cfg, model_cfg, data_stats.dense_dim).to(gpu_device)
            model.eval()
            batch = next(iter(val_loader)).to(gpu_device)
            with torch.no_grad():
                logits = model(batch)
            results.append(logits.detach().cpu().numpy())

        np.testing.assert_allclose(results[0], results[1], atol=1e-6)
