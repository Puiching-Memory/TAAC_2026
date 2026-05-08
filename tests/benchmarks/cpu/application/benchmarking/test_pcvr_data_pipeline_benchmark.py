from __future__ import annotations

from argparse import Namespace

from taac2026.application.benchmarking.pcvr_data_pipeline_benchmark import (
    _build_pipeline_config,
    _consume_measured_batches,
    _estimated_batches_per_pass,
)


def test_opt_augment_preset_uses_opt_cache_and_transforms() -> None:
    config = _build_pipeline_config(
        Namespace(
            pipeline_preset="opt-augment",
            cache_batches=512,
            views_per_row=2,
            seq_window_min_len=8,
            feature_mask_probability=0.03,
            domain_dropout_probability=0.03,
            seed=42,
            strict_time_filter=True,
        )
    )

    assert config.cache.mode == "opt"
    assert config.cache.max_batches == 512
    assert [transform.name for transform in config.transforms] == [
        "sequence_crop",
        "feature_mask",
        "domain_dropout",
    ]
    assert config.seed == 42


def test_cache_preset_uses_lru_cache() -> None:
    config = _build_pipeline_config(
        Namespace(
            pipeline_preset="cache",
            cache_batches=64,
            views_per_row=2,
            seq_window_min_len=8,
            feature_mask_probability=0.03,
            domain_dropout_probability=0.03,
            seed=42,
            strict_time_filter=True,
        )
    )

    assert config.cache.mode == "lru"
    assert config.cache.max_batches == 64
    assert config.transforms == ()


def test_cache_mode_overrides_preset_default() -> None:
    config = _build_pipeline_config(
        Namespace(
            pipeline_preset="cache",
            cache_mode="lfu",
            cache_batches=64,
            views_per_row=2,
            seq_window_min_len=8,
            feature_mask_probability=0.03,
            domain_dropout_probability=0.03,
            seed=42,
            strict_time_filter=True,
        )
    )

    assert config.cache.mode == "lfu"
    assert config.cache.max_batches == 64
    assert config.transforms == ()


def test_estimated_batches_per_pass_accounts_for_sequence_crop_views() -> None:
    config = _build_pipeline_config(
        Namespace(
            pipeline_preset="opt-augment",
            cache_batches=512,
            views_per_row=3,
            seq_window_min_len=8,
            feature_mask_probability=0.03,
            domain_dropout_probability=0.03,
            seed=42,
            strict_time_filter=True,
        )
    )

    assert _estimated_batches_per_pass(
        train_rows=1000,
        batch_size=128,
        pipeline_config=config,
    ) == 24


def test_max_batches_consumes_single_continuous_iterator() -> None:
    class Loader:
        def __init__(self) -> None:
            self.iterator_count = 0

        def __iter__(self):
            self.iterator_count += 1
            for _ in range(10):
                yield {"label": type("Label", (), {"shape": (3,)})()}

    loader = Loader()

    rows, batches, measured_passes = _consume_measured_batches(
        loader,
        batches_per_pass=2,
        passes=3,
        max_batches=5,
    )

    assert rows == 15
    assert batches == 5
    assert measured_passes == 3
    assert loader.iterator_count == 1
