"""PCVR Baseline+ experiment package."""

from __future__ import annotations

from pathlib import Path

from taac2026.api import (
    PCVRDataCacheConfig,
    PCVRDataConfig,
    PCVRDataPipelineConfig,
    PCVRDomainDropoutConfig,
    PCVRFeatureMaskConfig,
    PCVRModelConfig,
    PCVRNSConfig,
    PCVROptimizerConfig,
    PCVRSequenceCropConfig,
    PCVRSparseOptimizerConfig,
    PCVRTrainConfig,
)
from taac2026.api import create_pcvr_experiment
from taac2026.api import BinaryClassificationLossConfig, RuntimeExecutionConfig

NS_GROUP_METADATA = {
    "_purpose": "Shared PCVR non-sequential feature grouping for this experiment package. The concrete fid lists match the official TAAC PCVR schema and are converted to schema-entry indices at runtime.",
    "_format": "Top-level keys: user_ns_groups and item_ns_groups. Each group maps a semantic group name to a list of feature ids, using the numeric suffix from user_int_feats_{fid} or item_int_feats_{fid}.",
    "_usage": "The shared PCVR runtime reads this explicit config from PCVRNSConfig in __init__.py, and train_config.json persists the same grouping for evaluation and inference.",
    "_comment": "NS token groups for parquet data. Values are fids (the numeric suffix in column names, e.g. user_int_feats_{fid}); runtime converts them to schema entry indices at load time.",
    "_note_T": "For the official schema, num_ns = 7 user-int groups + 1 user-dense token + 4 item-int groups = 12 before model-specific tokenizer changes.",
    "_note_user_dense": "All user dense features are concatenated and projected to one dense NS token. They are not configured in this mapping.",
    "_note_shared_fids": "Fids 62/63/64/65/66/89/90/91 appear in both user_int_feats and user_dense_feats. The int and float parts jointly describe the same underlying signal. The int parts are grouped below; the float parts are included in the single user_dense NS token.",
}

USER_NS_GROUPS = {
    "U1": [1, 15],
    "U2": [48, 49, 89, 90, 91],
    "U3": [80],
    "U4": [51, 52, 53, 54, 86],
    "U5": [82, 92, 93],
    "U6": [50, 60, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    "U7": [3, 4, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66],
}

ITEM_NS_GROUPS = {
    "I1": [11, 13],
    "I2": [5, 6, 7, 8, 12],
    "I3": [16, 81, 83, 84, 85],
    "I4": [9, 10],
}

TRAIN_DEFAULTS = PCVRTrainConfig(
    data=PCVRDataConfig(
        batch_size=256,
        num_workers=8,
        buffer_batches=20,
        train_ratio=1.0,
        valid_ratio=0.1,
        eval_every_n_steps=0,
        seq_max_lens="seq_a:256,seq_b:256,seq_c:512,seq_d:512",
    ),
    data_pipeline=PCVRDataPipelineConfig(
        cache=PCVRDataCacheConfig(mode="opt", max_batches=512),
        transforms=(
            PCVRSequenceCropConfig(
                views_per_row=2,
                seq_window_mode="random_tail",
                seq_window_min_len=8,
            ),
            PCVRFeatureMaskConfig(probability=0.03),
            PCVRDomainDropoutConfig(probability=0.03),
        ),
        seed=42,
        strict_time_filter=True,
    ),
    optimizer=PCVROptimizerConfig(
        lr=1e-4,
        max_steps=0,
        patience=5,
        seed=42,
        device=None,
        dense_optimizer_type="adamw",
        scheduler_type="none",
        warmup_steps=0,
        min_lr_ratio=0.0,
    ),
    runtime=RuntimeExecutionConfig(amp=False, amp_dtype="bfloat16", compile=False),
    loss=BinaryClassificationLossConfig(
        loss_type="bce",
        focal_alpha=0.1,
        focal_gamma=2.0,
        pairwise_auc_weight=0.0,
        pairwise_auc_temperature=1.0,
    ),
    sparse_optimizer=PCVRSparseOptimizerConfig(
        sparse_lr=0.05,
        sparse_weight_decay=0.0,
        reinit_sparse_every_n_steps=0,
        reinit_cardinality_threshold=0,
    ),
    model=PCVRModelConfig(
        d_model=64,
        emb_dim=64,
        num_queries=2,
        num_blocks=2,
        num_heads=4,
        seq_encoder_type="transformer",
        hidden_mult=4,
        dropout_rate=0.01,
        seq_top_k=50,
        seq_causal=False,
        action_num=1,
        use_time_buckets=True,
        rank_mixer_mode="full",
        use_rope=False,
        rope_base=10000.0,
        emb_skip_threshold=1_000_000,
        seq_id_threshold=10000,
        gradient_checkpointing=False,
        flash_attention_backend="tilelang",
        rms_norm_backend="tilelang",
        rms_norm_block_rows=8,
    ),
    ns=PCVRNSConfig(
        grouping_strategy="explicit",
        metadata=NS_GROUP_METADATA,
        user_groups=USER_NS_GROUPS,
        item_groups=ITEM_NS_GROUPS,
        tokenizer_type="rankmixer",
        user_tokens=5,
        item_tokens=2,
    ),
)

EXPERIMENT = create_pcvr_experiment(
    name="pcvr_baseline_plus",
    package_dir=Path(__file__).resolve().parent,
    model_class_name="PCVRBaselinePlus",
    train_defaults=TRAIN_DEFAULTS,
)
TRAIN_HOOKS = EXPERIMENT.train_hooks
PREDICTION_HOOKS = EXPERIMENT.prediction_hooks
RUNTIME_HOOKS = EXPERIMENT.runtime_hooks

__all__ = ["EXPERIMENT", "PREDICTION_HOOKS", "RUNTIME_HOOKS", "TRAIN_DEFAULTS", "TRAIN_HOOKS"]
