from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from config.baseline.data import load_dataloaders
from taac2026.domain.config import DataConfig, ModelConfig, TrainConfig
from taac2026.domain.experiment import ExperimentSpec
from taac2026.domain.types import BatchTensors


# ---------------------------------------------------------------------------
# Schema contract: authoritative column-name expectations for the flat-column
# dataset format.  When the upstream HuggingFace schema changes, update these
# constants FIRST — any mismatch with ``build_row()`` or ``data.py`` will make
# the schema-contract tests fail immediately.
# ---------------------------------------------------------------------------

EXPECTED_SCALAR_COLUMNS: frozenset[str] = frozenset({
    "user_id",
    "item_id",
    "timestamp",
    "label_type",
    "label_time",
})

EXPECTED_USER_INT_PREFIX = "user_int_feats_"
EXPECTED_ITEM_INT_PREFIX = "item_int_feats_"

EXPECTED_DOMAIN_PREFIXES: dict[str, str] = {
    "domain_a": "domain_a_seq_",
    "domain_b": "domain_b_seq_",
    "domain_c": "domain_c_seq_",
    "domain_d": "domain_d_seq_",
}


def build_row(index: int, timestamp: int, positive: bool, user_id: str, item_id: int) -> dict[str, object]:
    return {
        "user_id": user_id,
        "item_id": item_id,
        "timestamp": timestamp,
        "label_type": 2 if positive else 1,
        "label_time": timestamp,
        # User int features
        "user_int_feats_8": (index % 2) + 1,
        # Item int features
        "item_int_feats_70": (index % 3) + 1,
        "item_int_feats_17": (index % 4) + 1,
        # Domain A sequences (post-like + timestamp)
        "domain_a_seq_11": [index + 1, index + 2, index + 3],
        "domain_a_seq_99": [timestamp - 30, timestamp - 20, timestamp - 10],
        # Domain B sequences (post-like + timestamp)
        "domain_b_seq_12": [index + 4, index + 5, index + 6],
        "domain_b_seq_99": [timestamp - 35, timestamp - 25, timestamp - 15],
        # Domain C sequences (post-like + timestamp)
        "domain_c_seq_13": [index + 7, index + 8, index + 9],
        "domain_c_seq_99": [timestamp - 40, timestamp - 30, timestamp - 20],
        # Domain D sequences (post-like + timestamp)
        "domain_d_seq_14": [index + 10, index + 11, index + 12],
        "domain_d_seq_99": [timestamp - 45, timestamp - 35, timestamp - 25],
    }


def write_dataset(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, row_group_size=max(1, min(2, len(rows))))


def build_edge_case_rows() -> list[dict[str, object]]:
    base_timestamp = 1_770_000_000
    sparse_row = build_row(0, base_timestamp + 90, True, "u_sparse", 301)
    for key in list(sparse_row.keys()):
        if key.startswith(("user_int_feats_", "item_int_feats_", "domain_")):
            sparse_row[key] = None

    short_sequence_row = build_row(1, base_timestamp + 120, False, "u_short", 302)
    for key in list(short_sequence_row.keys()):
        if key.startswith("domain_"):
            short_sequence_row[key] = None
    short_sequence_row["domain_a_seq_11"] = [41]
    short_sequence_row["domain_a_seq_99"] = [base_timestamp + 30]

    long_sequence_row = build_row(2, base_timestamp + 400, True, "u_long", 301)
    long_sequence_row["domain_a_seq_11"] = list(range(100, 108))
    long_sequence_row["domain_a_seq_99"] = [base_timestamp + offset for offset in range(8)]
    long_sequence_row["domain_b_seq_12"] = list(range(200, 208))
    long_sequence_row["domain_b_seq_99"] = [base_timestamp - 50 + offset for offset in range(8)]

    cold_item_row = build_row(3, base_timestamp + 500, False, "u_cold", 999)
    cold_item_row["label_type"] = 0
    return [sparse_row, short_sequence_row, long_sequence_row, cold_item_row]


def write_sample_dataset(path: Path, rows: list[dict[str, Any]] | None = None) -> None:
    base_timestamp = 1_770_000_000
    materialized_rows = rows or [
        build_row(0, base_timestamp + 300, True, "u1", 101),
        build_row(1, base_timestamp + 100, False, "u1", 102),
        build_row(2, base_timestamp + 500, True, "u2", 103),
        build_row(3, base_timestamp + 200, False, "u3", 101),
        build_row(4, base_timestamp + 600, True, "u2", 104),
        build_row(5, base_timestamp + 400, False, "u4", 105),
    ]
    write_dataset(path, materialized_rows)


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).float()
    summed = (tokens * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp_min(1.0)
    return summed / counts


def require_sequence_features(batch: BatchTensors) -> dict[str, Any]:
    if batch.sequence_features is None:
        raise RuntimeError("Batch is missing sequence_features")
    return batch.sequence_features.to_dict()


def require_sparse_features(batch: BatchTensors) -> dict[str, Any]:
    if batch.sparse_features is None:
        raise RuntimeError("Batch is missing sparse_features")
    return batch.sparse_features.to_dict()


def dense_sparse_tokens(
    sparse_by_key: dict[str, Any],
    name: str,
    desired_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    jagged = sparse_by_key[name]
    tokens = jagged.to_padded_dense(desired_length=desired_length, padding_value=0).to(dtype=torch.long)
    lengths = jagged.lengths().to(device=tokens.device)
    positions = torch.arange(desired_length, device=tokens.device).unsqueeze(0)
    return tokens, positions < lengths.unsqueeze(1)


def dense_sequence_tokens(
    sequence_by_key: dict[str, Any],
    name: str,
    desired_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    jagged = sequence_by_key[name]
    tokens = jagged.to_padded_dense(desired_length=desired_length, padding_value=0).to(dtype=torch.long)
    lengths = jagged.lengths().to(device=tokens.device)
    positions = torch.arange(desired_length, device=tokens.device).unsqueeze(0)
    return tokens, positions < lengths.unsqueeze(1)


def dense_sequence_grid(
    sequence_by_key: dict[str, Any],
    sequence_names: tuple[str, ...],
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_tokens: list[torch.Tensor] = []
    sequence_mask: list[torch.Tensor] = []
    for sequence_name in sequence_names:
        tokens, mask = dense_sequence_tokens(sequence_by_key, "sequence:" + sequence_name, max_seq_len)
        sequence_tokens.append(tokens)
        sequence_mask.append(mask)
    return torch.stack(sequence_tokens, dim=1), torch.stack(sequence_mask, dim=1)


class TinyExperimentModel(nn.Module):
    def __init__(self, data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> None:
        super().__init__()
        self.sequence_count = len(data_config.sequence_names)
        self.sequence_names = tuple(data_config.sequence_names)
        self.max_seq_len = data_config.max_seq_len
        self.max_feature_tokens = data_config.max_feature_tokens
        self.history_capacity = self.sequence_count * data_config.max_seq_len
        self.token_embedding = nn.Embedding(model_config.vocab_size, model_config.embedding_dim)
        self.token_projection = nn.Sequential(
            nn.Linear(model_config.embedding_dim, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        if self.sequence_count > 1:
            self.group_projection = nn.Sequential(
                nn.Linear(model_config.hidden_dim * self.sequence_count, model_config.hidden_dim),
                nn.LayerNorm(model_config.hidden_dim),
                nn.GELU(),
            )
        else:
            self.group_projection = nn.Identity()
        head_hidden_dim = model_config.head_hidden_dim or model_config.hidden_dim * 2
        self.output = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 7, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def _encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_projection(self.token_embedding(tokens))

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        sparse_by_key = require_sparse_features(batch)
        sequence_by_key = require_sequence_features(batch)
        candidate_tokens, candidate_mask = dense_sparse_tokens(sparse_by_key, "candidate_tokens", 1)
        context_tokens, context_mask = dense_sparse_tokens(sparse_by_key, "context_tokens", self.max_feature_tokens)
        history_tokens, history_mask = dense_sequence_tokens(sequence_by_key, "history_tokens", self.history_capacity)
        sequence_tokens, sequence_mask = dense_sequence_grid(sequence_by_key, self.sequence_names, self.max_seq_len)
        candidate = masked_mean(self._encode_tokens(candidate_tokens), candidate_mask)
        context = masked_mean(self._encode_tokens(context_tokens), context_mask)
        history = masked_mean(self._encode_tokens(history_tokens), history_mask)

        batch_size, sequence_count, sequence_len = sequence_tokens.shape
        flat_tokens = sequence_tokens.reshape(batch_size * sequence_count, sequence_len)
        flat_mask = sequence_mask.reshape(batch_size * sequence_count, sequence_len)
        flat_summary = masked_mean(self._encode_tokens(flat_tokens), flat_mask)
        grouped = self.group_projection(flat_summary.reshape(batch_size, sequence_count, -1).reshape(batch_size, -1))

        dense = self.dense_projection(batch.dense_features)
        interaction = candidate * history
        contrast = torch.abs(candidate - grouped)
        contextual = 0.5 * (context + dense)
        fused = torch.cat(
            [candidate, context, history, grouped, dense, interaction, contrast + contextual],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


class DisabledAuxiliaryLoss:
    enabled = False
    requires_aux = False


def build_local_data_pipeline(data_config, model_config, train_config):
    return load_dataloaders(
        config=data_config,
        vocab_size=model_config.vocab_size,
        batch_size=train_config.batch_size,
        eval_batch_size=train_config.resolved_eval_batch_size,
        num_workers=train_config.num_workers,
        seed=train_config.seed,
    )


def build_local_model_component(data_config, model_config, dense_dim):
    return TinyExperimentModel(data_config=data_config, model_config=model_config, dense_dim=dense_dim)


def build_local_loss_stack(data_config, model_config, train_config, data_stats, device):
    _ = data_config
    _ = model_config
    _ = train_config
    pos_weight = torch.tensor([data_stats.pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight), DisabledAuxiliaryLoss()


def build_local_optimizer_component(model, train_config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )


@dataclass(slots=True)
class TestWorkspace:
    __test__ = False

    root: Path
    dataset_path: Path
    data_config: DataConfig
    model_kwargs: dict[str, int | float]

    def write_experiment_package(
        self,
        *,
        hidden_dim: int = 16,
        embedding_dim: int = 16,
        num_heads: int = 4,
        switches: dict[str, bool] | None = None,
    ) -> Path:
        output_dir = self.root / "outputs"
        package_path = self.root / f"experiment_h{hidden_dim}"
        package_path.mkdir(parents=True, exist_ok=True)
        rendered_switches = switches or {"logging": False, "visualization": True}
        source = f'''
from __future__ import annotations

import torch
from torch import nn

from config.baseline.data import load_dataloaders
from taac2026.domain.config import DataConfig, ModelConfig, TrainConfig
from taac2026.domain.experiment import ExperimentSpec
from taac2026.domain.types import BatchTensors


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).float()
    summed = (tokens * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp_min(1.0)
    return summed / counts


def _require_sequence_features(batch: BatchTensors):
    if batch.sequence_features is None:
        raise RuntimeError("Batch is missing sequence_features")
    return batch.sequence_features.to_dict()


def _require_sparse_features(batch: BatchTensors):
    if batch.sparse_features is None:
        raise RuntimeError("Batch is missing sparse_features")
    return batch.sparse_features.to_dict()


def _dense_sparse_tokens(sparse_by_key, name: str, desired_length: int):
    jagged = sparse_by_key[name]
    tokens = jagged.to_padded_dense(desired_length=desired_length, padding_value=0).to(dtype=torch.long)
    lengths = jagged.lengths().to(device=tokens.device)
    positions = torch.arange(desired_length, device=tokens.device).unsqueeze(0)
    return tokens, positions < lengths.unsqueeze(1)


def _dense_sequence_tokens(sequence_by_key, name: str, desired_length: int):
    jagged = sequence_by_key[name]
    tokens = jagged.to_padded_dense(desired_length=desired_length, padding_value=0).to(dtype=torch.long)
    lengths = jagged.lengths().to(device=tokens.device)
    positions = torch.arange(desired_length, device=tokens.device).unsqueeze(0)
    return tokens, positions < lengths.unsqueeze(1)


def _dense_sequence_grid(sequence_by_key, sequence_names, max_seq_len: int):
    sequence_tokens = []
    sequence_mask = []
    for sequence_name in sequence_names:
        tokens, mask = _dense_sequence_tokens(sequence_by_key, "sequence:" + sequence_name, max_seq_len)
        sequence_tokens.append(tokens)
        sequence_mask.append(mask)
    return torch.stack(sequence_tokens, dim=1), torch.stack(sequence_mask, dim=1)


class _LocalExperimentModel(nn.Module):
    def __init__(self, data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> None:
        super().__init__()
        self.sequence_count = len(data_config.sequence_names)
        self.sequence_names = tuple(data_config.sequence_names)
        self.max_seq_len = data_config.max_seq_len
        self.max_feature_tokens = data_config.max_feature_tokens
        self.history_capacity = self.sequence_count * data_config.max_seq_len
        self.token_embedding = nn.Embedding(model_config.vocab_size, model_config.embedding_dim)
        self.token_projection = nn.Sequential(
            nn.Linear(model_config.embedding_dim, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        if self.sequence_count > 1:
            self.group_projection = nn.Sequential(
                nn.Linear(model_config.hidden_dim * self.sequence_count, model_config.hidden_dim),
                nn.LayerNorm(model_config.hidden_dim),
                nn.GELU(),
            )
        else:
            self.group_projection = nn.Identity()
        head_hidden_dim = model_config.head_hidden_dim or model_config.hidden_dim * 2
        self.output = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 7, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def _encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_projection(self.token_embedding(tokens))

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        sparse_by_key = _require_sparse_features(batch)
        sequence_by_key = _require_sequence_features(batch)
        candidate_tokens, candidate_mask = _dense_sparse_tokens(sparse_by_key, "candidate_tokens", 1)
        context_tokens, context_mask = _dense_sparse_tokens(sparse_by_key, "context_tokens", self.max_feature_tokens)
        history_tokens, history_mask = _dense_sequence_tokens(sequence_by_key, "history_tokens", self.history_capacity)
        sequence_tokens, sequence_mask = _dense_sequence_grid(sequence_by_key, self.sequence_names, self.max_seq_len)
        candidate = _masked_mean(self._encode_tokens(candidate_tokens), candidate_mask)
        context = _masked_mean(self._encode_tokens(context_tokens), context_mask)
        history = _masked_mean(self._encode_tokens(history_tokens), history_mask)
        batch_size, sequence_count, sequence_len = sequence_tokens.shape
        flat_tokens = sequence_tokens.reshape(batch_size * sequence_count, sequence_len)
        flat_mask = sequence_mask.reshape(batch_size * sequence_count, sequence_len)
        flat_summary = _masked_mean(self._encode_tokens(flat_tokens), flat_mask)
        grouped = self.group_projection(flat_summary.reshape(batch_size, sequence_count, -1).reshape(batch_size, -1))
        dense = self.dense_projection(batch.dense_features)
        interaction = candidate * history
        contrast = torch.abs(candidate - grouped)
        contextual = 0.5 * (context + dense)
        fused = torch.cat([candidate, context, history, grouped, dense, interaction, contrast + contextual], dim=-1)
        return self.output(fused).squeeze(-1)


class _DisabledAuxiliaryLoss:
    enabled = False
    requires_aux = False


def build_data_pipeline(data_config, model_config, train_config):
    return load_dataloaders(
        config=data_config,
        vocab_size=model_config.vocab_size,
        batch_size=train_config.batch_size,
        eval_batch_size=train_config.resolved_eval_batch_size,
        num_workers=train_config.num_workers,
        seed=train_config.seed,
    )


def build_model_component(data_config, model_config, dense_dim):
    return _LocalExperimentModel(data_config=data_config, model_config=model_config, dense_dim=dense_dim)


def build_loss_stack(data_config, model_config, train_config, data_stats, device):
    _ = data_config
    _ = model_config
    _ = train_config
    pos_weight = torch.tensor([data_stats.pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight), _DisabledAuxiliaryLoss()


def build_optimizer_component(model, train_config):
    return torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)


EXPERIMENT = ExperimentSpec(
    name="temp_experiment",
    data=DataConfig(
        dataset_path={str(self.dataset_path)!r},
        max_seq_len=4,
        max_feature_tokens=8,
        max_event_features=4,
        stream_batch_rows=2,
        val_ratio=0.34,
        label_action_type=2,
    ),
    model=ModelConfig(
        name="temp_experiment",
        vocab_size=257,
        embedding_dim={embedding_dim},
        hidden_dim={hidden_dim},
        dropout=0.0,
        num_layers=1,
        num_heads={num_heads},
        recent_seq_len=2,
        memory_slots=2,
        ffn_multiplier=2,
        feature_cross_layers=1,
        sequence_layers=1,
        static_layers=1,
        query_decoder_layers=1,
        fusion_layers=1,
        num_queries=2,
        head_hidden_dim={hidden_dim},
        segment_count=4,
    ),
    train=TrainConfig(
        seed=7,
        epochs=1,
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        output_dir={str(output_dir)!r},
        latency_warmup_steps=0,
        latency_measure_steps=1,
    ),
    build_data_pipeline=build_data_pipeline,
    build_model_component=build_model_component,
    build_loss_stack=build_loss_stack,
    build_optimizer_component=build_optimizer_component,
    switches={rendered_switches!r},
)
'''
        (package_path / "__init__.py").write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
        return package_path


def create_test_workspace(tmp_path: Path, *, rows: list[dict[str, Any]] | None = None) -> TestWorkspace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset_path = tmp_path / "sample.parquet"
    write_sample_dataset(dataset_path, rows=rows)
    data_config = DataConfig(
        dataset_path=str(dataset_path),
        max_seq_len=4,
        max_feature_tokens=8,
        max_event_features=4,
        stream_batch_rows=2,
        val_ratio=0.34,
    )
    model_kwargs: dict[str, int | float] = {
        "vocab_size": 257,
        "embedding_dim": 16,
        "hidden_dim": 16,
        "dropout": 0.0,
        "num_layers": 1,
        "num_heads": 4,
        "recent_seq_len": 2,
        "memory_slots": 2,
        "ffn_multiplier": 2,
        "feature_cross_layers": 1,
        "sequence_layers": 1,
        "static_layers": 1,
        "query_decoder_layers": 1,
        "fusion_layers": 1,
        "num_queries": 2,
        "head_hidden_dim": 16,
        "segment_count": 4,
    }
    return TestWorkspace(
        root=tmp_path,
        dataset_path=dataset_path,
        data_config=data_config,
        model_kwargs=model_kwargs,
    )


def prepare_experiment(experiment, test_workspace: TestWorkspace):
    experiment = experiment.clone()
    experiment.data.dataset_path = str(test_workspace.dataset_path)
    experiment.data.max_seq_len = test_workspace.data_config.max_seq_len
    experiment.data.max_feature_tokens = test_workspace.data_config.max_feature_tokens
    experiment.data.max_event_features = test_workspace.data_config.max_event_features
    experiment.data.stream_batch_rows = test_workspace.data_config.stream_batch_rows
    experiment.data.val_ratio = test_workspace.data_config.val_ratio
    experiment.train.batch_size = 2
    experiment.train.eval_batch_size = 2
    experiment.train.num_workers = 0
    return experiment


__all__ = [
    "DisabledAuxiliaryLoss",
    "EXPECTED_DOMAIN_PREFIXES",
    "EXPECTED_ITEM_INT_PREFIX",
    "EXPECTED_SCALAR_COLUMNS",
    "EXPECTED_USER_INT_PREFIX",
    "TestWorkspace",
    "TinyExperimentModel",
    "build_edge_case_rows",
    "build_local_data_pipeline",
    "build_local_loss_stack",
    "build_local_model_component",
    "build_local_optimizer_component",
    "build_row",
    "create_test_workspace",
    "prepare_experiment",
    "write_dataset",
    "write_sample_dataset",
]
