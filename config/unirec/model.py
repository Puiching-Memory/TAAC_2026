from __future__ import annotations

import math

import torch
from torch import nn

from taac2026.domain.config import DataConfig, ModelConfig
from taac2026.domain.features import build_default_feature_schema
from taac2026.domain.types import BatchTensors
from taac2026.infrastructure.nn.embedding import TorchRecEmbeddingBagAdapter
from taac2026.infrastructure.nn.hstu import (
	BlockAttnRes,
	HSTUBlock as UnifiedTransducerBlock,
	MixtureOfTransducers,
	build_unified_attention_mask,
)
from taac2026.infrastructure.nn.transformer import TaacTransformerBlock

from .data import TIME_GAP_BUCKET_COUNT


SPARSE_TABLE_NAMES = (
	"user_tokens",
	"context_tokens",
	"candidate_tokens",
	"candidate_post_tokens",
	"candidate_author_tokens",
)

SEQUENCE_FEATURE_KEYS = (
	"history_tokens",
	"history_post_tokens",
	"history_author_tokens",
	"history_action_tokens",
	"history_time_gap",
	"history_group_ids",
)


class UniRecModel(nn.Module):
	def __init__(self, data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> None:
		super().__init__()
		self.data_config = data_config
		self.model_config = model_config
		self.hidden_dim = model_config.hidden_dim
		self.sequence_count = len(data_config.sequence_names)
		self.history_capacity = self.sequence_count * data_config.max_seq_len
		self.feature_token_names = (
			"user",
			"context",
			"dense",
			"candidate",
			"candidate_post",
			"candidate_author",
		)
		self.n_feature_tokens = len(self.feature_token_names)
		self.global_window = max(128, self.n_feature_tokens + 8)
		self.local_window = 128
		self.truncated_seq_len = max(0, model_config.recent_seq_len)
		self.attn_res_block_size = max(1, model_config.memory_slots)
		self.sparse_embedding = TorchRecEmbeddingBagAdapter(
			feature_schema=build_default_feature_schema(data_config, model_config),
			table_names=SPARSE_TABLE_NAMES,
		)

		self.token_embedding = nn.Embedding(
			num_embeddings=model_config.vocab_size,
			embedding_dim=model_config.embedding_dim,
			padding_idx=0,
		)
		self.token_projection = (
			nn.Identity()
			if model_config.embedding_dim == model_config.hidden_dim
			else nn.Linear(model_config.embedding_dim, model_config.hidden_dim)
		)
		self.dense_projection = nn.Sequential(
			nn.Linear(dense_dim, model_config.hidden_dim),
			nn.LayerNorm(model_config.hidden_dim),
			nn.SiLU(),
		)
		self.feature_projections = nn.ModuleDict(
			{
				name: nn.Sequential(
					nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
					nn.LayerNorm(model_config.hidden_dim),
					nn.SiLU(),
				)
				for name in self.feature_token_names
			}
		)
		self.field_embedding = nn.Embedding(self.n_feature_tokens + 8, model_config.hidden_dim)
		self.segment_embedding = nn.Embedding(self.sequence_count + 2, model_config.hidden_dim)
		self.history_group_embedding = nn.Embedding(self.sequence_count + 1, model_config.hidden_dim, padding_idx=0)
		self.sequence_id_embedding = nn.Embedding(self.sequence_count + 1, model_config.hidden_dim, padding_idx=0)
		self.sequence_position_embedding = nn.Embedding(data_config.max_seq_len, model_config.hidden_dim)
		self.time_gap_embedding = nn.Embedding(TIME_GAP_BUCKET_COUNT + 1, model_config.hidden_dim, padding_idx=0)
		self.event_projection = nn.Sequential(
			nn.LayerNorm(model_config.hidden_dim * 6),
			nn.Linear(model_config.hidden_dim * 6, model_config.hidden_dim * 4),
			nn.SiLU(),
			nn.Dropout(model_config.dropout),
			nn.Linear(model_config.hidden_dim * 4, model_config.hidden_dim),
		)
		self.empty_sequence_tokens = nn.Parameter(torch.randn(self.sequence_count, model_config.hidden_dim) * 0.02)
		feature_cross_depth = max(1, model_config.feature_cross_layers)
		self.feature_cross_layers = nn.ModuleList(
			[
				TaacTransformerBlock(
					hidden_dim=model_config.hidden_dim,
					num_heads=model_config.num_heads,
					ffn_dim=int(model_config.hidden_dim * model_config.ffn_multiplier),
					dropout=model_config.dropout,
					attention_dropout=model_config.attention_dropout,
					norm_type="layernorm",
					ffn_type="gelu",
					attention_type="standard",
				)
				for _ in range(feature_cross_depth)
			]
		)
		self.interest_query = nn.Linear(model_config.hidden_dim, model_config.hidden_dim, bias=False)
		self.interest_key = nn.Linear(model_config.hidden_dim, model_config.hidden_dim, bias=False)
		self.interest_value = nn.Linear(model_config.hidden_dim, model_config.hidden_dim, bias=False)
		self.interest_out = nn.Sequential(
			nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
			nn.SiLU(),
			nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
		)
		self.target_fusion = nn.Sequential(
			nn.Linear(model_config.hidden_dim * 3, model_config.hidden_dim),
			nn.SiLU(),
			nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
		)

		mot_layers = max(0, model_config.sequence_layers)
		self.mot = None
		self.mot_projection = None
		if mot_layers > 0:
			self.mot = MixtureOfTransducers(model_config.hidden_dim, model_config.num_heads, mot_layers, model_config.dropout, self.sequence_count)
			self.mot_projection = nn.Linear(model_config.hidden_dim, model_config.hidden_dim)

		full_layers = model_config.static_layers if model_config.static_layers > 0 else max(1, model_config.num_layers)
		truncated_layers = model_config.fusion_layers if model_config.fusion_layers > 0 else max(0, model_config.num_layers - full_layers)
		self.full_blocks = nn.ModuleList(
			[
				UnifiedTransducerBlock(
					hidden_dim=model_config.hidden_dim,
					num_heads=model_config.num_heads,
					ffn_dim=int(model_config.hidden_dim * model_config.ffn_multiplier),
					dropout=model_config.dropout,
				)
				for _ in range(full_layers)
			]
		)
		self.truncated_blocks = nn.ModuleList(
			[
				UnifiedTransducerBlock(
					hidden_dim=model_config.hidden_dim,
					num_heads=model_config.num_heads,
					ffn_dim=int(model_config.hidden_dim * model_config.ffn_multiplier),
					dropout=model_config.dropout,
				)
				for _ in range(truncated_layers)
			]
		)
		self.block_attn_res = BlockAttnRes(model_config.hidden_dim, len(self.full_blocks) + len(self.truncated_blocks))
		self.final_norm = nn.LayerNorm(model_config.hidden_dim)
		head_hidden_dim = model_config.head_hidden_dim or model_config.hidden_dim * 2
		self.head = nn.Sequential(
			nn.Linear(model_config.hidden_dim, head_hidden_dim),
			nn.SiLU(),
			nn.Dropout(model_config.dropout),
			nn.Linear(head_hidden_dim, head_hidden_dim // 2),
			nn.SiLU(),
			nn.Dropout(model_config.dropout),
			nn.Linear(head_hidden_dim // 2, 1),
		)
		self._init_weights()

	def _init_weights(self) -> None:
		for module in self.modules():
			if isinstance(module, nn.Linear):
				nn.init.xavier_uniform_(module.weight)
				if module.bias is not None:
					nn.init.zeros_(module.bias)
			elif isinstance(module, nn.Embedding):
				nn.init.normal_(module.weight, std=0.02)
				if module.padding_idx is not None:
					nn.init.zeros_(module.weight[module.padding_idx])

	def _require(self, tensor: torch.Tensor | None, name: str) -> torch.Tensor:
		if tensor is None:
			raise RuntimeError(f"Batch is missing required tensor: {name}")
		return tensor

	def _require_sparse_features(self, batch: BatchTensors):
		if batch.sparse_features is None:
			raise RuntimeError("Batch is missing required TorchRec sparse feature tensor: sparse_features")
		return batch.sparse_features

	def _require_sequence_features(self, batch: BatchTensors):
		if batch.sequence_features is None:
			raise RuntimeError("Batch is missing required TorchRec sparse feature tensor: sequence_features")
		return batch.sequence_features

	def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
		return self.token_projection(self.token_embedding(tokens))

	def _dense_sequence_tokens(self, sequence_by_key, name: str) -> tuple[torch.Tensor, torch.Tensor]:
		jagged = sequence_by_key[name]
		tokens = jagged.to_padded_dense(desired_length=self.history_capacity, padding_value=0).to(dtype=torch.long)
		lengths = jagged.lengths().to(device=tokens.device)
		positions = torch.arange(self.history_capacity, device=tokens.device).unsqueeze(0)
		mask = positions < lengths.unsqueeze(1)
		return tokens, mask

	def _feature_summaries(self, batch: BatchTensors) -> dict[str, torch.Tensor]:
		pooled_sparse = self.sparse_embedding.forward_dict(self._require_sparse_features(batch))
		return {
			"user": self.feature_projections["user"](self.token_projection(pooled_sparse["user_tokens"])),
			"context": self.feature_projections["context"](self.token_projection(pooled_sparse["context_tokens"])),
			"dense": self.feature_projections["dense"](self.dense_projection(batch.dense_features)),
			"candidate": self.feature_projections["candidate"](self.token_projection(pooled_sparse["candidate_tokens"])),
			"candidate_post": self.feature_projections["candidate_post"](self.token_projection(pooled_sparse["candidate_post_tokens"])),
			"candidate_author": self.feature_projections["candidate_author"](self.token_projection(pooled_sparse["candidate_author_tokens"])),
		}

	def _build_feature_tokens(self, batch: BatchTensors) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		summaries = self._feature_summaries(batch)
		feature_tokens = torch.stack(
			[summaries[name] for name in self.feature_token_names],
			dim=1,
		)
		field_ids = torch.arange(self.n_feature_tokens, device=feature_tokens.device).unsqueeze(0).expand(batch.batch_size, -1)
		feature_tokens = feature_tokens + self.field_embedding(field_ids)
		feature_mask = torch.ones(batch.batch_size, self.n_feature_tokens, dtype=torch.bool, device=feature_tokens.device)
		for layer in self.feature_cross_layers:
			feature_tokens = layer(feature_tokens, feature_mask)

		user_pool = feature_tokens[:, :3, :].mean(dim=1)
		item_pool = feature_tokens[:, 3:, :].mean(dim=1)
		return feature_tokens, user_pool, item_pool

	def _build_event_tokens(self, batch: BatchTensors) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		sequence_by_key = self._require_sequence_features(batch).to_dict()
		missing_keys = [name for name in SEQUENCE_FEATURE_KEYS if name not in sequence_by_key]
		if missing_keys:
			missing = ", ".join(missing_keys)
			raise RuntimeError(f"Batch sequence_features is missing required keys: {missing}")

		history_tokens, history_mask = self._dense_sequence_tokens(sequence_by_key, "history_tokens")
		history_post_tokens, _ = self._dense_sequence_tokens(sequence_by_key, "history_post_tokens")
		history_author_tokens, _ = self._dense_sequence_tokens(sequence_by_key, "history_author_tokens")
		history_action_tokens, _ = self._dense_sequence_tokens(sequence_by_key, "history_action_tokens")
		history_time_gap, _ = self._dense_sequence_tokens(sequence_by_key, "history_time_gap")
		history_group_ids, _ = self._dense_sequence_tokens(sequence_by_key, "history_group_ids")

		history_hidden = self._embed_tokens(history_tokens)
		post_hidden = self._embed_tokens(history_post_tokens)
		author_hidden = self._embed_tokens(history_author_tokens)
		action_hidden = self._embed_tokens(history_action_tokens)
		gap_hidden = self.time_gap_embedding(history_time_gap.clamp(min=0, max=TIME_GAP_BUCKET_COUNT))
		group_hidden = self.history_group_embedding(history_group_ids)
		event_inputs = torch.cat(
			[
				history_hidden,
				post_hidden,
				author_hidden,
				action_hidden,
				gap_hidden,
				group_hidden,
			],
			dim=-1,
		)
		event_tokens = self.event_projection(event_inputs) * history_mask.unsqueeze(-1).float()
		return event_tokens, history_group_ids, history_time_gap, history_mask

	def _split_history_by_group(
		self,
		event_tokens: torch.Tensor,
		history_group_ids: torch.Tensor,
		history_time_gap: torch.Tensor,
		history_mask: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		batch_size, _, hidden_dim = event_tokens.shape
		max_seq_len = self.data_config.max_seq_len
		device = event_tokens.device
		branch_tokens = event_tokens.new_zeros(batch_size, self.sequence_count, max_seq_len, hidden_dim)
		branch_mask = torch.zeros(batch_size, self.sequence_count, max_seq_len, dtype=torch.bool, device=device)
		branch_times = event_tokens.new_zeros(batch_size, self.sequence_count, max_seq_len)

		for batch_index in range(batch_size):
			valid_positions = torch.nonzero(history_mask[batch_index], as_tuple=False).squeeze(-1)
			for sequence_index in range(self.sequence_count):
				if valid_positions.numel() > 0:
					selected_positions = valid_positions[
						history_group_ids[batch_index, valid_positions] == (sequence_index + 1)
					]
				else:
					selected_positions = valid_positions
				selected_positions = selected_positions[-max_seq_len:]
				selected_count = int(selected_positions.numel())
				if selected_count == 0:
					branch_tokens[batch_index, sequence_index, -1] = self.empty_sequence_tokens[sequence_index]
					branch_mask[batch_index, sequence_index, -1] = True
					continue

				start = max_seq_len - selected_count
				branch_tokens[batch_index, sequence_index, start:] = event_tokens[batch_index, selected_positions]
				branch_mask[batch_index, sequence_index, start:] = True
				gaps = history_time_gap[batch_index, selected_positions].float()
				recency = float(TIME_GAP_BUCKET_COUNT + 1) - gaps.clamp(min=0.0, max=float(TIME_GAP_BUCKET_COUNT + 1))
				base = torch.arange(selected_count, device=device, dtype=torch.float32) * float(TIME_GAP_BUCKET_COUNT + 1)
				branch_times[batch_index, sequence_index, start:] = base + recency

		positions = torch.arange(max_seq_len, device=device)
		branch_tokens = branch_tokens + self.sequence_position_embedding(positions).view(1, 1, max_seq_len, hidden_dim)
		sequence_ids = torch.arange(1, self.sequence_count + 1, device=device)
		branch_tokens = branch_tokens + self.sequence_id_embedding(sequence_ids).view(1, self.sequence_count, 1, hidden_dim)
		branch_tokens = branch_tokens * branch_mask.unsqueeze(-1).float()
		return branch_tokens, branch_mask, branch_times

	def _build_unified_sequence(self, batch: BatchTensors) -> tuple[torch.Tensor, torch.Tensor, int, int]:
		feature_tokens, user_pool, item_pool = self._build_feature_tokens(batch)
		event_tokens, history_group_ids, history_time_gap, history_mask = self._build_event_tokens(batch)
		branch_tokens, branch_mask, _ = self._split_history_by_group(
			event_tokens,
			history_group_ids,
			history_time_gap,
			history_mask,
		)

		per_branch_tokens = [branch_tokens[:, i] for i in range(self.sequence_count)]
		per_branch_mask = [branch_mask[:, i] for i in range(self.sequence_count)]

		all_sequence_tokens = torch.cat(per_branch_tokens, dim=1)
		all_sequence_mask = torch.cat(per_branch_mask, dim=1)
		query = self.interest_query(item_pool).unsqueeze(1)
		key = self.interest_key(all_sequence_tokens)
		value = self.interest_value(all_sequence_tokens)
		attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
		attention_scores = attention_scores.masked_fill(~all_sequence_mask.unsqueeze(1), -1.0e4)
		attention_weights = torch.softmax(attention_scores, dim=-1)
		attention_weights = attention_weights * all_sequence_mask.unsqueeze(1).float()
		attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
		interest = torch.matmul(attention_weights, value).squeeze(1)
		interest_token = self.interest_out(interest).unsqueeze(1)

		target_token = self.target_fusion(torch.cat([user_pool, item_pool, user_pool * item_pool], dim=-1)).unsqueeze(1)

		sequence_parts = [feature_tokens] + per_branch_tokens
		special_segment_id = self.sequence_count + 1
		segment_ids = [
			torch.zeros(batch.batch_size, self.n_feature_tokens, dtype=torch.long, device=feature_tokens.device),
		] + [
			torch.full((batch.batch_size, per_branch_tokens[i].shape[1]), i + 1, dtype=torch.long, device=feature_tokens.device)
			for i in range(self.sequence_count)
		]
		padding_parts = [
			torch.ones(batch.batch_size, self.n_feature_tokens, dtype=torch.bool, device=feature_tokens.device),
		] + list(per_branch_mask)

		if self.mot is not None and self.mot_projection is not None:
			mot_token = self.mot_projection(
				self.mot(
					per_branch_tokens,
					per_branch_mask,
				)
			).unsqueeze(1)
			sequence_parts.append(mot_token)
			segment_ids.append(torch.full((batch.batch_size, 1), special_segment_id, dtype=torch.long, device=feature_tokens.device))
			padding_parts.append(torch.ones(batch.batch_size, 1, dtype=torch.bool, device=feature_tokens.device))

		sequence_parts.append(interest_token)
		segment_ids.append(torch.full((batch.batch_size, 1), special_segment_id, dtype=torch.long, device=feature_tokens.device))
		padding_parts.append(torch.ones(batch.batch_size, 1, dtype=torch.bool, device=feature_tokens.device))

		sequence_parts.append(target_token)
		segment_ids.append(torch.full((batch.batch_size, 1), special_segment_id, dtype=torch.long, device=feature_tokens.device))
		padding_parts.append(torch.ones(batch.batch_size, 1, dtype=torch.bool, device=feature_tokens.device))

		unified_tokens = torch.cat(sequence_parts, dim=1)
		segments = torch.cat(segment_ids, dim=1)
		padding_mask = torch.cat(padding_parts, dim=1)
		unified_tokens = unified_tokens + self.segment_embedding(segments)
		n_special_tokens = 2 + (1 if self.mot is not None and self.mot_projection is not None else 0)
		return unified_tokens, padding_mask, self.n_feature_tokens, n_special_tokens

	def _compose_attention_mask(
		self,
		padding_mask: torch.Tensor,
		n_feature_tokens: int,
		n_special_tokens: int,
	) -> torch.Tensor:
		attn_mask = build_unified_attention_mask(
			seq_len=padding_mask.shape[1],
			n_feature_tokens=n_feature_tokens,
			n_special_tokens=n_special_tokens,
			global_window=self.global_window,
			local_window=self.local_window,
			device=padding_mask.device,
		)
		return attn_mask & padding_mask.unsqueeze(1).unsqueeze(-1) & padding_mask.unsqueeze(1).unsqueeze(2)

	def forward(self, batch: BatchTensors) -> torch.Tensor:
		hidden_states, padding_mask, n_feature_tokens, n_special_tokens = self._build_unified_sequence(batch)
		attention_mask = self._compose_attention_mask(padding_mask, n_feature_tokens, n_special_tokens)

		batch_size = hidden_states.shape[0]
		dim = hidden_states.shape[2]
		block_summaries: list[torch.Tensor] = []
		partial_sum = hidden_states.new_zeros((batch_size, dim))
		layer_idx = 0

		for block in self.full_blocks:
			hidden_states = block(hidden_states, attention_mask)
			hidden_states = hidden_states * padding_mask.unsqueeze(-1).float()
			hidden_states, partial_sum, _ = self.block_attn_res(
				layer_idx, hidden_states, block_summaries, partial_sum,
			)
			hidden_states = hidden_states * padding_mask.unsqueeze(-1).float()
			layer_idx += 1
			if layer_idx % self.attn_res_block_size == 0:
				block_summaries.append(partial_sum)
				partial_sum = hidden_states.new_zeros((batch_size, dim))

		# Save remaining partial_sum if there is an incomplete residual-attention block.
		if layer_idx % self.attn_res_block_size != 0:
			block_summaries.append(partial_sum)
			partial_sum = hidden_states.new_zeros((batch_size, dim))

		if self.truncated_blocks and self.truncated_seq_len > 0:
			special_start = hidden_states.shape[1] - n_special_tokens
			sequence_region = hidden_states[:, n_feature_tokens:special_start]
			sequence_mask = padding_mask[:, n_feature_tokens:special_start]
			recent_len = min(self.truncated_seq_len, sequence_region.shape[1])
			if recent_len > 0:
				recent_tokens = sequence_region[:, -recent_len:]
				recent_mask = sequence_mask[:, -recent_len:]
				truncated_tokens = torch.cat(
					[
						hidden_states[:, :n_feature_tokens],
						recent_tokens,
						hidden_states[:, special_start:],
					],
					dim=1,
				)
				truncated_padding = torch.cat(
					[
						padding_mask[:, :n_feature_tokens],
						recent_mask,
						padding_mask[:, special_start:],
					],
					dim=1,
				)
				truncated_mask = self._compose_attention_mask(truncated_padding, n_feature_tokens, n_special_tokens)
				for block in self.truncated_blocks:
					truncated_tokens = block(truncated_tokens, truncated_mask)
					truncated_tokens = truncated_tokens * truncated_padding.unsqueeze(-1).float()
					truncated_tokens, partial_sum, _ = self.block_attn_res(
						layer_idx, truncated_tokens, block_summaries, partial_sum,
					)
					truncated_tokens = truncated_tokens * truncated_padding.unsqueeze(-1).float()
					layer_idx += 1
					if layer_idx % self.attn_res_block_size == 0:
						block_summaries.append(partial_sum)
						partial_sum = hidden_states.new_zeros((batch_size, dim))

				updated_recent = truncated_tokens[:, n_feature_tokens:n_feature_tokens + recent_len]
				updated_special = truncated_tokens[:, -n_special_tokens:]
				frozen_prefix = sequence_region[:, :-recent_len]
				frozen_prefix_mask = sequence_mask[:, :-recent_len]
				hidden_states = torch.cat(
					[
						hidden_states[:, :n_feature_tokens],
						frozen_prefix,
						updated_recent,
						updated_special,
					],
					dim=1,
				)
				padding_mask = torch.cat(
					[
						padding_mask[:, :n_feature_tokens],
						frozen_prefix_mask,
						recent_mask,
						padding_mask[:, special_start:],
					],
					dim=1,
				)

		hidden_states = self.final_norm(hidden_states)
		return self.head(hidden_states[:, -1, :]).squeeze(-1)


def build_model_component(data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> UniRecModel:
	return UniRecModel(data_config=data_config, model_config=model_config, dense_dim=dense_dim)