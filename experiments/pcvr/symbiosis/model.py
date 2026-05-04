"""Symbiosis-style PCVR model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from taac2026.infrastructure.pcvr.modeling import (
	DenseTokenProjector,
	EmbeddingParameterMixin,
	ModelInput,
	NonSequentialTokenizer,
	RMSNorm,
	SequenceTokenizer,
	configure_rms_norm_runtime as _configure_rms_norm_runtime,
	maybe_gradient_checkpoint,
	choose_num_heads,
	make_padding_mask,
	masked_last,
	masked_mean,
	safe_key_padding_mask,
	sinusoidal_positions,
)


def configure_rms_norm_runtime(*, rms_norm_backend: str, rms_norm_block_rows: int) -> None:
	_configure_rms_norm_runtime(
		backend=rms_norm_backend,
		block_rows=rms_norm_block_rows,
	)


class SwiGLUFeedForward(nn.Module):
	def __init__(self, d_model: int, hidden_mult: int, dropout: float) -> None:
		super().__init__()
		hidden_dim = max(d_model, d_model * hidden_mult)
		self.gate_up = nn.Linear(d_model, hidden_dim * 2)
		self.dropout = nn.Dropout(dropout)
		self.down = nn.Linear(hidden_dim, d_model)

	def forward(self, tokens: torch.Tensor) -> torch.Tensor:
		gate, value = self.gate_up(tokens).chunk(2, dim=-1)
		return self.down(self.dropout(F.silu(gate) * value))


class FourierTimeEncoder(nn.Module):
	def __init__(self, d_model: int, num_bands: int = 4) -> None:
		super().__init__()
		self.register_buffer("frequencies", 2.0 ** torch.arange(num_bands, dtype=torch.float32), persistent=False)
		self.project = nn.Sequential(nn.Linear(num_bands * 2, d_model), nn.SiLU(), nn.LayerNorm(d_model))

	def forward(self, time_buckets: torch.Tensor | None, *, dtype: torch.dtype) -> torch.Tensor | None:
		if time_buckets is None:
			return None
		values = torch.log1p(time_buckets.to(dtype=torch.float32).clamp_min(0.0)).unsqueeze(-1)
		angles = values / self.frequencies.to(device=time_buckets.device).view(1, 1, -1)
		features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).to(dtype=dtype)
		return self.project(features)


class RotarySelfAttention(nn.Module):
	def __init__(self, d_model: int, num_heads: int, dropout: float, *, use_rope: bool, rope_base: float) -> None:
		super().__init__()
		self.num_heads = num_heads
		self.head_dim = d_model // num_heads
		self.dropout = dropout
		self.use_rope = use_rope and self.head_dim >= 2
		self.rope_base = rope_base
		self.rotary_dim = self.head_dim - self.head_dim % 2
		self.qkv = nn.Linear(d_model, d_model * 3)
		self.out = nn.Linear(d_model, d_model)

	def _split_heads(self, tokens: torch.Tensor) -> torch.Tensor:
		batch_size, token_count, d_model = tokens.shape
		return tokens.view(batch_size, token_count, self.num_heads, d_model // self.num_heads).transpose(1, 2)

	def _merge_heads(self, tokens: torch.Tensor) -> torch.Tensor:
		batch_size, _num_heads, token_count, head_dim = tokens.shape
		return tokens.transpose(1, 2).contiguous().view(batch_size, token_count, self.num_heads * head_dim)

	def _apply_rope(self, tokens: torch.Tensor) -> torch.Tensor:
		if not self.use_rope or self.rotary_dim < 2:
			return tokens
		token_count = tokens.shape[-2]
		device = tokens.device
		positions = torch.arange(token_count, dtype=torch.float32, device=device).unsqueeze(1)
		frequencies = torch.exp(
			torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device) * (-math.log(self.rope_base) / self.rotary_dim)
		)
		angles = positions * frequencies.unsqueeze(0)
		sin = angles.sin().view(1, 1, token_count, -1).to(dtype=tokens.dtype)
		cos = angles.cos().view(1, 1, token_count, -1).to(dtype=tokens.dtype)
		rotary = tokens[..., : self.rotary_dim]
		even = rotary[..., 0::2]
		odd = rotary[..., 1::2]
		rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
		return torch.cat([rotated, tokens[..., self.rotary_dim :]], dim=-1)

	def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
		query, key, value = (self._split_heads(part) for part in self.qkv(tokens).chunk(3, dim=-1))
		query = self._apply_rope(query)
		key = self._apply_rope(key)
		key_padding_mask = safe_key_padding_mask(padding_mask)
		valid_keys = ~key_padding_mask
		token_count = tokens.shape[1]
		attn_mask = valid_keys[:, None, None, :].expand(tokens.shape[0], self.num_heads, token_count, token_count)
		attended = F.scaled_dot_product_attention(
			query,
			key,
			value,
			attn_mask=attn_mask,
			dropout_p=self.dropout if self.training else 0.0,
		)
		return self.out(self._merge_heads(attended))


class UserItemGraphBlock(nn.Module):
	def __init__(self, d_model: int, num_heads: int, hidden_mult: int, dropout: float) -> None:
		super().__init__()
		self.user_norm = RMSNorm(d_model)
		self.item_norm = RMSNorm(d_model)
		self.user_from_item = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
		self.item_from_user = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
		self.user_gate = nn.Linear(d_model * 2, d_model)
		self.item_gate = nn.Linear(d_model * 2, d_model)
		self.user_ffn_norm = RMSNorm(d_model)
		self.item_ffn_norm = RMSNorm(d_model)
		self.user_ffn = SwiGLUFeedForward(d_model, hidden_mult, dropout)
		self.item_ffn = SwiGLUFeedForward(d_model, hidden_mult, dropout)

	def _update(
		self,
		target: torch.Tensor,
		source: torch.Tensor,
		target_norm: RMSNorm,
		source_norm: RMSNorm,
		attention: nn.MultiheadAttention,
		gate_layer: nn.Linear,
		ffn_norm: RMSNorm,
		ffn: SwiGLUFeedForward,
	) -> torch.Tensor:
		if target.shape[1] == 0:
			return target
		if source.shape[1] > 0:
			normalized_source = source_norm(source)
			attended, _weights = attention(target_norm(target), normalized_source, normalized_source, need_weights=False)
			gate = torch.sigmoid(gate_layer(torch.cat([target, attended], dim=-1)))
			target = target + gate * attended
		return target + ffn(ffn_norm(target))

	def forward(self, user_tokens: torch.Tensor, item_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		next_user_tokens = self._update(
			user_tokens,
			item_tokens,
			self.user_norm,
			self.item_norm,
			self.user_from_item,
			self.user_gate,
			self.user_ffn_norm,
			self.user_ffn,
		)
		next_item_tokens = self._update(
			item_tokens,
			user_tokens,
			self.item_norm,
			self.user_norm,
			self.item_from_user,
			self.item_gate,
			self.item_ffn_norm,
			self.item_ffn,
		)
		return next_user_tokens, next_item_tokens


class UnifiedBlock(nn.Module):
	def __init__(
		self,
		d_model: int,
		num_heads: int,
		hidden_mult: int,
		dropout: float,
		*,
		use_rope: bool,
		rope_base: float,
		use_domain_gate: bool,
	) -> None:
		super().__init__()
		self.use_domain_gate = use_domain_gate
		self.attn_norm = RMSNorm(d_model)
		self.attention = RotarySelfAttention(d_model, num_heads, dropout, use_rope=use_rope, rope_base=rope_base)
		self.film = nn.Linear(d_model, d_model * 2)
		self.sequence_query_norm = RMSNorm(d_model)
		self.sequence_norm = RMSNorm(d_model)
		self.sequence_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
		self.domain_gate = nn.Linear(d_model * 2, 1)
		self.sequence_gate = nn.Linear(d_model * 2, d_model)
		self.ffn_norm = RMSNorm(d_model)
		self.ffn = SwiGLUFeedForward(d_model, hidden_mult, dropout)

	def _attend_sequences(self, tokens: torch.Tensor, sequences: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
		if not sequences:
			return tokens.new_zeros(tokens.shape)
		query = self.sequence_query_norm(tokens)
		updates: list[torch.Tensor] = []
		scores: list[torch.Tensor] = []
		for sequence_tokens, sequence_mask in zip(sequences, masks, strict=True):
			if sequence_tokens.shape[1] == 0:
				continue
			attended, _weights = self.sequence_attention(
				query,
				self.sequence_norm(sequence_tokens),
				self.sequence_norm(sequence_tokens),
				key_padding_mask=safe_key_padding_mask(sequence_mask),
				need_weights=False,
			)
			valid_rows = (~sequence_mask).any(dim=1).to(attended.dtype).view(-1, 1, 1)
			updates.append(attended * valid_rows)
			if self.use_domain_gate:
				domain_score = self.domain_gate(torch.cat([tokens, attended], dim=-1))
				domain_score = domain_score.masked_fill(valid_rows <= 0, torch.finfo(domain_score.dtype).min)
				scores.append(domain_score)
		if not updates:
			return tokens.new_zeros(tokens.shape)
		if self.use_domain_gate:
			weights = torch.softmax(torch.stack(scores, dim=0), dim=0)
			return (torch.stack(updates, dim=0) * weights).sum(dim=0)
		return torch.stack(updates, dim=0).mean(dim=0)

	def forward(
		self,
		tokens: torch.Tensor,
		padding_mask: torch.Tensor,
		sequences: list[torch.Tensor],
		masks: list[torch.Tensor],
		modulation: torch.Tensor,
	) -> torch.Tensor:
		normalized_tokens = self.attn_norm(tokens)
		scale, shift = self.film(modulation).chunk(2, dim=-1)
		normalized_tokens = normalized_tokens * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(1)) + shift.unsqueeze(1)
		tokens = tokens + self.attention(normalized_tokens, padding_mask)
		sequence_update = self._attend_sequences(tokens, sequences, masks)
		sequence_gate = torch.sigmoid(self.sequence_gate(torch.cat([tokens, sequence_update], dim=-1)))
		tokens = tokens + sequence_gate * sequence_update
		return tokens + self.ffn(self.ffn_norm(tokens))


class ContextExchangeBlock(nn.Module):
	def __init__(self, d_model: int, num_heads: int, hidden_mult: int, dropout: float) -> None:
		super().__init__()
		self.query_norm = RMSNorm(d_model)
		self.sequence_norm = RMSNorm(d_model)
		self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
		self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
		self.ffn_norm = RMSNorm(d_model)
		self.ffn = SwiGLUFeedForward(d_model, hidden_mult, dropout)

	def forward(self, context: torch.Tensor, sequences: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
		if sequences:
			query = self.query_norm(context).unsqueeze(1)
			updates: list[torch.Tensor] = []
			for tokens, mask in zip(sequences, masks, strict=True):
				normalized_tokens = self.sequence_norm(tokens)
				attended, _weights = self.attention(
					query,
					normalized_tokens,
					normalized_tokens,
					key_padding_mask=safe_key_padding_mask(mask),
					need_weights=False,
				)
				updates.append(attended.squeeze(1))
			sequence_context = torch.stack(updates, dim=1).mean(dim=1)
			gate = self.gate(torch.cat([context, sequence_context], dim=-1))
			context = context + gate * sequence_context
		return context + self.ffn(self.ffn_norm(context))


class CandidateConditionedSequenceDecoder(nn.Module):
	def __init__(
		self,
		d_model: int,
		num_heads: int,
		hidden_mult: int,
		dropout: float,
		*,
		recent_tokens: int,
		memory_block_size: int,
		memory_top_k: int,
		use_compressed_memory: bool,
		use_attention_sink: bool,
	) -> None:
		super().__init__()
		self.recent_tokens = max(0, int(recent_tokens))
		self.memory_block_size = max(1, int(memory_block_size))
		self.memory_top_k = max(0, int(memory_top_k))
		self.use_compressed_memory = use_compressed_memory
		self.use_attention_sink = use_attention_sink
		self.query_norm = RMSNorm(d_model)
		self.memory_norm = RMSNorm(d_model)
		self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
		self.sink_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
		self.domain_gate = nn.Linear(d_model * 2, 1)
		self.ffn_norm = RMSNorm(d_model)
		self.ffn = SwiGLUFeedForward(d_model, hidden_mult, dropout)

	def _compressed_blocks(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		batch_size, token_count, d_model = tokens.shape
		pad_len = (-token_count) % self.memory_block_size
		if pad_len:
			tokens = F.pad(tokens, (0, 0, 0, pad_len))
			padding_mask = F.pad(padding_mask, (0, pad_len), value=True)
		blocks = tokens.view(batch_size, -1, self.memory_block_size, d_model)
		block_mask = padding_mask.view(batch_size, -1, self.memory_block_size)
		valid = (~block_mask).to(tokens.dtype).unsqueeze(-1)
		block_tokens = (blocks * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)
		return block_tokens, block_mask.all(dim=2)

	def _select_topk_blocks(
		self,
		query: torch.Tensor,
		blocks: torch.Tensor,
		block_mask: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor]:
		block_count = blocks.shape[1]
		if self.memory_top_k <= 0 or block_count <= self.memory_top_k:
			return blocks, block_mask
		scores = (self.memory_norm(blocks) * self.query_norm(query).unsqueeze(1)).sum(dim=-1)
		scores = scores.masked_fill(block_mask, torch.finfo(scores.dtype).min)
		indices = torch.topk(scores, k=self.memory_top_k, dim=1).indices
		gather_index = indices.unsqueeze(-1).expand(-1, -1, blocks.shape[-1])
		return blocks.gather(1, gather_index), block_mask.gather(1, indices)

	def _build_memory(
		self,
		query: torch.Tensor,
		tokens: torch.Tensor,
		padding_mask: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor]:
		memory_parts: list[torch.Tensor] = []
		mask_parts: list[torch.Tensor] = []
		if self.use_attention_sink:
			memory_parts.append(self.sink_token.expand(tokens.shape[0], -1, -1).to(dtype=tokens.dtype))
			mask_parts.append(torch.zeros(tokens.shape[0], 1, dtype=torch.bool, device=tokens.device))
		if self.recent_tokens > 0:
			recent_count = min(self.recent_tokens, tokens.shape[1])
			memory_parts.append(tokens[:, -recent_count:, :])
			mask_parts.append(padding_mask[:, -recent_count:])
		if self.use_compressed_memory:
			blocks, block_mask = self._compressed_blocks(tokens, padding_mask)
			blocks, block_mask = self._select_topk_blocks(query, blocks, block_mask)
			memory_parts.append(blocks)
			mask_parts.append(block_mask)
		global_token = masked_mean(tokens, padding_mask).unsqueeze(1)
		global_mask = padding_mask.all(dim=1, keepdim=True)
		memory_parts.append(global_token)
		mask_parts.append(global_mask)
		return torch.cat(memory_parts, dim=1), torch.cat(mask_parts, dim=1)

	def forward(self, query: torch.Tensor, sequences: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
		if not sequences:
			return query.new_zeros(query.shape)
		updates: list[torch.Tensor] = []
		scores: list[torch.Tensor] = []
		normalized_query = self.query_norm(query).unsqueeze(1)
		for sequence_tokens, sequence_mask in zip(sequences, masks, strict=True):
			memory, memory_mask = self._build_memory(query, sequence_tokens, sequence_mask)
			attended, _weights = self.attention(
				normalized_query,
				self.memory_norm(memory),
				self.memory_norm(memory),
				key_padding_mask=safe_key_padding_mask(memory_mask),
				need_weights=False,
			)
			attended = attended.squeeze(1)
			valid_rows = (~sequence_mask).any(dim=1).to(attended.dtype).unsqueeze(-1)
			attended = (attended + self.ffn(self.ffn_norm(attended))) * valid_rows
			updates.append(attended)
			domain_score = self.domain_gate(torch.cat([query, attended], dim=-1))
			domain_score = domain_score.masked_fill(valid_rows <= 0, torch.finfo(domain_score.dtype).min)
			scores.append(domain_score)
		weights = torch.softmax(torch.stack(scores, dim=1), dim=1)
		return (torch.stack(updates, dim=1) * weights).sum(dim=1)


class MultiLaneFusion(nn.Module):
	def __init__(self, d_model: int, lane_count: int) -> None:
		super().__init__()
		self.norm = RMSNorm(d_model * lane_count)
		self.weight_projection = nn.Linear(d_model * lane_count, lane_count)

	def forward(self, lanes: list[torch.Tensor]) -> torch.Tensor:
		stacked = torch.stack(lanes, dim=1)
		joined = torch.cat(lanes, dim=-1)
		weights = torch.softmax(self.weight_projection(self.norm(joined)), dim=-1)
		return (stacked * weights.unsqueeze(-1)).sum(dim=1)


class ActionConditionedHead(nn.Module):
	def __init__(self, d_model: int, action_num: int, hidden_mult: int, dropout: float) -> None:
		super().__init__()
		self.action_embeddings = nn.Parameter(torch.randn(action_num, d_model) * 0.02)
		self.context_projection = nn.Linear(d_model, d_model)
		self.norm = RMSNorm(d_model)
		self.ffn = SwiGLUFeedForward(d_model, hidden_mult, dropout)
		self.readout = nn.Linear(d_model, 1)

	def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
		action_tokens = self.context_projection(embeddings).unsqueeze(1) + self.action_embeddings.unsqueeze(0)
		action_tokens = action_tokens + self.ffn(self.norm(action_tokens))
		return self.readout(action_tokens).squeeze(-1)


class PCVRSymbiosis(EmbeddingParameterMixin, nn.Module):
	def __init__(
		self,
		user_int_feature_specs: list[tuple[int, int, int]],
		item_int_feature_specs: list[tuple[int, int, int]],
		user_dense_dim: int,
		item_dense_dim: int,
		seq_vocab_sizes: dict[str, list[int]],
		user_ns_groups: list[list[int]],
		item_ns_groups: list[list[int]],
		d_model: int = 64,
		emb_dim: int = 64,
		num_queries: int = 1,
		num_blocks: int = 2,
		num_heads: int = 4,
		seq_encoder_type: str = "transformer",
		hidden_mult: int = 4,
		dropout_rate: float = 0.01,
		seq_top_k: int = 50,
		seq_causal: bool = False,
		action_num: int = 1,
		num_time_buckets: int = 65,
		rank_mixer_mode: str = "full",
		use_rope: bool = False,
		rope_base: float = 10000.0,
		emb_skip_threshold: int = 0,
		seq_id_threshold: int = 10000,
		gradient_checkpointing: bool = False,
		ns_tokenizer_type: str = "rankmixer",
		user_ns_tokens: int = 5,
		item_ns_tokens: int = 2,
		symbiosis_use_user_item_graph: bool = True,
		symbiosis_use_fourier_time: bool = True,
		symbiosis_use_context_exchange: bool = True,
		symbiosis_use_multi_scale: bool = True,
		symbiosis_use_domain_gate: bool = True,
		symbiosis_use_candidate_decoder: bool = True,
		symbiosis_use_action_conditioning: bool = True,
		symbiosis_use_compressed_memory: bool = True,
		symbiosis_use_attention_sink: bool = True,
		symbiosis_use_lane_mixing: bool = True,
		symbiosis_use_semantic_id: bool = True,
		symbiosis_memory_block_size: int = 16,
		symbiosis_memory_top_k: int = 8,
		symbiosis_recent_tokens: int = 64,
	) -> None:
		super().__init__()
		del seq_encoder_type, seq_top_k, seq_causal, rank_mixer_mode, seq_id_threshold
		num_heads = choose_num_heads(d_model, num_heads)
		self.d_model = d_model
		self.action_num = action_num
		self.num_prompt_tokens = max(1, action_num, num_queries)
		self.gradient_checkpointing = bool(gradient_checkpointing)
		self.seq_domains = sorted(seq_vocab_sizes)
		self.symbiosis_use_user_item_graph = symbiosis_use_user_item_graph
		self.symbiosis_use_fourier_time = symbiosis_use_fourier_time
		self.symbiosis_use_context_exchange = symbiosis_use_context_exchange
		self.symbiosis_use_multi_scale = symbiosis_use_multi_scale
		self.symbiosis_use_domain_gate = symbiosis_use_domain_gate
		self.symbiosis_use_candidate_decoder = symbiosis_use_candidate_decoder
		self.symbiosis_use_action_conditioning = symbiosis_use_action_conditioning
		self.symbiosis_use_compressed_memory = symbiosis_use_compressed_memory
		self.symbiosis_use_attention_sink = symbiosis_use_attention_sink
		self.symbiosis_use_lane_mixing = symbiosis_use_lane_mixing
		self.symbiosis_use_semantic_id = symbiosis_use_semantic_id
		force_auto_split = ns_tokenizer_type == "rankmixer"
		self.user_tokenizer = NonSequentialTokenizer(user_int_feature_specs, user_ns_groups, emb_dim, d_model, user_ns_tokens, emb_skip_threshold, force_auto_split=force_auto_split)
		self.item_tokenizer = NonSequentialTokenizer(item_int_feature_specs, item_ns_groups, emb_dim, d_model, item_ns_tokens, emb_skip_threshold, force_auto_split=force_auto_split)
		self.user_dense = DenseTokenProjector(user_dense_dim, d_model)
		self.item_dense = DenseTokenProjector(item_dense_dim, d_model)
		self.semantic_projection = nn.Sequential(RMSNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU()) if symbiosis_use_semantic_id else None
		self.sequence_tokenizers = nn.ModuleDict(
			{domain: SequenceTokenizer(vocab_sizes, emb_dim, d_model, num_time_buckets, emb_skip_threshold) for domain, vocab_sizes in seq_vocab_sizes.items()}
		)
		self.time_encoders = nn.ModuleDict({domain: FourierTimeEncoder(d_model) for domain in self.seq_domains})
		self.num_ns = self.user_tokenizer.num_tokens + self.item_tokenizer.num_tokens
		self.num_ns += int(user_dense_dim > 0) + int(item_dense_dim > 0)
		self.num_ns += int(symbiosis_use_semantic_id)
		self.action_prompts = nn.Parameter(torch.randn(self.num_prompt_tokens, d_model) * 0.02)
		self.target_action_token = nn.Parameter(torch.randn(1, d_model) * 0.02)
		self.action_query_projection = nn.Sequential(RMSNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU())
		self.graph_blocks = nn.ModuleList(
			UserItemGraphBlock(d_model, num_heads, hidden_mult, dropout_rate) for _ in range(max(1, num_blocks)) if symbiosis_use_user_item_graph
		)
		self.unified_blocks = nn.ModuleList(
			UnifiedBlock(
				d_model,
				num_heads,
				hidden_mult,
				dropout_rate,
				use_rope=use_rope,
				rope_base=rope_base,
				use_domain_gate=symbiosis_use_domain_gate,
			)
			for _ in range(max(1, num_blocks))
		)
		self.context_blocks = nn.ModuleList(
			ContextExchangeBlock(d_model, num_heads, hidden_mult, dropout_rate) for _ in range(max(1, num_blocks)) if symbiosis_use_context_exchange
		)
		self.candidate_query_projection = nn.Sequential(RMSNorm(d_model * 4), nn.Linear(d_model * 4, d_model), nn.SiLU())
		self.candidate_decoder = (
			CandidateConditionedSequenceDecoder(
				d_model,
				num_heads,
				hidden_mult,
				dropout_rate,
				recent_tokens=symbiosis_recent_tokens,
				memory_block_size=symbiosis_memory_block_size,
				memory_top_k=symbiosis_memory_top_k,
				use_compressed_memory=symbiosis_use_compressed_memory,
				use_attention_sink=symbiosis_use_attention_sink,
			)
			if symbiosis_use_candidate_decoder
			else None
		)
		self.lane_mixer = MultiLaneFusion(d_model, 5) if symbiosis_use_lane_mixing else None
		self.unified_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
		self.scale_projection = nn.Sequential(RMSNorm(d_model * 3), nn.Linear(d_model * 3, d_model), nn.SiLU())
		self.fusion_projection = nn.Sequential(RMSNorm(d_model * 5), nn.Linear(d_model * 5, d_model), nn.SiLU())
		self.fusion_gate = nn.Sequential(nn.Linear(d_model * 5, d_model), nn.Sigmoid())
		self.out_norm = RMSNorm(d_model)
		self.classifier = ActionConditionedHead(d_model, action_num, hidden_mult, dropout_rate)

	def _encode_non_sequence_parts(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
		user_parts = [self.user_tokenizer(inputs.user_int_feats)]
		user_dense = self.user_dense(inputs.user_dense_feats)
		if user_dense is not None:
			user_parts.append(user_dense)
		item_parts = [self.item_tokenizer(inputs.item_int_feats)]
		item_dense = self.item_dense(inputs.item_dense_feats)
		if item_dense is not None:
			item_parts.append(item_dense)
		user_tokens = torch.cat(user_parts, dim=1)
		item_tokens = torch.cat(item_parts, dim=1)
		if self.semantic_projection is not None:
			semantic_token = self.semantic_projection(masked_mean(item_tokens)).unsqueeze(1)
			item_tokens = torch.cat([item_tokens, semantic_token], dim=1)
		return user_tokens, item_tokens

	def _encode_non_sequence(self, inputs: ModelInput) -> torch.Tensor:
		user_tokens, item_tokens = self._encode_non_sequence_parts(inputs)
		return torch.cat([user_tokens, item_tokens], dim=1)

	def _encode_sequences(self, inputs: ModelInput) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
		sequences: list[torch.Tensor] = []
		masks: list[torch.Tensor] = []
		lengths: list[torch.Tensor] = []
		for domain in self.seq_domains:
			raw_sequence = inputs.seq_data[domain]
			seq_len = inputs.seq_lens[domain].to(raw_sequence.device)
			time_buckets = inputs.seq_time_buckets.get(domain)
			tokens = self.sequence_tokenizers[domain](raw_sequence, time_buckets)
			tokens = tokens + sinusoidal_positions(tokens.shape[1], self.d_model, tokens.device).unsqueeze(0)
			if self.symbiosis_use_fourier_time:
				time_tokens = self.time_encoders[domain](time_buckets, dtype=tokens.dtype)
				if time_tokens is not None:
					tokens = tokens + time_tokens
			sequences.append(tokens)
			masks.append(make_padding_mask(seq_len, raw_sequence.shape[2]))
			lengths.append(seq_len)
		return sequences, masks, lengths

	def _multi_scale_context(
		self,
		sequences: list[torch.Tensor],
		masks: list[torch.Tensor],
		lengths: list[torch.Tensor],
		ns_tokens: torch.Tensor,
	) -> torch.Tensor:
		if not sequences:
			return masked_mean(ns_tokens)
		means = torch.stack([masked_mean(tokens, mask) for tokens, mask in zip(sequences, masks, strict=True)], dim=1).mean(dim=1)
		lasts = torch.stack([masked_last(tokens, seq_len) for tokens, seq_len in zip(sequences, lengths, strict=True)], dim=1).mean(dim=1)
		recents = torch.stack(
			[masked_mean(tokens[:, tokens.shape[1] // 2 :, :], mask[:, mask.shape[1] // 2 :]) for tokens, mask in zip(sequences, masks, strict=True)],
			dim=1,
		).mean(dim=1)
		return self.scale_projection(torch.cat([means, recents, lasts], dim=-1))

	def _action_context(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
		if not self.symbiosis_use_action_conditioning:
			return torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)
		action_token = self.target_action_token.to(device=device, dtype=dtype).expand(batch_size, -1)
		return self.action_query_projection(action_token)

	def _candidate_query(
		self,
		user_context: torch.Tensor,
		item_context: torch.Tensor,
		graph_context: torch.Tensor,
		action_context: torch.Tensor,
	) -> torch.Tensor:
		return self.candidate_query_projection(torch.cat([user_context, item_context, graph_context, action_context], dim=-1))

	def _embed(self, inputs: ModelInput) -> torch.Tensor:
		user_tokens, item_tokens = self._encode_non_sequence_parts(inputs)
		for block in self.graph_blocks:
			user_tokens, item_tokens = maybe_gradient_checkpoint(
				block,
				user_tokens,
				item_tokens,
				enabled=self.gradient_checkpointing,
			)
		ns_tokens = torch.cat([user_tokens, item_tokens], dim=1)
		user_context = masked_mean(user_tokens)
		item_context = masked_mean(item_tokens)
		graph_context = masked_mean(ns_tokens)
		sequences, masks, lengths = self._encode_sequences(inputs)
		action_context = self._action_context(ns_tokens.shape[0], ns_tokens.device, ns_tokens.dtype)
		candidate_query = self._candidate_query(user_context, item_context, graph_context, action_context)
		if self.candidate_decoder is not None:
			candidate_sequence_context = self.candidate_decoder(candidate_query, sequences, masks)
		else:
			candidate_sequence_context = torch.zeros_like(graph_context)
		ns_mask = torch.zeros(ns_tokens.shape[0], ns_tokens.shape[1], dtype=torch.bool, device=ns_tokens.device)
		prompt_tokens = self.action_prompts.unsqueeze(0).expand(ns_tokens.shape[0], -1, -1)
		if self.symbiosis_use_action_conditioning:
			prompt_tokens = prompt_tokens + action_context.unsqueeze(1)
		prompt_mask = torch.zeros(prompt_tokens.shape[0], prompt_tokens.shape[1], dtype=torch.bool, device=ns_tokens.device)
		unified_tokens = torch.cat([prompt_tokens, ns_tokens], dim=1)
		unified_mask = torch.cat([prompt_mask, ns_mask], dim=1)
		for block in self.unified_blocks:
			unified_tokens = maybe_gradient_checkpoint(
				block,
				unified_tokens,
				unified_mask,
				sequences,
				masks,
				candidate_query,
				enabled=self.gradient_checkpointing,
			)
		prompt_context = masked_mean(unified_tokens[:, : self.num_prompt_tokens, :])
		token_context = masked_mean(unified_tokens[:, self.num_prompt_tokens :, :], unified_mask[:, self.num_prompt_tokens :])
		unified_gate = self.unified_gate(torch.cat([prompt_context, token_context], dim=-1))
		unified_context = unified_gate * prompt_context + (1.0 - unified_gate) * token_context
		context = graph_context
		for block in self.context_blocks:
			context = maybe_gradient_checkpoint(
				block,
				context,
				sequences,
				masks,
				enabled=self.gradient_checkpointing,
			)
		if self.symbiosis_use_multi_scale:
			scale_context = self._multi_scale_context(sequences, masks, lengths, ns_tokens)
		else:
			scale_context = torch.zeros_like(graph_context)
		lanes = [unified_context, context, scale_context, graph_context, candidate_sequence_context]
		joined = torch.cat(lanes, dim=-1)
		candidate = self.fusion_projection(joined)
		if self.lane_mixer is not None:
			blended = self.lane_mixer(lanes)
		else:
			blended = torch.stack(lanes, dim=0).mean(dim=0)
		gate = self.fusion_gate(joined)
		return self.out_norm(gate * candidate + (1.0 - gate) * blended)

	def forward(self, inputs: ModelInput) -> torch.Tensor:
		return self.classifier(self._embed(inputs))

	def predict(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
		embeddings = self._embed(inputs)
		return self.classifier(embeddings), embeddings


__all__ = ["ModelInput", "PCVRSymbiosis"]