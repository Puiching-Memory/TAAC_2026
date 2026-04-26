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
	choose_num_heads,
	make_padding_mask,
	masked_last,
	masked_mean,
	safe_key_padding_mask,
	sinusoidal_positions,
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
	def __init__(self, d_model: int, num_heads: int, hidden_mult: int, dropout: float, *, use_rope: bool, rope_base: float) -> None:
		super().__init__()
		self.attn_norm = RMSNorm(d_model)
		self.attention = RotarySelfAttention(d_model, num_heads, dropout, use_rope=use_rope, rope_base=rope_base)
		self.film = nn.Linear(d_model, d_model * 2)
		self.sequence_query_norm = RMSNorm(d_model)
		self.sequence_norm = RMSNorm(d_model)
		self.sequence_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
		self.sequence_gate = nn.Linear(d_model * 2, d_model)
		self.ffn_norm = RMSNorm(d_model)
		self.ffn = SwiGLUFeedForward(d_model, hidden_mult, dropout)

	def _attend_sequences(self, tokens: torch.Tensor, sequences: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
		if not sequences:
			return tokens.new_zeros(tokens.shape)
		query = self.sequence_query_norm(tokens)
		updates: list[torch.Tensor] = []
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
		if not updates:
			return tokens.new_zeros(tokens.shape)
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
		ns_tokenizer_type: str = "rankmixer",
		user_ns_tokens: int = 5,
		item_ns_tokens: int = 2,
	) -> None:
		super().__init__()
		del seq_encoder_type, seq_top_k, seq_causal, rank_mixer_mode, seq_id_threshold
		num_heads = choose_num_heads(d_model, num_heads)
		self.d_model = d_model
		self.action_num = action_num
		self.num_prompt_tokens = max(1, action_num, num_queries)
		self.seq_domains = sorted(seq_vocab_sizes)
		force_auto_split = ns_tokenizer_type == "rankmixer"
		self.user_tokenizer = NonSequentialTokenizer(user_int_feature_specs, user_ns_groups, emb_dim, d_model, user_ns_tokens, emb_skip_threshold, force_auto_split=force_auto_split)
		self.item_tokenizer = NonSequentialTokenizer(item_int_feature_specs, item_ns_groups, emb_dim, d_model, item_ns_tokens, emb_skip_threshold, force_auto_split=force_auto_split)
		self.user_dense = DenseTokenProjector(user_dense_dim, d_model)
		self.item_dense = DenseTokenProjector(item_dense_dim, d_model)
		self.sequence_tokenizers = nn.ModuleDict(
			{domain: SequenceTokenizer(vocab_sizes, emb_dim, d_model, num_time_buckets, emb_skip_threshold) for domain, vocab_sizes in seq_vocab_sizes.items()}
		)
		self.time_encoders = nn.ModuleDict({domain: FourierTimeEncoder(d_model) for domain in self.seq_domains})
		self.num_ns = self.user_tokenizer.num_tokens + self.item_tokenizer.num_tokens
		self.num_ns += int(user_dense_dim > 0) + int(item_dense_dim > 0)
		self.action_prompts = nn.Parameter(torch.randn(self.num_prompt_tokens, d_model) * 0.02)
		self.graph_blocks = nn.ModuleList(UserItemGraphBlock(d_model, num_heads, hidden_mult, dropout_rate) for _ in range(max(1, num_blocks)))
		self.unified_blocks = nn.ModuleList(
			UnifiedBlock(d_model, num_heads, hidden_mult, dropout_rate, use_rope=use_rope, rope_base=rope_base) for _ in range(max(1, num_blocks))
		)
		self.context_blocks = nn.ModuleList(ContextExchangeBlock(d_model, num_heads, hidden_mult, dropout_rate) for _ in range(max(1, num_blocks)))
		self.unified_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
		self.scale_projection = nn.Sequential(RMSNorm(d_model * 3), nn.Linear(d_model * 3, d_model), nn.SiLU())
		self.fusion_projection = nn.Sequential(RMSNorm(d_model * 4), nn.Linear(d_model * 4, d_model), nn.SiLU())
		self.fusion_gate = nn.Sequential(nn.Linear(d_model * 4, d_model), nn.Sigmoid())
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
		return torch.cat(user_parts, dim=1), torch.cat(item_parts, dim=1)

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
			max_len = int(seq_len.max().item()) if seq_len.numel() > 0 else 0
			if max_len <= 0:
				max_len = 1
			if raw_sequence.shape[2] > max_len:
				raw_sequence = raw_sequence[:, :, :max_len]
				time_buckets = inputs.seq_time_buckets.get(domain)
				if time_buckets is not None:
					time_buckets = time_buckets[:, :max_len]
			else:
				time_buckets = inputs.seq_time_buckets.get(domain)
			tokens = self.sequence_tokenizers[domain](raw_sequence, time_buckets)
			tokens = tokens + sinusoidal_positions(tokens.shape[1], self.d_model, tokens.device).unsqueeze(0)
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

	def _embed(self, inputs: ModelInput) -> torch.Tensor:
		user_tokens, item_tokens = self._encode_non_sequence_parts(inputs)
		for block in self.graph_blocks:
			user_tokens, item_tokens = block(user_tokens, item_tokens)
		ns_tokens = torch.cat([user_tokens, item_tokens], dim=1)
		graph_context = masked_mean(ns_tokens)
		sequences, masks, lengths = self._encode_sequences(inputs)
		ns_mask = torch.zeros(ns_tokens.shape[0], ns_tokens.shape[1], dtype=torch.bool, device=ns_tokens.device)
		prompt_tokens = self.action_prompts.unsqueeze(0).expand(ns_tokens.shape[0], -1, -1)
		prompt_mask = torch.zeros(prompt_tokens.shape[0], prompt_tokens.shape[1], dtype=torch.bool, device=ns_tokens.device)
		unified_tokens = torch.cat([prompt_tokens, ns_tokens], dim=1)
		unified_mask = torch.cat([prompt_mask, ns_mask], dim=1)
		for block in self.unified_blocks:
			unified_tokens = block(unified_tokens, unified_mask, sequences, masks, graph_context)
		prompt_context = masked_mean(unified_tokens[:, : self.num_prompt_tokens, :])
		token_context = masked_mean(unified_tokens[:, self.num_prompt_tokens :, :], unified_mask[:, self.num_prompt_tokens :])
		unified_gate = self.unified_gate(torch.cat([prompt_context, token_context], dim=-1))
		unified_context = unified_gate * prompt_context + (1.0 - unified_gate) * token_context
		context = graph_context
		for block in self.context_blocks:
			context = block(context, sequences, masks)
		scale_context = self._multi_scale_context(sequences, masks, lengths, ns_tokens)
		joined = torch.cat([unified_context, context, scale_context, graph_context], dim=-1)
		candidate = self.fusion_projection(joined)
		blended = (unified_context + context + scale_context + graph_context) * 0.25
		gate = self.fusion_gate(joined)
		return self.out_norm(gate * candidate + (1.0 - gate) * blended)

	def forward(self, inputs: ModelInput) -> torch.Tensor:
		return self.classifier(self._embed(inputs))

	def predict(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
		embeddings = self._embed(inputs)
		return self.classifier(embeddings), embeddings


__all__ = ["ModelInput", "PCVRSymbiosis"]