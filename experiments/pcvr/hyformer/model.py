"""HyFormer-style PCVR model."""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import DenseTokenProjector, EmbeddingParameterMixin, NonSequentialTokenizer, SequenceTokenizer, configure_rms_norm_runtime as _configure_rms_norm_runtime
from taac2026.infrastructure.pcvr.modeling import (
	ModelInput,
	maybe_gradient_checkpoint,
	choose_num_heads,
	make_padding_mask,
	masked_mean,
	safe_key_padding_mask,
	sinusoidal_positions,
)


def configure_rms_norm_runtime(*, rms_norm_backend: str, rms_norm_block_rows: int) -> None:
	_configure_rms_norm_runtime(
		backend=rms_norm_backend,
		block_rows=rms_norm_block_rows,
	)


class QueryDecodeBoostBlock(nn.Module):
	def __init__(self, d_model: int, num_heads: int, hidden_mult: int, dropout: float) -> None:
		super().__init__()
		self.query_norm = nn.LayerNorm(d_model)
		self.sequence_norm = nn.LayerNorm(d_model)
		self.decode_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
		self.mixer_norm = nn.LayerNorm(d_model)
		self.mixer = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
		self.ffn = nn.Sequential(
			nn.LayerNorm(d_model),
			nn.Linear(d_model, d_model * hidden_mult),
			nn.SiLU(),
			nn.Dropout(dropout),
			nn.Linear(d_model * hidden_mult, d_model),
		)

	def forward(
		self,
		queries: torch.Tensor,
		ns_tokens: torch.Tensor,
		sequences: list[torch.Tensor],
		masks: list[torch.Tensor],
	) -> tuple[torch.Tensor, torch.Tensor]:
		if sequences:
			decoded: list[torch.Tensor] = []
			normalized_queries = self.query_norm(queries)
			for tokens, mask in zip(sequences, masks, strict=True):
				normalized_tokens = self.sequence_norm(tokens)
				attended, _weights = self.decode_attention(
					normalized_queries,
					normalized_tokens,
					normalized_tokens,
					key_padding_mask=safe_key_padding_mask(mask),
					need_weights=False,
				)
				decoded.append(attended)
			queries = queries + torch.stack(decoded, dim=1).mean(dim=1)
		mixed_input = torch.cat([queries, ns_tokens], dim=1)
		normalized_mixed = self.mixer_norm(mixed_input)
		mixed, _weights = self.mixer(normalized_mixed, normalized_mixed, normalized_mixed, need_weights=False)
		mixed = mixed_input + mixed
		mixed = mixed + self.ffn(mixed)
		return mixed[:, : queries.shape[1]], mixed[:, queries.shape[1] :]


class PCVRHyFormer(EmbeddingParameterMixin, nn.Module):
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
	) -> None:
		super().__init__()
		del seq_encoder_type, seq_top_k, seq_causal, rank_mixer_mode, use_rope, rope_base, seq_id_threshold
		num_heads = choose_num_heads(d_model, num_heads)
		self.d_model = d_model
		self.action_num = action_num
		self.num_queries = max(1, num_queries)
		self.gradient_checkpointing = bool(gradient_checkpointing)
		self.seq_domains = sorted(seq_vocab_sizes)
		force_auto_split = ns_tokenizer_type == "rankmixer"
		self.user_tokenizer = NonSequentialTokenizer(user_int_feature_specs, user_ns_groups, emb_dim, d_model, user_ns_tokens, emb_skip_threshold, force_auto_split=force_auto_split)
		self.item_tokenizer = NonSequentialTokenizer(item_int_feature_specs, item_ns_groups, emb_dim, d_model, item_ns_tokens, emb_skip_threshold, force_auto_split=force_auto_split)
		self.user_dense = DenseTokenProjector(user_dense_dim, d_model)
		self.item_dense = DenseTokenProjector(item_dense_dim, d_model)
		self.sequence_tokenizers = nn.ModuleDict(
			{domain: SequenceTokenizer(vocab_sizes, emb_dim, d_model, num_time_buckets, emb_skip_threshold) for domain, vocab_sizes in seq_vocab_sizes.items()}
		)
		self.num_ns = self.user_tokenizer.num_tokens + self.item_tokenizer.num_tokens
		self.num_ns += int(user_dense_dim > 0) + int(item_dense_dim > 0)
		self.query_projection = nn.Linear(d_model, self.num_queries * d_model)
		self.blocks = nn.ModuleList(QueryDecodeBoostBlock(d_model, num_heads, hidden_mult, dropout_rate) for _ in range(max(1, num_blocks)))
		self.output_projection = nn.Sequential(nn.LayerNorm(self.num_queries * d_model), nn.Linear(self.num_queries * d_model, d_model), nn.SiLU())
		self.classifier = nn.Sequential(
			nn.LayerNorm(d_model),
			nn.Linear(d_model, d_model * hidden_mult),
			nn.SiLU(),
			nn.Dropout(dropout_rate),
			nn.Linear(d_model * hidden_mult, action_num),
		)

	def _encode_non_sequence(self, inputs: ModelInput) -> torch.Tensor:
		parts = [self.user_tokenizer(inputs.user_int_feats)]
		user_dense = self.user_dense(inputs.user_dense_feats)
		if user_dense is not None:
			parts.append(user_dense)
		parts.append(self.item_tokenizer(inputs.item_int_feats))
		item_dense = self.item_dense(inputs.item_dense_feats)
		if item_dense is not None:
			parts.append(item_dense)
		return torch.cat(parts, dim=1)

	def _encode_sequences(self, inputs: ModelInput) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
		sequences: list[torch.Tensor] = []
		masks: list[torch.Tensor] = []
		for domain in self.seq_domains:
			raw_sequence = inputs.seq_data[domain]
			seq_len = inputs.seq_lens[domain].to(raw_sequence.device)
			tokens = self.sequence_tokenizers[domain](raw_sequence, inputs.seq_time_buckets.get(domain))
			tokens = tokens + sinusoidal_positions(tokens.shape[1], self.d_model, tokens.device).unsqueeze(0)
			sequences.append(tokens)
			masks.append(make_padding_mask(seq_len, raw_sequence.shape[2]))
		return sequences, masks

	def _embed(self, inputs: ModelInput) -> torch.Tensor:
		ns_tokens = self._encode_non_sequence(inputs)
		sequences, masks = self._encode_sequences(inputs)
		queries = self.query_projection(masked_mean(ns_tokens)).view(ns_tokens.shape[0], self.num_queries, self.d_model)
		for block in self.blocks:
			queries, ns_tokens = maybe_gradient_checkpoint(
				block,
				queries,
				ns_tokens,
				sequences,
				masks,
				enabled=self.gradient_checkpointing,
			)
		return self.output_projection(queries.reshape(queries.shape[0], -1))

	def forward(self, inputs: ModelInput) -> torch.Tensor:
		return self.classifier(self._embed(inputs))

	def predict(self, inputs: ModelInput) -> tuple[torch.Tensor, torch.Tensor]:
		embeddings = self._embed(inputs)
		return self.classifier(embeddings), embeddings


__all__ = ["ModelInput", "PCVRHyFormer"]