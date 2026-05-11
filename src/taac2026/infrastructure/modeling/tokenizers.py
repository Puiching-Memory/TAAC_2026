"""Tokenizer model primitives."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from taac2026.infrastructure.modeling.embeddings import FeatureEmbeddingBank, hash_compress_ids


class NonSequentialTokenizer(nn.Module):
	def __init__(
		self,
		feature_specs: list[tuple[int, int, int]],
		groups: list[list[int]],
		emb_dim: int,
		d_model: int,
		num_tokens: int = 0,
		emb_skip_threshold: int = 0,
		compress_high_cardinality: bool = False,
		force_auto_split: bool = False,
	) -> None:
		super().__init__()
		self.bank = FeatureEmbeddingBank(
			feature_specs,
			emb_dim,
			emb_skip_threshold,
			compress_high_cardinality=compress_high_cardinality,
		)
		self.groups = [list(group) for group in groups] or [[index] for index in range(len(feature_specs))]
		self.feature_count = len(feature_specs)
		self.num_tokens = int(num_tokens) if num_tokens > 0 else len(self.groups)
		self.auto_split = force_auto_split or self.num_tokens != len(self.groups)
		if self.auto_split:
			input_dim = max(1, self.feature_count * emb_dim)
			self.project = nn.Sequential(
				nn.Linear(input_dim, self.num_tokens * d_model),
				nn.SiLU(),
				nn.LayerNorm(self.num_tokens * d_model),
			)
		else:
			self.project = nn.Sequential(nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model))
		self.d_model = d_model

	@property
	def embeddings(self) -> Iterable[nn.Embedding]:
		return self.bank.embeddings

	def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
		batch_size = int_feats.shape[0]
		feature_tokens = self.bank(int_feats)
		if self.num_tokens <= 0:
			return int_feats.new_zeros(batch_size, 0, self.d_model, dtype=torch.float32)
		if self.auto_split:
			if feature_tokens.shape[1] == 0:
				flat = int_feats.new_zeros(batch_size, 1, dtype=torch.float32)
			else:
				flat = feature_tokens.reshape(batch_size, -1)
			return self.project(flat).view(batch_size, self.num_tokens, self.d_model)
		grouped_tokens: list[torch.Tensor] = []
		for group in self.groups:
			valid_indices = [index for index in group if 0 <= index < feature_tokens.shape[1]]
			if valid_indices:
				grouped_tokens.append(feature_tokens[:, valid_indices, :].mean(dim=1))
			else:
				grouped_tokens.append(int_feats.new_zeros(batch_size, self.bank.output_dim, dtype=torch.float32))
		return self.project(torch.stack(grouped_tokens, dim=1))


class DenseTokenProjector(nn.Module):
	def __init__(self, input_dim: int, d_model: int) -> None:
		super().__init__()
		self.input_dim = input_dim
		if input_dim > 0:
			self.project = nn.Sequential(nn.Linear(input_dim, d_model), nn.SiLU(), nn.LayerNorm(d_model))
		else:
			self.project = None

	def forward(self, features: torch.Tensor) -> torch.Tensor | None:
		if self.project is None:
			return None
		return self.project(features).unsqueeze(1)


class SequenceTokenizer(nn.Module):
	def __init__(
		self,
		vocab_sizes: list[int],
		emb_dim: int,
		d_model: int,
		num_time_buckets: int = 0,
		emb_skip_threshold: int = 0,
		compress_high_cardinality: bool = False,
	) -> None:
		super().__init__()
		self.vocab_sizes = [int(value) for value in vocab_sizes]
		self.emb_dim = emb_dim
		self.compress_high_cardinality = bool(compress_high_cardinality)
		self.embeddings = nn.ModuleList()
		self.compressed_embeddings = nn.ModuleList()
		self._embedding_index: list[int] = []
		self._compressed_embedding_index: list[int] = []
		for vocab_size in self.vocab_sizes:
			if vocab_size <= 0:
				self._embedding_index.append(-1)
				self._compressed_embedding_index.append(-1)
			elif emb_skip_threshold > 0 and vocab_size > emb_skip_threshold and self.compress_high_cardinality:
				self._embedding_index.append(-1)
				self._compressed_embedding_index.append(len(self.compressed_embeddings))
				self.compressed_embeddings.append(nn.Embedding(int(emb_skip_threshold) + 1, emb_dim, padding_idx=0))
			elif emb_skip_threshold > 0 and vocab_size > emb_skip_threshold:
				self._embedding_index.append(-1)
				self._compressed_embedding_index.append(-1)
			else:
				self._embedding_index.append(len(self.embeddings))
				self._compressed_embedding_index.append(-1)
				self.embeddings.append(nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0))
		input_dim = max(1, len(self.vocab_sizes) * emb_dim)
		self.project = nn.Sequential(nn.Linear(input_dim, d_model), nn.SiLU(), nn.LayerNorm(d_model))
		self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0) if num_time_buckets > 0 else None
		self.reset_parameters()

	def reset_parameters(self) -> None:
		for embedding in [*self.embeddings, *self.compressed_embeddings]:
			nn.init.xavier_normal_(embedding.weight)
			embedding.weight.data[0].zero_()
		if self.time_embedding is not None:
			nn.init.xavier_normal_(self.time_embedding.weight)
			self.time_embedding.weight.data[0].zero_()

	def forward(self, sequence: torch.Tensor, time_buckets: torch.Tensor | None = None) -> torch.Tensor:
		batch_size, feature_count, seq_len = sequence.shape
		pieces: list[torch.Tensor] = []
		for feature_index in range(feature_count):
			embedding_index = self._embedding_index[feature_index] if feature_index < len(self._embedding_index) else -1
			compressed_index = self._compressed_embedding_index[feature_index] if feature_index < len(self._compressed_embedding_index) else -1
			if embedding_index >= 0:
				vocab_size = self.vocab_sizes[feature_index]
				values = sequence[:, feature_index, :].to(torch.long).clamp(min=0, max=vocab_size)
				pieces.append(self.embeddings[embedding_index](values))
			elif compressed_index >= 0:
				embedding = self.compressed_embeddings[compressed_index]
				values = sequence[:, feature_index, :].to(torch.long).clamp(min=0)
				pieces.append(embedding(hash_compress_ids(values, embedding.num_embeddings - 1)))
			else:
				pieces.append(sequence.new_zeros(batch_size, seq_len, self.emb_dim, dtype=torch.float32))
		if pieces:
			token_input = torch.cat(pieces, dim=-1)
		else:
			token_input = sequence.new_zeros(batch_size, seq_len, 1, dtype=torch.float32)
		tokens = self.project(token_input)
		if self.time_embedding is not None and time_buckets is not None:
			time_values = time_buckets.to(torch.long).clamp(min=0, max=self.time_embedding.num_embeddings - 1)
			tokens = tokens + self.time_embedding(time_values)
		return tokens


__all__ = ["DenseTokenProjector", "NonSequentialTokenizer", "SequenceTokenizer"]