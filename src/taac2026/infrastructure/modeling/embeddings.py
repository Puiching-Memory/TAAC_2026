"""Embedding model primitives."""

from __future__ import annotations

import torch
import torch.nn as nn

from taac2026.infrastructure.accelerators.embedding.embedding_bag import embedding_bag_mean


def hash_compress_ids(values: torch.Tensor, num_buckets: int) -> torch.Tensor:
	if num_buckets <= 0:
		return values.new_zeros(values.shape)
	positive = values > 0
	compressed = torch.remainder(values.clamp_min(1) - 1, int(num_buckets)) + 1
	return torch.where(positive, compressed, values.new_zeros(values.shape))


class FeatureEmbeddingBank(nn.Module):
	def __init__(
		self,
		feature_specs: list[tuple[int, int, int]],
		emb_dim: int,
		emb_skip_threshold: int = 0,
		*,
		compress_high_cardinality: bool = False,
	) -> None:
		super().__init__()
		self.feature_specs = list(feature_specs)
		self.emb_dim = emb_dim
		self.compress_high_cardinality = bool(compress_high_cardinality)
		self.embeddings = nn.ModuleList()
		self.compressed_embeddings = nn.ModuleList()
		self._embedding_index: list[int] = []
		self._compressed_embedding_index: list[int] = []
		for vocab_size, _offset, _length in self.feature_specs:
			if int(vocab_size) <= 0:
				self._embedding_index.append(-1)
				self._compressed_embedding_index.append(-1)
			elif emb_skip_threshold > 0 and int(vocab_size) > emb_skip_threshold and self.compress_high_cardinality:
				self._embedding_index.append(-1)
				self._compressed_embedding_index.append(len(self.compressed_embeddings))
				self.compressed_embeddings.append(nn.Embedding(int(emb_skip_threshold) + 1, emb_dim, padding_idx=0))
			elif emb_skip_threshold > 0 and int(vocab_size) > emb_skip_threshold:
				self._embedding_index.append(-1)
				self._compressed_embedding_index.append(-1)
			else:
				self._embedding_index.append(len(self.embeddings))
				self._compressed_embedding_index.append(-1)
				self.embeddings.append(nn.Embedding(int(vocab_size) + 1, emb_dim, padding_idx=0))
		self.reset_parameters()

	@property
	def output_dim(self) -> int:
		return self.emb_dim

	def reset_parameters(self) -> None:
		for embedding in [*self.embeddings, *self.compressed_embeddings]:
			nn.init.xavier_normal_(embedding.weight)
			embedding.weight.data[0].zero_()

	def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
		batch_size = int_feats.shape[0]
		if not self.feature_specs:
			return int_feats.new_zeros(batch_size, 0, self.emb_dim, dtype=torch.float32)
		tokens: list[torch.Tensor] = []
		for feature_index, (vocab_size, offset, length) in enumerate(self.feature_specs):
			embedding_index = self._embedding_index[feature_index]
			compressed_index = self._compressed_embedding_index[feature_index]
			if embedding_index >= 0:
				values = int_feats[:, offset : offset + length].to(torch.long).clamp(min=0, max=int(vocab_size))
				tokens.append(embedding_bag_mean(self.embeddings[embedding_index].weight, values))
			elif compressed_index >= 0:
				embedding = self.compressed_embeddings[compressed_index]
				values = int_feats[:, offset : offset + length].to(torch.long).clamp(min=0)
				compressed_values = hash_compress_ids(values, embedding.num_embeddings - 1)
				tokens.append(embedding_bag_mean(embedding.weight, compressed_values))
			else:
				tokens.append(int_feats.new_zeros(batch_size, self.emb_dim, dtype=torch.float32))
		return torch.stack(tokens, dim=1)


class EmbeddingParameterMixin:
	def get_sparse_params(self) -> list[nn.Parameter]:
		sparse_ptrs = {module.weight.data_ptr() for module in self.modules() if isinstance(module, nn.Embedding)}
		return [parameter for parameter in self.parameters() if parameter.data_ptr() in sparse_ptrs]

	def get_dense_params(self) -> list[nn.Parameter]:
		sparse_ptrs = {parameter.data_ptr() for parameter in self.get_sparse_params()}
		return [parameter for parameter in self.parameters() if parameter.data_ptr() not in sparse_ptrs]

	def reinit_high_cardinality_params(self, cardinality_threshold: int = 10000) -> set[int]:
		reinitialized: set[int] = set()
		for module in self.modules():
			if not isinstance(module, nn.Embedding):
				continue
			if module.num_embeddings - 1 <= cardinality_threshold:
				continue
			nn.init.xavier_normal_(module.weight)
			module.weight.data[0].zero_()
			reinitialized.add(module.weight.data_ptr())
		return reinitialized


__all__ = ["EmbeddingParameterMixin", "FeatureEmbeddingBank", "hash_compress_ids"]