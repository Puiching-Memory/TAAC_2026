"""PCVR batch transform components."""

from __future__ import annotations

import torch

from taac2026.domain.config import (
    PCVRDataPipelineConfig,
    PCVRDataTransformConfig,
    PCVRDomainDropoutConfig,
    PCVRFeatureMaskConfig,
    PCVRSequenceCropConfig,
)
from taac2026.infrastructure.data.batches import PCVRBatch, PCVRBatchTransform, clone_pcvr_batch, repeat_pcvr_rows


class PCVRSequenceCropTransform:
    """Create time-safe sequence views from each batch."""

    def __init__(self, config: PCVRSequenceCropConfig) -> None:
        self.config = config

    def __call__(self, batch: PCVRBatch, *, generator: torch.Generator) -> PCVRBatch:
        if not self.config.enabled:
            return clone_pcvr_batch(batch)

        augmented = repeat_pcvr_rows(batch, max(1, int(self.config.views_per_row)))
        self._apply_sequence_crop(augmented, generator=generator)
        return augmented

    def _apply_sequence_crop(self, batch: PCVRBatch, *, generator: torch.Generator) -> None:
        for domain in batch.get("_seq_domains", []):
            sequence = batch.get(domain)
            lengths = batch.get(f"{domain}_len")
            time_buckets = batch.get(f"{domain}_time_bucket")
            if not isinstance(sequence, torch.Tensor) or not isinstance(lengths, torch.Tensor):
                continue
            if not isinstance(time_buckets, torch.Tensor):
                time_buckets = torch.zeros(sequence.shape[0], sequence.shape[2], dtype=torch.long)
                batch[f"{domain}_time_bucket"] = time_buckets
            self._crop_domain(sequence, lengths, time_buckets, generator=generator)

    def _crop_domain(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        time_buckets: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> None:
        max_len = int(sequence.shape[2])
        if max_len <= 0 or sequence.shape[0] == 0:
            return

        row_count = int(sequence.shape[0])
        device = sequence.device
        length_values = lengths.clamp(min=0, max=max_len)
        active_rows = length_values > 0
        if not bool(active_rows.any()):
            return

        min_len_config = max(1, int(self.config.seq_window_min_len))
        min_lengths = torch.minimum(torch.full_like(length_values, min_len_config), length_values).clamp(min=1)

        if self.config.seq_window_mode == "tail":
            window_lengths = length_values
        else:
            span = (length_values - min_lengths + 1).clamp(min=1)
            draws = torch.floor(torch.rand(row_count, generator=generator, device=device) * span.to(torch.float32)).to(
                length_values.dtype
            )
            window_lengths = torch.where(active_rows, min_lengths + draws, torch.zeros_like(length_values))

        if self.config.seq_window_mode == "rolling":
            max_starts = (length_values - window_lengths).clamp(min=0)
            start_draws = torch.floor(
                torch.rand(row_count, generator=generator, device=device) * (max_starts + 1).to(torch.float32)
            ).to(length_values.dtype)
            starts = torch.where(active_rows, start_draws, torch.zeros_like(length_values))
        else:
            starts = (length_values - window_lengths).clamp(min=0)

        positions = starts.view(-1, 1) + torch.arange(max_len, device=device).view(1, -1)
        positions = positions.clamp(max=max_len - 1).to(torch.long)
        valid_positions = torch.arange(max_len, device=device).view(1, -1) < window_lengths.view(-1, 1)

        sequence_positions = positions.view(row_count, 1, max_len).expand(row_count, sequence.shape[1], max_len)
        cropped_sequence = sequence.gather(2, sequence_positions)
        sequence.copy_(cropped_sequence.masked_fill(~valid_positions.view(row_count, 1, max_len), 0))

        cropped_time = time_buckets.gather(1, positions)
        time_buckets.copy_(cropped_time.masked_fill(~valid_positions, 0))
        lengths.copy_(window_lengths.to(lengths.dtype))


class PCVRDomainDropoutTransform:
    """Drop whole sequence domains for selected rows."""

    def __init__(self, config: PCVRDomainDropoutConfig) -> None:
        self.config = config

    def __call__(self, batch: PCVRBatch, *, generator: torch.Generator) -> PCVRBatch:
        if not self.config.enabled:
            return clone_pcvr_batch(batch)
        augmented = clone_pcvr_batch(batch)
        self._apply_domain_dropout(augmented, generator=generator)
        return augmented

    def _apply_domain_dropout(self, batch: PCVRBatch, *, generator: torch.Generator) -> None:
        probability = float(self.config.probability)
        if probability <= 0.0:
            return
        for domain in batch.get("_seq_domains", []):
            sequence = batch.get(domain)
            lengths = batch.get(f"{domain}_len")
            time_buckets = batch.get(f"{domain}_time_bucket")
            if not isinstance(sequence, torch.Tensor) or not isinstance(lengths, torch.Tensor):
                continue
            drop_mask = torch.rand(sequence.shape[0], generator=generator) < probability
            if not bool(drop_mask.any()):
                continue
            sequence[drop_mask] = 0
            lengths[drop_mask] = 0
            if isinstance(time_buckets, torch.Tensor):
                time_buckets[drop_mask] = 0


class PCVRFeatureMaskTransform:
    """Mask sparse features and sequence events in a batch."""

    def __init__(self, config: PCVRFeatureMaskConfig) -> None:
        self.config = config

    def __call__(self, batch: PCVRBatch, *, generator: torch.Generator) -> PCVRBatch:
        if not self.config.enabled:
            return clone_pcvr_batch(batch)
        augmented = clone_pcvr_batch(batch)
        self._apply_feature_masking(augmented, generator=generator)
        return augmented

    def _apply_feature_masking(self, batch: PCVRBatch, *, generator: torch.Generator) -> None:
        probability = float(self.config.probability)
        if probability <= 0.0:
            return

        for feature_key in ("user_int_feats", "item_int_feats"):
            features = batch.get(feature_key)
            if isinstance(features, torch.Tensor) and features.numel() > 0:
                mask = torch.rand(features.shape, generator=generator) < probability
                features[mask] = 0

        for domain in batch.get("_seq_domains", []):
            sequence = batch.get(domain)
            lengths = batch.get(f"{domain}_len")
            time_buckets = batch.get(f"{domain}_time_bucket")
            if not isinstance(sequence, torch.Tensor) or not isinstance(lengths, torch.Tensor):
                continue
            valid_positions = torch.arange(sequence.shape[2]).view(1, -1) < lengths.view(-1, 1)
            event_mask = (torch.rand(valid_positions.shape, generator=generator) < probability) & valid_positions
            if bool(event_mask.any()):
                expanded_mask = event_mask.unsqueeze(1).expand_as(sequence)
                sequence[expanded_mask] = 0
                if isinstance(time_buckets, torch.Tensor):
                    time_buckets[event_mask] = 0
                self._compact_domain(
                    sequence,
                    lengths,
                    time_buckets if isinstance(time_buckets, torch.Tensor) else None,
                    row_mask=event_mask.any(dim=1),
                )

    def _compact_domain(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        time_buckets: torch.Tensor | None,
        *,
        row_mask: torch.Tensor | None = None,
    ) -> None:
        max_len = int(sequence.shape[2])
        valid_positions = (sequence > 0).any(dim=1)
        if row_mask is None:
            row_indices = range(sequence.shape[0])
        else:
            row_indices = torch.nonzero(row_mask, as_tuple=False).flatten().tolist()
        for row_index in row_indices:
            row_positions = torch.nonzero(valid_positions[row_index], as_tuple=False).flatten()
            new_len = min(int(row_positions.numel()), max_len)
            if new_len <= 0:
                sequence[row_index].zero_()
                lengths[row_index] = 0
                if time_buckets is not None:
                    time_buckets[row_index].zero_()
                continue
            selected_sequence = sequence[row_index, :, row_positions[:new_len]].clone()
            sequence[row_index].zero_()
            sequence[row_index, :, :new_len] = selected_sequence
            lengths[row_index] = new_len
            if time_buckets is not None:
                selected_time = time_buckets[row_index, row_positions[:new_len]].clone()
                time_buckets[row_index].zero_()
                time_buckets[row_index, :new_len] = selected_time


def build_pcvr_batch_transform(config: PCVRDataTransformConfig) -> PCVRBatchTransform:
    if isinstance(config, PCVRSequenceCropConfig):
        return PCVRSequenceCropTransform(config)
    if isinstance(config, PCVRFeatureMaskConfig):
        return PCVRFeatureMaskTransform(config)
    if isinstance(config, PCVRDomainDropoutConfig):
        return PCVRDomainDropoutTransform(config)
    raise TypeError(f"unsupported PCVR data transform config: {type(config).__name__}")


def build_pcvr_batch_transforms(config: PCVRDataPipelineConfig) -> tuple[PCVRBatchTransform, ...]:
    return tuple(
        build_pcvr_batch_transform(transform)
        for transform in config.transforms
        if transform.enabled and not _is_noop_pcvr_batch_transform_config(transform)
    )


def _is_noop_pcvr_batch_transform_config(config: PCVRDataTransformConfig) -> bool:
    if isinstance(config, PCVRSequenceCropConfig):
        return config.views_per_row <= 1 and config.seq_window_mode == "tail"
    if isinstance(config, (PCVRFeatureMaskConfig, PCVRDomainDropoutConfig)):
        return float(config.probability) <= 0.0
    return False


__all__ = [
    "PCVRDomainDropoutTransform",
    "PCVRFeatureMaskTransform",
    "PCVRSequenceCropTransform",
    "build_pcvr_batch_transform",
    "build_pcvr_batch_transforms",
]
