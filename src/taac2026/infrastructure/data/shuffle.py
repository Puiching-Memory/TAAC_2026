"""PCVR row-level shuffle buffer."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from taac2026.infrastructure.data.batches import (
    PCVRBatch,
    concat_pcvr_batches,
    pcvr_batch_row_count,
    take_pcvr_rows,
)


class PCVRShuffleBuffer:
    """Row-level shuffle buffer for pre-batched PCVR tensor dictionaries."""

    def __init__(self, *, batch_size: int, buffer_batches: int, shuffle: bool) -> None:
        self.batch_size = int(batch_size)
        self.buffer_batches = int(buffer_batches)
        self.shuffle = shuffle
        self._buffer: list[PCVRBatch] = []

    @property
    def requires_generator(self) -> bool:
        return self.shuffle and self.buffer_batches > 1

    def push(self, batch: PCVRBatch, *, generator: torch.Generator | None = None) -> Iterator[PCVRBatch]:
        if self.shuffle and self.buffer_batches > 1:
            self._buffer.append(batch)
            if len(self._buffer) >= self.buffer_batches:
                yield from self.flush(generator=generator)
        else:
            yield batch

    def flush(self, *, generator: torch.Generator | None = None) -> Iterator[PCVRBatch]:
        if not self._buffer:
            return
        merged = concat_pcvr_batches(self._buffer)
        total_rows = pcvr_batch_row_count(merged)
        if self.shuffle:
            row_order = torch.randperm(total_rows, generator=generator)
        else:
            row_order = torch.arange(total_rows)
        for start in range(0, total_rows, self.batch_size):
            end = min(start + self.batch_size, total_rows)
            yield take_pcvr_rows(merged, row_order[start:end])
        self._buffer.clear()


__all__ = ["PCVRShuffleBuffer"]
