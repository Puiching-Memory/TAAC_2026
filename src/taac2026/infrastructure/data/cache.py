"""PCVR data cache components."""

from __future__ import annotations

import multiprocessing as mp
from collections import OrderedDict
from collections.abc import Hashable, Iterable
from typing import Any

import torch

from taac2026.domain.config import PCVRDataCacheConfig
from taac2026.infrastructure.data.batches import (
    PCVRBatch,
    PCVRSharedTensorSpec,
    clone_pcvr_batch,
    pcvr_batch_row_count,
)


class PCVRMemoryBatchCache:
    """Small per-process LRU cache for converted base PCVR batches."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        max_batches: int = 0,
        policy: str = "lru",
    ) -> None:
        self.enabled = enabled
        self.max_batches = max(0, int(max_batches))
        self.policy = policy if policy == "opt" else "lru"
        self._items: OrderedDict[Hashable, PCVRBatch] = OrderedDict()
        self._trace_positions: dict[Hashable, int] = {}
        self._trace_length = 0
        self._trace_cyclic = True
        self._access_count = 0
        self._opt_fallback = False

    @classmethod
    def from_config(cls, config: PCVRDataCacheConfig | None) -> PCVRMemoryBatchCache:
        if config is None:
            return cls()
        return cls(
            enabled=config.enabled,
            max_batches=config.max_batches,
            policy="opt" if config.mode == "opt" else "lru",
        )

    def configure_access_trace(self, trace: Iterable[Hashable], *, cyclic: bool = True) -> None:
        if not self._opt_enabled or self.max_batches <= 0:
            return

        trace_positions: dict[Hashable, int] = {}
        trace_length = 0
        for key in trace:
            if key in trace_positions:
                self._opt_fallback = True
                return
            trace_positions[key] = trace_length
            trace_length += 1

        if trace_length == 0:
            self._trace_positions = {}
            self._trace_length = 0
            self._access_count = 0
            self._opt_fallback = False
            return

        if trace_length == self._trace_length and cyclic == self._trace_cyclic and trace_positions == self._trace_positions:
            return

        self._items.clear()
        self._trace_positions = trace_positions
        self._trace_length = trace_length
        self._trace_cyclic = cyclic
        self._access_count = 0
        self._opt_fallback = False

    def get(self, key: Hashable) -> PCVRBatch | None:
        if not self.enabled:
            return None
        self._record_access(key)
        batch = self._items.get(key)
        if batch is None:
            return None
        if not self._opt_active:
            self._items.move_to_end(key)
        return clone_pcvr_batch(batch)

    def put(self, key: Hashable, batch: PCVRBatch) -> None:
        if not self.enabled:
            return
        self._items[key] = clone_pcvr_batch(batch)
        if not self._opt_active:
            self._items.move_to_end(key)
            while self.max_batches > 0 and len(self._items) > self.max_batches:
                self._items.popitem(last=False)
            return

        while self.max_batches > 0 and len(self._items) > self.max_batches:
            victim = self._select_opt_victim()
            del self._items[victim]

    def __len__(self) -> int:
        return len(self._items)

    @property
    def _opt_enabled(self) -> bool:
        return self.enabled and self.policy == "opt"

    @property
    def _opt_active(self) -> bool:
        return self._opt_enabled and self._trace_length > 0 and not self._opt_fallback

    def _record_access(self, key: Hashable) -> None:
        if not self._opt_active:
            return
        expected_position = self._access_count % self._trace_length
        if self._trace_positions.get(key) != expected_position:
            self._opt_fallback = True
            return
        self._access_count += 1

    def _next_use(self, key: Hashable) -> float:
        position = self._trace_positions.get(key)
        if position is None or self._trace_length <= 0:
            return float("inf")
        cycle_index, cycle_position = divmod(self._access_count, self._trace_length)
        if position >= cycle_position:
            return float(cycle_index * self._trace_length + position)
        if self._trace_cyclic:
            return float((cycle_index + 1) * self._trace_length + position)
        return float("inf")

    def _select_opt_victim(self) -> Hashable:
        victim: Hashable | None = None
        farthest = float("-inf")
        for key in self._items:
            next_use = self._next_use(key)
            if next_use > farthest:
                farthest = next_use
                victim = key
        if victim is None:
            raise RuntimeError("OPT cache eviction requested for empty cache")
        return victim


class PCVRSharedBatchCache:
    """Shared-memory CPU batch cache for multi-worker single-card training."""

    uses_global_access_trace = True

    def __init__(
        self,
        *,
        enabled: bool = False,
        max_batches: int = 0,
        policy: str = "lru",
        tensor_specs: dict[str, PCVRSharedTensorSpec] | None = None,
        static_values: dict[str, Any] | None = None,
    ) -> None:
        self.enabled = enabled
        self.max_batches = max(0, int(max_batches))
        self.policy = policy if policy == "opt" else "lru"
        self.static_values = dict(static_values or {})
        self.tensor_specs = dict(tensor_specs or {})
        self._manager = mp.Manager()
        self._lock = mp.RLock()
        self._key_to_slot = self._manager.dict()
        self._slot_to_key = self._manager.list([None] * self.max_batches)
        self._row_counts = torch.zeros(self.max_batches, dtype=torch.int64).share_memory_()
        self._last_access = torch.zeros(self.max_batches, dtype=torch.int64).share_memory_()
        self._touch_counter = mp.Value("q", 0)
        self._access_count = mp.Value("q", 0)
        self._opt_fallback = mp.Value("b", False)
        self._trace_positions: dict[Hashable, int] = {}
        self._trace_length = 0
        self._trace_cyclic = True
        self._storage = {
            key: torch.empty((self.max_batches, *spec.shape), dtype=spec.dtype).share_memory_()
            for key, spec in self.tensor_specs.items()
        }

    def __len__(self) -> int:
        return len(self._key_to_slot)

    @property
    def _opt_enabled(self) -> bool:
        return self.enabled and self.policy == "opt"

    @property
    def _opt_active(self) -> bool:
        return self._opt_enabled and self._trace_length > 0 and not bool(self._opt_fallback.value)

    def configure_access_trace(self, trace: Iterable[Hashable], *, cyclic: bool = True) -> None:
        if not self._opt_enabled or self.max_batches <= 0:
            return
        trace_positions: dict[Hashable, int] = {}
        trace_length = 0
        for key in trace:
            if key in trace_positions:
                with self._opt_fallback.get_lock():
                    self._opt_fallback.value = True
                return
            trace_positions[key] = trace_length
            trace_length += 1

        if not trace_positions:
            self._trace_positions = {}
            self._trace_length = 0
            with self._access_count.get_lock():
                self._access_count.value = 0
            with self._opt_fallback.get_lock():
                self._opt_fallback.value = False
            return

        if trace_length == self._trace_length and cyclic == self._trace_cyclic and trace_positions == self._trace_positions:
            return

        self._trace_positions = trace_positions
        self._trace_length = trace_length
        self._trace_cyclic = cyclic
        with self._access_count.get_lock():
            self._access_count.value = 0
        with self._opt_fallback.get_lock():
            self._opt_fallback.value = False

    def get(self, key: Hashable) -> PCVRBatch | None:
        if not self.enabled or self.max_batches <= 0:
            return None
        with self._lock:
            self._record_access(key)
            slot = self._key_to_slot.get(key)
            if slot is None:
                return None
            slot_index = int(slot)
            self._touch_slot(slot_index)
            row_count = int(self._row_counts[slot_index].item())
            return self._materialize_slot(slot_index, row_count)

    def put(self, key: Hashable, batch: PCVRBatch) -> None:
        if not self.enabled or self.max_batches <= 0:
            return
        with self._lock:
            slot = self._key_to_slot.get(key)
            if slot is None:
                slot_index = self._allocate_slot_for_key(key)
                if slot_index is None:
                    return
            else:
                slot_index = int(slot)
            self._write_slot(slot_index, key, batch)

    def _allocate_slot_for_key(self, key: Hashable) -> int | None:
        for index, slot_key in enumerate(list(self._slot_to_key)):
            if slot_key is None:
                return index

        if self._opt_active:
            victim_key = self._select_opt_victim()
            victim_next_use = self._next_use(victim_key)
            candidate_next_use = self._next_use(key)
            if candidate_next_use >= victim_next_use:
                return None
            victim_slot = int(self._key_to_slot[victim_key])
        else:
            victim_slot = self._select_lru_victim_slot()
            victim_key = self._slot_to_key[victim_slot]

        if victim_key is not None:
            self._key_to_slot.pop(victim_key, None)
            self._slot_to_key[victim_slot] = None
        return victim_slot

    def _write_slot(self, slot_index: int, key: Hashable, batch: PCVRBatch) -> None:
        row_count = pcvr_batch_row_count(batch)
        for tensor_key, storage in self._storage.items():
            value = batch.get(tensor_key)
            if not isinstance(value, torch.Tensor):
                raise KeyError(f"shared cache tensor key missing from batch: {tensor_key}")
            target = storage[slot_index]
            target.zero_()
            if row_count > 0:
                target[:row_count].copy_(value)
        self._row_counts[slot_index] = row_count
        self._slot_to_key[slot_index] = key
        self._key_to_slot[key] = slot_index
        self._touch_slot(slot_index)

    def _materialize_slot(self, slot_index: int, row_count: int) -> PCVRBatch:
        batch: PCVRBatch = dict(self.static_values)
        for tensor_key, storage in self._storage.items():
            batch[tensor_key] = storage[slot_index][:row_count].clone()
        return batch

    def _touch_slot(self, slot_index: int) -> None:
        with self._touch_counter.get_lock():
            self._touch_counter.value += 1
            self._last_access[slot_index] = self._touch_counter.value

    def _record_access(self, key: Hashable) -> None:
        if not self._opt_active:
            return
        expected_position = self._access_count.value % self._trace_length
        if self._trace_positions.get(key) != expected_position:
            with self._opt_fallback.get_lock():
                self._opt_fallback.value = True
            return
        with self._access_count.get_lock():
            self._access_count.value += 1

    def _next_use(self, key: Hashable) -> float:
        position = self._trace_positions.get(key)
        if position is None or self._trace_length <= 0:
            return float("inf")
        access_count = self._access_count.value
        cycle_index, cycle_position = divmod(access_count, self._trace_length)
        if position >= cycle_position:
            return float(cycle_index * self._trace_length + position)
        if self._trace_cyclic:
            return float((cycle_index + 1) * self._trace_length + position)
        return float("inf")

    def _select_opt_victim(self) -> Hashable:
        victim: Hashable | None = None
        farthest = float("-inf")
        for key in list(self._key_to_slot.keys()):
            next_use = self._next_use(key)
            if next_use > farthest:
                farthest = next_use
                victim = key
        if victim is None:
            raise RuntimeError("shared OPT cache eviction requested for empty cache")
        return victim

    def _select_lru_victim_slot(self) -> int:
        victim_slot = -1
        oldest = None
        for slot_index, key in enumerate(list(self._slot_to_key)):
            if key is None:
                return slot_index
            slot_access = int(self._last_access[slot_index].item())
            if oldest is None or slot_access < oldest:
                oldest = slot_access
                victim_slot = slot_index
        if victim_slot < 0:
            raise RuntimeError("shared cache LRU eviction requested for empty cache")
        return victim_slot


__all__ = ["PCVRMemoryBatchCache", "PCVRSharedBatchCache", "PCVRSharedTensorSpec"]
