"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""

import gc
import logging
import random
import zlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from numpy.typing import NDArray
from torch.utils.data import IterableDataset, DataLoader

from taac2026.infrastructure.io.json_utils import dumps, read_path
from taac2026.infrastructure.pcvr.config import PCVRDataPipelineConfig
from taac2026.infrastructure.pcvr.data_observation import (
    PCVRRowGroupSplitPlan,
    build_pcvr_observed_schema_report,
    collect_pcvr_row_groups,
    plan_pcvr_row_group_split,
)
from taac2026.infrastructure.pcvr.data_pipeline import (
    PCVRBatchTransform,
    PCVRDataPipeline,
    PCVRMemoryBatchCache,
    PCVRSharedBatchCache,
    PCVRSharedTensorSpec,
    PCVRShuffleBuffer,
    build_pcvr_batch_transforms,
    stable_pcvr_batch_seed_from_path_crc,
)
from taac2026.infrastructure.pcvr.data_schema import (
    BUCKET_BOUNDARIES,
    FeatureSchema,
    NUM_TIME_BUCKETS,
)


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: dict[str, int] | None = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: tuple[int, int] | None = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        data_pipeline_config: PCVRDataPipelineConfig | None = None,
        transforms: Sequence[PCVRBatchTransform] | None = None,
        dataset_role: str = "dataset",
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.
            data_pipeline_config: optional cache and transform pipeline settings.
            transforms: optional additional batch transforms applied after
                conversion and before shuffle buffering.
        """
        super().__init__()

        # Accept either a directory or a single file path.
        parquet_root = Path(parquet_path).expanduser()
        if parquet_root.is_dir():
            files = sorted(parquet_root.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = [str(path) for path in files]
        else:
            self._parquet_files = [str(parquet_root)]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        self.dataset_role = str(dataset_role).strip() or "dataset"
        self.row_group_range = row_group_range
        self.schema_path = Path(schema_path).expanduser().resolve()
        self.data_pipeline_config = data_pipeline_config or PCVRDataPipelineConfig()
        self._strict_time_filter = bool(
            self.is_training
            and self.data_pipeline_config.enabled
            and self.data_pipeline_config.strict_time_filter
        )
        # Out-of-bound statistics:
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: dict[tuple[str, int], dict[str, int]] = {}

        # Build the list of Row Groups.
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)
        self._global_batch_keys: tuple[tuple[str, int, int], ...] | None = None
        self._scheduled_num_workers: int | None = None
        self._scheduled_cyclic = False

        # Load schema.json.
        self._load_schema(schema_path, seq_max_lens or {})

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- Pre-allocate numpy buffers ----
        B = batch_size
        self._buf_user_int = np.zeros(
            (B, self.user_int_schema.total_dim), dtype=np.int64
        )
        self._buf_item_int = np.zeros(
            (B, self.item_int_schema.total_dim), dtype=np.int64
        )
        self._buf_user_dense = np.zeros(
            (B, self.user_dense_schema.total_dim), dtype=np.float32
        )
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- Pre-compute (col_idx, offset, vocab_size) plans for int columns ----
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f"user_int_feats_{fid}")
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f"item_int_feats_{fid}")
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f"user_dense_feats_{fid}")
            self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        # Sequence column plan: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f"{prefix}_{fid}")
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = (
                self._col_idx.get(f"{prefix}_{ts_fid}") if ts_fid is not None else None
            )
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}"
        )

        pipeline_transforms = list(transforms or [])
        if self.is_training:
            pipeline_transforms.extend(
                build_pcvr_batch_transforms(self.data_pipeline_config)
            )
        self.pipeline = PCVRDataPipeline(
            cache=PCVRMemoryBatchCache.from_config(self.data_pipeline_config.cache),
            transforms=tuple(pipeline_transforms),
        )

    def _load_schema(self, schema_path: str, seq_max_lens: dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``."""
        resolved_schema_path = Path(schema_path).expanduser().resolve()
        raw = read_path(resolved_schema_path)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: list[list[int]] = raw["user_int"]
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: list[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)

        # ---- item_int ----
        self._item_int_cols: list[list[int]] = raw["item_int"]
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: list[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        # ---- user_dense: [[fid, dim], ...] ----
        self._user_dense_cols: list[list[int]] = raw["user_dense"]
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)

        # ---- item_dense (empty) ----
        self.item_dense_schema: FeatureSchema = FeatureSchema()

        # ---- sequence domains ----
        self._seq_cfg: dict[str, dict[str, Any]] = raw["seq"]
        self.seq_domains: list[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: dict[str, list[int]] = {}
        self.seq_vocab_sizes: dict[str, dict[int, int]] = {}
        self.seq_domain_vocab_sizes: dict[str, list[int]] = {}
        self.ts_fids: dict[str, int | None] = {}
        self.sideinfo_fids: dict[str, list[int]] = {}
        self._seq_prefix: dict[str, str] = {}
        self._seq_maxlen: dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg["prefix"]
            ts_fid = cfg["ts_fid"]
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg["features"]]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg["features"]}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len: from seq_max_lens arg; unspecified domains fall back to 256.
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

        logging.info(
            "Loaded PCVR schema for %s dataset: path=%s, row_groups=%s, user_int=%d (%d dims), item_int=%d (%d dims), user_dense=%d (%d dims), seq_domains=%s",
            self.dataset_role,
            resolved_schema_path,
            self.row_group_range if self.row_group_range is not None else "all",
            len(self._user_int_cols),
            self.user_int_schema.total_dim,
            len(self._item_int_cols),
            self.item_int_schema.total_dim,
            len(self._user_dense_cols),
            self.user_dense_schema.total_dim,
            ", ".join(self.seq_domains) if self.seq_domains else "<none>",
        )
        logging.info(
            "PCVR %s schema payload: %s",
            self.dataset_role,
            dumps(raw),
        )

    def __len__(self) -> int:
        # Ceiling per Row Group; this is an upper bound on the true batch count.
        return sum(
            (n + self.batch_size - 1) // self.batch_size for _, _, n in self._rg_list
        )

    def _iter_base_batch_keys(
        self, rg_list: Sequence[tuple[str, int, int]]
    ) -> Iterator[tuple[str, int, int]]:
        for file_path, rg_idx, row_count in rg_list:
            batch_count = (int(row_count) + self.batch_size - 1) // self.batch_size
            for batch_index in range(batch_count):
                yield (file_path, rg_idx, batch_index)

    def _resolved_global_batch_keys(self) -> tuple[tuple[str, int, int], ...]:
        if self._global_batch_keys is None:
            self._global_batch_keys = tuple(self._iter_base_batch_keys(self._rg_list))
        return self._global_batch_keys

    def configure_global_batch_schedule(
        self, *, num_workers: int, cyclic: bool = True
    ) -> None:
        self._scheduled_num_workers = max(1, int(num_workers))
        self._scheduled_cyclic = bool(cyclic)

    def _iter_worker_scheduled_batch_keys(
        self,
        *,
        worker_id: int,
        num_workers: int,
        cyclic: bool,
    ) -> Iterator[tuple[int, tuple[str, int, int]]]:
        global_batch_keys = self._resolved_global_batch_keys()
        worker_keys = [
            (batch_position, batch_key)
            for batch_position, batch_key in enumerate(global_batch_keys)
            if batch_position % num_workers == worker_id
        ]
        if not worker_keys:
            return

        while True:
            yield from worker_keys
            if not cyclic:
                return

    def _read_record_batch_for_key(
        self,
        *,
        parquet_files: dict[str, pq.ParquetFile],
        row_group_iterators: dict[tuple[str, int], tuple[Iterator[pa.RecordBatch], int]],
        file_path: str,
        row_group_index: int,
        batch_index: int,
    ) -> pa.RecordBatch:
        parquet_file = parquet_files.get(file_path)
        if parquet_file is None:
            parquet_file = pq.ParquetFile(file_path)
            parquet_files[file_path] = parquet_file

        iterator_key = (file_path, row_group_index)
        iterator_state = row_group_iterators.get(iterator_key)
        next_batch_index = 0
        if iterator_state is not None:
            iterator, next_batch_index = iterator_state
        else:
            iterator = parquet_file.iter_batches(
                batch_size=self.batch_size,
                row_groups=[row_group_index],
            )

        if batch_index < next_batch_index:
            iterator = parquet_file.iter_batches(
                batch_size=self.batch_size,
                row_groups=[row_group_index],
            )
            next_batch_index = 0

        while next_batch_index <= batch_index:
            try:
                record_batch = next(iterator)
            except StopIteration as exc:
                raise IndexError(
                    f"batch_index {batch_index} out of range for row group {row_group_index} in {file_path}"
                ) from exc
            if next_batch_index == batch_index:
                row_group_iterators[iterator_key] = (iterator, next_batch_index + 1)
                return record_batch
            next_batch_index += 1

        raise RuntimeError("failed to materialize requested parquet batch")

    def build_shared_opt_cache(self, num_workers: int) -> PCVRSharedBatchCache:
        tensor_specs: dict[str, PCVRSharedTensorSpec] = {
            "user_int_feats": PCVRSharedTensorSpec(
                shape=(self.batch_size, self.user_int_schema.total_dim),
                dtype=torch.long,
            ),
            "user_dense_feats": PCVRSharedTensorSpec(
                shape=(self.batch_size, self.user_dense_schema.total_dim),
                dtype=torch.float32,
            ),
            "item_int_feats": PCVRSharedTensorSpec(
                shape=(self.batch_size, self.item_int_schema.total_dim),
                dtype=torch.long,
            ),
            "item_dense_feats": PCVRSharedTensorSpec(
                shape=(self.batch_size, 0),
                dtype=torch.float32,
            ),
            "label": PCVRSharedTensorSpec(
                shape=(self.batch_size,),
                dtype=torch.long,
            ),
            "timestamp": PCVRSharedTensorSpec(
                shape=(self.batch_size,),
                dtype=torch.long,
            ),
        }
        for domain in self.seq_domains:
            tensor_specs[domain] = PCVRSharedTensorSpec(
                shape=(self.batch_size, len(self.sideinfo_fids[domain]), self._seq_maxlen[domain]),
                dtype=torch.long,
            )
            tensor_specs[f"{domain}_len"] = PCVRSharedTensorSpec(
                shape=(self.batch_size,),
                dtype=torch.long,
            )
            tensor_specs[f"{domain}_time_bucket"] = PCVRSharedTensorSpec(
                shape=(self.batch_size, self._seq_maxlen[domain]),
                dtype=torch.long,
            )

        cache = PCVRSharedBatchCache(
            enabled=self.data_pipeline_config.cache.enabled,
            max_batches=self.data_pipeline_config.cache.max_batches,
            policy="opt",
            tensor_specs=tensor_specs,
            static_values={"_seq_domains": list(self.seq_domains)},
        )
        cache.configure_access_trace(
            self._resolved_global_batch_keys(),
            cyclic=True,
        )
        return cache

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        scheduled_batch_keys: Iterator[tuple[int, tuple[str, int, int]]] | None = None
        if (
            worker_info is not None
            and self._scheduled_num_workers is not None
            and worker_info.num_workers == self._scheduled_num_workers
        ):
            scheduled_batch_keys = self._iter_worker_scheduled_batch_keys(
                worker_id=worker_id,
                num_workers=worker_info.num_workers,
                cyclic=self._scheduled_cyclic,
            )

        rg_list = self._rg_list
        if scheduled_batch_keys is None and worker_info is not None and worker_info.num_workers > 1:
            rg_list = [
                rg
                for i, rg in enumerate(rg_list)
                if i % worker_info.num_workers == worker_info.id
            ]

        if not getattr(self.pipeline.cache, "uses_global_access_trace", False):
            self.pipeline.configure_access_trace(
                self._iter_base_batch_keys(rg_list),
                cyclic=True,
            )

        shuffle_buffer = PCVRShuffleBuffer(
            batch_size=self.batch_size,
            buffer_batches=self.buffer_batches,
            shuffle=self.shuffle,
        )
        needs_generator = (
            self.pipeline.requires_generator or shuffle_buffer.requires_generator
        )
        base_seed = (
            self.data_pipeline_config.seed
            if self.data_pipeline_config.seed is not None
            else 0
        )
        path_crc_cache: dict[str, int] = {}
        parquet_files: dict[str, pq.ParquetFile] = {}
        row_group_iterators: dict[tuple[str, int], tuple[Iterator[pa.RecordBatch], int]] = {}

        if scheduled_batch_keys is not None:
            last_generator: torch.Generator | None = None
            for batch_position, (file_path, rg_idx, batch_index) in scheduled_batch_keys:
                path_crc = path_crc_cache.get(file_path)
                if path_crc is None:
                    path_crc = zlib.crc32(file_path.encode("utf-8"))
                    path_crc_cache[file_path] = path_crc
                cache_key = (file_path, rg_idx, batch_index)
                generator: torch.Generator | None = None
                if needs_generator:
                    batch_seed = stable_pcvr_batch_seed_from_path_crc(
                        base_seed=base_seed,
                        worker_id=batch_position,
                        path_crc=path_crc,
                        row_group_index=rg_idx,
                        batch_index=batch_index,
                    )
                    generator = torch.Generator().manual_seed(batch_seed)
                    last_generator = generator
                batch_dict = self.pipeline.read_base_batch(
                    cache_key,
                    lambda file_path=file_path, rg_idx=rg_idx, batch_index=batch_index: self._convert_batch(
                        self._read_record_batch_for_key(
                            parquet_files=parquet_files,
                            row_group_iterators=row_group_iterators,
                            file_path=file_path,
                            row_group_index=rg_idx,
                            batch_index=batch_index,
                        )
                    ),
                )
                batch_dict = self.pipeline.apply_transforms(
                    batch_dict, generator=generator
                )
                yield from shuffle_buffer.push(batch_dict, generator=generator)

            yield from shuffle_buffer.flush(generator=last_generator)

            del shuffle_buffer
            gc.collect()
            return

        current_file_path: str | None = None
        current_parquet_file: pq.ParquetFile | None = None
        current_path_crc = 0
        last_generator: torch.Generator | None = None
        for file_path, rg_idx, _ in rg_list:
            if file_path != current_file_path:
                current_file_path = file_path
                current_parquet_file = pq.ParquetFile(file_path)
                current_path_crc = zlib.crc32(file_path.encode("utf-8"))
            if current_parquet_file is None:
                continue
            for batch_index, batch in enumerate(
                current_parquet_file.iter_batches(
                    batch_size=self.batch_size, row_groups=[rg_idx]
                )
            ):
                cache_key = (file_path, rg_idx, batch_index)
                generator: torch.Generator | None = None
                if needs_generator:
                    batch_seed = stable_pcvr_batch_seed_from_path_crc(
                        base_seed=base_seed,
                        worker_id=worker_id,
                        path_crc=current_path_crc,
                        row_group_index=rg_idx,
                        batch_index=batch_index,
                    )
                    generator = torch.Generator().manual_seed(batch_seed)
                    last_generator = generator
                batch_dict = self.pipeline.read_base_batch(
                    cache_key, lambda batch=batch: self._convert_batch(batch)
                )
                batch_dict = self.pipeline.apply_transforms(
                    batch_dict, generator=generator
                )
                yield from shuffle_buffer.push(batch_dict, generator=generator)

        yield from shuffle_buffer.flush(generator=last_generator)

        del shuffle_buffer
        gc.collect()

    def _flush_buffer(self, buffer: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        shuffle_buffer = PCVRShuffleBuffer(
            batch_size=self.batch_size,
            buffer_batches=max(1, len(buffer)),
            shuffle=self.shuffle,
        )
        for batch in buffer:
            yield from shuffle_buffer.push(batch)
        yield from shuffle_buffer.flush()

    # ---- Helpers ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: NDArray[np.int64],
        vocab_size: int,
    ) -> None:
        """Record out-of-bound indices and (optionally) clip them to 0,
        without printing to the console.
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s["count"] += n
            s["max"] = max(s["max"], mx)
            s["min_oob"] = min(s["min_oob"], mn)
        else:
            self._oob_stats[key] = {
                "count": n,
                "max": mx,
                "min_oob": mn,
                "vocab": vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json"
            )

    def dump_oob_stats(self, path: str | None = None) -> None:
        """Dump out-of-bound statistics to a file if ``path`` is provided,
        otherwise to ``logging.info``.
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s["min_oob"] >= s["vocab"] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}"
            )
        msg = "\n".join(lines)
        if path:
            with Path(path).open("w") as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Pad an Arrow ``ListArray`` of ints to shape ``[B, max_len]``.

        Values <= 0 are mapped to 0 (padding). Note: the raw data contains -1
        (missing); currently treated the same way as 0 (padding).

        Returns:
            A tuple ``(padded, lengths)`` where ``padded`` has shape
            ``[B, max_len]`` and ``lengths`` has shape ``[B]``.
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[i, :use_len] = values[start : start + use_len]
            lengths[i] = use_len

        padded[padded <= 0] = 0
        return padded, lengths

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> NDArray[np.float32]:
        """Pad an Arrow ``ListArray<float>`` to shape ``[B, max_dim]``."""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_dim), dtype=np.float32)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start : start + use_len]

        return padded

    def _convert_batch(self, batch: "pa.RecordBatch") -> dict[str, Any]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors."""
        B = batch.num_rows

        # ---- meta ----
        timestamps = (
            batch.column(self._col_idx["timestamp"]).to_numpy().astype(np.int64)
        )
        if self.is_training:
            labels = (
                batch.column(self._col_idx["label_type"])
                .fill_null(0)
                .to_numpy(zero_copy_only=False)
                .astype(np.int64)
                == 2
            ).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
        user_ids = batch.column(self._col_idx["user_id"]).to_pylist()

        # ---- user_int: write into pre-allocated buffer ----
        # Note: null -> 0 (via fill_null), -1 -> 0 (via arr<=0); missing values
        # are treated the same as padding. Features with vs==0 have no vocab
        # information and are forced to 0 on the dataset side so that the
        # model's 1-slot Embedding (created for vs=0) is never indexed out of
        # range.
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob("user_int", ci, arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob("user_int", ci, padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset : offset + dim] = padded

        # ---- item_int ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob("item_int", ci, arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob("item_int", ci, padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset : offset + dim] = padded

        # ---- user_dense ----
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset : offset + dim] = padded

        result = {
            "user_int_feats": torch.from_numpy(user_int.copy()),
            "user_dense_feats": torch.from_numpy(user_dense.copy()),
            "item_int_feats": torch.from_numpy(item_int.copy()),
            "item_dense_feats": torch.zeros(B, 0, dtype=torch.float32),
            "label": torch.from_numpy(labels),
            "timestamp": torch.from_numpy(timestamps),
            "user_id": user_ids,
            "_seq_domains": self.seq_domains,
        }

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # Write directly into the pre-allocated 3D buffer.
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # Fused path: first collect (offsets, values, vocab_size, col_idx)
            # for every side-info column, then fill the buffer in a single pass.
            col_data = []
            for ci, _slot, vs in side_plan:
                col = batch.column(ci)
                col_data.append((col.offsets.to_numpy(), col.values.to_numpy(), vs, ci))

            ts_padded = np.zeros((B, max_len), dtype=np.int64)
            if self._strict_time_filter and ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offsets = ts_col.offsets.to_numpy()
                ts_values = ts_col.values.to_numpy()
                for row_index in range(B):
                    timestamp_start = int(ts_offsets[row_index])
                    timestamp_end = int(ts_offsets[row_index + 1])
                    if timestamp_end <= timestamp_start:
                        continue
                    row_timestamps = ts_values[timestamp_start:timestamp_end]
                    valid_positions = np.flatnonzero(
                        (row_timestamps > 0) & (row_timestamps < timestamps[row_index])
                    )
                    if valid_positions.size == 0:
                        continue
                    if valid_positions.size > max_len:
                        valid_positions = valid_positions[-max_len:]

                    view_len = int(valid_positions.size)
                    lengths[row_index] = view_len
                    ts_padded[row_index, :view_len] = row_timestamps[valid_positions]
                    for feature_index, (
                        side_offsets,
                        side_values,
                        _vocab_size,
                        _col_idx,
                    ) in enumerate(col_data):
                        side_start = int(side_offsets[row_index])
                        side_end = int(side_offsets[row_index + 1])
                        side_len = side_end - side_start
                        if side_len <= 0:
                            continue
                        side_positions = valid_positions[valid_positions < side_len]
                        if side_positions.size == 0:
                            continue
                        out[row_index, feature_index, : side_positions.size] = (
                            side_values[side_start + side_positions]
                        )
            else:
                for feature_index, (
                    side_offsets,
                    side_values,
                    _vocab_size,
                    _col_idx,
                ) in enumerate(col_data):
                    for row_index in range(B):
                        start = int(side_offsets[row_index])
                        end = int(side_offsets[row_index + 1])
                        raw_len = end - start
                        if raw_len <= 0:
                            continue
                        use_len = min(raw_len, max_len)
                        out[row_index, feature_index, :use_len] = side_values[
                            start : start + use_len
                        ]
                        if use_len > lengths[row_index]:
                            lengths[row_index] = use_len

                if ts_ci is not None:
                    ts_col = batch.column(ts_ci)
                    ts_offsets = ts_col.offsets.to_numpy()
                    ts_values = ts_col.values.to_numpy()
                    for row_index in range(B):
                        start = int(ts_offsets[row_index])
                        end = int(ts_offsets[row_index + 1])
                        raw_len = end - start
                        if raw_len <= 0:
                            continue
                        use_len = min(raw_len, max_len)
                        ts_padded[row_index, :use_len] = ts_values[
                            start : start + use_len
                        ]

            # Values <= 0 -> 0.
            out[out <= 0] = 0

            # Check out-of-bound values per feature's vocab_size.
            # vs==0 means no vocab info; force the whole slice to 0 so that
            # the model's 1-slot Embedding is never indexed out of range.
            for c, (_, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f"seq_{domain}", ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            result[domain] = torch.from_numpy(out.copy())
            result[f"{domain}_len"] = torch.from_numpy(lengths.copy())

            # Time bucketing.
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            if ts_ci is not None:
                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                # np.searchsorted returns values in [0, len(BUCKET_BOUNDARIES)].
                # After +1 the nominal range is [1, len(BUCKET_BOUNDARIES)+1];
                # the upper bound only appears when time_diff exceeds the
                # largest boundary (~1 year) and would index past
                # nn.Embedding(NUM_TIME_BUCKETS=len(BUCKET_BOUNDARIES)+1).
                # Clip raw result to [0, len(BUCKET_BOUNDARIES)-1] so the final
                # bucket id (after +1) stays within [1, len(BUCKET_BOUNDARIES)]
                # and is always a valid Embedding index. Time-diffs beyond the
                # largest boundary collapse into the last bucket.
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0,
                    len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets

            result[f"{domain}_time_bucket"] = torch.from_numpy(time_bucket.copy())

        return result


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: dict[str, int] | None = None,
    data_pipeline_config: PCVRDataPipelineConfig | None = None,
    **kwargs: Any,
) -> tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    The validation split is taken as the last ``valid_ratio`` fraction of Row
    Groups (in the file order returned by ``glob``).

    Returns:
        A tuple ``(train_loader, valid_loader, train_dataset)``. The third
        element is returned so the caller can access the feature schema
        (``user_int_schema``, ``item_int_schema``, ...) needed to construct
        the model.
    """
    random.seed(seed)

    rg_info = collect_pcvr_row_groups(data_dir)
    split_plan = plan_pcvr_row_group_split(
        rg_info, valid_ratio=valid_ratio, train_ratio=train_ratio
    )

    # train_ratio: use only the first N% of the training Row Groups.
    if train_ratio < 1.0 and not split_plan.reuse_train_for_valid:
        logging.info(
            f"train_ratio={train_ratio}: using {split_plan.train_row_groups} train Row Groups"
        )

    if split_plan.reuse_train_for_valid:
        logging.warning(
            "Single Row Group parquet detected; reusing the same Row Group for train and valid "
            "to keep smoke runs functional"
        )

    logging.info(
        f"Row Group split: {split_plan.train_row_groups} train ({split_plan.train_rows} rows), "
        f"{split_plan.valid_row_groups} valid ({split_plan.valid_rows} rows)"
    )

    train_pipeline_config = data_pipeline_config or PCVRDataPipelineConfig()
    valid_pipeline_config = PCVRDataPipelineConfig(cache=train_pipeline_config.cache)

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=split_plan.train_row_group_range,
        clip_vocab=clip_vocab,
        data_pipeline_config=train_pipeline_config,
        is_training=True,
        dataset_role="train",
    )

    if num_workers > 1 and train_pipeline_config.cache.mode == "opt":
        train_dataset.pipeline.cache = train_dataset.build_shared_opt_cache(
            num_workers=num_workers,
        )
        train_dataset.configure_global_batch_schedule(
            num_workers=num_workers,
            cyclic=True,
        )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw["persistent_workers"] = True
        _train_kw["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=use_cuda,
        **_train_kw,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=split_plan.valid_row_group_range,
        clip_vocab=clip_vocab,
        data_pipeline_config=valid_pipeline_config,
        is_training=True,
        dataset_role="valid",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=use_cuda,
    )

    logging.info(
        f"Parquet train: {split_plan.train_rows} rows, valid: {split_plan.valid_rows} rows, "
        f"batch_size={batch_size}, buffer_batches={buffer_batches}"
    )

    return train_loader, valid_loader, train_dataset


__all__ = [
    "BUCKET_BOUNDARIES",
    "FeatureSchema",
    "NUM_TIME_BUCKETS",
    "PCVRParquetDataset",
    "PCVRRowGroupSplitPlan",
    "build_pcvr_observed_schema_report",
    "collect_pcvr_row_groups",
    "get_pcvr_data",
    "plan_pcvr_row_group_split",
]
