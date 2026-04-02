from __future__ import annotations

from torch import nn

from ..config import ModelConfig
from .baseline import GrokDINReadoutBaseline, GrokUnifiedBaseline
from .din import CreatorwyxDINAdapter, CreatorwyxGroupedDINAdapter
from .sequence import RetrievalStyleAdapter, TencentSASRecAdapter
from .unified import DeepContextNet, UniRecDINReadoutModel, UniRecModel, UniScaleFormer


def build_model(config: ModelConfig, dense_dim: int, max_seq_len: int) -> nn.Module:
    name = "grok_baseline" if config.name == "baseline" else config.name
    if name == "grok_baseline":
        return GrokUnifiedBaseline(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
    if name == "grok_din_readout":
        return GrokDINReadoutBaseline(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
    if name == "creatorwyx_din_adapter":
        return CreatorwyxDINAdapter(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
    if name == "creatorwyx_grouped_din_adapter":
        return CreatorwyxGroupedDINAdapter(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
    if name == "tencent_sasrec_adapter":
        return TencentSASRecAdapter(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
    if name in {"zcyeee_retrieval_adapter", "o_o_retrieval_adapter", "omnigenrec_adapter"}:
        return RetrievalStyleAdapter(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len, variant=name)
    if name == "deep_context_net":
        return DeepContextNet(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
    if name == "unirec":
        return UniRecModel(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
    if name == "unirec_din_readout":
        return UniRecDINReadoutModel(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
    if name == "uniscaleformer":
        return UniScaleFormer(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
    raise ValueError(f"Unsupported model name: {config.name}")


__all__ = [
    "CreatorwyxDINAdapter",
    "CreatorwyxGroupedDINAdapter",
    "DeepContextNet",
    "GrokDINReadoutBaseline",
    "GrokUnifiedBaseline",
    "TencentSASRecAdapter",
    "UniRecDINReadoutModel",
    "UniRecModel",
    "UniScaleFormer",
    "build_model",
]