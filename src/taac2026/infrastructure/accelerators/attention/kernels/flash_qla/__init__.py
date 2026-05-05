__version__ = "0.1.0"

from .ops.gated_delta_rule.chunk import (
    flash_qla_available,
    flash_qla_device_compute_version,
    flash_qla_target_compute_version,
    chunk_gated_delta_rule_fwd,
    chunk_gated_delta_rule_bwd,
    chunk_gated_delta_rule,
)

__all__ = [
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_bwd",
    "chunk_gated_delta_rule_fwd",
    "flash_qla_available",
    "flash_qla_device_compute_version",
    "flash_qla_target_compute_version",
]
