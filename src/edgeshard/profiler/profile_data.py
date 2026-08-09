"""Profile data models — profiling result storage.

This module defines data structures for profiling results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ProfileResult:
    """Result of profiling a model on a specific device.

    Keyed by (model_name, model_revision, dtype, device_type, device_name).
    """

    # Key fields
    model_name: str
    model_revision: str = ""
    dtype: str = "float16"
    device_type: str = "cuda:0"  # "cuda:0", "jetson:0", "cpu:0"
    device_name: str = ""  # "NVIDIA GeForce RTX 4090"
    device_memory_mb: int = 0

    # Measurements
    layer_forward_ms: float = 0.0  # Average single-layer forward latency (ms)
    kv_cache_per_token_mb: float = 0.0  # KV cache memory per token (MB)
    prefill_tokens_per_sec: float = 0.0  # Prefill throughput
    decode_tokens_per_sec: float = 0.0  # Decode throughput
    total_model_memory_mb: float = 0.0  # Total model weight memory

    # Metadata
    num_layers_profiled: int = 0
    num_runs: int = 10  # Number of measurement runs (averaged)
    timestamp: float = 0.0  # Unix timestamp

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    @property
    def profile_key(self) -> tuple[str, str, str, str, str]:
        """Return the unique key for this profile."""
        return (
            self.model_name,
            self.model_revision,
            self.dtype,
            self.device_type,
            self.device_name,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "dtype": self.dtype,
            "device_type": self.device_type,
            "device_name": self.device_name,
            "device_memory_mb": self.device_memory_mb,
            "layer_forward_ms": self.layer_forward_ms,
            "kv_cache_per_token_mb": self.kv_cache_per_token_mb,
            "prefill_tokens_per_sec": self.prefill_tokens_per_sec,
            "decode_tokens_per_sec": self.decode_tokens_per_sec,
            "total_model_memory_mb": self.total_model_memory_mb,
            "num_layers_profiled": self.num_layers_profiled,
            "num_runs": self.num_runs,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ProfileResult:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
