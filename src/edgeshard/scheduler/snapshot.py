"""Cluster and profile snapshots consumed by the Scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from edgeshard.common.identifiers import WorkerId


@dataclass
class DeviceInfo:
    """Normalized hardware information for a single device."""

    device_id: str
    device_type: str  # "cuda", "jetson", "cpu"
    name: str
    total_memory_mb: int
    compute_capability: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerState:
    """Snapshot of a single Worker's resources and status."""

    worker_id: WorkerId
    hostname: str
    devices: list[DeviceInfo] = field(default_factory=list)
    available_memory_mb: int = 0
    cpu_count: int = 1
    status: str = "online"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterSnapshot:
    """Immutable snapshot of all Workers and their resources.

    This is the input to the Scheduler — never raw monitoring data.
    """

    workers: list[WorkerState] = field(default_factory=list)
    timestamp: float = 0.0

    def total_memory_mb(self) -> int:
        return sum(w.available_memory_mb for w in self.workers)


@dataclass
class ProfileEntry:
    """Profiling result for a model/device/dtype combination."""

    model_name: str
    model_revision: str
    device_fingerprint: str
    dtype: str
    runtime_version: str
    layer_forward_ms: float = 0.0
    kv_cache_per_token_mb: float = 0.0
    activation_memory_mb: float = 0.0
    prefill_tokens_per_sec: float = 0.0
    decode_tokens_per_sec: float = 0.0


@dataclass
class ProfileSnapshot:
    """Collection of profiling results available to the Scheduler."""

    entries: list[ProfileEntry] = field(default_factory=list)

    def get(
        self,
        model_name: str,
        device_fingerprint: str,
        dtype: str,
    ) -> ProfileEntry | None:
        for entry in self.entries:
            if (
                entry.model_name == model_name
                and entry.device_fingerprint == device_fingerprint
                and entry.dtype == dtype
            ):
                return entry
        return None


@dataclass
class SchedulingPolicy:
    """Policy configuration for the Scheduler."""

    name: str = "default"
    params: dict[str, Any] = field(default_factory=dict)
