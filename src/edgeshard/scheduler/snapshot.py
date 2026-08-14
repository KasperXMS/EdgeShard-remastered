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
    available_memory_mb: int = 0  # Dynamic: free memory from latest heartbeat
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

    Attributes:
        workers: List of WorkerState snapshots.
        timestamp: Unix timestamp when snapshot was taken.
        bandwidth_matrix: Maps (worker_id_a, worker_id_b) → bandwidth in MB/s.
            Populated from M5 NetworkMetrics. Defaults to 100 MB/s for
            unknown pairs (conservative estimate).
        source_worker_id: The worker where input tokens originate.
            The scheduler enforces a privacy constraint: the first model
            layer must be placed on this worker. If None, the first
            worker in the list is used as source.
    """

    workers: list[WorkerState] = field(default_factory=list)
    timestamp: float = 0.0
    bandwidth_matrix: dict[tuple[str, str], float] = field(default_factory=dict)
    source_worker_id: str | None = None

    def total_memory_mb(self) -> int:
        return sum(w.available_memory_mb for w in self.workers)

    def get_bandwidth(self, worker_a: str, worker_b: str) -> float:
        """Get bandwidth between two workers in MB/s.

        Returns the measured bandwidth if available, otherwise a
        conservative default of 100 MB/s.
        """
        if worker_a == worker_b:
            return float("inf")  # Same worker, no network transfer
        key = (worker_a, worker_b)
        rev_key = (worker_b, worker_a)
        bw = self.bandwidth_matrix.get(key)
        if bw is not None:
            return bw
        bw = self.bandwidth_matrix.get(rev_key)
        if bw is not None:
            return bw
        return 100.0  # Conservative default: 100 MB/s

    def get_source_worker(self) -> str:
        """Get the source worker ID.

        Returns the explicitly set source_worker_id, or the first
        worker in the list if not set.
        """
        if self.source_worker_id is not None:
            return self.source_worker_id
        if self.workers:
            return str(self.workers[0].worker_id)
        raise ValueError("ClusterSnapshot has no workers")


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
