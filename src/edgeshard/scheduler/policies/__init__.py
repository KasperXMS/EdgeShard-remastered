"""Scheduling policy interface.

All scheduling policies implement SchedulingPolicyBase.solve().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from edgeshard.scheduler.model_info import ModelSchedulingInfo
from edgeshard.scheduler.snapshot import (
    ClusterSnapshot,
    ProfileSnapshot,
    SchedulingPolicy,
)


@dataclass(frozen=True)
class DeviceSlot:
    """A usable device in the cluster.

    Attributes:
        worker_id: Worker that owns this device.
        device_id: Device identifier on the worker (e.g. "cuda:0").
        device_type: "cuda", "jetson", "cpu".
        device_name: Human-readable name (e.g. "RTX 4090").
        memory_budget_mb: Available memory for model + KV cache.
        layer_forward_ms: Per-layer compute time (from profiling).
    """

    worker_id: str
    device_id: str
    device_type: str
    device_name: str
    memory_budget_mb: float
    layer_forward_ms: float = 0.0


@dataclass(frozen=True)
class PartitionResult:
    """Result of the partitioning algorithm.

    Attributes:
        assignments: List of (layer_start, layer_end, device_slot) tuples.
            Each tuple maps a contiguous layer range to a device.
        estimated_latency_ms: Total estimated sequential latency (ms).
        estimated_throughput_tps: Estimated pipeline throughput (tokens/s).
        total_communication_ms: Total cross-device communication time.
    """

    assignments: list[tuple[int, int, DeviceSlot]]
    estimated_latency_ms: float = 0.0
    estimated_throughput_tps: float = 0.0
    total_communication_ms: float = 0.0


class SchedulingPolicyBase(ABC):
    """Base class for scheduling policies.

    A policy takes model info, cluster state, and profile data,
    then produces a partition plan (which layers go where).
    """

    @abstractmethod
    def solve(
        self,
        model_info: ModelSchedulingInfo,
        cluster: ClusterSnapshot,
        profiles: ProfileSnapshot,
        policy_config: SchedulingPolicy,
        max_seq_len: int = 512,
    ) -> PartitionResult:
        """Compute optimal layer partition and device assignment.

        Args:
            model_info: Model architecture metadata.
            cluster: Current cluster resources.
            profiles: Available profiling data.
            policy_config: Policy name and parameters.
            max_seq_len: Maximum sequence length for KV cache estimation.

        Returns:
            PartitionResult with layer->device assignments.

        Raises:
            SchedulerError: If no valid placement exists.
        """
