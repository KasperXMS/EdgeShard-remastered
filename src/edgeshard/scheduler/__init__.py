"""EdgeShard Scheduler — placement planning for distributed inference.

Public API:
    schedule() — Core scheduling entry point (pure function)
    PlacementPlan, ShardPlacement — Immutable output data models
    ClusterSnapshot, ProfileSnapshot — Input data models
    DeviceSlot, PartitionResult — Internal algorithm types
"""

from edgeshard.scheduler.placement import PlacementPlan, ShardPlacement
from edgeshard.scheduler.planner import schedule
from edgeshard.scheduler.policies import DeviceSlot, PartitionResult
from edgeshard.scheduler.snapshot import (
    ClusterSnapshot,
    DeviceInfo,
    ProfileEntry,
    ProfileSnapshot,
    SchedulingPolicy,
    WorkerState,
)

__all__ = [
    "schedule",
    "PlacementPlan",
    "ShardPlacement",
    "ClusterSnapshot",
    "WorkerState",
    "DeviceInfo",
    "ProfileSnapshot",
    "ProfileEntry",
    "SchedulingPolicy",
    "DeviceSlot",
    "PartitionResult",
]
