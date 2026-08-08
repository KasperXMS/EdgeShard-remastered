"""Core scheduling logic — pure planner, no side effects.

The Scheduler is a pure function:
    (ModelSpec, ClusterSnapshot, ProfileSnapshot, SchedulingPolicy) -> PlacementPlan

It may not start containers, mutate Workers, or perform I/O.
"""

from __future__ import annotations

from edgeshard.common.config import ModelSpec
from edgeshard.scheduler.placement import PlacementPlan
from edgeshard.scheduler.snapshot import (
    ClusterSnapshot,
    ProfileSnapshot,
    SchedulingPolicy,
)


def schedule(
    model: ModelSpec,
    cluster: ClusterSnapshot,
    profiles: ProfileSnapshot,
    policy: SchedulingPolicy,
) -> PlacementPlan:
    """Plan model sharding and placement across the cluster.

    This is the core scheduling entry point. It consumes immutable
    snapshots and produces an immutable PlacementPlan.

    Args:
        model: Which model to serve.
        cluster: Current cluster resources.
        profiles: Available profiling data.
        policy: Scheduling policy configuration.

    Returns:
        Immutable PlacementPlan.

    Raises:
        SchedulerError: If no valid placement exists.
    """
    # TODO (M7): Implement actual scheduling logic.
    # For now, return an empty plan.
    raise NotImplementedError(
        "Scheduler not yet implemented. See milestone M7."
    )
