"""PlacementPlan data model — the immutable output of scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field

from edgeshard.common.config import ServiceSpec
from edgeshard.common.identifiers import ServiceName, WorkerId


@dataclass(frozen=True)
class ShardPlacement:
    """Where and what a single Shard should run.

    Attributes:
        shard_index: 0-based index within the service.
        worker_id: Target Worker node.
        layer_start: Inclusive start layer index in the full model.
        layer_end: Exclusive end layer index in the full model.
        device: Target device on the Worker (e.g. "cuda:0", "cpu").
    """

    shard_index: int
    worker_id: WorkerId
    layer_start: int
    layer_end: int
    device: str


@dataclass(frozen=True)
class PlacementPlan:
    """Immutable deployment and execution plan produced by the Scheduler.

    A PlacementPlan is the contract between Scheduler and Deployment.
    Once created, it may not be modified. If conditions change, a new
    plan is produced.
    """

    service_name: ServiceName
    service_spec: ServiceSpec
    shards: list[ShardPlacement] = field(default_factory=list)
    version: int = 1

    def shard_for_layer(self, layer_idx: int) -> ShardPlacement | None:
        """Find which shard owns a given layer."""
        for shard in self.shards:
            if shard.layer_start <= layer_idx < shard.layer_end:
                return shard
        return None

    def shards_on_worker(self, worker_id: WorkerId) -> list[ShardPlacement]:
        """List all shards assigned to a specific Worker."""
        return [s for s in self.shards if s.worker_id == worker_id]
