"""PlacementPlan data model — the immutable output of scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

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

    def to_yaml(self, path: str | Path) -> None:
        """Serialize plan to a YAML file.

        Args:
            path: Output file path.
        """
        data = {
            "service_name": self.service_name.value,
            "version": self.version,
            "model": self.service_spec.model.name,
            "model_revision": self.service_spec.model.revision,
            "dtype": self.service_spec.model.dtype,
            "shards": [
                {
                    "shard_index": s.shard_index,
                    "worker_id": str(s.worker_id),
                    "layer_start": s.layer_start,
                    "layer_end": s.layer_end,
                    "device": s.device,
                }
                for s in self.shards
            ],
            "metadata": {
                "total_layers": self.shards[-1].layer_end if self.shards else 0,
                "num_shards": len(self.shards),
                "num_workers": len(set(str(s.worker_id) for s in self.shards)),
            },
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PlacementPlan:
        """Deserialize a PlacementPlan from a YAML file.

        Args:
            path: Input YAML file path.

        Returns:
            PlacementPlan loaded from YAML.
        """
        from edgeshard.common.config import ModelSpec

        with open(Path(path), encoding="utf-8") as f:
            data = yaml.safe_load(f)

        shards = [
            ShardPlacement(
                shard_index=s["shard_index"],
                worker_id=WorkerId(s["worker_id"]),
                layer_start=s["layer_start"],
                layer_end=s["layer_end"],
                device=s["device"],
            )
            for s in data.get("shards", [])
        ]

        model_spec = ModelSpec(
            name=data.get("model", ""),
            revision=data.get("model_revision", "main"),
            dtype=data.get("dtype", "float16"),
        )

        service_spec = ServiceSpec(
            name=data.get("service_name", "unknown"),
            model=model_spec,
        )

        return cls(
            service_name=ServiceName(data.get("service_name", "unknown")),
            service_spec=service_spec,
            shards=shards,
            version=data.get("version", 1),
        )
