"""Deployment backend interface.

A DeploymentBackend knows how to start and stop shard processes on a
specific infrastructure (local process, Docker container, Kubernetes pod, etc.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edgeshard.scheduler.placement import PlacementPlan, ShardPlacement


class ShardStatus(str, Enum):
    """Status of a deployed shard."""

    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


@dataclass
class ShardHandle:
    """Handle to a running shard.

    Attributes:
        shard_id: Unique shard identifier.
        worker_id: Worker where the shard runs.
        status: Current shard status.
        data_address: Data plane address (host:port) for shard-to-shard communication.
        model_name: Model being served.
        layer_start: First layer index (inclusive).
        layer_end: Last layer index (exclusive).
        device: Target device (e.g., "cuda:0").
        error_message: Error details if status is FAILED.
    """

    shard_id: str
    worker_id: str
    status: ShardStatus = ShardStatus.STARTING
    data_address: str = ""
    model_name: str = ""
    layer_start: int = 0
    layer_end: int = 0
    device: str = ""
    error_message: str = ""
    metadata: dict = field(default_factory=dict)


class DeploymentBackend(ABC):
    """Abstract base class for deployment backends.

    A backend knows how to start/stop shard processes on a specific
    infrastructure. Implementations include:
    - LocalBackend: Spawn shard as a local subprocess
    - RemoteBackend: Send gRPC command to Worker to spawn shard
    - DockerBackend: Run shard in a Docker container
    """

    @abstractmethod
    async def start_shard(
        self,
        shard_placement: "ShardPlacement",
        model_name: str,
        model_revision: str = "main",
        dtype: str = "float16",
        data_host: str = "0.0.0.0",
        data_port: int = 50100,
        master_address: str = "localhost:10500",
        is_first: bool = False,
        is_last: bool = False,
    ) -> ShardHandle:
        """Start a shard process.

        Args:
            shard_placement: Placement info (worker, layers, device).
            model_name: HuggingFace model ID or local path.
            model_revision: Model revision/branch.
            dtype: Weight dtype.
            data_host: Data plane host.
            data_port: Data plane port.
            master_address: Master address for registration.
            is_first: True if this is the first shard (has embedding).
            is_last: True if this is the last shard (has LM head).

        Returns:
            ShardHandle with status and data address.
        """
        ...

    @abstractmethod
    async def stop_shard(self, shard_id: str) -> bool:
        """Stop a running shard.

        Args:
            shard_id: Shard to stop.

        Returns:
            True if stopped successfully.
        """
        ...

    @abstractmethod
    async def list_shards(self) -> list[ShardHandle]:
        """List all running shards.

        Returns:
            List of ShardHandle objects.
        """
        ...

    @abstractmethod
    async def get_shard(self, shard_id: str) -> ShardHandle | None:
        """Get a specific shard handle.

        Args:
            shard_id: Shard to look up.

        Returns:
            ShardHandle if found, None otherwise.
        """
        ...

    async def wait_ready(self, shard_id: str, timeout_seconds: float = 60.0) -> bool:
        """Wait for a shard to become ready.

        Default implementation polls get_shard() until READY or timeout.

        Args:
            shard_id: Shard to wait for.
            timeout_seconds: Maximum wait time.

        Returns:
            True if shard became ready, False if timed out.
        """
        import asyncio
        import time

        start = time.monotonic()
        while time.monotonic() - start < timeout_seconds:
            handle = await self.get_shard(shard_id)
            if handle is None:
                return False
            if handle.status == ShardStatus.READY:
                return True
            if handle.status == ShardStatus.FAILED:
                return False
            await asyncio.sleep(1.0)
        return False

    async def deploy_plan(
        self,
        plan: "PlacementPlan",
        base_port: int = 50100,
    ) -> list[ShardHandle]:
        """Deploy all shards in a placement plan.

        Args:
            plan: The placement plan to deploy.
            base_port: Starting port for data plane.

        Returns:
            List of ShardHandle objects for all shards.
        """
        handles = []
        total_shards = len(plan.shards)
        for i, shard in enumerate(plan.shards):
            handle = await self.start_shard(
                shard_placement=shard,
                model_name=plan.service_spec.model.name,
                model_revision=plan.service_spec.model.revision,
                dtype=plan.service_spec.model.dtype,
                data_port=base_port + i,
                is_first=(i == 0),
                is_last=(i == total_shards - 1),
            )
            handles.append(handle)

        # Wait for all shards to be ready
        all_ready = True
        for handle in handles:
            ready = await self.wait_ready(handle.shard_id, timeout_seconds=120.0)
            if not ready:
                all_ready = False

        return handles
