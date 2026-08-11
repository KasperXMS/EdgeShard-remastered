"""Deployment manager — orchestrate shard deployment across Workers.

The DeploymentManager coordinates deployment of a PlacementPlan by:
1. Reading the plan to determine which shards go where
2. Using the appropriate DeploymentBackend to start each shard
3. Monitoring shard health and readiness
4. Handling failures and retries
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from edgeshard.common.logging import get_logger
from edgeshard.deployment.backend import DeploymentBackend, ShardHandle, ShardStatus

if TYPE_CHECKING:
    from edgeshard.scheduler.placement import PlacementPlan

logger = get_logger(__name__)


@dataclass
class DeploymentState:
    """Current state of a deployment.

    Attributes:
        service_name: Name of the deployed service.
        plan: The placement plan being deployed.
        shards: Map of shard_id → ShardHandle.
        status: Overall deployment status.
        message: Human-readable status message.
    """

    service_name: str
    plan: "PlacementPlan | None" = None
    shards: dict[str, ShardHandle] = field(default_factory=dict)
    status: str = "idle"  # idle, deploying, ready, failed, stopping
    message: str = ""


class DeploymentManager:
    """Orchestrates shard deployment across Workers.

    The manager uses a DeploymentBackend to start/stop shards and tracks
    the overall deployment state.
    """

    def __init__(self, backend: DeploymentBackend) -> None:
        """Initialize deployment manager.

        Args:
            backend: The deployment backend to use.
        """
        self._backend = backend
        self._deployments: dict[str, DeploymentState] = {}

    async def deploy(
        self,
        plan: "PlacementPlan",
        base_port: int = 50100,
        wait_for_ready: bool = True,
        timeout_seconds: float = 120.0,
    ) -> DeploymentState:
        """Deploy a placement plan.

        Args:
            plan: The placement plan to deploy.
            base_port: Starting port for data plane.
            wait_for_ready: Whether to wait for all shards to be ready.
            timeout_seconds: Maximum time to wait for readiness.

        Returns:
            DeploymentState with current status.
        """
        service_name = plan.service_name.value
        state = DeploymentState(
            service_name=service_name,
            plan=plan,
            status="deploying",
            message=f"Deploying {len(plan.shards)} shard(s)...",
        )
        self._deployments[service_name] = state

        logger.info(
            f"Deploying service '{service_name}' with {len(plan.shards)} shard(s)"
        )

        # Start all shards
        for i, shard_placement in enumerate(plan.shards):
            shard_id = f"shard-{shard_placement.shard_index}"
            data_port = base_port + i

            # Determine if this is the last shard
            is_last = i == len(plan.shards) - 1

            logger.info(
                f"Starting shard {shard_id} on {shard_placement.worker_id} "
                f"(layers {shard_placement.layer_start}-{shard_placement.layer_end}, "
                f"device {shard_placement.device})"
            )

            handle = await self._backend.start_shard(
                shard_placement=shard_placement,
                model_name=plan.service_spec.model.name,
                model_revision=plan.service_spec.model.revision,
                dtype=plan.service_spec.model.dtype,
                data_port=data_port,
            )

            # Update is_last_shard flag (need to restart if not set correctly)
            # For now, we'll handle this in the backend

            state.shards[shard_id] = handle

            if handle.status == ShardStatus.FAILED:
                state.status = "failed"
                state.message = f"Shard {shard_id} failed to start: {handle.error_message}"
                logger.error(state.message)
                return state

        # Wait for all shards to be ready
        if wait_for_ready:
            state.message = "Waiting for shards to be ready..."
            all_ready = True

            for shard_id in state.shards:
                ready = await self._backend.wait_ready(
                    shard_id, timeout_seconds=timeout_seconds
                )
                if not ready:
                    all_ready = False
                    handle = state.shards[shard_id]
                    state.status = "failed"
                    state.message = f"Shard {shard_id} did not become ready"
                    logger.error(state.message)
                    return state

            if all_ready:
                state.status = "ready"
                state.message = f"All {len(state.shards)} shard(s) ready"
                logger.info(state.message)

        return state

    async def stop(self, service_name: str) -> bool:
        """Stop a deployed service.

        Args:
            service_name: Name of the service to stop.

        Returns:
            True if stopped successfully.
        """
        state = self._deployments.get(service_name)
        if state is None:
            logger.warning(f"Service '{service_name}' not found")
            return False

        state.status = "stopping"
        state.message = "Stopping all shards..."

        success = True
        for shard_id in state.shards:
            if not await self._backend.stop_shard(shard_id):
                success = False

        if success:
            state.status = "stopped"
            state.message = "All shards stopped"
            logger.info(f"Service '{service_name}' stopped")
        else:
            state.status = "failed"
            state.message = "Some shards failed to stop"
            logger.error(f"Service '{service_name}' failed to stop cleanly")

        return success

    async def stop_all(self) -> None:
        """Stop all deployed services."""
        for service_name in list(self._deployments.keys()):
            await self.stop(service_name)

    def get_deployment(self, service_name: str) -> DeploymentState | None:
        """Get deployment state for a service.

        Args:
            service_name: Name of the service.

        Returns:
            DeploymentState if found, None otherwise.
        """
        return self._deployments.get(service_name)

    def list_deployments(self) -> list[DeploymentState]:
        """List all deployments.

        Returns:
            List of DeploymentState objects.
        """
        return list(self._deployments.values())

    async def refresh_status(self, service_name: str) -> DeploymentState | None:
        """Refresh the status of a deployment.

        Args:
            service_name: Name of the service.

        Returns:
            Updated DeploymentState, or None if not found.
        """
        state = self._deployments.get(service_name)
        if state is None:
            return None

        # Refresh shard statuses from backend
        for shard_id in state.shards:
            handle = await self._backend.get_shard(shard_id)
            if handle is not None:
                state.shards[shard_id] = handle

        # Update overall status
        statuses = [h.status for h in state.shards.values()]
        if all(s == ShardStatus.READY for s in statuses):
            state.status = "ready"
        elif all(s == ShardStatus.STOPPED for s in statuses):
            state.status = "stopped"
        elif any(s == ShardStatus.FAILED for s in statuses):
            state.status = "failed"
        else:
            state.status = "deploying"

        return state
