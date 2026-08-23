"""Remote deployment backend — send shard commands to Workers via gRPC.

This backend is used by the Master to deploy shards on remote Workers.
The Worker receives the StartShard RPC and spawns a ShardDaemon locally.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import grpc

from edgeshard._grpc import edgeshard_pb2, edgeshard_pb2_grpc
from edgeshard.common.logging import get_logger
from edgeshard.deployment.backend import DeploymentBackend, ShardHandle, ShardStatus

if TYPE_CHECKING:
    from edgeshard.scheduler.placement import ShardPlacement

logger = get_logger(__name__)


class RemoteDeploymentBackend(DeploymentBackend):
    """Deploy shards on remote Workers via gRPC.

    The Master uses this backend to tell Workers to start shard processes.
    Each Worker receives a StartShard RPC and spawns a ShardDaemon locally.
    """

    def __init__(
        self,
        worker_addresses: dict[str, str] | None = None,
        default_port: int = 10500,
    ):
        """Initialize remote deployment backend.

        Args:
            worker_addresses: Map of worker_id → gRPC address (host:port).
                If None, must be provided per-call via metadata.
            default_port: Default gRPC port if not in worker_addresses.
        """
        self._worker_addresses = worker_addresses or {}
        self._default_port = default_port
        self._shards: dict[str, ShardHandle] = {}
        self._channels: dict[str, grpc.Channel] = {}

    def _get_address(self, worker_id: str) -> str:
        """Get gRPC address for a worker."""
        if worker_id in self._worker_addresses:
            return self._worker_addresses[worker_id]
        # Assume worker_id can be resolved via hostname
        return f"{worker_id}:{self._default_port}"

    def _get_channel(self, worker_id: str) -> grpc.Channel:
        """Get or create a gRPC channel to a worker."""
        if worker_id not in self._channels:
            address = self._get_address(worker_id)
            self._channels[worker_id] = grpc.insecure_channel(address)
        return self._channels[worker_id]

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
        """Start a shard on a remote Worker via gRPC."""
        worker_id = str(shard_placement.worker_id)
        shard_id = f"shard-{shard_placement.shard_index}"

        # Create shard handle (starting state)
        handle = ShardHandle(
            shard_id=shard_id,
            worker_id=worker_id,
            status=ShardStatus.STARTING,
            model_name=model_name,
            layer_start=shard_placement.layer_start,
            layer_end=shard_placement.layer_end,
            device=shard_placement.device,
        )
        self._shards[shard_id] = handle

        try:
            channel = self._get_channel(worker_id)
            stub = edgeshard_pb2_grpc.WorkerServiceStub(channel)

            request = edgeshard_pb2.StartShardRequest(
                shard_id=shard_id,
                model_name=model_name,
                model_revision=model_revision,
                dtype=dtype,
                layer_start=shard_placement.layer_start,
                layer_end=shard_placement.layer_end,
                device=shard_placement.device,
                data_host=data_host,
                data_port=data_port,
                is_first_shard=is_first,
                is_last_shard=is_last,
                master_address=master_address,
            )

            # Run gRPC call in executor to avoid blocking
            # Use longer timeout — model loading can take several minutes
            # (especially for large models or when downloading from HuggingFace)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: stub.StartShard(request, timeout=300.0),
            )

            if response.success:
                handle.status = ShardStatus.READY
                handle.data_address = response.data_address
                logger.info(
                    f"Shard {shard_id} started on {worker_id} "
                    f"at {response.data_address}"
                )
            else:
                handle.status = ShardStatus.FAILED
                handle.error_message = response.message
                logger.error(f"Failed to start shard {shard_id}: {response.message}")

        except grpc.RpcError as e:
            handle.status = ShardStatus.FAILED
            handle.error_message = f"gRPC error: {e.details()}"
            logger.error(f"gRPC error starting shard {shard_id}: {e.details()}")
        except Exception as e:
            handle.status = ShardStatus.FAILED
            handle.error_message = str(e)
            logger.error(f"Error starting shard {shard_id}: {e}")

        return handle

    async def stop_shard(self, shard_id: str) -> bool:
        """Stop a shard on a remote Worker."""
        handle = self._shards.get(shard_id)
        if handle is None:
            logger.warning(f"Shard {shard_id} not found")
            return False

        try:
            channel = self._get_channel(handle.worker_id)
            stub = edgeshard_pb2_grpc.WorkerServiceStub(channel)

            request = edgeshard_pb2.StopShardRequest(shard_id=shard_id)

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: stub.StopShard(request, timeout=10.0),
            )

            if response.success:
                handle.status = ShardStatus.STOPPED
                logger.info(f"Shard {shard_id} stopped")
            else:
                logger.error(f"Failed to stop shard {shard_id}: {response.message}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error stopping shard {shard_id}: {e}")
            return False

    async def list_shards(self) -> list[ShardHandle]:
        """List all shards managed by this backend."""
        return list(self._shards.values())

    async def get_shard(self, shard_id: str) -> ShardHandle | None:
        """Get a specific shard handle."""
        return self._shards.get(shard_id)

    def update_worker_address(self, worker_id: str, address: str) -> None:
        """Update the gRPC address for a worker.

        Useful when worker addresses are discovered dynamically.
        """
        self._worker_addresses[worker_id] = address
        # Close old channel if exists
        if worker_id in self._channels:
            self._channels[worker_id].close()
            del self._channels[worker_id]

    def close(self) -> None:
        """Close all gRPC channels."""
        for channel in self._channels.values():
            channel.close()
        self._channels.clear()
