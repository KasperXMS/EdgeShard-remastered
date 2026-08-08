"""Worker manager — tracks registered Workers and their state.

The WorkerManager maintains:
- Registry of active Workers
- Last heartbeat timestamps
- Hardware and resource information
- Health status

It provides methods for:
- Registering/unregistering Workers
- Updating heartbeat state
- Querying cluster state
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from edgeshard._grpc import edgeshard_pb2
from edgeshard.common.identifiers import WorkerId
from edgeshard.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WorkerRecord:
    """Internal record of a registered Worker."""

    worker_id: WorkerId
    hostname: str
    devices: list[edgeshard_pb2.DeviceInfo]
    available_memory_mb: int
    cpu_count: int
    status: str = "online"
    last_heartbeat: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)

    def is_alive(self, timeout_seconds: float = 30.0) -> bool:
        """Check if Worker is alive based on last heartbeat."""
        return (time.time() - self.last_heartbeat) < timeout_seconds

    def to_proto(self) -> edgeshard_pb2.WorkerState:
        """Convert to protobuf WorkerState message."""
        return edgeshard_pb2.WorkerState(
            worker_id=str(self.worker_id),
            hostname=self.hostname,
            devices=self.devices,
            available_memory_mb=self.available_memory_mb,
            cpu_count=self.cpu_count,
            status=self.status,
            metadata=self.metadata,
            last_heartbeat=int(self.last_heartbeat),
        )


class WorkerManager:
    """Manages registered Workers and their state."""

    def __init__(self) -> None:
        self._workers: dict[str, WorkerRecord] = {}

    def register_worker(
        self,
        worker_id: str,
        hostname: str,
        devices: list[edgeshard_pb2.DeviceInfo],
        available_memory_mb: int,
        cpu_count: int,
        metadata: dict[str, str] | None = None,
    ) -> bool:
        """Register a new Worker or update existing one.

        Args:
            worker_id: Unique Worker identifier.
            hostname: Worker hostname.
            devices: List of devices on the Worker.
            available_memory_mb: Available system memory.
            cpu_count: Number of CPU cores.
            metadata: Optional metadata key-value pairs.

        Returns:
            True if registration succeeded.
        """
        wid = WorkerId(worker_id)

        if worker_id in self._workers:
            logger.info(f"Worker {worker_id} re-registering")
        else:
            logger.info(f"Registering new Worker {worker_id} from {hostname}")

        self._workers[worker_id] = WorkerRecord(
            worker_id=wid,
            hostname=hostname,
            devices=devices,
            available_memory_mb=available_memory_mb,
            cpu_count=cpu_count,
            status="online",
            last_heartbeat=time.time(),
            metadata=metadata or {},
        )

        return True

    def unregister_worker(self, worker_id: str) -> bool:
        """Unregister a Worker.

        Args:
            worker_id: Worker identifier.

        Returns:
            True if unregistration succeeded.
        """
        if worker_id not in self._workers:
            logger.warning(f"Worker {worker_id} not found for unregistration")
            return False

        del self._workers[worker_id]
        logger.info(f"Unregistered Worker {worker_id}")
        return True

    def update_heartbeat(
        self,
        worker_id: str,
        available_memory_mb: int,
        status: str,
        metadata: dict[str, str] | None = None,
    ) -> bool:
        """Update Worker heartbeat and status.

        Args:
            worker_id: Worker identifier.
            available_memory_mb: Current available memory.
            status: Worker status ("online", "busy", etc.).
            metadata: Optional metadata updates.

        Returns:
            True if heartbeat was accepted.
        """
        if worker_id not in self._workers:
            logger.warning(f"Heartbeat from unknown Worker {worker_id}")
            return False

        worker = self._workers[worker_id]
        worker.last_heartbeat = time.time()
        worker.available_memory_mb = available_memory_mb
        worker.status = status

        if metadata:
            worker.metadata.update(metadata)

        return True

    def get_worker(self, worker_id: str) -> WorkerRecord | None:
        """Get a Worker record by ID.

        Args:
            worker_id: Worker identifier.

        Returns:
            WorkerRecord or None if not found.
        """
        return self._workers.get(worker_id)

    def list_workers(self) -> list[WorkerRecord]:
        """List all registered Workers.

        Returns:
            List of WorkerRecords.
        """
        return list(self._workers.values())

    def get_alive_workers(self, timeout_seconds: float = 30.0) -> list[WorkerRecord]:
        """Get list of Workers that are currently alive.

        Args:
            timeout_seconds: Heartbeat timeout threshold.

        Returns:
            List of alive WorkerRecords.
        """
        return [w for w in self._workers.values() if w.is_alive(timeout_seconds)]

    def get_cluster_snapshot(self) -> dict:
        """Get a snapshot of cluster state for scheduling.

        Returns:
            Dict with cluster information.
        """
        alive_workers = self.get_alive_workers()
        total_memory_mb = sum(w.available_memory_mb for w in alive_workers)
        total_devices = sum(len(w.devices) for w in alive_workers)

        return {
            "num_workers": len(alive_workers),
            "total_memory_mb": total_memory_mb,
            "total_devices": total_devices,
            "workers": [w.to_proto() for w in alive_workers],
        }
