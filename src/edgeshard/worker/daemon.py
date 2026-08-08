"""Worker daemon — node-level execution daemon for EdgeShard.

The Worker daemon:
1. Probes local hardware
2. Registers with Master via gRPC
3. Sends periodic heartbeats
4. Manages local model cache
5. Executes deployment commands from Master
6. Hosts Shard runtime instances

One Worker may host multiple Shards and profiler jobs.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import grpc

from edgeshard._grpc import edgeshard_pb2, edgeshard_pb2_grpc
from edgeshard.common.config import WorkerConfig
from edgeshard.common.logging import get_logger, setup_logging
from edgeshard.worker.hardware_probe import (
    get_available_memory_mb,
    get_cpu_count,
    get_hostname,
    probe_hardware,
)

logger = get_logger(__name__)


class WorkerDaemon:
    """Worker daemon that registers with Master and maintains heartbeat."""

    def __init__(self, config: WorkerConfig) -> None:
        self._config = config
        self._worker_id = f"w-{uuid.uuid4().hex[:12]}"
        self._hostname = get_hostname()
        self._devices = probe_hardware()
        self._channel: grpc.Channel | None = None
        self._stub: edgeshard_pb2_grpc.WorkerServiceStub | None = None
        self._running = False
        self._heartbeat_task: asyncio.Task | None = None

    @property
    def worker_id(self) -> str:
        return self._worker_id

    async def start(self) -> None:
        """Start the Worker daemon."""
        logger.info(f"Starting Worker {self._worker_id} on {self._hostname}")
        logger.info(f"Discovered {len(self._devices)} device(s)")

        # Connect to Master
        master_address = self._config.registration.master_address
        logger.info(f"Connecting to Master at {master_address}")

        self._channel = grpc.insecure_channel(master_address)
        self._stub = edgeshard_pb2_grpc.WorkerServiceStub(self._channel)

        # Register with Master
        await self._register()

        # Start heartbeat loop
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info("Worker started successfully")

    async def stop(self) -> None:
        """Stop the Worker daemon."""
        logger.info("Stopping Worker...")
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Unregister from Master
        if self._stub:
            await self._unregister()

        if self._channel:
            self._channel.close()

        logger.info("Worker stopped")

    async def _register(self) -> None:
        """Register this Worker with the Master."""
        request = edgeshard_pb2.RegisterWorkerRequest(
            worker_id=self._worker_id,
            hostname=self._hostname,
            available_memory_mb=get_available_memory_mb(),
            cpu_count=get_cpu_count(),
        )

        # Add devices
        for device in self._devices:
            request.devices.append(device)

        try:
            response = await self._stub.RegisterWorker(request)
            if response.success:
                logger.info(f"Registered with Master: {response.message}")
            else:
                logger.error(f"Registration failed: {response.message}")
                raise RuntimeError(f"Registration failed: {response.message}")
        except grpc.RpcError as e:
            logger.error(f"Registration RPC error: {e}")
            raise

    async def _unregister(self) -> None:
        """Unregister this Worker from the Master."""
        request = edgeshard_pb2.UnregisterWorkerRequest(
            worker_id=self._worker_id,
        )

        try:
            response = await self._stub.UnregisterWorker(request)
            if response.success:
                logger.info(f"Unregistered from Master: {response.message}")
            else:
                logger.warning(f"Unregister failed: {response.message}")
        except grpc.RpcError as e:
            logger.warning(f"Unregister RPC error: {e}")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to Master."""
        interval = self._config.registration.heartbeat_interval_seconds

        while self._running:
            try:
                await self._send_heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")

            await asyncio.sleep(interval)

    async def _send_heartbeat(self) -> None:
        """Send a single heartbeat to Master."""
        request = edgeshard_pb2.HeartbeatRequest(
            worker_id=self._worker_id,
            available_memory_mb=get_available_memory_mb(),
            status="online",
        )

        try:
            response = await self._stub.Heartbeat(request)
            if not response.success:
                logger.warning(f"Heartbeat rejected: {response.message}")
        except grpc.RpcError as e:
            logger.error(f"Heartbeat RPC error: {e}")
            raise
