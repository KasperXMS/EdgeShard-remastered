"""Worker daemon — node-level execution daemon for EdgeShard.

The Worker daemon:
1. Probes local hardware
2. Registers with Master via gRPC
3. Sends periodic heartbeats (including dynamic metrics)
4. Manages local model cache
5. Executes deployment commands from Master
6. Hosts Shard runtime instances

One Worker may host multiple Shards and profiler jobs.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
import uuid
from pathlib import Path
from concurrent import futures
from typing import Any

import grpc

from edgeshard._grpc import edgeshard_pb2, edgeshard_pb2_grpc
from edgeshard.common.config import WorkerConfig
from edgeshard.common.logging import get_logger, setup_logging
from edgeshard.worker.hardware_probe import (
    collect_all_device_metrics,
    get_available_memory_mb,
    get_cpu_count,
    get_hostname,
    probe_hardware,
)
from edgeshard.worker.network_probe import NetworkProbe

logger = get_logger(__name__)


class WorkerServicer(edgeshard_pb2_grpc.WorkerServiceServicer):
    """gRPC servicer for Worker-side commands (M8 deployment).

    This handles StartShard, StopShard, ListShards RPCs from the Master.
    """

    def __init__(self, daemon: "WorkerDaemon") -> None:
        self._daemon = daemon

    async def StartShard(
        self,
        request: edgeshard_pb2.StartShardRequest,
        context: grpc.ServicerContext,
    ) -> edgeshard_pb2.StartShardResponse:
        """Handle StartShard RPC from Master."""
        # Auto-detect first/last if not explicitly set by older Master code
        is_first = request.is_first_shard
        is_last = request.is_last_shard
        if not is_first and not is_last:
            # Fallback: if layer range starts at 0 → first shard
            # We can't know "last" without model info, but first is safe
            if request.layer_start == 0:
                is_first = True
            logger.info(
                f"Master didn't set is_first/is_last flags; "
                f"inferred is_first={is_first}, is_last={is_last}"
            )

        logger.info(
            f"StartShard request: {request.shard_id} "
            f"(layers {request.layer_start}:{request.layer_end}, "
            f"device {request.device}, "
            f"first={is_first}, last={is_last})"
        )

        try:
            # Spawn shard as subprocess
            data_address = await self._daemon.spawn_shard(
                shard_id=request.shard_id,
                model_name=request.model_name,
                model_revision=request.model_revision,
                dtype=request.dtype,
                layer_start=request.layer_start,
                layer_end=request.layer_end,
                device=request.device,
                data_host=request.data_host or "0.0.0.0",
                data_port=request.data_port or 50100,
                is_first_shard=is_first,
                is_last_shard=is_last,
                master_address=request.master_address,
            )

            return edgeshard_pb2.StartShardResponse(
                success=True,
                message=f"Shard {request.shard_id} started",
                shard_id=request.shard_id,
                data_address=data_address,
            )

        except Exception as e:
            logger.error(f"Failed to start shard {request.shard_id}: {e}")
            return edgeshard_pb2.StartShardResponse(
                success=False,
                message=f"Failed to start shard: {e}",
                shard_id=request.shard_id,
            )

    async def StopShard(
        self,
        request: edgeshard_pb2.StopShardRequest,
        context: grpc.ServicerContext,
    ) -> edgeshard_pb2.StopShardResponse:
        """Handle StopShard RPC from Master."""
        logger.info(f"StopShard request: {request.shard_id}")

        try:
            success = await self._daemon.stop_shard(request.shard_id)
            return edgeshard_pb2.StopShardResponse(
                success=success,
                message="Shard stopped" if success else "Shard not found",
            )
        except Exception as e:
            logger.error(f"Failed to stop shard {request.shard_id}: {e}")
            return edgeshard_pb2.StopShardResponse(
                success=False,
                message=f"Failed to stop shard: {e}",
            )

    async def ListShards(
        self,
        request: edgeshard_pb2.ListShardsRequest,
        context: grpc.ServicerContext,
    ) -> edgeshard_pb2.ListShardsResponse:
        """Handle ListShards RPC from Master."""
        shards = self._daemon.list_shards()
        shard_infos = []

        for shard in shards:
            shard_infos.append(
                edgeshard_pb2.ShardInfo(
                    shard_id=shard["shard_id"],
                    model_name=shard.get("model_name", ""),
                    layer_start=shard.get("layer_start", 0),
                    layer_end=shard.get("layer_end", 0),
                    device=shard.get("device", ""),
                    status=shard.get("status", "unknown"),
                    data_address=shard.get("data_address", ""),
                    start_time=shard.get("start_time", 0),
                    pid=str(shard.get("pid", "")),
                )
            )

        return edgeshard_pb2.ListShardsResponse(shards=shard_infos)


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
        self._registered = False  # Track if registration succeeded
        self._heartbeat_task: asyncio.Task | None = None
        self._network_probe = NetworkProbe(self._worker_id)
        self._known_workers: dict[str, tuple[str, int]] = {}  # worker_id -> (host, port)

        # M8: Shard management
        self._server: grpc.aio.Server | None = None
        self._shards: dict[str, dict] = {}  # shard_id -> shard info
        self._shard_processes: dict[str, subprocess.Popen] = {}  # shard_id -> process

    @property
    def worker_id(self) -> str:
        return self._worker_id

    async def start(self) -> None:
        """Start the Worker daemon."""
        logger.info(f"Starting Worker {self._worker_id} on {self._hostname}")
        logger.info(f"Discovered {len(self._devices)} device(s)")

        # Start Worker gRPC server (for receiving deployment commands from Master)
        await self._start_grpc_server()

        # Connect to Master
        master_address = self._config.registration.master_address
        logger.info(f"Connecting to Master at {master_address}")

        self._channel = grpc.aio.insecure_channel(master_address)
        self._stub = edgeshard_pb2_grpc.WorkerServiceStub(self._channel)

        # Register with Master
        await self._register()

        # Discover other workers and measure network topology
        await self._discover_workers()

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

        # Stop all shards
        for shard_id in list(self._shards.keys()):
            await self.stop_shard(shard_id)

        # Unregister from Master FIRST (before stopping server/channel)
        # Only if registration succeeded
        if self._registered and self._stub:
            await self._unregister()

        # Stop gRPC server
        if self._server:
            await self._server.stop(grace=2.0)

        # Close channel
        if self._channel:
            await self._channel.close()

        logger.info("Worker stopped")

    async def _start_grpc_server(self) -> None:
        """Start the Worker gRPC server for deployment commands."""
        # Use a port offset from the master port
        # Master uses 10500, Worker uses 10600 by default
        worker_port = 10600

        self._server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=4))

        # Add servicer
        servicer = WorkerServicer(self)
        edgeshard_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, self._server)

        bind_address = f"0.0.0.0:{worker_port}"
        self._server.add_insecure_port(bind_address)
        await self._server.start()

        logger.info(f"Worker gRPC server listening on {bind_address}")

    # -------------------------------------------------------------------------
    # M8: Shard management
    # -------------------------------------------------------------------------

    async def spawn_shard(
        self,
        shard_id: str,
        model_name: str,
        model_revision: str,
        dtype: str,
        layer_start: int,
        layer_end: int,
        device: str,
        data_host: str,
        data_port: int,
        is_first_shard: bool,
        is_last_shard: bool,
        master_address: str,
    ) -> str:
        """Spawn a shard subprocess.

        Returns:
            Data address (host:port) of the shard.
        """
        if shard_id in self._shard_processes:
            raise RuntimeError(f"Shard {shard_id} already running")

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "edgeshard",
            "shard",
            "start",
            model_name,
            "--shard-id",
            shard_id,
            "--layers",
            f"{layer_start}:{layer_end}",
            "--dtype",
            dtype,
            "--host",
            data_host,
            "--port",
            str(data_port),
        ]

        if is_first_shard:
            cmd.append("--first")
        if is_last_shard:
            cmd.append("--last")

        logger.info(f"Spawning shard: {' '.join(cmd)}")

        # Write shard output to a log file for debugging
        # (Rich traceback formatting is too verbose for in-memory capture)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{shard_id}.log"
        log_fh = open(log_file, "w")
        logger.info(f"Shard log: {log_file.resolve()}")

        # Start subprocess — stderr goes to log file for full traceback
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=log_fh,
        )

        self._shard_processes[shard_id] = proc
        self._shards[shard_id] = {
            "shard_id": shard_id,
            "model_name": model_name,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "device": device,
            "status": "starting",
            "data_address": f"{data_host}:{data_port}",
            "start_time": int(time.time()),
            "pid": proc.pid,
        }

        # Poll for shard readiness — wait up to 300 seconds for model loading/download
        # Large models (7B+) can take several minutes to load from disk or download
        ready = False
        for _ in range(300):
            await asyncio.sleep(1.0)

            # Check if process died
            if proc.poll() is not None:
                # Process exited
                break

            # Try to connect to the data port to verify shard is ready
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                result = sock.connect_ex(("127.0.0.1", data_port))
                sock.close()
                if result == 0:
                    ready = True
                    break
            except Exception:
                pass

        if ready:
            self._shards[shard_id]["status"] = "ready"
            logger.info(f"Shard {shard_id} started and ready (PID: {proc.pid})")
        else:
            self._shards[shard_id]["status"] = "failed"
            # Read the last part of the log file for error summary
            error_msg = f"Shard did not become ready within 300s. Check log: {log_file.resolve()}"
            try:
                log_fh.close()
                with open(log_file, errors="replace") as f:
                    lines = f.readlines()
                    # Get last 30 lines for error context
                    tail = "".join(lines[-30:])
                    if tail:
                        error_msg = f"{error_msg}\n--- Last lines from {log_file.name} ---\n{tail}"
            except Exception:
                pass
            logger.error(f"Shard {shard_id} failed to start: {error_msg}")
            raise RuntimeError(f"Shard failed to start: {error_msg}")

        return f"{data_host}:{data_port}"

    async def stop_shard(self, shard_id: str) -> bool:
        """Stop a running shard."""
        proc = self._shard_processes.get(shard_id)
        if proc is None:
            return False

        try:
            proc.terminate()
            proc.wait(timeout=10.0)
            if shard_id in self._shards:
                self._shards[shard_id]["status"] = "stopped"
            del self._shard_processes[shard_id]
            logger.info(f"Shard {shard_id} stopped")
            return True
        except subprocess.TimeoutExpired:
            proc.kill()
            if shard_id in self._shards:
                self._shards[shard_id]["status"] = "stopped"
            del self._shard_processes[shard_id]
            return True
        except Exception as e:
            logger.error(f"Error stopping shard {shard_id}: {e}")
            return False

    def list_shards(self) -> list[dict]:
        """List all shards on this Worker."""
        # Update status based on process state
        for shard_id, proc in list(self._shard_processes.items()):
            if proc.poll() is not None:
                if shard_id in self._shards:
                    self._shards[shard_id]["status"] = "stopped"
                del self._shard_processes[shard_id]

        return list(self._shards.values())

    async def _register(self) -> None:
        """Register this Worker with the Master. Retries with backoff."""
        request = edgeshard_pb2.RegisterWorkerRequest(
            worker_id=self._worker_id,
            hostname=self._hostname,
            available_memory_mb=get_available_memory_mb(),
            cpu_count=get_cpu_count(),
        )

        # Add devices
        for device in self._devices:
            request.devices.append(device)

        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self._stub.RegisterWorker(request),
                    timeout=10.0,
                )
                if response.success:
                    logger.info(f"Registered with Master: {response.message}")
                    self._registered = True
                    return
                else:
                    logger.error(f"Registration failed: {response.message}")
                    raise RuntimeError(f"Registration failed: {response.message}")
            except asyncio.TimeoutError:
                logger.warning(
                    f"Registration timed out (attempt {attempt}/{max_retries}), "
                    f"retrying in {attempt * 3}s..."
                )
            except grpc.RpcError as e:
                logger.warning(
                    f"Registration RPC error (attempt {attempt}/{max_retries}): "
                    f"{e.code().name} — {e.details()}. "
                    f"Retrying in {attempt * 3}s..."
                )
            await asyncio.sleep(attempt * 3)

        raise RuntimeError(
            f"Failed to register with Master after {max_retries} attempts. "
            f"Check that the Master is running and reachable."
        )

    async def _unregister(self) -> None:
        """Unregister this Worker from the Master."""
        request = edgeshard_pb2.UnregisterWorkerRequest(
            worker_id=self._worker_id,
        )

        try:
            # Use timeout to prevent hanging if master is down
            response = await asyncio.wait_for(
                self._stub.UnregisterWorker(request),
                timeout=5.0,
            )
            if response.success:
                logger.info(f"Unregistered from Master: {response.message}")
            else:
                logger.warning(f"Unregister failed: {response.message}")
        except asyncio.TimeoutError:
            logger.error("Unregister timed out (master may be down)")
        except grpc.RpcError as e:
            logger.error(f"Unregister RPC error: {e}")
        except Exception as e:
            logger.error(f"Unregister failed: {e}")

    async def _discover_workers(self) -> None:
        """Discover other workers and measure network latency."""
        if not self._stub:
            return

        try:
            # Get list of all workers
            request = edgeshard_pb2.ListWorkersRequest()
            response = await self._stub.ListWorkers(request)

            for worker in response.workers:
                if worker.worker_id == self._worker_id:
                    continue  # Skip self

                # Store known worker (use hostname:10500 as default port for probing)
                # In practice, we'd need a separate data port for tensor transfer
                self._known_workers[worker.worker_id] = (worker.hostname, 10500)

            # Measure latency to known workers
            if self._known_workers:
                logger.info(f"Discovering network topology to {len(self._known_workers)} worker(s)...")
                await self._network_probe.measure_latency_to_workers(self._known_workers)

                # Estimate bandwidth
                for worker_id, (host, port) in self._known_workers.items():
                    self._network_probe.estimate_bandwidth(host, port, worker_id)
                    break  # Just need one estimate

        except grpc.RpcError as e:
            logger.warning(f"Failed to discover workers: {e}")

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
        """Send a single heartbeat to Master with dynamic metrics."""
        # Collect device metrics
        device_metrics = collect_all_device_metrics(self._devices)

        # Get network metrics
        network_metrics = self._network_probe.get_network_metrics()

        request = edgeshard_pb2.HeartbeatRequest(
            worker_id=self._worker_id,
            available_memory_mb=get_available_memory_mb(),
            status="online",
        )

        # Add device metrics
        for dm in device_metrics:
            request.device_metrics.append(dm)

        # Add network metrics if available
        if network_metrics.latency_ms_to_worker or network_metrics.estimated_bandwidth_mbps:
            request.network_metrics.CopyFrom(network_metrics)

        try:
            response = await self._stub.Heartbeat(request)
            if not response.success:
                logger.warning(f"Heartbeat rejected: {response.message}")
        except grpc.RpcError as e:
            logger.error(f"Heartbeat RPC error: {e}")
            raise
