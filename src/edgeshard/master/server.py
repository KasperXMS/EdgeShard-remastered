"""Master gRPC server — control plane for EdgeShard cluster.

The Master server:
1. Accepts Worker registrations
2. Processes heartbeats
3. Provides cluster discovery API
4. Manages service lifecycle (placeholder for M8)
5. Coordinates profiling (placeholder for M6)

The Master never participates in the token data path.
"""

from __future__ import annotations

import asyncio
from concurrent import futures

import grpc

from edgeshard._grpc import edgeshard_pb2, edgeshard_pb2_grpc
from edgeshard.common.config import MasterConfig
from edgeshard.common.logging import get_logger, setup_logging
from edgeshard.master.worker_manager import WorkerManager

logger = get_logger(__name__)


class EdgeShardMasterServicer(edgeshard_pb2_grpc.WorkerServiceServicer):
    """gRPC servicer for Master node."""

    def __init__(self) -> None:
        self._worker_manager = WorkerManager()

    async def RegisterWorker(
        self,
        request: edgeshard_pb2.RegisterWorkerRequest,
        context: grpc.ServicerContext,
    ) -> edgeshard_pb2.RegisterWorkerResponse:
        """Handle Worker registration."""
        logger.info(f"Registration request from Worker {request.worker_id}")

        success = self._worker_manager.register_worker(
            worker_id=request.worker_id,
            hostname=request.hostname,
            devices=list(request.devices),
            available_memory_mb=request.available_memory_mb,
            cpu_count=request.cpu_count,
            metadata=dict(request.metadata),
        )

        if success:
            return edgeshard_pb2.RegisterWorkerResponse(
                success=True,
                message=f"Worker {request.worker_id} registered successfully",
                master_id="master-001",  # TODO: generate unique master ID
            )
        else:
            return edgeshard_pb2.RegisterWorkerResponse(
                success=False,
                message="Registration failed",
            )

    async def Heartbeat(
        self,
        request: edgeshard_pb2.HeartbeatRequest,
        context: grpc.ServicerContext,
    ) -> edgeshard_pb2.HeartbeatResponse:
        """Handle Worker heartbeat."""
        success = self._worker_manager.update_heartbeat(
            worker_id=request.worker_id,
            available_memory_mb=request.available_memory_mb,
            status=request.status,
            metadata=dict(request.metadata),
            device_metrics=list(request.device_metrics),
            network_metrics=request.network_metrics if request.HasField("network_metrics") else None,
        )

        if success:
            return edgeshard_pb2.HeartbeatResponse(
                success=True,
                message="Heartbeat accepted",
            )
        else:
            return edgeshard_pb2.HeartbeatResponse(
                success=False,
                message="Worker not registered",
            )

    async def UnregisterWorker(
        self,
        request: edgeshard_pb2.UnregisterWorkerRequest,
        context: grpc.ServicerContext,
    ) -> edgeshard_pb2.UnregisterWorkerResponse:
        """Handle Worker unregistration."""
        logger.info(f"Unregister request from Worker {request.worker_id}")

        success = self._worker_manager.unregister_worker(request.worker_id)

        return edgeshard_pb2.UnregisterWorkerResponse(
            success=success,
            message="Unregistered" if success else "Worker not found",
        )

    async def ListWorkers(
        self,
        request: edgeshard_pb2.ListWorkersRequest,
        context: grpc.ServicerContext,
    ) -> edgeshard_pb2.ListWorkersResponse:
        """List all registered Workers."""
        workers = self._worker_manager.list_workers()
        worker_states = [w.to_proto() for w in workers]

        return edgeshard_pb2.ListWorkersResponse(workers=worker_states)

    async def GetWorker(
        self,
        request: edgeshard_pb2.GetWorkerRequest,
        context: grpc.ServicerContext,
    ) -> edgeshard_pb2.GetWorkerResponse:
        """Get a specific Worker's state."""
        worker = self._worker_manager.get_worker(request.worker_id)

        if worker:
            return edgeshard_pb2.GetWorkerResponse(worker=worker.to_proto())
        else:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Worker {request.worker_id} not found")
            return edgeshard_pb2.GetWorkerResponse()

    async def DeployService(
        self,
        request: edgeshard_pb2.DeployServiceRequest,
        context: grpc.ServicerContext,
    ) -> edgeshard_pb2.DeployServiceResponse:
        """Deploy a service (placeholder for M8)."""
        logger.info(f"Deploy service request: {request.spec.name}")
        # TODO (M8): Implement actual deployment logic
        return edgeshard_pb2.DeployServiceResponse(
            success=False,
            message="Deployment not yet implemented (M8)",
        )

    async def ListServices(
        self,
        request: edgeshard_pb2.ListServicesRequest,
        context: grpc.ServicerContext,
    ) -> edgeshard_pb2.ListServicesResponse:
        """List deployed services (placeholder for M8)."""
        # TODO (M8): Implement service listing
        return edgeshard_pb2.ListServicesResponse(service_names=[])

    async def GetServiceStatus(
        self,
        request: edgeshard_pb2.GetServiceStatusRequest,
        context: grpc.ServicerContext,
    ) -> edgeshard_pb2.GetServiceStatusResponse:
        """Get service status (placeholder for M8)."""
        # TODO (M8): Implement service status
        return edgeshard_pb2.GetServiceStatusResponse(
            service_name=request.service_name,
            status="not_found",
        )

    async def StopService(
        self,
        request: edgeshard_pb2.StopServiceRequest,
        context: grpc.ServicerContext,
    ) -> edgeshard_pb2.StopServiceResponse:
        """Stop a service (placeholder for M8)."""
        logger.info(f"Stop service request: {request.service_name}")
        # TODO (M8): Implement service stop
        return edgeshard_pb2.StopServiceResponse(
            success=False,
            message="Service stop not yet implemented (M8)",
        )

    async def Profile(
        self,
        request: edgeshard_pb2.ProfileRequest,
        context: grpc.ServicerContext,
    ) -> edgeshard_pb2.ProfileResponse:
        """Run profiling (placeholder for M6)."""
        logger.info(f"Profile request for {request.model_name}")
        # TODO (M6): Implement profiling
        return edgeshard_pb2.ProfileResponse(
            success=False,
            message="Profiling not yet implemented (M6)",
        )


class MasterServer:
    """Master gRPC server."""

    def __init__(self, config: MasterConfig) -> None:
        self._config = config
        self._server: grpc.aio.Server | None = None
        self._servicer = EdgeShardMasterServicer()

    async def start(self) -> None:
        """Start the Master gRPC server."""
        logger.info("Starting Master gRPC server")

        self._server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=self._config.grpc.max_workers)
        )

        # Add servicer
        edgeshard_pb2_grpc.add_WorkerServiceServicer_to_server(
            self._servicer, self._server
        )

        # Bind address
        bind_address = f"{self._config.grpc.host}:{self._config.grpc.port}"
        self._server.add_insecure_port(bind_address)

        await self._server.start()
        logger.info(f"Master listening on {bind_address}")

    async def stop(self, grace: float = 5.0) -> None:
        """Stop the Master gRPC server."""
        logger.info("Stopping Master gRPC server")
        if self._server:
            await self._server.stop(grace)
        logger.info("Master stopped")

    async def wait_for_termination(self) -> None:
        """Wait for server to terminate."""
        if self._server:
            await self._server.wait_for_termination()
