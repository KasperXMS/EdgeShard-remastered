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
        # M8: Deployment state tracking
        self._deployments: dict[str, dict] = {}  # service_name -> deployment info

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
        """Deploy a service (M8)."""
        logger.info(f"Deploy service request: {request.spec.name}")
        # Store deployment info
        self._deployments[request.spec.name] = {
            "spec": request.spec,
            "status": "deploying",
            "shards": [],
        }
        return edgeshard_pb2.DeployServiceResponse(
            success=True,
            message=f"Service {request.spec.name} deployment initiated",
            service_name=request.spec.name,
        )

    async def ListServices(
        self,
        request: edgeshard_pb2.ListServicesRequest,
        context: grpc.ServicerContext,
    ) -> edgeshard_pb2.ListServicesResponse:
        """List deployed services (M8)."""
        return edgeshard_pb2.ListServicesResponse(
            service_names=list(self._deployments.keys())
        )

    async def GetServiceStatus(
        self,
        request: edgeshard_pb2.GetServiceStatusRequest,
        context: grpc.ServicerContext,
    ) -> edgeshard_pb2.GetServiceStatusResponse:
        """Get service status (M8)."""
        deployment = self._deployments.get(request.service_name)
        if deployment is None:
            return edgeshard_pb2.GetServiceStatusResponse(
                service_name=request.service_name,
                status="not_found",
            )
        return edgeshard_pb2.GetServiceStatusResponse(
            service_name=request.service_name,
            status=deployment.get("status", "unknown"),
            shard_count=len(deployment.get("shards", [])),
            message=f"Service {request.service_name}",
        )

    async def StopService(
        self,
        request: edgeshard_pb2.StopServiceRequest,
        context: grpc.ServicerContext,
    ) -> edgeshard_pb2.StopServiceResponse:
        """Stop a service (M8)."""
        logger.info(f"Stop service request: {request.service_name}")
        if request.service_name in self._deployments:
            self._deployments[request.service_name]["status"] = "stopped"
            return edgeshard_pb2.StopServiceResponse(
                success=True,
                message=f"Service {request.service_name} stopped",
            )
        return edgeshard_pb2.StopServiceResponse(
            success=False,
            message=f"Service {request.service_name} not found",
        )

    async def RegisterShards(
        self,
        request: edgeshard_pb2.RegisterShardsRequest,
        context: grpc.ServicerContext,
    ) -> edgeshard_pb2.RegisterShardsResponse:
        """Register deployed shards for a service (M8).

        Called by `edgeshard deploy` after shards are started.
        """
        service_name = request.service_name
        if service_name not in self._deployments:
            self._deployments[service_name] = {
                "spec": None,
                "status": "ready",
                "shards": [],
            }

        # Store shard endpoints
        self._deployments[service_name]["shards"] = [
            {
                "shard_id": s.shard_id,
                "worker_id": s.worker_id,
                "data_address": s.data_address,
                "layer_start": s.layer_start,
                "layer_end": s.layer_end,
                "device": s.device,
                "model_name": s.model_name,
                "is_first_shard": s.is_first_shard,
                "is_last_shard": s.is_last_shard,
            }
            for s in request.shards
        ]
        self._deployments[service_name]["status"] = "ready"

        logger.info(
            f"Registered {len(request.shards)} shard(s) for service {service_name}"
        )

        return edgeshard_pb2.RegisterShardsResponse(
            success=True,
            message=f"Registered {len(request.shards)} shard(s)",
        )

    async def GetShardEndpoints(
        self,
        request: edgeshard_pb2.GetShardEndpointsRequest,
        context: grpc.ServicerContext,
    ) -> edgeshard_pb2.GetShardEndpointsResponse:
        """Get shard endpoints for inference (M8).

        Called by `edgeshard infer` to discover shard addresses.
        """
        service_name = request.service_name
        deployment = self._deployments.get(service_name)

        if deployment is None:
            # If no service name specified, try to find any ready deployment
            if not service_name:
                for name, dep in self._deployments.items():
                    if dep.get("status") == "ready" and dep.get("shards"):
                        deployment = dep
                        service_name = name
                        break

            if deployment is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"No deployed service found")
                return edgeshard_pb2.GetShardEndpointsResponse()

        # Build endpoint list
        endpoints = []
        for shard in deployment.get("shards", []):
            endpoints.append(
                edgeshard_pb2.ShardEndpoint(
                    shard_id=shard["shard_id"],
                    worker_id=shard["worker_id"],
                    data_address=shard["data_address"],
                    layer_start=shard["layer_start"],
                    layer_end=shard["layer_end"],
                    device=shard["device"],
                    model_name=shard.get("model_name", ""),
                    is_first_shard=shard.get("is_first_shard", False),
                    is_last_shard=shard.get("is_last_shard", False),
                )
            )

        return edgeshard_pb2.GetShardEndpointsResponse(
            service_name=service_name,
            endpoints=endpoints,
        )

    async def Profile(
        self,
        request: edgeshard_pb2.ProfileRequest,
        context: grpc.ServicerContext,
    ) -> edgeshard_pb2.ProfileResponse:
        """Run profiling on a model."""
        logger.info(f"Profile request for {request.model_name} on {request.device}")

        try:
            import torch
            from edgeshard.profiler.executor import ProfileExecutor
            from edgeshard.profiler.store import ProfileStore

            # Parse dtype
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(request.dtype, torch.float16)
            torch_device = torch.device(request.device)

            # Run profiling
            executor = ProfileExecutor()
            result = await executor.profile(
                model_path=request.model_name,
                dtype=torch_dtype,
                device=torch_device,
            )

            # Save to store
            store = ProfileStore()
            store.save(result)

            return edgeshard_pb2.ProfileResponse(
                success=True,
                message="Profiling complete",
                layer_forward_ms=result.layer_forward_ms,
                kv_cache_per_token_mb=result.kv_cache_per_token_mb,
                prefill_tokens_per_sec=result.prefill_tokens_per_sec,
                decode_tokens_per_sec=result.decode_tokens_per_sec,
                num_layers_profiled=result.num_layers_profiled,
                total_model_memory_mb=result.total_model_memory_mb,
                device_name=result.device_name,
            )

        except Exception as e:
            logger.error(f"Profiling failed: {e}")
            return edgeshard_pb2.ProfileResponse(
                success=False,
                message=f"Profiling failed: {e}",
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
