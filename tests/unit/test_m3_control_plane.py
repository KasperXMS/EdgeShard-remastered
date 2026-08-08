"""Test M3: Master/Worker control plane.

This test verifies:
1. Master gRPC server starts correctly
2. Worker can register with Master
3. Worker heartbeat works
4. Worker can be unregistered
5. Cluster discovery returns correct state
"""

from __future__ import annotations

import asyncio
import time

import pytest

from edgeshard._grpc import edgeshard_pb2, edgeshard_pb2_grpc
from edgeshard.common.config import MasterConfig, WorkerConfig
from edgeshard.master.server import MasterServer
from edgeshard.worker.daemon import WorkerDaemon


@pytest.mark.asyncio
async def test_master_server_start_stop():
    """Test that Master server can start and stop."""
    config = MasterConfig()
    config.grpc.port = 50051  # Use different port for testing

    server = MasterServer(config)
    await server.start()

    # Give server time to start
    await asyncio.sleep(0.5)

    # Verify server is running by trying to connect
    import grpc

    channel = grpc.aio.insecure_channel(f"localhost:{config.grpc.port}")
    stub = edgeshard_pb2_grpc.WorkerServiceStub(channel)

    # Call ListWorkers (should return empty list)
    request = edgeshard_pb2.ListWorkersRequest()
    response = await stub.ListWorkers(request)
    assert len(response.workers) == 0

    await channel.close()
    await server.stop()


@pytest.mark.asyncio
async def test_worker_registration():
    """Test Worker registration with Master."""
    # Start Master
    master_config = MasterConfig()
    master_config.grpc.port = 50052
    master_server = MasterServer(master_config)
    await master_server.start()
    await asyncio.sleep(0.5)

    try:
        # Start Worker
        worker_config = WorkerConfig()
        worker_config.registration.master_address = f"localhost:{master_config.grpc.port}"
        worker_config.registration.heartbeat_interval_seconds = 1.0

        worker = WorkerDaemon(worker_config)
        await worker.start()
        await asyncio.sleep(1.0)  # Wait for registration

        # Verify Worker is registered
        import grpc

        channel = grpc.aio.insecure_channel(f"localhost:{master_config.grpc.port}")
        stub = edgeshard_pb2_grpc.WorkerServiceStub(channel)

        request = edgeshard_pb2.ListWorkersRequest()
        response = await stub.ListWorkers(request)

        assert len(response.workers) == 1
        assert response.workers[0].worker_id == worker.worker_id
        assert response.workers[0].status == "online"

        await channel.close()

        # Stop Worker
        await worker.stop()
        await asyncio.sleep(0.5)

        # Verify Worker is unregistered
        channel = grpc.aio.insecure_channel(f"localhost:{master_config.grpc.port}")
        stub = edgeshard_pb2_grpc.WorkerServiceStub(channel)

        response = await stub.ListWorkers(request)
        assert len(response.workers) == 0

        await channel.close()

    finally:
        await master_server.stop()


@pytest.mark.asyncio
async def test_worker_heartbeat():
    """Test Worker heartbeat updates."""
    # Start Master
    master_config = MasterConfig()
    master_config.grpc.port = 50053
    master_server = MasterServer(master_config)
    await master_server.start()
    await asyncio.sleep(0.5)

    try:
        # Start Worker with fast heartbeat
        worker_config = WorkerConfig()
        worker_config.registration.master_address = f"localhost:{master_config.grpc.port}"
        worker_config.registration.heartbeat_interval_seconds = 0.5

        worker = WorkerDaemon(worker_config)
        await worker.start()

        # Wait for registration + 2 heartbeats
        await asyncio.sleep(2.0)

        # Verify Worker is still registered and alive
        import grpc

        channel = grpc.aio.insecure_channel(f"localhost:{master_config.grpc.port}")
        stub = edgeshard_pb2_grpc.WorkerServiceStub(channel)

        request = edgeshard_pb2.GetWorkerRequest(worker_id=worker.worker_id)
        response = await stub.GetWorker(request)

        assert response.worker.worker_id == worker.worker_id
        assert response.worker.status == "online"
        assert response.worker.last_heartbeat > 0

        await channel.close()
        await worker.stop()

    finally:
        await master_server.stop()


@pytest.mark.asyncio
async def test_multiple_workers():
    """Test multiple Workers registering with Master."""
    # Start Master
    master_config = MasterConfig()
    master_config.grpc.port = 50054
    master_server = MasterServer(master_config)
    await master_server.start()
    await asyncio.sleep(0.5)

    workers = []
    try:
        # Start 3 Workers
        for i in range(3):
            worker_config = WorkerConfig()
            worker_config.registration.master_address = (
                f"localhost:{master_config.grpc.port}"
            )
            worker_config.registration.heartbeat_interval_seconds = 1.0

            worker = WorkerDaemon(worker_config)
            await worker.start()
            workers.append(worker)

        # Wait for all registrations
        await asyncio.sleep(2.0)

        # Verify all Workers are registered
        import grpc

        channel = grpc.aio.insecure_channel(f"localhost:{master_config.grpc.port}")
        stub = edgeshard_pb2_grpc.WorkerServiceStub(channel)

        request = edgeshard_pb2.ListWorkersRequest()
        response = await stub.ListWorkers(request)

        assert len(response.workers) == 3

        # All should be online
        for worker_state in response.workers:
            assert worker_state.status == "online"

        await channel.close()

    finally:
        # Stop all workers
        for worker in workers:
            await worker.stop()

        await asyncio.sleep(0.5)
        await master_server.stop()


if __name__ == "__main__":
    print("Running M3 Master/Worker control plane tests...")
    print("(These tests require grpcio and grpcio-tools)")

    asyncio.run(test_master_server_start_stop())
    print("[PASS] test_master_server_start_stop")

    asyncio.run(test_worker_registration())
    print("[PASS] test_worker_registration")

    asyncio.run(test_worker_heartbeat())
    print("[PASS] test_worker_heartbeat")

    asyncio.run(test_multiple_workers())
    print("[PASS] test_multiple_workers")

    print("\nAll M3 tests passed!")
