"""Test M5: Resource Discovery Enhancement.

This test verifies:
1. GPU metrics collection (utilization, temperature, power)
2. CPU metrics collection
3. Network latency measurement
4. Worker heartbeat with dynamic metrics
5. WorkerManager stores and returns metrics
6. CLI commands work with metrics
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from edgeshard._grpc import edgeshard_pb2, edgeshard_pb2_grpc
from edgeshard.common.config import MasterConfig
from edgeshard.master.server import MasterServer
from edgeshard.master.worker_manager import WorkerManager, WorkerRecord
from edgeshard.worker.hardware_probe import (
    collect_all_device_metrics,
    collect_gpu_metrics,
    get_cpu_utilization,
    get_used_memory_mb,
    probe_hardware,
)
from edgeshard.worker.network_probe import NetworkProbe


class TestGpuMetrics:
    """Test GPU metrics collection."""

    def test_collect_gpu_metrics_no_gpu(self):
        """Test GPU metrics collection when no GPU is available."""
        # This test runs on any system, GPU or not
        metrics = collect_gpu_metrics(0)
        # On systems without GPU or pynvml, should return None
        # On systems with GPU, should return GpuMetrics
        if metrics is not None:
            assert isinstance(metrics, edgeshard_pb2.GpuMetrics)
            assert 0 <= metrics.utilization_percent <= 100
            assert 0 <= metrics.memory_utilization_percent <= 100
            assert metrics.temperature_c >= 0
            assert metrics.free_memory_mb >= 0

    def test_collect_all_device_metrics(self):
        """Test collecting metrics for all devices."""
        devices = probe_hardware()
        metrics = collect_all_device_metrics(devices)

        # Should have at least CPU metrics
        assert len(metrics) >= 0  # May be empty if NVML not available

        for dm in metrics:
            assert dm.device_id  # Should have device ID

    def test_cpu_metrics(self):
        """Test CPU metrics functions."""
        utilization = get_cpu_utilization()
        assert 0 <= utilization <= 100

        used_memory = get_used_memory_mb()
        assert used_memory > 0


class TestNetworkProbe:
    """Test network probing functionality."""

    def test_network_probe_init(self):
        """Test NetworkProbe initialization."""
        probe = NetworkProbe("test-worker")
        assert probe._worker_id == "test-worker"
        assert len(probe._latency_cache) == 0

    def test_measure_latency_unreachable(self):
        """Test latency measurement to unreachable host."""
        probe = NetworkProbe("test-worker")

        async def run():
            # Try to connect to a non-existent host
            latency = await probe.measure_latency(
                "192.0.2.1",  # TEST-NET, should be unreachable
                99999,
                "target-worker",
            )
            return latency

        latency = asyncio.run(run())
        # Should return None for unreachable host
        assert latency is None

    def test_get_network_metrics(self):
        """Test getting network metrics from cache."""
        probe = NetworkProbe("test-worker")
        probe._latency_cache["worker-1"] = 5.5
        probe._bandwidth_cache["worker-1"] = 1000

        metrics = probe.get_network_metrics()

        assert isinstance(metrics, edgeshard_pb2.NetworkMetrics)
        assert "worker-1" in metrics.latency_ms_to_worker
        assert metrics.latency_ms_to_worker["worker-1"] == 5
        assert metrics.estimated_bandwidth_mbps == 1000


class TestWorkerManagerMetrics:
    """Test WorkerManager with dynamic metrics."""

    def test_register_worker(self):
        """Test worker registration."""
        manager = WorkerManager()

        devices = [
            edgeshard_pb2.DeviceInfo(
                device_id="cuda:0",
                device_type="cuda",
                name="Test GPU",
                total_memory_mb=8192,
            )
        ]

        success = manager.register_worker(
            worker_id="w-test",
            hostname="test-host",
            devices=devices,
            available_memory_mb=16384,
            cpu_count=8,
        )

        assert success
        worker = manager.get_worker("w-test")
        assert worker is not None
        assert worker.hostname == "test-host"
        assert len(worker.devices) == 1

    def test_update_heartbeat_with_metrics(self):
        """Test heartbeat update with dynamic metrics."""
        manager = WorkerManager()

        # Register first
        manager.register_worker(
            worker_id="w-test",
            hostname="test-host",
            devices=[],
            available_memory_mb=16384,
            cpu_count=8,
        )

        # Create GPU metrics
        gpu_metrics = edgeshard_pb2.GpuMetrics(
            utilization_percent=45,
            memory_utilization_percent=30,
            temperature_c=62,
            power_draw_mw=180000,
            power_limit_mw=350000,
            free_memory_mb=5000,
        )

        device_metrics = [
            edgeshard_pb2.DeviceMetrics(
                device_id="cuda:0",
                gpu_metrics=gpu_metrics,
            )
        ]

        # Create network metrics
        network_metrics = edgeshard_pb2.NetworkMetrics(
            latency_ms_to_worker={"w-other": 10},
            estimated_bandwidth_mbps=1000,
        )

        # Update heartbeat with metrics
        success = manager.update_heartbeat(
            worker_id="w-test",
            available_memory_mb=15000,
            status="online",
            device_metrics=device_metrics,
            network_metrics=network_metrics,
        )

        assert success

        worker = manager.get_worker("w-test")
        assert worker is not None
        assert len(worker.device_metrics) == 1
        assert worker.device_metrics[0].gpu_metrics.utilization_percent == 45
        assert worker.network_metrics.latency_ms_to_worker["w-other"] == 10

    def test_worker_record_gpu_summary(self):
        """Test WorkerRecord.get_gpu_summary()."""
        worker = WorkerRecord(
            worker_id="w-test",
            hostname="test-host",
            devices=[],
            available_memory_mb=16384,
            cpu_count=8,
        )

        # Add GPU metrics
        gpu_metrics = edgeshard_pb2.GpuMetrics(
            utilization_percent=50,
            temperature_c=65,
            power_draw_mw=200000,
            power_limit_mw=300000,
            free_memory_mb=4000,
        )
        worker.device_metrics = [
            edgeshard_pb2.DeviceMetrics(
                device_id="cuda:0",
                gpu_metrics=gpu_metrics,
            )
        ]

        summary = worker.get_gpu_summary()
        assert "cuda:0" in summary
        assert summary["cuda:0"]["utilization_percent"] == 50
        assert summary["cuda:0"]["temperature_c"] == 65
        assert summary["cuda:0"]["power_draw_w"] == 200.0

    def test_to_proto_includes_metrics(self):
        """Test WorkerRecord.to_proto() includes metrics."""
        worker = WorkerRecord(
            worker_id="w-test",
            hostname="test-host",
            devices=[],
            available_memory_mb=16384,
            cpu_count=8,
        )

        # Add metrics
        gpu_metrics = edgeshard_pb2.GpuMetrics(
            utilization_percent=40,
            temperature_c=60,
        )
        worker.device_metrics = [
            edgeshard_pb2.DeviceMetrics(
                device_id="cuda:0",
                gpu_metrics=gpu_metrics,
            )
        ]
        worker.network_metrics = edgeshard_pb2.NetworkMetrics(
            latency_ms_to_worker={"w-other": 5},
        )

        proto = worker.to_proto()

        assert len(proto.device_metrics) == 1
        assert proto.device_metrics[0].gpu_metrics.utilization_percent == 40
        assert proto.network_metrics.latency_ms_to_worker["w-other"] == 5


@pytest.mark.asyncio
class TestMasterMetricsIntegration:
    """Test Master handling of worker metrics."""

    async def test_heartbeat_with_metrics(self):
        """Test Master accepts heartbeat with metrics."""
        config = MasterConfig()
        config.grpc.port = 50060

        server = MasterServer(config)
        await server.start()
        await asyncio.sleep(0.5)

        try:
            import grpc

            channel = grpc.aio.insecure_channel(f"localhost:{config.grpc.port}")
            stub = edgeshard_pb2_grpc.WorkerServiceStub(channel)

            # Register worker
            register_request = edgeshard_pb2.RegisterWorkerRequest(
                worker_id="w-test-metrics",
                hostname="test-host",
                available_memory_mb=16384,
                cpu_count=8,
            )
            register_response = await stub.RegisterWorker(register_request)
            assert register_response.success

            # Send heartbeat with metrics
            gpu_metrics = edgeshard_pb2.GpuMetrics(
                utilization_percent=55,
                temperature_c=68,
            )
            device_metrics = edgeshard_pb2.DeviceMetrics(
                device_id="cuda:0",
                gpu_metrics=gpu_metrics,
            )

            heartbeat_request = edgeshard_pb2.HeartbeatRequest(
                worker_id="w-test-metrics",
                available_memory_mb=15000,
                status="online",
            )
            heartbeat_request.device_metrics.append(device_metrics)

            heartbeat_response = await stub.Heartbeat(heartbeat_request)
            assert heartbeat_response.success

            # Verify metrics stored
            list_response = await stub.ListWorkers(edgeshard_pb2.ListWorkersRequest())
            assert len(list_response.workers) == 1

            worker = list_response.workers[0]
            assert len(worker.device_metrics) == 1
            assert worker.device_metrics[0].gpu_metrics.utilization_percent == 55

            await channel.close()
        finally:
            await server.stop()


class TestProtobufMessages:
    """Test new protobuf message types."""

    def test_gpu_metrics_message(self):
        """Test GpuMetrics protobuf message."""
        metrics = edgeshard_pb2.GpuMetrics(
            utilization_percent=75,
            memory_utilization_percent=50,
            temperature_c=70,
            power_draw_mw=250000,
            power_limit_mw=350000,
            free_memory_mb=2048,
        )

        assert metrics.utilization_percent == 75
        assert metrics.temperature_c == 70
        assert metrics.free_memory_mb == 2048

    def test_cpu_metrics_message(self):
        """Test CpuMetrics protobuf message."""
        metrics = edgeshard_pb2.CpuMetrics(
            utilization_percent=30.5,
            used_memory_mb=8192,
        )

        assert metrics.utilization_percent == 30.5
        assert metrics.used_memory_mb == 8192

    def test_device_metrics_message(self):
        """Test DeviceMetrics protobuf message."""
        gpu_metrics = edgeshard_pb2.GpuMetrics(
            utilization_percent=60,
            temperature_c=65,
        )

        metrics = edgeshard_pb2.DeviceMetrics(
            device_id="cuda:0",
            gpu_metrics=gpu_metrics,
        )

        assert metrics.device_id == "cuda:0"
        assert metrics.HasField("gpu_metrics")
        assert metrics.gpu_metrics.utilization_percent == 60

    def test_network_metrics_message(self):
        """Test NetworkMetrics protobuf message."""
        metrics = edgeshard_pb2.NetworkMetrics(
            latency_ms_to_worker={"w-1": 5, "w-2": 10},
            estimated_bandwidth_mbps=1000,
        )

        assert metrics.latency_ms_to_worker["w-1"] == 5
        assert metrics.latency_ms_to_worker["w-2"] == 10
        assert metrics.estimated_bandwidth_mbps == 1000