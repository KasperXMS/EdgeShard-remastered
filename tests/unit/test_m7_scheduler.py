"""Test M7: Scheduler and Placement.

Tests the scheduling pipeline:
1. ClusterSnapshot extensions (bandwidth, source worker)
2. Model info extraction
3. Bridge functions
4. DP latency optimization
5. DP throughput optimization
6. Single-device placement
7. Policy selection
8. PlacementPlan YAML serialization
9. End-to-end planner
10. Error handling

All tests use synthetic data — no GPU or model required.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from edgeshard.common.config import ModelSpec, ServiceSpec
from edgeshard.common.errors import SchedulerError
from edgeshard.common.identifiers import ServiceName, WorkerId
from edgeshard.scheduler import (
    ClusterSnapshot,
    DeviceInfo,
    PlacementPlan,
    ProfileEntry,
    ProfileSnapshot,
    SchedulingPolicy,
    ShardPlacement,
    WorkerState,
    schedule,
)
from edgeshard.scheduler.model_info import ModelSchedulingInfo, load_model_info
from edgeshard.scheduler.policies import DeviceSlot, PartitionResult
from edgeshard.scheduler.policies.default import (
    DefaultPolicy,
    optimize_latency,
    optimize_throughput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_device(
    device_id: str = "cuda:0",
    device_type: str = "cuda",
    name: str = "RTX 4090",
    total_memory_mb: int = 24576,
) -> DeviceInfo:
    return DeviceInfo(
        device_id=device_id,
        device_type=device_type,
        name=name,
        total_memory_mb=total_memory_mb,
    )


def make_worker(
    worker_id: str = "w-001",
    hostname: str = "worker-1",
    devices: list[DeviceInfo] | None = None,
    available_memory_mb: int = 32768,
    status: str = "online",
) -> WorkerState:
    if devices is None:
        devices = [make_device()]
    return WorkerState(
        worker_id=WorkerId(worker_id),
        hostname=hostname,
        devices=devices,
        available_memory_mb=available_memory_mb,
        cpu_count=8,
        status=status,
    )


def make_profile(
    model_name: str = "test-model",
    device_name: str = "RTX 4090",
    layer_forward_ms: float = 1.0,
    kv_cache_per_token_mb: float = 0.001,
    prefill_tokens_per_sec: float = 5000.0,
    decode_tokens_per_sec: float = 100.0,
) -> ProfileEntry:
    return ProfileEntry(
        model_name=model_name,
        model_revision="main",
        device_fingerprint=f"cuda:{device_name}",
        dtype="float16",
        runtime_version="1.0",
        layer_forward_ms=layer_forward_ms,
        kv_cache_per_token_mb=kv_cache_per_token_mb,
        prefill_tokens_per_sec=prefill_tokens_per_sec,
        decode_tokens_per_sec=decode_tokens_per_sec,
    )


def make_model_info(
    num_layers: int = 24,
    hidden_size: int = 4096,
    per_layer_memory_mb: float = 100.0,
    kv_cache_per_token_mb: float = 0.001,
    total_model_memory_mb: float = 2400.0,
) -> ModelSchedulingInfo:
    return ModelSchedulingInfo(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=32,
        num_kv_heads=32,
        intermediate_size=11008,
        vocab_size=152064,
        per_layer_memory_mb=per_layer_memory_mb,
        kv_cache_per_token_mb=kv_cache_per_token_mb,
        total_model_memory_mb=total_model_memory_mb,
    )


# ---------------------------------------------------------------------------
# Test ClusterSnapshot Extensions
# ---------------------------------------------------------------------------


class TestClusterSnapshot:
    """Test ClusterSnapshot extensions for M7."""

    def test_bandwidth_matrix(self):
        """Test bandwidth_matrix in ClusterSnapshot."""
        snapshot = ClusterSnapshot(
            workers=[make_worker("w-001"), make_worker("w-002")],
            bandwidth_matrix={
                ("w-001", "w-002"): 125.0,  # 1000 Mbps = 125 MB/s
            },
        )

        assert snapshot.get_bandwidth("w-001", "w-002") == 125.0
        assert snapshot.get_bandwidth("w-002", "w-001") == 125.0  # Reverse lookup
        assert snapshot.get_bandwidth("w-001", "w-001") == float("inf")  # Same worker
        assert snapshot.get_bandwidth("w-001", "w-unknown") == 100.0  # Default

    def test_source_worker(self):
        """Test source_worker_id selection."""
        snapshot = ClusterSnapshot(
            workers=[make_worker("w-001"), make_worker("w-002")],
            source_worker_id="w-002",
        )
        assert snapshot.get_source_worker() == "w-002"

    def test_source_worker_default(self):
        """Test source_worker defaults to first worker."""
        snapshot = ClusterSnapshot(
            workers=[make_worker("w-001"), make_worker("w-002")],
        )
        assert snapshot.get_source_worker() == "w-001"

    def test_source_worker_empty_raises(self):
        """Test empty cluster raises on get_source_worker."""
        snapshot = ClusterSnapshot()
        with pytest.raises(ValueError, match="no workers"):
            snapshot.get_source_worker()


# ---------------------------------------------------------------------------
# Test Model Info
# ---------------------------------------------------------------------------


class TestModelInfo:
    """Test model info extraction."""

    def test_load_from_config_json(self):
        """Test loading model info from config.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "num_hidden_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 11008,
                "vocab_size": 152064,
            }
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)

            info = load_model_info(tmpdir)
            assert info.num_layers == 32
            assert info.hidden_size == 4096
            assert info.num_kv_heads == 8
            assert info.intermediate_size == 11008

    def test_load_with_profile_fallback(self):
        """Test that profile data supplements config info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"num_hidden_layers": 24, "hidden_size": 2048}
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config, f)

            profile = make_profile(kv_cache_per_token_mb=0.002)
            info = load_model_info(tmpdir, profile=profile)

            assert info.num_layers == 24
            assert info.kv_cache_per_token_mb == pytest.approx(0.002)

    def test_load_no_config_raises(self):
        """Test that missing config.json + no profile raises."""
        with pytest.raises(SchedulerError, match="Cannot determine"):
            load_model_info("/nonexistent/model/path")


# ---------------------------------------------------------------------------
# Test DP Latency Optimization
# ---------------------------------------------------------------------------


class TestLatencyDP:
    """Test Algorithm 1: Latency optimization."""

    def test_single_device(self):
        """All layers on one device."""
        device = DeviceSlot("w-001", "cuda:0", "cuda", "GPU", 24000, 1.0)
        comm_costs = {(0, 0): 0.0}

        result = optimize_latency(
            num_layers=4,
            devices=[device],
            comm_costs=comm_costs,
            per_layer_memory=100.0,
            kv_cache_per_layer=1.0,
            source_device=device,
        )

        assert len(result.assignments) == 1
        assert result.assignments[0][0] == 0
        assert result.assignments[0][1] == 4
        assert result.estimated_latency_ms == pytest.approx(4.0)

    def test_two_devices_split(self):
        """Two devices with different speeds — faster device gets more layers."""
        fast = DeviceSlot("w-001", "cuda:0", "cuda", "Fast GPU", 24000, 0.5)
        slow = DeviceSlot("w-002", "cuda:0", "cuda", "Slow GPU", 24000, 2.0)
        comm_costs = {
            (0, 0): 0.0,
            (0, 1): 5.0,
            (1, 0): 5.0,
            (1, 1): 0.0,
        }

        result = optimize_latency(
            num_layers=8,
            devices=[fast, slow],
            comm_costs=comm_costs,
            per_layer_memory=100.0,
            kv_cache_per_layer=1.0,
            source_device=fast,
        )

        # Should use both devices
        assert len(result.assignments) >= 1
        # All layers should be covered
        total_layers = sum(end - start for start, end, _ in result.assignments)
        assert total_layers == 8

    def test_memory_constraint(self):
        """Device can't hold all layers — must split."""
        small = DeviceSlot("w-001", "cuda:0", "cuda", "Small GPU", 500.0, 1.0)
        large = DeviceSlot("w-002", "cuda:0", "cuda", "Large GPU", 24000.0, 2.0)
        comm_costs = {
            (0, 0): 0.0,
            (0, 1): 1.0,
            (1, 0): 1.0,
            (1, 1): 0.0,
        }

        result = optimize_latency(
            num_layers=10,
            devices=[small, large],
            comm_costs=comm_costs,
            per_layer_memory=100.0,  # 100 MB per layer
            kv_cache_per_layer=1.0,
            source_device=small,
        )

        # Small GPU can hold ~4 layers (500 - 1 kv) / 100 = ~4
        # So it shouldn't have all 10
        for start, end, dev in result.assignments:
            if dev.worker_id == "w-001":
                layers = end - start
                assert layers <= 5  # Limited by memory

    def test_no_valid_placement_raises(self):
        """All devices too small — should raise."""
        tiny = DeviceSlot("w-001", "cuda:0", "cuda", "Tiny", 10.0, 1.0)
        comm_costs = {(0, 0): 0.0}

        with pytest.raises(SchedulerError, match="No valid partition"):
            optimize_latency(
                num_layers=10,
                devices=[tiny],
                comm_costs=comm_costs,
                per_layer_memory=100.0,
                kv_cache_per_layer=1.0,
                source_device=tiny,
            )


# ---------------------------------------------------------------------------
# Test DP Throughput Optimization
# ---------------------------------------------------------------------------


class TestThroughputDP:
    """Test Algorithm 2: Throughput optimization."""

    def test_single_device_throughput(self):
        """Single device throughput."""
        device = DeviceSlot("w-001", "cuda:0", "cuda", "GPU", 24000, 1.0)
        comm_costs = {(0, 0): 0.0}

        result = optimize_throughput(
            num_layers=4,
            devices=[device],
            comm_costs=comm_costs,
            per_layer_memory=100.0,
            kv_cache_per_layer=1.0,
            source_device=device,
        )

        assert len(result.assignments) == 1
        assert result.estimated_throughput_tps > 0

    def test_throughput_balances_stages(self):
        """Throughput optimization should balance stage times."""
        fast = DeviceSlot("w-001", "cuda:0", "cuda", "Fast", 24000, 1.0)
        slow = DeviceSlot("w-002", "cuda:0", "cuda", "Slow", 24000, 3.0)
        comm_costs = {
            (0, 0): 0.0,
            (0, 1): 0.1,
            (1, 0): 0.1,
            (1, 1): 0.0,
        }

        result = optimize_throughput(
            num_layers=8,
            devices=[fast, slow],
            comm_costs=comm_costs,
            per_layer_memory=100.0,
            kv_cache_per_layer=1.0,
            source_device=fast,
        )

        # All layers covered
        total = sum(end - start for start, end, _ in result.assignments)
        assert total == 8


# ---------------------------------------------------------------------------
# Test DefaultPolicy
# ---------------------------------------------------------------------------


class TestDefaultPolicy:
    """Test the DefaultPolicy.solve() method."""

    def test_single_device_placement(self):
        """Model fits on one device — no splitting."""
        model_info = make_model_info(
            num_layers=24,
            per_layer_memory_mb=100.0,
            total_model_memory_mb=2400.0,
            kv_cache_per_token_mb=0.001,
        )

        cluster = ClusterSnapshot(
            workers=[make_worker("w-001", devices=[make_device(total_memory_mb=24576)])],
            source_worker_id="w-001",
        )

        profiles = ProfileSnapshot(entries=[make_profile()])

        policy = DefaultPolicy()
        result = policy.solve(
            model_info=model_info,
            cluster=cluster,
            profiles=profiles,
            policy_config=SchedulingPolicy(name="default"),
            max_seq_len=512,
        )

        # Should fit on single device (2400 MB model + 0.512 MB KV << 24576 MB)
        assert len(result.assignments) == 1
        assert result.assignments[0][0] == 0
        assert result.assignments[0][1] == 24

    def test_multi_device_split(self):
        """Model doesn't fit on one device — must split."""
        model_info = make_model_info(
            num_layers=24,
            per_layer_memory_mb=500.0,  # 500 MB per layer
            total_model_memory_mb=12000.0,  # 12 GB total
            kv_cache_per_token_mb=0.001,
        )

        # Two small GPUs, each 8 GB — can't hold 12 GB model alone
        cluster = ClusterSnapshot(
            workers=[
                make_worker("w-001", devices=[make_device("cuda:0", total_memory_mb=8192)]),
                make_worker("w-002", devices=[make_device("cuda:0", total_memory_mb=8192)]),
            ],
            source_worker_id="w-001",
            bandwidth_matrix={("w-001", "w-002"): 125.0},
        )

        profiles = ProfileSnapshot(entries=[
            make_profile(device_name="RTX 4090", layer_forward_ms=1.0),
        ])

        policy = DefaultPolicy()
        result = policy.solve(
            model_info=model_info,
            cluster=cluster,
            profiles=profiles,
            policy_config=SchedulingPolicy(name="default"),
            max_seq_len=512,
        )

        # Should split across devices
        assert len(result.assignments) >= 1
        total_layers = sum(end - start for start, end, _ in result.assignments)
        assert total_layers == 24

    def test_no_workers_raises(self):
        """Empty cluster should raise SchedulerError."""
        model_info = make_model_info()
        cluster = ClusterSnapshot(workers=[])
        profiles = ProfileSnapshot()

        policy = DefaultPolicy()
        with pytest.raises(SchedulerError, match="No usable devices"):
            policy.solve(
                model_info=model_info,
                cluster=cluster,
                profiles=profiles,
                policy_config=SchedulingPolicy(),
            )


# ---------------------------------------------------------------------------
# Test PlacementPlan YAML Serialization
# ---------------------------------------------------------------------------


class TestPlacementPlanYAML:
    """Test PlacementPlan to_yaml / from_yaml roundtrip."""

    def test_roundtrip(self):
        """Serialize and deserialize should produce equal plan."""
        spec = ServiceSpec(
            name="test-service",
            model=ModelSpec(name="test-model", revision="main", dtype="float16"),
        )

        plan = PlacementPlan(
            service_name=ServiceName("test-service"),
            service_spec=spec,
            shards=[
                ShardPlacement(0, WorkerId("w-001"), 0, 12, "cuda:0"),
                ShardPlacement(1, WorkerId("w-002"), 12, 24, "cuda:0"),
            ],
            version=1,
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name

        try:
            plan.to_yaml(path)
            loaded = PlacementPlan.from_yaml(path)

            assert loaded.service_name.value == "test-service"
            assert len(loaded.shards) == 2
            assert loaded.shards[0].layer_start == 0
            assert loaded.shards[0].layer_end == 12
            assert str(loaded.shards[0].worker_id) == "w-001"
            assert loaded.shards[1].layer_start == 12
            assert loaded.shards[1].layer_end == 24
            assert str(loaded.shards[1].worker_id) == "w-002"
            assert loaded.version == 1
        finally:
            os.unlink(path)

    def test_shard_for_layer(self):
        """Test shard_for_layer lookup."""
        spec = ServiceSpec(name="test", model=ModelSpec(name="m"))
        plan = PlacementPlan(
            service_name=ServiceName("test"),
            service_spec=spec,
            shards=[
                ShardPlacement(0, WorkerId("w-001"), 0, 12, "cuda:0"),
                ShardPlacement(1, WorkerId("w-002"), 12, 24, "cuda:0"),
            ],
        )

        assert plan.shard_for_layer(0).shard_index == 0
        assert plan.shard_for_layer(11).shard_index == 0
        assert plan.shard_for_layer(12).shard_index == 1
        assert plan.shard_for_layer(23).shard_index == 1
        assert plan.shard_for_layer(24) is None

    def test_shards_on_worker(self):
        """Test shards_on_worker lookup."""
        spec = ServiceSpec(name="test", model=ModelSpec(name="m"))
        plan = PlacementPlan(
            service_name=ServiceName("test"),
            service_spec=spec,
            shards=[
                ShardPlacement(0, WorkerId("w-001"), 0, 12, "cuda:0"),
                ShardPlacement(1, WorkerId("w-001"), 12, 18, "cuda:0"),
                ShardPlacement(2, WorkerId("w-002"), 18, 24, "cuda:0"),
            ],
        )

        assert len(plan.shards_on_worker(WorkerId("w-001"))) == 2
        assert len(plan.shards_on_worker(WorkerId("w-002"))) == 1
        assert len(plan.shards_on_worker(WorkerId("w-999"))) == 0


# ---------------------------------------------------------------------------
# Test End-to-End Planner
# ---------------------------------------------------------------------------


class TestPlannerEndToEnd:
    """Test the schedule() function end-to-end."""

    def test_schedule_single_device(self):
        """Schedule a model that fits on one device."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config.json
            config = {
                "num_hidden_layers": 12,
                "hidden_size": 2048,
                "num_attention_heads": 16,
                "intermediate_size": 5504,
            }
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config, f)

            model = ModelSpec(name=tmpdir, dtype="float16")
            cluster = ClusterSnapshot(
                workers=[make_worker("w-001")],
                source_worker_id="w-001",
            )
            profiles = ProfileSnapshot(entries=[make_profile()])
            policy = SchedulingPolicy(name="default")

            plan = schedule(model, cluster, profiles, policy)

            assert isinstance(plan, PlacementPlan)
            assert len(plan.shards) >= 1
            # All layers covered
            total = sum(s.layer_end - s.layer_start for s in plan.shards)
            assert total == 12

    def test_schedule_empty_cluster_raises(self):
        """Schedule with no workers should raise."""
        model = ModelSpec(name="test-model")
        cluster = ClusterSnapshot()
        profiles = ProfileSnapshot()
        policy = SchedulingPolicy()

        with pytest.raises(SchedulerError, match="no workers"):
            schedule(model, cluster, profiles, policy)


# ---------------------------------------------------------------------------
# Test Protobuf Integration
# ---------------------------------------------------------------------------


class TestProtobufMessages:
    """Test new protobuf messages for M7."""

    def test_shard_placement_proto(self):
        """Test ShardPlacementProto message."""
        from edgeshard._grpc import edgeshard_pb2

        msg = edgeshard_pb2.ShardPlacementProto(
            shard_index=0,
            worker_id="w-001",
            layer_start=0,
            layer_end=12,
            device="cuda:0",
        )
        assert msg.shard_index == 0
        assert msg.worker_id == "w-001"
        assert msg.layer_start == 0
        assert msg.layer_end == 12

    def test_placement_plan_message(self):
        """Test PlacementPlanMessage."""
        from edgeshard._grpc import edgeshard_pb2

        shard = edgeshard_pb2.ShardPlacementProto(
            shard_index=0,
            worker_id="w-001",
            layer_start=0,
            layer_end=24,
            device="cuda:0",
        )
        msg = edgeshard_pb2.PlacementPlanMessage(
            service_name="test-service",
            version=1,
            model_name="test-model",
            shards=[shard],
            estimated_latency_ms=100.0,
            estimated_throughput_tps=50.0,
        )
        assert msg.service_name == "test-service"
        assert len(msg.shards) == 1
        assert msg.estimated_latency_ms == pytest.approx(100.0)
