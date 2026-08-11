"""Tests for M8: Deployment module.

Tests the deployment backend interface, local backend, remote backend,
and deployment manager.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from edgeshard._grpc import edgeshard_pb2, edgeshard_pb2_grpc
from edgeshard.common.config import ModelSpec, ServiceSpec
from edgeshard.common.identifiers import ServiceName, WorkerId
from edgeshard.deployment.backend import DeploymentBackend, ShardHandle, ShardStatus
from edgeshard.deployment.local import LocalDeploymentBackend
from edgeshard.deployment.manager import DeploymentManager, DeploymentState
from edgeshard.deployment.remote import RemoteDeploymentBackend
from edgeshard.scheduler.placement import PlacementPlan, ShardPlacement


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_plan() -> PlacementPlan:
    """Create a sample placement plan for testing."""
    model_spec = ModelSpec(name="test-model", revision="main", dtype="float16")
    service_spec = ServiceSpec(name="test-service", model=model_spec)

    shards = [
        ShardPlacement(
            shard_index=0,
            worker_id=WorkerId("worker-1"),
            layer_start=0,
            layer_end=14,
            device="cuda:0",
        ),
        ShardPlacement(
            shard_index=1,
            worker_id=WorkerId("worker-2"),
            layer_start=14,
            layer_end=28,
            device="cuda:0",
        ),
    ]

    return PlacementPlan(
        service_name=ServiceName("test-service"),
        service_spec=service_spec,
        shards=shards,
        version=1,
    )


# ---------------------------------------------------------------------------
# Test Proto Messages (M8)
# ---------------------------------------------------------------------------


class TestM8ProtoMessages:
    """Test M8 protobuf messages."""

    def test_start_shard_request(self):
        """Test StartShardRequest message creation."""
        request = edgeshard_pb2.StartShardRequest(
            shard_id="shard-0",
            model_name="Qwen/Qwen2.5-7B-Instruct",
            model_revision="main",
            dtype="float16",
            layer_start=0,
            layer_end=14,
            device="cuda:0",
            data_host="0.0.0.0",
            data_port=50100,
            is_first_shard=True,
            is_last_shard=False,
            master_address="localhost:10500",
        )

        assert request.shard_id == "shard-0"
        assert request.model_name == "Qwen/Qwen2.5-7B-Instruct"
        assert request.layer_start == 0
        assert request.layer_end == 14
        assert request.is_first_shard is True
        assert request.is_last_shard is False

    def test_start_shard_response(self):
        """Test StartShardResponse message creation."""
        response = edgeshard_pb2.StartShardResponse(
            success=True,
            message="Shard started",
            shard_id="shard-0",
            data_address="0.0.0.0:50100",
        )

        assert response.success is True
        assert response.shard_id == "shard-0"
        assert response.data_address == "0.0.0.0:50100"

    def test_stop_shard_request(self):
        """Test StopShardRequest message creation."""
        request = edgeshard_pb2.StopShardRequest(shard_id="shard-0")
        assert request.shard_id == "shard-0"

    def test_list_shards_response(self):
        """Test ListShardsResponse with ShardInfo."""
        shard_info = edgeshard_pb2.ShardInfo(
            shard_id="shard-0",
            model_name="test-model",
            layer_start=0,
            layer_end=14,
            device="cuda:0",
            status="ready",
            data_address="0.0.0.0:50100",
            start_time=1234567890,
            pid="12345",
        )

        response = edgeshard_pb2.ListShardsResponse(shards=[shard_info])

        assert len(response.shards) == 1
        assert response.shards[0].shard_id == "shard-0"
        assert response.shards[0].status == "ready"


# ---------------------------------------------------------------------------
# Test ShardHandle and ShardStatus
# ---------------------------------------------------------------------------


class TestShardHandle:
    """Test ShardHandle dataclass."""

    def test_shard_handle_creation(self):
        """Test ShardHandle creation."""
        handle = ShardHandle(
            shard_id="shard-0",
            worker_id="worker-1",
            status=ShardStatus.STARTING,
            data_address="0.0.0.0:50100",
            model_name="test-model",
            layer_start=0,
            layer_end=14,
            device="cuda:0",
        )

        assert handle.shard_id == "shard-0"
        assert handle.worker_id == "worker-1"
        assert handle.status == ShardStatus.STARTING

    def test_shard_status_enum(self):
        """Test ShardStatus enum values."""
        assert ShardStatus.STARTING.value == "starting"
        assert ShardStatus.READY.value == "ready"
        assert ShardStatus.FAILED.value == "failed"
        assert ShardStatus.STOPPED.value == "stopped"


# ---------------------------------------------------------------------------
# Test LocalDeploymentBackend
# ---------------------------------------------------------------------------


class TestLocalDeploymentBackend:
    """Test LocalDeploymentBackend."""

    @pytest.fixture
    def backend(self) -> LocalDeploymentBackend:
        """Create a local deployment backend."""
        return LocalDeploymentBackend()

    def test_backend_creation(self, backend: LocalDeploymentBackend):
        """Test backend is created correctly."""
        assert backend._shards == {}
        assert backend._processes == {}

    @pytest.mark.asyncio
    async def test_list_shards_empty(self, backend: LocalDeploymentBackend):
        """Test listing shards when none exist."""
        shards = await backend.list_shards()
        assert shards == []

    @pytest.mark.asyncio
    async def test_get_shard_not_found(self, backend: LocalDeploymentBackend):
        """Test getting a non-existent shard."""
        handle = await backend.get_shard("nonexistent")
        assert handle is None


# ---------------------------------------------------------------------------
# Test RemoteDeploymentBackend
# ---------------------------------------------------------------------------


class TestRemoteDeploymentBackend:
    """Test RemoteDeploymentBackend."""

    @pytest.fixture
    def backend(self) -> RemoteDeploymentBackend:
        """Create a remote deployment backend."""
        return RemoteDeploymentBackend(
            worker_addresses={
                "worker-1": "host1:10600",
                "worker-2": "host2:10600",
            }
        )

    def test_backend_creation(self, backend: RemoteDeploymentBackend):
        """Test backend is created with worker addresses."""
        assert backend._worker_addresses["worker-1"] == "host1:10600"
        assert backend._worker_addresses["worker-2"] == "host2:10600"

    def test_get_address(self, backend: RemoteDeploymentBackend):
        """Test getting worker address."""
        assert backend._get_address("worker-1") == "host1:10600"
        assert backend._get_address("unknown") == "unknown:10500"  # default

    def test_update_worker_address(self, backend: RemoteDeploymentBackend):
        """Test updating worker address."""
        backend.update_worker_address("worker-1", "newhost:10600")
        assert backend._worker_addresses["worker-1"] == "newhost:10600"

    @pytest.mark.asyncio
    async def test_list_shards_empty(self, backend: RemoteDeploymentBackend):
        """Test listing shards when none exist."""
        shards = await backend.list_shards()
        assert shards == []

    @pytest.mark.asyncio
    async def test_get_shard_not_found(self, backend: RemoteDeploymentBackend):
        """Test getting a non-existent shard."""
        handle = await backend.get_shard("nonexistent")
        assert handle is None

    def test_close(self, backend: RemoteDeploymentBackend):
        """Test closing backend."""
        backend.close()
        assert backend._channels == {}


# ---------------------------------------------------------------------------
# Test DeploymentManager
# ---------------------------------------------------------------------------


class TestDeploymentManager:
    """Test DeploymentManager."""

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Create a mock deployment backend."""
        backend = MagicMock(spec=DeploymentBackend)
        backend.start_shard = AsyncMock()
        backend.stop_shard = AsyncMock(return_value=True)
        backend.list_shards = AsyncMock(return_value=[])
        backend.get_shard = AsyncMock()
        backend.wait_ready = AsyncMock(return_value=True)
        return backend

    @pytest.fixture
    def manager(self, mock_backend: MagicMock) -> DeploymentManager:
        """Create a deployment manager with mock backend."""
        return DeploymentManager(mock_backend)

    def test_manager_creation(self, manager: DeploymentManager):
        """Test manager is created correctly."""
        assert manager._deployments == {}

    def test_get_deployment_not_found(self, manager: DeploymentManager):
        """Test getting non-existent deployment."""
        state = manager.get_deployment("nonexistent")
        assert state is None

    def test_list_deployments_empty(self, manager: DeploymentManager):
        """Test listing deployments when none exist."""
        deployments = manager.list_deployments()
        assert deployments == []

    @pytest.mark.asyncio
    async def test_deploy_creates_state(
        self, manager: DeploymentManager, mock_backend: MagicMock, sample_plan: PlacementPlan
    ):
        """Test that deploy creates a deployment state."""
        # Mock start_shard to return a ready handle
        mock_backend.start_shard.return_value = ShardHandle(
            shard_id="shard-0",
            worker_id="worker-1",
            status=ShardStatus.READY,
            data_address="0.0.0.0:50100",
        )

        state = await manager.deploy(sample_plan, wait_for_ready=False)

        assert state.service_name == "test-service"
        assert state.status in ["deploying", "ready"]
        assert len(state.shards) == 2

    @pytest.mark.asyncio
    async def test_stop_deployment(
        self, manager: DeploymentManager, mock_backend: MagicMock, sample_plan: PlacementPlan
    ):
        """Test stopping a deployment."""
        # First deploy
        mock_backend.start_shard.return_value = ShardHandle(
            shard_id="shard-0",
            worker_id="worker-1",
            status=ShardStatus.READY,
        )
        await manager.deploy(sample_plan, wait_for_ready=False)

        # Then stop
        success = await manager.stop("test-service")
        assert success is True

    @pytest.mark.asyncio
    async def test_stop_nonexistent(self, manager: DeploymentManager):
        """Test stopping a non-existent deployment."""
        success = await manager.stop("nonexistent")
        assert success is False


# ---------------------------------------------------------------------------
# Test DeploymentState
# ---------------------------------------------------------------------------


class TestDeploymentState:
    """Test DeploymentState dataclass."""

    def test_state_creation(self):
        """Test DeploymentState creation."""
        state = DeploymentState(
            service_name="test-service",
            status="deploying",
            message="Starting shards...",
        )

        assert state.service_name == "test-service"
        assert state.status == "deploying"
        assert state.shards == {}

    def test_state_defaults(self):
        """Test DeploymentState defaults."""
        state = DeploymentState(service_name="test")
        assert state.plan is None
        assert state.shards == {}
        assert state.status == "idle"
        assert state.message == ""
