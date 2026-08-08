"""EdgeShard REST API schemas (FastAPI request/response models)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class NodeInfo(BaseModel):
    """Information about a registered Worker node."""

    worker_id: str
    hostname: str
    gpu_devices: list[str] = Field(default_factory=list)
    total_memory_gb: float | None = None
    status: str = "unknown"


class NodeListResponse(BaseModel):
    """Response for listing all registered Workers."""

    nodes: list[NodeInfo] = Field(default_factory=list)


class ServiceStatus(BaseModel):
    """Status of a deployed service."""

    name: str
    model: str
    status: str = "unknown"
    shard_count: int = 0
