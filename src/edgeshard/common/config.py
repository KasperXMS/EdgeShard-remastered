"""Configuration schemas for Master, Worker, and Service.

These Pydantic models validate YAML configuration files and provide
sensible defaults. They are the single source of truth for config shape.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

class MasterGrpcConfig(BaseModel):
    """gRPC control-plane listener for the Master."""

    host: str = "0.0.0.0"
    port: int = 10500
    max_workers: int = 8


class MasterApiConfig(BaseModel):
    """REST API listener (FastAPI) for CLI / external clients."""

    host: str = "0.0.0.0"
    port: int = 10501


class MasterStateConfig(BaseModel):
    """State store configuration."""

    path: Path = Path("edgeshard_state/master.db")


class MasterConfig(BaseModel):
    """Top-level Master configuration."""

    grpc: MasterGrpcConfig = Field(default_factory=MasterGrpcConfig)
    api: MasterApiConfig = Field(default_factory=MasterApiConfig)
    state: MasterStateConfig = Field(default_factory=MasterStateConfig)
    log_level: str = "INFO"
    log_json: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> MasterConfig:
        raw = yaml.safe_load(path.read_text())
        return cls.model_validate(raw or {})


# ---------------------------------------------------------------------------
# Worker config
# ---------------------------------------------------------------------------

class WorkerRegistrationConfig(BaseModel):
    """How the Worker connects to the Master."""

    master_address: str = "localhost:10500"
    reconnect_interval_seconds: float = 5.0
    heartbeat_interval_seconds: float = 10.0


class WorkerModelCacheConfig(BaseModel):
    """Local model weight cache."""

    cache_dir: Path = Path("models_cache")
    max_size_gb: float | None = None


class WorkerConfig(BaseModel):
    """Top-level Worker configuration."""

    registration: WorkerRegistrationConfig = Field(
        default_factory=WorkerRegistrationConfig
    )
    model_cache: WorkerModelCacheConfig = Field(
        default_factory=WorkerModelCacheConfig
    )
    log_level: str = "INFO"
    log_json: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> WorkerConfig:
        raw = yaml.safe_load(path.read_text())
        return cls.model_validate(raw or {})


# ---------------------------------------------------------------------------
# Service config (user-facing YAML for deployment)
# ---------------------------------------------------------------------------

class ModelSpec(BaseModel):
    """Which model to serve."""

    name: str = Field(description="Hugging Face model ID or local path")
    revision: str = "main"
    dtype: str = "float16"


class RuntimeSpec(BaseModel):
    """Runtime backend settings."""

    backend: str = "torch"
    max_batch_size: int = 1
    max_sequence_length: int = 4096


class SchedulingPolicyName(str, Enum):
    """Available scheduling policies."""

    DEFAULT = "default"
    LATENCY_FIRST = "latency-first"
    MEMORY_BALANCED = "memory-balanced"


class SchedulingSpec(BaseModel):
    """Scheduling hints (not hard constraints)."""

    policy: SchedulingPolicyName = SchedulingPolicyName.DEFAULT
    hints: dict[str, Any] = Field(default_factory=dict)


class DeploymentSpec(BaseModel):
    """Deployment backend settings."""

    backend: str = "docker"
    shard_image: str = "edgeshard/shard:latest"
    resource_limits: dict[str, Any] = Field(default_factory=dict)


class ServiceSpec(BaseModel):
    """User-facing service specification.

    This expresses *what* to serve, not *how* to shard it.
    The Scheduler fills in the layer→worker mapping.
    """

    name: str = Field(description="Unique service name")
    model: ModelSpec
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    scheduling: SchedulingSpec = Field(default_factory=SchedulingSpec)
    deployment: DeploymentSpec = Field(default_factory=DeploymentSpec)

    @classmethod
    def from_yaml(cls, path: Path) -> ServiceSpec:
        raw = yaml.safe_load(path.read_text())
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file as a dict."""
    return yaml.safe_load(path.read_text()) or {}
