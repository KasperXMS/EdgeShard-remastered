"""Strongly-typed identifiers for EdgeShard entities."""

from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerId:
    """Unique identifier for a Worker node."""

    value: str

    @classmethod
    def generate(cls) -> WorkerId:
        return cls(value=f"w-{uuid.uuid4().hex[:12]}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ShardId:
    """Unique identifier for a deployed model Shard."""

    service_name: str
    shard_index: int

    def __str__(self) -> str:
        return f"{self.service_name}/shard-{self.shard_index}"


@dataclass(frozen=True)
class SessionId:
    """Unique identifier for an inference Session."""

    value: str

    @classmethod
    def generate(cls) -> SessionId:
        return cls(value=f"s-{uuid.uuid4().hex[:16]}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ServiceName:
    """Name of a deployed inference service."""

    value: str

    def __str__(self) -> str:
        return self.value
