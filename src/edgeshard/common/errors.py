"""EdgeShard exception hierarchy."""


class EdgeShardError(Exception):
    """Base exception for all EdgeShard errors."""


class ConfigError(EdgeShardError):
    """Invalid or missing configuration."""


class WorkerError(EdgeShardError):
    """Worker-level failure (registration, heartbeat, deployment)."""


class ShardError(EdgeShardError):
    """Shard runtime failure (model loading, forward, KV cache)."""


class SchedulerError(EdgeShardError):
    """Scheduling or placement planning failure."""


class ProfileError(EdgeShardError):
    """Profiling execution or result retrieval failure."""


class TransportError(EdgeShardError):
    """Tensor data-plane transport failure."""


class DeploymentError(EdgeShardError):
    """Container or deployment backend failure."""


class SessionError(EdgeShardError):
    """Inference session lifecycle failure."""
