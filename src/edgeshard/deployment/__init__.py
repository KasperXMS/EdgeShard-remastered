"""Deployment module — shard lifecycle management.

The deployment module is responsible for starting and stopping shard
processes on Worker nodes according to a PlacementPlan.
"""

from edgeshard.deployment.backend import DeploymentBackend, ShardHandle, ShardStatus
from edgeshard.deployment.manager import DeploymentManager

__all__ = [
    "DeploymentBackend",
    "ShardHandle",
    "ShardStatus",
    "DeploymentManager",
]
