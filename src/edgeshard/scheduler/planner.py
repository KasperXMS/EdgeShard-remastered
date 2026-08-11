"""Core scheduling logic — pure planner, no side effects.

The Scheduler is a pure function:
    (ModelSpec, ClusterSnapshot, ProfileSnapshot, SchedulingPolicy) -> PlacementPlan

It may not start containers, mutate Workers, or perform I/O.
"""

from __future__ import annotations

from edgeshard.common.config import ModelSpec, ServiceSpec
from edgeshard.common.errors import SchedulerError
from edgeshard.common.identifiers import ServiceName, WorkerId
from edgeshard.common.logging import get_logger
from edgeshard.scheduler.model_info import load_model_info
from edgeshard.scheduler.placement import PlacementPlan, ShardPlacement
from edgeshard.scheduler.policies import PartitionResult
from edgeshard.scheduler.policies.default import DefaultPolicy
from edgeshard.scheduler.snapshot import (
    ClusterSnapshot,
    ProfileSnapshot,
    SchedulingPolicy,
)

logger = get_logger(__name__)


def schedule(
    model: ModelSpec,
    cluster: ClusterSnapshot,
    profiles: ProfileSnapshot,
    policy: SchedulingPolicy,
    service_name: str = "default",
    max_seq_len: int = 512,
) -> PlacementPlan:
    """Plan model sharding and placement across the cluster.

    This is the core scheduling entry point. It consumes immutable
    snapshots and produces an immutable PlacementPlan.

    Args:
        model: Which model to serve.
        cluster: Current cluster resources.
        profiles: Available profiling data.
        policy: Scheduling policy configuration.
        service_name: Name for the service (used in PlacementPlan).
        max_seq_len: Maximum sequence length for KV cache estimation.

    Returns:
        Immutable PlacementPlan.

    Raises:
        SchedulerError: If no valid placement exists.
    """
    if not cluster.workers:
        raise SchedulerError("Cluster has no workers. Cannot schedule.")

    logger.info(
        f"Scheduling model '{model.name}' across "
        f"{len(cluster.workers)} workers, policy='{policy.name}'"
    )

    # 1. Load model info (num_layers, hidden_size, etc.)
    # Try to find a matching profile for model info
    matching_profile = _find_matching_profile(profiles, model)

    model_info = load_model_info(
        model_path=model.name,
        profile=matching_profile,
        dtype_bytes=_dtype_to_bytes(model.dtype),
    )

    # Tag model_name into model_info for profile matching
    model_info.__dict__["_model_name"] = model.name

    logger.info(
        f"Model info: {model_info.num_layers} layers, "
        f"hidden_size={model_info.hidden_size}, "
        f"per_layer_mem={model_info.per_layer_memory_mb:.1f} MB, "
        f"kv_cache/token={model_info.kv_cache_per_token_mb:.4f} MB"
    )

    # 2. Select and run policy
    policy_impl = _get_policy(policy.name)
    result = policy_impl.solve(
        model_info=model_info,
        cluster=cluster,
        profiles=profiles,
        policy_config=policy,
        max_seq_len=max_seq_len,
    )

    logger.info(
        f"Placement plan: {len(result.assignments)} shards, "
        f"est. latency={result.estimated_latency_ms:.1f} ms, "
        f"est. throughput={result.estimated_throughput_tps:.1f} tok/s"
    )

    # 3. Convert PartitionResult → PlacementPlan
    shards = _build_shard_placements(result, service_name)

    # 4. Build ServiceSpec for the plan
    service_spec = ServiceSpec(
        name=service_name,
        model=model,
    )

    return PlacementPlan(
        service_name=ServiceName(service_name),
        service_spec=service_spec,
        shards=shards,
        version=1,
    )


def _find_matching_profile(
    profiles: ProfileSnapshot, model: ModelSpec
) -> any:
    """Find the best matching profile for a model."""
    # Try exact match on model_name
    for entry in profiles.entries:
        if entry.model_name == model.name and entry.dtype == model.dtype:
            return entry

    # Try partial match (model name contains)
    for entry in profiles.entries:
        if model.name in entry.model_name or entry.model_name in model.name:
            return entry

    # Return first profile for the model (any device)
    for entry in profiles.entries:
        if entry.model_name == model.name:
            return entry

    return None


def _get_policy(name: str) -> DefaultPolicy:
    """Get the policy implementation by name.

    Currently only the default policy is implemented. The default policy
    handles all scheduling modes (latency-first, memory-balanced, etc.)
    via its internal mode selection.

    Future: latency-first and memory-balanced could have separate
    implementations optimized for their specific objectives.
    """
    return DefaultPolicy()


def _dtype_to_bytes(dtype: str) -> int:
    """Convert dtype string to bytes per parameter."""
    dtype_map = {
        "float16": 2,
        "fp16": 2,
        "bfloat16": 2,
        "bf16": 2,
        "float32": 4,
        "fp32": 4,
        "int8": 1,
        "int4": 1,  # Approximate
    }
    return dtype_map.get(dtype.lower(), 2)


def _build_shard_placements(
    result: PartitionResult,
    service_name: str,  # noqa: ARG001 — reserved for future use
) -> list[ShardPlacement]:
    """Convert PartitionResult assignments to ShardPlacement list."""
    shards: list[ShardPlacement] = []

    for idx, (layer_start, layer_end, device) in enumerate(result.assignments):
        shards.append(
            ShardPlacement(
                shard_index=idx,
                worker_id=WorkerId(device.worker_id),
                layer_start=layer_start,
                layer_end=layer_end,
                device=device.device_id,
            )
        )

    return shards
