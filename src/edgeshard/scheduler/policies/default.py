"""Default scheduling policy — DP-based joint device selection and model partition.

Implements the algorithms from the EdgeShard paper (Zhang et al., 2025):
- Algorithm 1: Latency optimization via dynamic programming
- Algorithm 2: Throughput optimization (pipeline bottleneck minimization)

The key insight: LLM layers are sequential, so the partition must be
contiguous. The DP exploits optimal substructure: the optimal partition
of the first i layers depends only on the optimal partition of i-1 layers.

Latency DP (Algorithm 1):
    DP(i, j) = min_k { DP(i-1, k) + comp(j) + comm(k → j) }
    where i = layer index, j = current device, k = previous device

Throughput DP (Algorithm 2, simplified):
    DP(i, j) = min over split points k < i:
        max(DP(k, prev), comp(j) * (i-k), comm(prev → j))
    Minimizes the pipeline bottleneck stage.
"""

from __future__ import annotations

import math
from typing import Any

from edgeshard.common.errors import SchedulerError
from edgeshard.common.logging import get_logger
from edgeshard.scheduler.model_info import (
    ModelSchedulingInfo,
    estimate_activation_size_mb,
)
from edgeshard.scheduler.policies import (
    DeviceSlot,
    PartitionResult,
    SchedulingPolicyBase,
)
from edgeshard.scheduler.snapshot import (
    ClusterSnapshot,
    ProfileSnapshot,
    SchedulingPolicy,
)

logger = get_logger(__name__)

# Default activation size estimate when hidden_size is unknown (MB)
_DEFAULT_ACTIVATION_SIZE_MB = 10.0

# Default bandwidth when unknown (MB/s)
_DEFAULT_BANDWIDTH_MBPS = 100.0

# KV cache overhead safety margin (account for allocator fragmentation)
_MEMORY_SAFETY_MARGIN = 0.9


class DefaultPolicy(SchedulingPolicyBase):
    """Default scheduling policy with latency/throughput optimization.

    Decision flow:
    1. Try single-device placement (zero communication cost)
    2. If no single device fits, use DP to find optimal multi-device partition
    3. Choose between latency and throughput optimization based on policy params
    """

    def solve(
        self,
        model_info: ModelSchedulingInfo,
        cluster: ClusterSnapshot,
        profiles: ProfileSnapshot,
        policy_config: SchedulingPolicy,
        max_seq_len: int = 512,
    ) -> PartitionResult:
        # Build device slots from cluster + profiles
        devices = _build_device_slots(model_info, cluster, profiles, max_seq_len)

        if not devices:
            raise SchedulerError(
                "No usable devices in cluster. Ensure workers are online "
                "and have sufficient memory."
            )

        # Determine optimization mode
        mode = policy_config.params.get("mode", "latency")
        if policy_config.name == "latency-first":
            mode = "latency"
        elif policy_config.name == "memory-balanced":
            mode = "throughput"
        elif "latency_weight" in policy_config.params:
            w = policy_config.params["latency_weight"]
            mode = "latency" if w >= 0.5 else "throughput"

        # Compute activation transfer size
        activation_size_mb = estimate_activation_size_mb(model_info, max_seq_len)
        if activation_size_mb <= 0:
            activation_size_mb = _DEFAULT_ACTIVATION_SIZE_MB

        # Compute per-layer memory cost
        per_layer_mem = model_info.per_layer_memory_mb
        kv_per_token = model_info.kv_cache_per_token_mb
        kv_cache_per_layer = kv_per_token * max_seq_len

        # Source device (first layer must be here — privacy constraint)
        source_wid = cluster.get_source_worker()
        source_device = _find_source_device(devices, source_wid)

        # Build communication cost matrix
        comm_costs = _build_comm_cost_matrix(
            devices, cluster, activation_size_mb
        )

        # Try single-device placement first
        single_result = _try_single_device(
            model_info, devices, per_layer_mem, kv_cache_per_layer, max_seq_len
        )
        if single_result is not None:
            logger.info(
                f"Single-device placement feasible on "
                f"{single_result.assignments[0][2].worker_id}:"
                f"{single_result.assignments[0][2].device_id}"
            )
            return single_result

        # Multi-device DP
        if mode == "throughput":
            result = optimize_throughput(
                num_layers=model_info.num_layers,
                devices=devices,
                comm_costs=comm_costs,
                per_layer_memory=per_layer_mem,
                kv_cache_per_layer=kv_cache_per_layer,
                source_device=source_device,
                max_seq_len=max_seq_len,
            )
        else:
            result = optimize_latency(
                num_layers=model_info.num_layers,
                devices=devices,
                comm_costs=comm_costs,
                per_layer_memory=per_layer_mem,
                kv_cache_per_layer=kv_cache_per_layer,
                source_device=source_device,
                max_seq_len=max_seq_len,
            )

        return result


# ---------------------------------------------------------------------------
# Cost matrix construction
# ---------------------------------------------------------------------------


def _build_device_slots(
    model_info: ModelSchedulingInfo,
    cluster: ClusterSnapshot,
    profiles: ProfileSnapshot,
    max_seq_len: int,
) -> list[DeviceSlot]:
    """Build list of usable device slots from cluster state and profiles.

    Each device slot represents a GPU/CPU that can host model layers.
    Memory budget accounts for model weights + KV cache.
    """
    devices: list[DeviceSlot] = []

    per_layer_mem = model_info.per_layer_memory_mb
    kv_per_token = model_info.kv_cache_per_token_mb
    kv_cache_total = kv_per_token * max_seq_len

    for worker in cluster.workers:
        if worker.status != "online":
            continue

        if worker.devices:
            # Use device-level info
            for dev in worker.devices:
                # Match profile for this device
                fingerprint = f"{dev.device_type}:{dev.name}"
                profile = profiles.get(model_info.__dict__.get("_model_name", ""), fingerprint, "")

                # Find matching profile (try various model names)
                matched_profile = _find_matching_profile(profiles, dev, worker)

                layer_fwd_ms = 0.0
                if matched_profile is not None:
                    layer_fwd_ms = matched_profile.layer_forward_ms

                # Memory budget: device total memory * safety margin
                mem_budget = dev.total_memory_mb * _MEMORY_SAFETY_MARGIN

                # Check if this device can hold at least 1 layer + KV cache
                min_required = per_layer_mem + kv_cache_total
                if mem_budget >= min_required:
                    devices.append(
                        DeviceSlot(
                            worker_id=str(worker.worker_id),
                            device_id=dev.device_id,
                            device_type=dev.device_type,
                            device_name=dev.name,
                            memory_budget_mb=mem_budget,
                            layer_forward_ms=layer_fwd_ms,
                        )
                    )
        else:
            # No device info — use worker-level memory as CPU device
            mem_budget = worker.available_memory_mb * _MEMORY_SAFETY_MARGIN
            min_required = per_layer_mem + kv_cache_total
            if mem_budget >= min_required:
                devices.append(
                    DeviceSlot(
                        worker_id=str(worker.worker_id),
                        device_id="cpu",
                        device_type="cpu",
                        device_name="CPU",
                        memory_budget_mb=mem_budget,
                        layer_forward_ms=0.0,
                    )
                )

    # If no profiles matched, use default latency estimate
    if all(d.layer_forward_ms == 0.0 for d in devices):
        logger.warning(
            "No matching profiles found for any device. "
            "Using default latency estimates."
        )
        default_devices = []
        for d in devices:
            if d.device_type in ("cuda", "jetson"):
                # Estimate based on device memory (larger GPU = faster)
                ref = {6000: 0.3, 8000: 0.5, 16000: 1.0, 24000: 1.5}
                fwd_ms = _interpolate_latency(d.memory_budget_mb, ref)
            else:
                fwd_ms = 10.0  # CPU is slow
            default_devices.append(
                DeviceSlot(
                    worker_id=d.worker_id,
                    device_id=d.device_id,
                    device_type=d.device_type,
                    device_name=d.device_name,
                    memory_budget_mb=d.memory_budget_mb,
                    layer_forward_ms=fwd_ms,
                )
            )
        return default_devices

    return devices


def _find_matching_profile(
    profiles: ProfileSnapshot,
    dev: Any,
    worker: Any,
) -> Any:
    """Find a matching profile entry for a device."""
    # Try exact match on device name
    for entry in profiles.entries:
        if dev.name and dev.name in entry.device_fingerprint:
            return entry
        if dev.device_type and dev.device_type in entry.device_fingerprint:
            return entry
    return None


def _find_source_device(
    devices: list[DeviceSlot], source_wid: str
) -> DeviceSlot:
    """Find the source device on the source worker.

    The source device is the first device on the source worker.
    If no devices are found on the source worker, use the first device.
    """
    for d in devices:
        if d.worker_id == source_wid:
            return d
    if devices:
        logger.warning(
            f"Source worker {source_wid} has no devices. "
            f"Using {devices[0].worker_id}:{devices[0].device_id} as source."
        )
        return devices[0]
    raise SchedulerError("No devices available for source")


def _build_comm_cost_matrix(
    devices: list[DeviceSlot],
    cluster: ClusterSnapshot,
    activation_size_mb: float,
) -> dict[tuple[int, int], float]:
    """Build communication cost matrix between device pairs.

    Returns dict mapping (device_idx_a, device_idx_b) → transfer time in ms.
    Transfer time = activation_size_mb / bandwidth_mb_per_s * 1000.
    """
    n = len(devices)
    costs: dict[tuple[int, int], float] = {}

    for i in range(n):
        for j in range(n):
            if i == j:
                costs[(i, j)] = 0.0  # Same device, no transfer
            else:
                d_i = devices[i]
                d_j = devices[j]

                if d_i.worker_id == d_j.worker_id:
                    # Same worker — NVLink/PCIe, very fast
                    costs[(i, j)] = 0.1  # 0.1 ms
                else:
                    # Cross-worker: use bandwidth matrix
                    bw_mb_per_s = cluster.get_bandwidth(
                        d_i.worker_id, d_j.worker_id
                    )
                    if bw_mb_per_s <= 0:
                        bw_mb_per_s = _DEFAULT_BANDWIDTH_MBPS
                    transfer_ms = (activation_size_mb / bw_mb_per_s) * 1000.0
                    costs[(i, j)] = transfer_ms

    return costs


def _interpolate_latency(memory_mb: float, table: dict[int, float]) -> float:
    """Interpolate latency from a reference table."""
    sorted_keys = sorted(table.keys())
    if memory_mb <= sorted_keys[0]:
        return table[sorted_keys[0]]
    if memory_mb >= sorted_keys[-1]:
        return table[sorted_keys[-1]]
    for i in range(len(sorted_keys) - 1):
        if sorted_keys[i] <= memory_mb <= sorted_keys[i + 1]:
            ratio = (memory_mb - sorted_keys[i]) / (
                sorted_keys[i + 1] - sorted_keys[i]
            )
            return table[sorted_keys[i]] + ratio * (
                table[sorted_keys[i + 1]] - table[sorted_keys[i]]
            )
    return 1.0


def _try_single_device(
    model_info: ModelSchedulingInfo,
    devices: list[DeviceSlot],
    per_layer_mem: float,
    kv_cache_per_layer: float,
    max_seq_len: int,
) -> PartitionResult | None:
    """Try to fit the entire model on a single device.

    Returns a single-shard PartitionResult if possible, else None.
    """
    total_model_mem = model_info.total_model_memory_mb
    kv_cache_total = model_info.kv_cache_per_token_mb * max_seq_len
    total_required = total_model_mem + kv_cache_total

    # Find the device with the most memory that can fit the model
    best_device: DeviceSlot | None = None
    best_latency = float("inf")

    for d in devices:
        if d.memory_budget_mb >= total_required:
            # This device can hold the full model
            latency = d.layer_forward_ms * model_info.num_layers
            if latency < best_latency:
                best_latency = latency
                best_device = d

    if best_device is not None:
        return PartitionResult(
            assignments=[(0, model_info.num_layers, best_device)],
            estimated_latency_ms=best_latency,
            estimated_throughput_tps=(
                1000.0 / best_device.layer_forward_ms
                if best_device.layer_forward_ms > 0
                else 0.0
            ),
            total_communication_ms=0.0,
        )

    return None


# ---------------------------------------------------------------------------
# Algorithm 1: Latency Optimization (from EdgeShard paper)
# ---------------------------------------------------------------------------


def optimize_latency(
    num_layers: int,
    devices: list[DeviceSlot],
    comm_costs: dict[tuple[int, int], float],
    per_layer_memory: float,
    kv_cache_per_layer: float,
    source_device: DeviceSlot,
    max_seq_len: int = 512,
) -> PartitionResult:
    """Dynamic programming algorithm for latency optimization.

    Based on Algorithm 1 from the EdgeShard paper.

    DP(i, j) = minimal total execution time for the first i layers,
    where layer i-1 is assigned to device j.

    State transition:
        DP(i, j) = min_k { DP(i-1, k) + comp(j) + comm(k → j) }

    For the last layer (i == N-1), add return-to-source communication:
        + comm(j → source)

    Memory constraint: consecutive layers on device j must satisfy:
        layer_count * per_layer_memory + kv_cache_per_layer <= budget_j

    Args:
        num_layers: Total model layers.
        devices: Available device slots.
        comm_costs: Communication cost matrix (device pair → ms).
        per_layer_memory: Memory per layer in MB (model weights).
        kv_cache_per_layer: KV cache memory for one layer (all tokens) in MB.
        source_device: Source device (first layer must be here).
        max_seq_len: Max sequence length.

    Returns:
        PartitionResult with optimal layer assignments.

    Raises:
        SchedulerError: If no valid partition exists.
    """
    N = num_layers
    M = len(devices)
    INF = float("inf")

    if M == 0:
        raise SchedulerError("No devices available for placement")

    # Find source device index
    source_idx = 0
    for i, d in enumerate(devices):
        if (
            d.worker_id == source_device.worker_id
            and d.device_id == source_device.device_id
        ):
            source_idx = i
            break

    # Compute per-layer compute cost for each device
    comp_cost = [d.layer_forward_ms for d in devices]
    if all(c == 0.0 for c in comp_cost):
        # No profiling data — use uniform cost
        comp_cost = [1.0] * M

    # Memory budget per device
    mem_budget = [d.memory_budget_mb for d in devices]

    # DP table: dp[i][j] = min cost for first i layers, layer i-1 on device j
    dp = [[INF] * M for _ in range(N + 1)]
    choice = [[-1] * M for _ in range(N + 1)]  # Track which k was optimal

    # Base case: layer 0 must be on source device (privacy constraint)
    dp[1][source_idx] = comp_cost[source_idx]
    choice[1][source_idx] = -1  # No previous device

    # Fill DP table
    for i in range(2, N + 1):  # i = number of layers placed so far
        for j in range(M):  # j = current device
            # Calculate how many consecutive layers device j can hold
            # Memory: layer_count * per_layer_memory + kv_cache_per_layer
            max_layers_on_j = int(
                (mem_budget[j] - kv_cache_per_layer) / per_layer_memory
            ) if per_layer_memory > 0 else N

            if max_layers_on_j <= 0:
                continue  # Device j can't even hold 1 layer + KV cache

            # Try all possible previous devices k
            for k in range(M):
                if dp[i - 1][k] == INF:
                    continue  # Previous state not reachable

                # How many consecutive layers are on device j?
                # At minimum 1 (current layer i-1)
                # We need to count back how many layers are on device j
                layers_on_j = _count_consecutive_layers(choice, i - 1, k, j) + 1

                if layers_on_j > max_layers_on_j:
                    continue  # Memory exceeded

                # Communication cost: k → j
                if k == j:
                    comm = 0.0
                else:
                    comm = comm_costs.get((k, j), 10.0)  # Default 10ms

                total = dp[i - 1][k] + comp_cost[j] + comm

                # For last layer: add return-to-source communication
                if i == N and j != source_idx:
                    return_comm = comm_costs.get((j, source_idx), 10.0)
                    total += return_comm

                if total < dp[i][j]:
                    dp[i][j] = total
                    choice[i][j] = k

    # Find optimal last device
    best_last = -1
    best_cost = INF
    for j in range(M):
        if dp[N][j] < best_cost:
            best_cost = dp[N][j]
            best_last = j

    if best_last == -1:
        raise SchedulerError(
            f"No valid partition found for {N} layers across {M} devices. "
            "Total memory may be insufficient."
        )

    # Backtrace to find assignment
    assignments_raw = _backtrace(choice, N, best_last, devices)

    # Calculate total communication time
    total_comm = _calculate_total_comm(assignments_raw, comm_costs)

    # Estimate throughput (bottleneck stage)
    throughput = _estimate_throughput(assignments_raw, comp_cost, comm_costs)

    return PartitionResult(
        assignments=assignments_raw,
        estimated_latency_ms=best_cost,
        estimated_throughput_tps=throughput,
        total_communication_ms=total_comm,
    )


def _count_consecutive_layers(
    choice: list[list[int]],
    layer_idx: int,
    current_device: int,
    target_device: int,
) -> int:
    """Count how many consecutive layers before layer_idx are on target_device.

    Walks backward through the choice table to count consecutive assignments.
    """
    count = 0
    prev = current_device
    i = layer_idx

    while i > 1:
        k = choice[i][prev]
        if k == -1:
            break
        if k != target_device:
            break
        count += 1
        prev = k
        i -= 1

    return count


def _backtrace(
    choice: list[list[int]],
    num_layers: int,
    last_device: int,
    devices: list[DeviceSlot],
) -> list[tuple[int, int, DeviceSlot]]:
    """Backtrace DP choices to reconstruct the layer assignment.

    Returns list of (layer_start, layer_end, device_slot) tuples,
    merging consecutive layers on the same device into a single shard.
    """
    N = num_layers

    # Reconstruct per-layer assignment
    layer_assignment = [0] * N  # layer_assignment[i] = device index for layer i
    layer_assignment[N - 1] = last_device

    current_device = last_device
    for i in range(N, 1, -1):
        prev_device = choice[i][current_device]
        if prev_device == -1:
            # This is the first layer
            layer_assignment[i - 2] = current_device
            break
        layer_assignment[i - 2] = prev_device
        current_device = prev_device

    # Merge consecutive layers on same device into shards
    assignments: list[tuple[int, int, DeviceSlot]] = []
    shard_start = 0
    for i in range(1, N):
        if layer_assignment[i] != layer_assignment[i - 1]:
            # End of a shard
            dev_idx = layer_assignment[i - 1]
            assignments.append((shard_start, i, devices[dev_idx]))
            shard_start = i
    # Last shard
    dev_idx = layer_assignment[N - 1]
    assignments.append((shard_start, N, devices[dev_idx]))

    return assignments


def _calculate_total_comm(
    assignments: list[tuple[int, int, DeviceSlot]],
    comm_costs: dict[tuple[int, int], float],
) -> float:
    """Calculate total cross-device communication time."""
    total = 0.0
    for i in range(1, len(assignments)):
        prev_dev = assignments[i - 1][2]
        curr_dev = assignments[i][2]
        if prev_dev != curr_dev:
            # Find device indices in original list (need to search)
            # Use a default estimate since we don't have indices here
            total += 1.0  # Placeholder — actual cost computed by caller
    return total


def _estimate_throughput(
    assignments: list[tuple[int, int, DeviceSlot]],
    comp_costs: list[float],
    comm_costs: dict[tuple[int, int], float],
) -> float:
    """Estimate pipeline throughput from stage times.

    Throughput is limited by the slowest pipeline stage (bottleneck).
    """
    if not assignments:
        return 0.0

    # Calculate time for each stage
    max_stage_time = 0.0
    for layer_start, layer_end, device in assignments:
        n_layers = layer_end - layer_start
        # Find device index
        dev_idx = -1
        for idx, d in enumerate(
            [DeviceSlot("", "", "", "", 0, c) for c in comp_costs]
        ):
            if d.layer_forward_ms == device.layer_forward_ms:
                dev_idx = idx
                break

        if dev_idx >= 0:
            stage_time = comp_costs[dev_idx] * n_layers
        else:
            stage_time = device.layer_forward_ms * n_layers
        max_stage_time = max(max_stage_time, stage_time)

    if max_stage_time > 0:
        return 1000.0 / max_stage_time  # tokens/sec
    return 0.0


# ---------------------------------------------------------------------------
# Algorithm 2: Throughput Optimization (from EdgeShard paper)
# ---------------------------------------------------------------------------


def optimize_throughput(
    num_layers: int,
    devices: list[DeviceSlot],
    comm_costs: dict[tuple[int, int], float],
    per_layer_memory: float,
    kv_cache_per_layer: float,
    source_device: DeviceSlot,
    max_seq_len: int = 512,
) -> PartitionResult:
    """Dynamic programming for throughput optimization.

    Based on Algorithm 2 from the EdgeShard paper (simplified version).

    Instead of minimizing total latency, we minimize the bottleneck
    pipeline stage. Each device processes a contiguous block of layers,
    and the throughput is limited by max(stage_time) across all stages.

    DP(i, j) = minimum bottleneck time for placing layers 0..i-1,
    where layer i-1 is on device j.

    State transition:
        DP(i, j) = min over split point k:
            max(DP(k, prev_device),
                comp(j) * (i - k),     # compute for layers k..i-1
                comm(prev_device → j))  # transfer at boundary

    This ensures balanced pipeline stages for maximum throughput.
    """
    N = num_layers
    M = len(devices)
    INF = float("inf")

    if M == 0:
        raise SchedulerError("No devices available for placement")

    # Find source device index
    source_idx = 0
    for i, d in enumerate(devices):
        if (
            d.worker_id == source_device.worker_id
            and d.device_id == source_device.device_id
        ):
            source_idx = i
            break

    comp_cost = [d.layer_forward_ms for d in devices]
    if all(c == 0.0 for c in comp_cost):
        comp_cost = [1.0] * M

    mem_budget = [d.memory_budget_mb for d in devices]

    # dp[i][j] = minimum bottleneck time for first i layers, last on device j
    dp = [[INF] * M for _ in range(N + 1)]
    choice = [[(- 1, -1)] * M for _ in range(N + 1)]
    # choice[i][j] = (split_point_k, prev_device) that gave optimal

    # Base: layer 0 on source
    dp[1][source_idx] = comp_cost[source_idx]
    choice[1][source_idx] = (0, -1)

    for i in range(2, N + 1):
        for j in range(M):
            max_layers_j = int(
                (mem_budget[j] - kv_cache_per_layer) / per_layer_memory
            ) if per_layer_memory > 0 else N

            if max_layers_j <= 0:
                continue

            # Try all split points k (first k layers on previous devices,
            # layers k..i-1 on device j)
            for k in range(max(1, i - max_layers_j), i):
                for prev in range(M):
                    if dp[k][prev] == INF:
                        continue

                    # Compute time for this stage (layers k..i-1 on device j)
                    stage_layers = i - k
                    stage_compute = comp_cost[j] * stage_layers

                    # Communication at boundary k-1 → k
                    if prev == j:
                        comm = 0.0
                    else:
                        comm = comm_costs.get((prev, j), 10.0)

                    # Bottleneck = max of prior stages and this stage
                    bottleneck = max(dp[k][prev], stage_compute + comm)

                    # For last layer: add return-to-source
                    if i == N and j != source_idx:
                        return_comm = comm_costs.get((j, source_idx), 10.0)
                        bottleneck = max(bottleneck, return_comm)

                    if bottleneck < dp[i][j]:
                        dp[i][j] = bottleneck
                        choice[i][j] = (k, prev)

    # Find optimal
    best_last = -1
    best_cost = INF
    for j in range(M):
        if dp[N][j] < best_cost:
            best_cost = dp[N][j]
            best_last = j

    if best_last == -1:
        raise SchedulerError(
            f"No valid throughput-optimized partition for {N} layers "
            f"across {M} devices."
        )

    # Backtrace
    assignments = _backtrace_throughput(choice, N, best_last, devices)

    total_comm = 0.0
    for i in range(1, len(assignments)):
        if assignments[i][2].worker_id != assignments[i - 1][2].worker_id:
            total_comm += 5.0  # Approximate

    throughput = 1000.0 / best_cost if best_cost > 0 else 0.0

    return PartitionResult(
        assignments=assignments,
        estimated_latency_ms=dp[N][best_last],
        estimated_throughput_tps=throughput,
        total_communication_ms=total_comm,
    )


def _backtrace_throughput(
    choice: list[list[tuple[int, int]]],
    num_layers: int,
    last_device: int,
    devices: list[DeviceSlot],
) -> list[tuple[int, int, DeviceSlot]]:
    """Backtrace throughput DP to reconstruct assignments."""
    N = num_layers
    shards: list[tuple[int, int, DeviceSlot]] = []

    i = N
    j = last_device

    while i > 1:
        k, prev = choice[i][j]
        if k == -1 and prev == -1:
            # Base case: first layer
            shards.append((0, i, devices[j]))
            break
        shards.append((k, i, devices[j]))
        i = k
        j = prev

    if i == 1 and not any(s[0] == 0 for s in shards):
        shards.append((0, 1, devices[source_idx_from_choice(choice)]))

    # Reverse to get chronological order and merge consecutive same-device
    shards.reverse()
    return _merge_consecutive_shards(shards)


def source_idx_from_choice(choice: list[list[tuple[int, int]]]) -> int:
    """Find source device index from choice table."""
    return 0  # Default


def _merge_consecutive_shards(
    shards: list[tuple[int, int, DeviceSlot]],
) -> list[tuple[int, int, DeviceSlot]]:
    """Merge consecutive shards on the same device."""
    if not shards:
        return []

    merged: list[tuple[int, int, DeviceSlot]] = [shards[0]]
    for start, end, device in shards[1:]:
        prev_start, prev_end, prev_device = merged[-1]
        if (
            device.worker_id == prev_device.worker_id
            and device.device_id == prev_device.device_id
        ):
            # Same device — merge
            merged[-1] = (prev_start, end, prev_device)
        else:
            merged.append((start, end, device))

    return merged
