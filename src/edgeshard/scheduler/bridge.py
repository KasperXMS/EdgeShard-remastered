"""Bridge functions — convert runtime data to scheduler snapshot types.

These bridges isolate the scheduler from protobuf and runtime internals.
The scheduler only sees typed, immutable dataclass snapshots.
"""

from __future__ import annotations

import time

from edgeshard.common.identifiers import WorkerId
from edgeshard.common.logging import get_logger
from edgeshard.scheduler.snapshot import (
    ClusterSnapshot,
    DeviceInfo,
    ProfileEntry,
    ProfileSnapshot,
    WorkerState,
)

logger = get_logger(__name__)


def worker_records_to_cluster_snapshot(
    records: list,
) -> ClusterSnapshot:
    """Convert WorkerManager WorkerRecords to a typed ClusterSnapshot.

    Args:
        records: List of WorkerRecord objects from WorkerManager.
            Each record has proto DeviceInfo, DeviceMetrics, NetworkMetrics.

    Returns:
        ClusterSnapshot ready for the scheduler.
    """
    from edgeshard._grpc import edgeshard_pb2

    workers: list[WorkerState] = []
    bandwidth_matrix: dict[tuple[str, str], float] = {}
    source_worker_id: str | None = None

    for rec in records:
        # Skip dead workers
        if not rec.is_alive():
            continue

        # Convert proto DeviceInfo → scheduler DeviceInfo
        devices: list[DeviceInfo] = []
        for pb_dev in rec.devices:
            devices.append(
                DeviceInfo(
                    device_id=pb_dev.device_id,
                    device_type=pb_dev.device_type,
                    name=pb_dev.name,
                    total_memory_mb=pb_dev.total_memory_mb,
                    compute_capability=pb_dev.compute_capability or None,
                    properties=dict(pb_dev.properties),
                )
            )

        worker_state = WorkerState(
            worker_id=rec.worker_id,
            hostname=rec.hostname,
            devices=devices,
            available_memory_mb=rec.available_memory_mb,
            cpu_count=rec.cpu_count,
            status=rec.status,
            metadata=dict(rec.metadata),
        )
        workers.append(worker_state)

        # Set source worker (first alive worker)
        if source_worker_id is None:
            source_worker_id = str(rec.worker_id)

        # Extract bandwidth from NetworkMetrics
        if rec.network_metrics is not None:
            nm = rec.network_metrics
            for other_wid, latency_ms in nm.latency_ms_to_worker.items():
                # Convert latency to a bandwidth estimate
                # We don't have direct bandwidth, but we can use the
                # estimated_bandwidth_mbps field
                if nm.estimated_bandwidth_mbps > 0:
                    bw_mbps = float(nm.estimated_bandwidth_mbps)
                    # Convert Mbps → MB/s (divide by 8)
                    bandwidth_matrix[(str(rec.worker_id), other_wid)] = bw_mbps / 8.0

    return ClusterSnapshot(
        workers=workers,
        timestamp=time.time(),
        bandwidth_matrix=bandwidth_matrix,
        source_worker_id=source_worker_id,
    )


def profile_results_to_snapshot(
    profiles: list,
) -> ProfileSnapshot:
    """Convert ProfileResult objects to a typed ProfileSnapshot.

    Args:
        profiles: List of ProfileResult objects from ProfileStore.

    Returns:
        ProfileSnapshot ready for the scheduler.
    """
    entries: list[ProfileEntry] = []

    for p in profiles:
        # Create device fingerprint from device_type + device_name
        device_fingerprint = f"{p.device_type}:{p.device_name}"

        entry = ProfileEntry(
            model_name=p.model_name,
            model_revision=p.model_revision,
            device_fingerprint=device_fingerprint,
            dtype=p.dtype,
            runtime_version="",  # Not tracked in ProfileResult
            layer_forward_ms=p.layer_forward_ms,
            kv_cache_per_token_mb=p.kv_cache_per_token_mb,
            activation_memory_mb=0.0,  # Not directly measured
            prefill_tokens_per_sec=p.prefill_tokens_per_sec,
            decode_tokens_per_sec=p.decode_tokens_per_sec,
        )
        entries.append(entry)

    return ProfileSnapshot(entries=entries)


def cluster_snapshot_from_yaml(path: str) -> ClusterSnapshot:
    """Load a ClusterSnapshot from a YAML file.

    Useful for offline planning without a running Master.

    Args:
        path: Path to cluster YAML file.

    Returns:
        ClusterSnapshot loaded from YAML.
    """
    import yaml
    from pathlib import Path

    with open(Path(path), encoding="utf-8") as f:
        data = yaml.safe_load(f)

    workers: list[WorkerState] = []
    for w_data in data.get("workers", []):
        devices = [
            DeviceInfo(
                device_id=d.get("device_id", ""),
                device_type=d.get("device_type", "cpu"),
                name=d.get("name", ""),
                total_memory_mb=d.get("total_memory_mb", 0),
                compute_capability=d.get("compute_capability"),
            )
            for d in w_data.get("devices", [])
        ]
        workers.append(
            WorkerState(
                worker_id=WorkerId(w_data.get("worker_id", "unknown")),
                hostname=w_data.get("hostname", "localhost"),
                devices=devices,
                available_memory_mb=w_data.get("available_memory_mb", 0),
                cpu_count=w_data.get("cpu_count", 1),
                status=w_data.get("status", "online"),
            )
        )

    # Parse bandwidth matrix
    bandwidth_matrix: dict[tuple[str, str], float] = {}
    for bw_entry in data.get("bandwidth_matrix", []):
        key = (bw_entry["from"], bw_entry["to"])
        bandwidth_matrix[key] = float(bw_entry["bandwidth_mb_per_s"])

    return ClusterSnapshot(
        workers=workers,
        timestamp=data.get("timestamp", time.time()),
        bandwidth_matrix=bandwidth_matrix,
        source_worker_id=data.get("source_worker_id"),
    )
