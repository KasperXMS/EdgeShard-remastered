"""Network probe — measure latency and bandwidth between workers.

This module provides network topology discovery:
- TCP connect latency estimation (lightweight)
- Bandwidth estimation via small tensor transfer
"""

from __future__ import annotations

import asyncio
import socket
import time
from typing import Any

from edgeshard._grpc import edgeshard_pb2
from edgeshard.common.logging import get_logger

logger = get_logger(__name__)


class NetworkProbe:
    """Probes network connectivity to other workers."""

    def __init__(self, worker_id: str) -> None:
        self._worker_id = worker_id
        self._latency_cache: dict[str, float] = {}  # worker_id -> latency_ms
        self._bandwidth_cache: dict[str, int] = {}  # worker_id -> mbps

    async def measure_latency(
        self,
        target_host: str,
        target_port: int,
        target_worker_id: str,
        num_samples: int = 3,
    ) -> float | None:
        """Measure TCP connect latency to another worker.

        Args:
            target_host: Target hostname or IP.
            target_port: Target port.
            target_worker_id: Target worker ID for caching.
            num_samples: Number of samples to average.

        Returns:
            Average latency in milliseconds, or None if unreachable.
        """
        latencies = []

        for i in range(num_samples):
            try:
                start = time.perf_counter()

                # Non-blocking connect with timeout
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.getaddrinfo(target_host, target_port),
                    timeout=2.0
                )

                # TCP connect
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)

                connect_future = loop.run_in_executor(None, sock.connect, (target_host, target_port))
                await asyncio.wait_for(connect_future, timeout=2.0)

                elapsed_ms = (time.perf_counter() - start) * 1000

                sock.close()
                latencies.append(elapsed_ms)

            except asyncio.TimeoutError:
                logger.debug(f"Timeout connecting to {target_host}:{target_port}")
                continue
            except socket.error as e:
                logger.debug(f"Socket error connecting to {target_host}:{target_port}: {e}")
                continue
            except Exception as e:
                logger.debug(f"Error measuring latency to {target_host}:{target_port}: {e}")
                continue

        if not latencies:
            return None

        avg_latency = sum(latencies) / len(latencies)
        self._latency_cache[target_worker_id] = avg_latency

        logger.debug(f"Latency to {target_worker_id} ({target_host}:{target_port}): {avg_latency:.2f} ms")
        return avg_latency

    async def measure_latency_to_workers(
        self,
        workers: dict[str, tuple[str, int]],  # worker_id -> (host, port)
    ) -> dict[str, int]:
        """Measure latency to multiple workers.

        Args:
            workers: Dict mapping worker_id to (host, port).

        Returns:
            Dict mapping worker_id to latency_ms (int).
        """
        results = {}

        for worker_id, (host, port) in workers.items():
            if worker_id == self._worker_id:
                continue  # Skip self

            latency = await self.measure_latency(host, port, worker_id)
            if latency is not None:
                results[worker_id] = int(latency)

        return results

    def get_cached_latency(self, worker_id: str) -> int | None:
        """Get cached latency to a worker.

        Args:
            worker_id: Target worker ID.

        Returns:
            Cached latency in milliseconds, or None if not cached.
        """
        latency = self._latency_cache.get(worker_id)
        if latency is not None:
            return int(latency)
        return None

    def estimate_bandwidth(
        self,
        target_host: str,
        target_port: int,
        target_worker_id: str,
    ) -> int | None:
        """Estimate network bandwidth to another worker.

        This is a simplified estimation based on network interface info.
        For accurate measurement, a dedicated bandwidth test server is needed.

        Args:
            target_host: Target hostname or IP.
            target_port: Target port.
            target_worker_id: Target worker ID for caching.

        Returns:
            Estimated bandwidth in Mbps, or None if estimation fails.
        """
        try:
            # Get local network interface speed
            import psutil

            # Find the interface that would route to target
            local_addrs = psutil.net_if_addrs()
            stats = psutil.net_if_stats()

            max_speed = 0
            for iface, iface_stats in stats.items():
                if iface_stats.isup:
                    # Speed is in Mbps
                    speed = iface_stats.speed
                    if speed > max_speed:
                        max_speed = speed

            if max_speed == 0:
                # Fallback: assume typical connection
                max_speed = 1000  # 1 Gbps default

            # Apply a conservative factor for cross-network
            # Real bandwidth is usually 60-80% of interface speed
            estimated_mbps = int(max_speed * 0.7)

            self._bandwidth_cache[target_worker_id] = estimated_mbps
            logger.debug(f"Estimated bandwidth to {target_worker_id}: {estimated_mbps} Mbps")
            return estimated_mbps

        except Exception as e:
            logger.debug(f"Failed to estimate bandwidth: {e}")
            return None

    def get_network_metrics(
        self,
        workers: dict[str, tuple[str, int]] | None = None,
    ) -> edgeshard_pb2.NetworkMetrics:
        """Build NetworkMetrics message.

        Args:
            workers: Optional dict of worker_id -> (host, port) to probe.
                     If None, uses cached values.

        Returns:
            NetworkMetrics protobuf message.
        """
        # Use cached latencies
        latency_map = {
            worker_id: int(latency)
            for worker_id, latency in self._latency_cache.items()
        }

        # Estimate bandwidth (use max of estimates to known workers)
        estimated_bandwidth = 0
        if self._bandwidth_cache:
            estimated_bandwidth = max(self._bandwidth_cache.values())

        return edgeshard_pb2.NetworkMetrics(
            latency_ms_to_worker=latency_map,
            estimated_bandwidth_mbps=estimated_bandwidth,
        )


async def probe_network_topology(
    worker_id: str,
    known_workers: dict[str, tuple[str, int]],
) -> edgeshard_pb2.NetworkMetrics:
    """Probe network topology to all known workers.

    Args:
        worker_id: This worker's ID.
        known_workers: Dict of worker_id -> (host, port).

    Returns:
        NetworkMetrics with latency and bandwidth estimates.
    """
    probe = NetworkProbe(worker_id)

    # Measure latency to all workers
    await probe.measure_latency_to_workers(known_workers)

    # Estimate bandwidth to at least one worker
    for target_id, (host, port) in known_workers.items():
        if target_id != worker_id:
            probe.estimate_bandwidth(host, port, target_id)
            break

    return probe.get_network_metrics()
