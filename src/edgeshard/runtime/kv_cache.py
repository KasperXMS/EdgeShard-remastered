"""KV cache manager — memory tracking and session cleanup.

This module provides utilities for managing KV cache memory across sessions.
For M2, we focus on:
- Memory usage tracking per session
- Automatic cleanup on session release
- Debugging utilities to inspect KV cache state

Future extensions:
- Eviction policies (LRU, priority-based)
- Memory pressure monitoring
- Cross-shard KV cache coordination
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from edgeshard.common.identifiers import SessionId
from edgeshard.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KVCacheStats:
    """Statistics for KV cache memory usage."""

    num_sessions: int
    total_memory_mb: float
    per_session_memory_mb: dict[str, float]


class KVCacheManager:
    """Manager for KV cache memory across multiple sessions.

    This is a lightweight wrapper that tracks memory usage and provides
    debugging utilities. The actual KV cache is stored in SessionState
    within ModelShard.
    """

    def __init__(self) -> None:
        self._session_memory: dict[str, float] = {}  # session_id -> MB
        self._total_memory_mb = 0.0

    def register_session(
        self,
        session_id: SessionId,
        kv_cache: Any,
        device: torch.device,
    ) -> None:
        """Register a new session and estimate its KV cache memory.

        Args:
            session_id: Session identifier.
            kv_cache: KV cache structure (model-specific).
            device: Device where KV cache is stored.
        """
        memory_mb = self._estimate_kv_cache_memory(kv_cache)
        self._session_memory[str(session_id)] = memory_mb
        self._total_memory_mb += memory_mb

        logger.debug(
            f"Session {session_id} registered: {memory_mb:.2f} MB KV cache"
        )

    def unregister_session(self, session_id: SessionId) -> None:
        """Unregister a session and free its memory tracking.

        Args:
            session_id: Session identifier.
        """
        sid = str(session_id)
        if sid in self._session_memory:
            memory_mb = self._session_memory[sid]
            del self._session_memory[sid]
            self._total_memory_mb -= memory_mb
            logger.debug(
                f"Session {session_id} unregistered: freed {memory_mb:.2f} MB"
            )

    def get_stats(self) -> KVCacheStats:
        """Get current KV cache memory statistics.

        Returns:
            KVCacheStats with memory usage breakdown.
        """
        return KVCacheStats(
            num_sessions=len(self._session_memory),
            total_memory_mb=self._total_memory_mb,
            per_session_memory_mb=self._session_memory.copy(),
        )

    def _estimate_kv_cache_memory(self, kv_cache: Any) -> float:
        """Estimate memory usage of a KV cache structure.

        Args:
            kv_cache: Model-specific KV cache (e.g., list of tuples).

        Returns:
            Estimated memory in MB.
        """
        total_bytes = 0

        if isinstance(kv_cache, list):
            for layer_cache in kv_cache:
                if layer_cache is None:
                    continue
                if isinstance(layer_cache, tuple):
                    for tensor in layer_cache:
                        if isinstance(tensor, torch.Tensor):
                            total_bytes += tensor.numel() * tensor.element_size()
                elif isinstance(layer_cache, torch.Tensor):
                    total_bytes += layer_cache.numel() * layer_cache.element_size()
        elif isinstance(kv_cache, torch.Tensor):
            total_bytes = kv_cache.numel() * kv_cache.element_size()

        return total_bytes / (1024 * 1024)  # Convert to MB

    def print_debug_info(self) -> None:
        """Print debug information about KV cache state."""
        stats = self.get_stats()
        print(f"\n=== KV Cache Manager Debug ===")
        print(f"Active sessions: {stats.num_sessions}")
        print(f"Total memory: {stats.total_memory_mb:.2f} MB")
        if stats.per_session_memory_mb:
            print("Per-session breakdown:")
            for sid, mem in stats.per_session_memory_mb.items():
                print(f"  {sid}: {mem:.2f} MB")
        print("=" * 30)


# Global manager instance (can be replaced with dependency injection)
_global_manager = KVCacheManager()


def get_kv_cache_manager() -> KVCacheManager:
    """Get the global KV cache manager instance."""
    return _global_manager
