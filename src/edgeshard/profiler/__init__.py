"""Profiler module — model performance profiling for EdgeShard.

Provides:
- ProfileResult: Data model for profiling results
- ProfileExecutor: Measures model performance on a device (requires torch)
- ProfileStore: SQLite persistence for profiles
"""

from edgeshard.profiler.profile_data import ProfileResult
from edgeshard.profiler.store import ProfileStore

# Lazy import for ProfileExecutor — it requires torch which may not be
# available in all environments (e.g., CPU-only, or planning-only usage).
# Use __getattr__ for PEP 562 lazy module loading.
def __getattr__(name: str):
    if name == "ProfileExecutor":
        from edgeshard.profiler.executor import ProfileExecutor
        return ProfileExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ProfileResult",
    "ProfileExecutor",
    "ProfileStore",
]
