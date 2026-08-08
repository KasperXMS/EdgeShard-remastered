"""Model family registry — maps model types to their adapters."""

from __future__ import annotations

from typing import Type

from edgeshard.runtime.adapters.base import ModelAdapter

# Registry of model type -> adapter class
_ADAPTER_REGISTRY: dict[str, Type[ModelAdapter]] = {}


def register_adapter(model_type: str, adapter_class: Type[ModelAdapter]) -> None:
    """Register a ModelAdapter for a model type.

    Args:
        model_type: Model type identifier (e.g. "qwen2", "llama").
        adapter_class: ModelAdapter implementation class.
    """
    _ADAPTER_REGISTRY[model_type] = adapter_class


def get_adapter(model_type: str) -> ModelAdapter:
    """Get a ModelAdapter instance for a model type.

    Args:
        model_type: Model type identifier.

    Returns:
        ModelAdapter instance.

    Raises:
        KeyError: If model type is not registered.
    """
    if model_type not in _ADAPTER_REGISTRY:
        raise KeyError(
            f"Model type '{model_type}' not registered. "
            f"Available: {list(_ADAPTER_REGISTRY.keys())}"
        )
    return _ADAPTER_REGISTRY[model_type]()


def list_adapters() -> list[str]:
    """List all registered model types."""
    return list(_ADAPTER_REGISTRY.keys())


def _register_builtin_adapters() -> None:
    """Register built-in adapters."""
    try:
        from edgeshard.runtime.adapters.qwen2 import Qwen2Adapter

        register_adapter("qwen2", Qwen2Adapter)
        register_adapter("qwen2.5", Qwen2Adapter)
    except ImportError:
        pass  # Qwen2 adapter not available


# Register built-in adapters on import
_register_builtin_adapters()
