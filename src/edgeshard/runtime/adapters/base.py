"""ModelAdapter ABC — isolates model-specific implementation.

Each supported model family (Qwen2, LLaMA, etc.) implements this interface.
The Shard Runtime uses ModelAdapter to load weights, run forward passes,
and manage KV cache without knowing model internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class ModelAdapter(ABC):
    """Abstract interface for model-specific logic.

    Implementations handle:
    - Loading weights from checkpoints
    - Running forward passes for specific layer ranges
    - Managing model-specific KV cache structures
    - Providing model metadata (num_layers, hidden_size, etc.)
    """

    @abstractmethod
    def load(
        self,
        model_path: str,
        layer_start: int,
        layer_end: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Load a subset of model layers.

        Args:
            model_path: Path to model weights or Hugging Face ID.
            layer_start: Inclusive start layer index.
            layer_end: Exclusive end layer index.
            dtype: Target dtype for weights.
            device: Target device.
        """

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Any,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        """Run forward pass through loaded layers.

        Args:
            hidden_states: Input hidden states [batch, seq, hidden].
            kv_cache: Model-specific KV cache structure.
            position_ids: Position IDs for RoPE/attention.

        Returns:
            (output_hidden_states, updated_kv_cache)
        """

    @abstractmethod
    def init_kv_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
    ) -> Any:
        """Initialize an empty KV cache for this model.

        Args:
            batch_size: Batch size.
            max_seq_len: Maximum sequence length.
            device: Target device.

        Returns:
            Model-specific KV cache structure.
        """

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata.

        Returns:
            Dict with keys like: num_layers, hidden_size, num_heads,
            head_dim, vocab_size, etc.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release model weights and free memory."""
