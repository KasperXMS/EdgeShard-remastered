"""Qwen2/Qwen2.5 ModelAdapter implementation.

This adapter handles:
- Loading a subset of transformer layers for a shard
- Managing model-specific KV cache structure
- Forward pass through the loaded layers
- Integration with Hugging Face Transformers

For a full model split across shards:
- First shard: includes embedding layer + layers [0, k1)
- Middle shards: layers [k1, k2), [k2, k3), etc.
- Last shard: layers [kn, num_layers) + LM head
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file

from edgeshard.common.errors import ShardError
from edgeshard.common.logging import get_logger
from edgeshard.runtime.adapters.base import ModelAdapter

logger = get_logger(__name__)


class Qwen2Adapter(ModelAdapter):
    """ModelAdapter for Qwen2 and Qwen2.5 model families."""

    def __init__(self) -> None:
        self._layers: list[nn.Module] = []
        self._embed_tokens: nn.Embedding | None = None
        self._lm_head: nn.Linear | None = None
        self._norm: nn.Module | None = None
        self._config: dict[str, Any] = {}
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32
        self._loaded = False

    def load(
        self,
        model_path: str,
        layer_start: int,
        layer_end: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Load a subset of Qwen2 model layers.

        Args:
            model_path: Path to model directory or Hugging Face ID.
            layer_start: Inclusive start layer index.
            layer_end: Exclusive end layer index.
            dtype: Target dtype for weights.
            device: Target device.
        """
        logger.info(
            f"Loading Qwen2 layers [{layer_start}, {layer_end}) from {model_path}"
        )

        self._device = device
        self._dtype = dtype

        # Load config
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self._config = json.load(f)
        else:
            raise ShardError(f"Config not found: {config_path}")

        num_layers = self._config.get("num_hidden_layers", 0)
        if layer_end > num_layers:
            raise ShardError(
                f"layer_end {layer_end} exceeds num_layers {num_layers}"
            )

        # Load model weights
        weights_path = Path(model_path) / "model.safetensors"
        if not weights_path.exists():
            # Try sharded checkpoints
            weights_path = Path(model_path) / "model-00001-of-00001.safetensors"
            if not weights_path.exists():
                raise ShardError(f"Weights not found in {model_path}")

        logger.info(f"Loading weights from {weights_path}")
        state_dict = load_file(str(weights_path), device=str(device))

        # Filter weights for this shard's layer range
        shard_weights = {}
        for key, value in state_dict.items():
            if "model.layers." in key:
                # Extract layer index
                layer_idx = int(key.split("model.layers.")[1].split(".")[0])
                if layer_start <= layer_idx < layer_end:
                    # Remap to 0-based for this shard
                    new_key = key.replace(
                        f"model.layers.{layer_idx}.",
                        f"model.layers.{layer_idx - layer_start}.",
                    )
                    shard_weights[new_key] = value.to(dtype)
            elif key == "model.embed_tokens.weight" and layer_start == 0:
                shard_weights[key] = value.to(dtype)
            elif key == "model.norm.weight" and layer_end == num_layers:
                shard_weights[key] = value.to(dtype)
            elif key == "lm_head.weight" and layer_end == num_layers:
                shard_weights[key] = value.to(dtype)

        # Build model components
        self._build_model(shard_weights, layer_start, layer_end, num_layers)
        self._loaded = True
        logger.info(f"Loaded {len(self._layers)} layers on {device}")

    def _build_model(
        self,
        weights: dict[str, torch.Tensor],
        layer_start: int,
        layer_end: int,
        num_layers: int,
    ) -> None:
        """Build model components from filtered weights."""
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2DecoderLayer,
            Qwen2Config,
            Qwen2RMSNorm,
        )

        config = Qwen2Config(**self._config)

        # Embedding layer (only on first shard)
        if layer_start == 0 and "model.embed_tokens.weight" in weights:
            vocab_size = config.vocab_size
            hidden_size = config.hidden_size
            self._embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self._embed_tokens.weight.data = weights["model.embed_tokens.weight"]
            self._embed_tokens.to(self._device, self._dtype)

        # Transformer layers
        self._layers = []
        for i in range(layer_end - layer_start):
            layer = Qwen2DecoderLayer(config, layer_idx=layer_start + i)
            # Load weights for this layer
            layer_weights = {
                k.replace(f"model.layers.{i}.", ""): v
                for k, v in weights.items()
                if k.startswith(f"model.layers.{i}.")
            }
            layer.load_state_dict(layer_weights, strict=False)
            layer.to(self._device, self._dtype)
            layer.eval()
            self._layers.append(layer)

        # Final norm (only on last shard)
        if layer_end == num_layers and "model.norm.weight" in weights:
            self._norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self._norm.weight.data = weights["model.norm.weight"]
            self._norm.to(self._device, self._dtype)

        # LM head (only on last shard)
        if layer_end == num_layers and "lm_head.weight" in weights:
            self._lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self._lm_head.weight.data = weights["lm_head.weight"]
            self._lm_head.to(self._device, self._dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run forward pass through loaded layers.

        Args:
            hidden_states: Input hidden states [batch, seq, hidden].
            kv_cache: List of (key, value) tuples per layer.
            position_ids: Position IDs for RoPE [batch, seq].

        Returns:
            (output_hidden_states, updated_kv_cache)
        """
        if not self._loaded:
            raise ShardError("Model not loaded")

        # Compute RoPE embeddings (required for newer transformers versions)
        position_embeddings = self._compute_position_embeddings(position_ids)

        new_kv_cache = []
        for i, layer in enumerate(self._layers):
            # Get KV cache for this layer
            layer_kv = kv_cache[i] if i < len(kv_cache) else None

            # Run layer
            outputs = layer(
                hidden_states,
                past_key_value=layer_kv,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=True,
            )
            hidden_states = outputs[0]
            new_kv = outputs[1] if len(outputs) > 1 else None
            new_kv_cache.append(new_kv)

        # Apply final norm if this is the last shard
        if self._norm is not None:
            hidden_states = self._norm(hidden_states)

        return hidden_states, new_kv_cache

    def _compute_position_embeddings(
        self, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE position embeddings (cos, sin).

        Args:
            position_ids: Position IDs [batch, seq].

        Returns:
            Tuple of (cos, sin) tensors.
        """
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

        # Create config object
        config = Qwen2Config(**self._config)

        # Create rotary embedding from config
        rot_emb = Qwen2RotaryEmbedding(config)
        rot_emb = rot_emb.to(self._device, self._dtype)

        # Compute cos and sin
        position_embeddings = rot_emb(hidden_states, position_ids)
        return position_embeddings

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings.

        Args:
            input_ids: Token IDs [batch, seq].

        Returns:
            Embeddings [batch, seq, hidden].
        """
        if self._embed_tokens is None:
            raise ShardError("This shard does not have embedding layer")
        return self._embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states.

        Args:
            hidden_states: Hidden states [batch, seq, hidden].

        Returns:
            Logits [batch, seq, vocab].
        """
        if self._lm_head is None:
            raise ShardError("This shard does not have LM head")
        return self._lm_head(hidden_states)

    def init_kv_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Initialize empty KV cache for Qwen2.

        Returns:
            List of (key, value) tuples, one per layer. Each is None initially.
        """
        # Qwen2 uses dynamic cache, so we start with None for each layer
        return [None] * len(self._layers)

    def get_model_info(self) -> dict[str, Any]:
        """Return Qwen2 model metadata."""
        return {
            "model_type": "qwen2",
            "num_layers": self._config.get("num_hidden_layers", 0),
            "hidden_size": self._config.get("hidden_size", 0),
            "num_attention_heads": self._config.get("num_attention_heads", 0),
            "num_key_value_heads": self._config.get("num_key_value_heads", 0),
            "head_dim": self._config.get("head_dim", 0),
            "vocab_size": self._config.get("vocab_size", 0),
            "max_position_embeddings": self._config.get(
                "max_position_embeddings", 0
            ),
            "loaded_layers": len(self._layers),
            "has_embedding": self._embed_tokens is not None,
            "has_lm_head": self._lm_head is not None,
        }

    def unload(self) -> None:
        """Release model weights and free memory."""
        self._layers.clear()
        self._embed_tokens = None
        self._lm_head = None
        self._norm = None
        self._config.clear()
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Qwen2 adapter unloaded")
