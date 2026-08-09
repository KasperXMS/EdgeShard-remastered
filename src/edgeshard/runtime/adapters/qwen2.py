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
        self._layer_start: int = 0
        self._layer_end: int = 0
        self._loaded = False

    def load(
        self,
        model_path: str,
        layer_start: int,
        layer_end: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Load a subset of Qwen2 model layers."""
        logger.info(f"Loading Qwen2 layers [{layer_start}, {layer_end}) from {model_path}")

        self._device = device
        self._dtype = dtype
        self._layer_start = layer_start
        self._layer_end = layer_end

        # Load config
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self._config = json.load(f)
        else:
            raise ShardError(f"Config not found: {config_path}")

        num_layers = self._config.get("num_hidden_layers", 0)
        logger.info(f"Model config: num_hidden_layers={num_layers}, layer_range=[{layer_start}, {layer_end})")

        if layer_end > num_layers:
            raise ShardError(f"layer_end {layer_end} exceeds num_layers {num_layers}")

        # Load model weights to CPU first to avoid GPU memory bloat
        state_dict = self._load_weights(Path(model_path), torch.device("cpu"))

        # Filter weights for this shard's layer range, then move to target device
        shard_weights = {}
        for key, value in state_dict.items():
            if "model.layers." in key:
                layer_idx = int(key.split("model.layers.")[1].split(".")[0])
                if layer_start <= layer_idx < layer_end:
                    new_key = key.replace(
                        f"model.layers.{layer_idx}.",
                        f"model.layers.{layer_idx - layer_start}.",
                    )
                    shard_weights[new_key] = value.to(dtype=dtype, device=device)
            elif key == "model.embed_tokens.weight" and layer_start == 0:
                shard_weights[key] = value.to(dtype=dtype, device=device)
            elif key == "model.norm.weight" and layer_end == num_layers:
                shard_weights[key] = value.to(dtype=dtype, device=device)
            elif key == "lm_head.weight" and layer_end == num_layers:
                shard_weights[key] = value.to(dtype=dtype, device=device)

        # Free the full state dict from CPU memory
        del state_dict

        # Build model components
        self._build_model(shard_weights, layer_start, layer_end, num_layers)
        self._loaded = True
        logger.info(f"Loaded {len(self._layers)} layers on {device}")

    def _load_weights(self, model_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
        """Load weights from single or sharded checkpoint."""
        device_str = str(device)

        # Try single file first
        single_file = model_path / "model.safetensors"
        if single_file.exists():
            logger.info(f"Loading weights from {single_file} to {device_str}")
            return load_file(str(single_file), device=device_str)

        # Try sharded checkpoint
        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            logger.info(f"Loading sharded weights from {model_path} to {device_str}")
            with open(index_file) as f:
                index = json.load(f)

            shard_files = set(index["weight_map"].values())
            state_dict = {}

            for shard_file in shard_files:
                shard_path = model_path / shard_file
                logger.info(f"  Loading shard {shard_file}")
                shard_data = load_file(str(shard_path), device=device_str)
                state_dict.update(shard_data)

            return state_dict

        # Try legacy naming
        legacy_file = model_path / "model-00001-of-00001.safetensors"
        if legacy_file.exists():
            logger.info(f"Loading weights from {legacy_file} to {device_str}")
            return load_file(str(legacy_file), device=device_str)

        raise ShardError(f"No weights found in {model_path}")

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
            missing, unexpected = layer.load_state_dict(layer_weights, strict=False)
            if missing:
                logger.warning(f"Layer {i} missing keys: {len(missing)} keys")
            if unexpected:
                logger.warning(f"Layer {i} unexpected keys: {len(unexpected)} keys")

            layer.to(self._device, self._dtype)
            layer.eval()
            self._layers.append(layer)

        # Final norm (only on last shard)
        if layer_end == num_layers and "model.norm.weight" in weights:
            self._norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self._norm.weight.data = weights["model.norm.weight"]
            self._norm.to(self._device, self._dtype)
            logger.debug(f"Loaded final norm (layer_end={layer_end}, num_layers={num_layers})")

        # LM head (only on last shard)
        if layer_end == num_layers and "lm_head.weight" in weights:
            self._lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self._lm_head.weight.data = weights["lm_head.weight"]
            self._lm_head.to(self._device, self._dtype)
            logger.debug(f"Loaded LM head (layer_end={layer_end}, num_layers={num_layers})")
        elif layer_end == num_layers:
            logger.warning(f"LM head not found in weights for last shard (layer_end={layer_end}, num_layers={num_layers})")

    def _apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to query and key tensors.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
            cos: Cosine embeddings [seq_len, head_dim] or [1, 1, seq_len, head_dim]
            sin: Sine embeddings [seq_len, head_dim] or [1, 1, seq_len, head_dim]

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        # Ensure cos/sin have correct shape for broadcasting
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim]
            sin = sin.unsqueeze(0).unsqueeze(0)

        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        return q_embed, k_embed

    def _compute_rotary_embeddings(
        self,
        position_ids: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary position embeddings (cos, sin) for the given position_ids.

        Args:
            position_ids: Position IDs [batch, seq_len]
            seq_len: Sequence length.

        Returns:
            Tuple of (cos, sin) tensors.
        """
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2Config,
            rotate_half,
        )

        config = Qwen2Config(**self._config)
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta
        max_position_embeddings = config.max_position_embeddings

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=self._device) / head_dim)
        )

        # Compute cos/sin for the given position_ids
        # position_ids: [batch, seq_len]
        # We need to compute freqs for each position
        batch_size, seq_len_actual = position_ids.shape

        # Create position indices [seq_len]
        position_ids_flat = position_ids.reshape(-1)  # [batch * seq_len]

        # Compute outer product: [batch * seq_len, head_dim/2]
        freqs = torch.outer(position_ids_flat.float(), inv_freq)

        # Reshape to [batch, seq_len, head_dim/2]
        freqs = freqs.reshape(batch_size, seq_len_actual, -1)

        # Create cos/sin embeddings [batch, seq_len, head_dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        # Reshape for broadcasting: [batch, 1, seq_len, head_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        return cos, sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run forward pass through loaded layers."""
        if not self._loaded:
            raise ShardError("Model not loaded")

        # Compute rotary embeddings once for all layers
        seq_len = hidden_states.shape[1]
        cos, sin = self._compute_rotary_embeddings(position_ids, seq_len)
        position_embeddings = (cos, sin)

        new_kv_cache = []

        for i, layer in enumerate(self._layers):
            # Get KV cache for this layer
            layer_kv = kv_cache[i] if i < len(kv_cache) else None

            # Call the layer with the standard API for transformers 4.44.0
            try:
                outputs = layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=layer_kv,
                    use_cache=True,
                    position_embeddings=position_embeddings,
                )
                hidden_states = outputs[0]
                new_kv = outputs[1] if len(outputs) > 1 else None
            except Exception as e:
                logger.error(f"Layer {i} forward failed: {e}")
                raise

            new_kv_cache.append(new_kv)

        # Apply final norm if this is the last shard
        if self._norm is not None:
            hidden_states = self._norm(hidden_states)

        return hidden_states, new_kv_cache

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings."""
        if self._embed_tokens is None:
            raise ShardError("This shard does not have embedding layer")
        return self._embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states."""
        if self._lm_head is None:
            raise ShardError("This shard does not have LM head")
        return self._lm_head(hidden_states)

    def init_kv_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Initialize empty KV cache for Qwen2."""
        return [None] * len(self._layers)

    def get_model_info(self) -> dict[str, Any]:
        """Return Qwen2 model metadata."""
        hidden_size = self._config.get("hidden_size", 0)
        num_attention_heads = self._config.get("num_attention_heads", 0)
        # head_dim might not be in config, compute it if missing
        head_dim = self._config.get("head_dim", 0)
        if head_dim == 0 and hidden_size > 0 and num_attention_heads > 0:
            head_dim = hidden_size // num_attention_heads

        return {
            "model_type": "qwen2",
            "num_layers": self._config.get("num_hidden_layers", 0),
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": self._config.get("num_key_value_heads", 0),
            "head_dim": head_dim,
            "vocab_size": self._config.get("vocab_size", 0),
            "max_position_embeddings": self._config.get("max_position_embeddings", 0),
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

    def get_device(self) -> torch.device:
        """Return the device where model weights are loaded."""
        return self._device
