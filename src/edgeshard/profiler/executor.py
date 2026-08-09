"""Profile executor — measures model performance on a device.

This module provides profiling capabilities:
- Layer forward pass latency
- KV cache memory cost per token
- Prefill throughput (tokens/sec)
- Decode throughput (tokens/sec)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from edgeshard.common.logging import get_logger
from edgeshard.profiler.profile_data import ProfileResult
from edgeshard.runtime.adapters.base import ModelAdapter

logger = get_logger(__name__)


class ProfileExecutor:
    """Executes profiling on a model/device combination.

    Profiling loads the model (or a subset of layers) and measures:
    - Single-layer forward latency
    - KV cache memory per token
    - Prefill throughput
    - Decode throughput
    """

    def __init__(
        self,
        num_warmup: int = 3,
        num_runs: int = 10,
        prefill_seq_len: int = 512,
        decode_steps: int = 20,
    ) -> None:
        """Initialize ProfileExecutor.

        Args:
            num_warmup: Number of warmup runs before timing.
            num_runs: Number of timed runs for averaging.
            prefill_seq_len: Sequence length for prefill benchmark.
            decode_steps: Number of decode steps for throughput measurement.
        """
        self._num_warmup = num_warmup
        self._num_runs = num_runs
        self._prefill_seq_len = prefill_seq_len
        self._decode_steps = decode_steps

    async def profile(
        self,
        model_path: str,
        dtype: torch.dtype,
        device: torch.device,
        layer_start: int = 0,
        layer_end: int | None = None,
        batch_size: int = 1,
    ) -> ProfileResult:
        """Run full profiling on a model.

        Args:
            model_path: Path to model weights or Hugging Face ID.
            dtype: Target dtype for weights.
            device: Target device.
            layer_start: Start layer index (for partial profiling).
            layer_end: End layer index (None = all layers).
            batch_size: Batch size for profiling.

        Returns:
            ProfileResult with all measurements.
        """
        from edgeshard.runtime.adapters.qwen2 import Qwen2Adapter
        from edgeshard.runtime.kv_cache import get_kv_cache_manager

        logger.info(f"Profiling {model_path} on {device}...")

        # Load adapter
        adapter = Qwen2Adapter()

        # Determine layer range
        model_info = self._get_model_info(model_path)
        num_layers = model_info.get("num_hidden_layers", 24)

        if layer_end is None:
            layer_end = num_layers

        num_layers_to_load = layer_end - layer_start
        logger.info(f"Loading layers [{layer_start}, {layer_end})")

        adapter.load(model_path, layer_start, layer_end, dtype, device)

        # Get device info
        device_name = self._get_device_name(device)
        device_memory_mb = self._get_device_memory_mb(device)

        # Run measurements
        logger.info("Measuring layer forward latency...")
        layer_forward_ms = await self._measure_layer_forward(adapter, device, dtype, batch_size)

        logger.info("Estimating KV cache memory per token...")
        kv_cache_per_token_mb = self._estimate_kv_cache_per_token(adapter, device, dtype, batch_size)

        logger.info("Measuring prefill throughput...")
        prefill_tps = await self._measure_prefill_throughput(
            adapter, device, dtype, batch_size, self._prefill_seq_len
        )

        logger.info("Measuring decode throughput...")
        decode_tps = await self._measure_decode_throughput(
            adapter, device, dtype, batch_size, self._decode_steps
        )

        # Estimate total model memory
        total_model_memory_mb = self._estimate_model_memory(adapter)

        # Clean up
        adapter.unload()

        # Build result
        result = ProfileResult(
            model_name=str(model_path),
            model_revision="",
            dtype=str(dtype).replace("torch.", ""),
            device_type=str(device),
            device_name=device_name,
            device_memory_mb=device_memory_mb,
            layer_forward_ms=layer_forward_ms,
            kv_cache_per_token_mb=kv_cache_per_token_mb,
            prefill_tokens_per_sec=prefill_tps,
            decode_tokens_per_sec=decode_tps,
            total_model_memory_mb=total_model_memory_mb,
            num_layers_profiled=num_layers_to_load,
            num_runs=self._num_runs,
        )

        logger.info(
            f"Profiling complete: layer_forward={layer_forward_ms:.2f}ms, "
            f"prefill={prefill_tps:.1f} tok/s, decode={decode_tps:.1f} tok/s"
        )

        return result

    def _get_model_info(self, model_path: str) -> dict:
        """Read model config to get metadata."""
        import json

        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}

    def _get_device_name(self, device: torch.device) -> str:
        """Get human-readable device name."""
        if device.type == "cuda":
            try:
                return torch.cuda.get_device_name(device)
            except Exception:
                return "Unknown GPU"
        return str(device)

    def _get_device_memory_mb(self, device: torch.device) -> int:
        """Get total device memory in MB."""
        if device.type == "cuda":
            try:
                props = torch.cuda.get_device_properties(device)
                return props.total_mem // (1024 * 1024)
            except Exception:
                return 0
        return 0

    async def _measure_layer_forward(
        self,
        adapter: ModelAdapter,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> float:
        """Measure average single-layer forward latency in milliseconds.

        Args:
            adapter: Loaded model adapter.
            device: Device.
            dtype: Dtype.
            batch_size: Batch size.

        Returns:
            Average forward latency per layer (ms).
        """
        if not adapter._layers:
            return 0.0

        hidden_size = adapter.get_model_info()["hidden_size"]
        seq_len = 1  # Single token for decode-like measurement

        # Create dummy hidden states
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size,
            device=device, dtype=dtype,
        )
        position_ids = torch.tensor([[0]], device=device)

        # Use first layer only for per-layer measurement
        layer = adapter._layers[0]
        kv_cache = adapter.init_kv_cache(batch_size, seq_len, device)

        # Compute rotary embeddings
        cos, sin = adapter._compute_rotary_embeddings(position_ids, seq_len)

        # Build attention mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)
        cache_position = torch.arange(seq_len, device=device)

        # Warmup
        for _ in range(self._num_warmup):
            with torch.no_grad():
                layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=kv_cache,
                    use_cache=True,
                    position_embeddings=(cos, sin),
                    cache_position=cache_position,
                )

        # Synchronize before timing
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # Timed runs
        start_time = time.perf_counter()
        for _ in range(self._num_runs):
            kv_cache = adapter.init_kv_cache(batch_size, seq_len, device)
            with torch.no_grad():
                layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=kv_cache,
                    use_cache=True,
                    position_embeddings=(cos, sin),
                    cache_position=cache_position,
                )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return elapsed_ms / self._num_runs

    def _estimate_kv_cache_per_token(
        self,
        adapter: ModelAdapter,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> float:
        """Estimate KV cache memory per token in MB.

        Args:
            adapter: Loaded model adapter.
            device: Device.
            dtype: Dtype.
            batch_size: Batch size.

        Returns:
            Estimated KV cache memory per token (MB).
        """
        model_info = adapter.get_model_info()
        num_layers = model_info.get("loaded_layers", len(adapter._layers))
        num_kv_heads = model_info.get("num_key_value_heads", model_info.get("num_attention_heads", 1))
        head_dim = model_info.get("head_dim", 0)
        if head_dim == 0:
            hidden_size = model_info.get("hidden_size", 0)
            num_attn_heads = model_info.get("num_attention_heads", 1)
            head_dim = hidden_size // num_attn_heads if num_attn_heads > 0 else 0

        # KV cache per token per layer:
        # 2 (K + V) * num_kv_heads * head_dim * dtype_size
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        kv_per_token_per_layer = 2 * num_kv_heads * head_dim * dtype_size
        kv_per_token_total = kv_per_token_per_layer * num_layers

        # Convert to MB
        return (kv_per_token_total * batch_size) / (1024 * 1024)

    async def _measure_prefill_throughput(
        self,
        adapter: ModelAdapter,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        seq_len: int,
    ) -> float:
        """Measure prefill throughput in tokens/sec.

        Args:
            adapter: Loaded model adapter.
            device: Device.
            dtype: Dtype.
            batch_size: Batch size.
            seq_len: Sequence length for prefill.

        Returns:
            Prefill throughput (tokens/sec).
        """
        hidden_size = adapter.get_model_info()["hidden_size"]

        # Create dummy hidden states (simulating after embedding)
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size,
            device=device, dtype=dtype,
        )
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        kv_cache = adapter.init_kv_cache(batch_size, seq_len + 10, device)

        # Warmup
        for _ in range(self._num_warmup):
            test_cache = adapter.init_kv_cache(batch_size, seq_len + 10, device)
            with torch.no_grad():
                adapter.forward(hidden_states, test_cache, position_ids)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # Timed runs
        total_tokens = 0
        start_time = time.perf_counter()

        for _ in range(self._num_runs):
            kv_cache = adapter.init_kv_cache(batch_size, seq_len + 10, device)
            with torch.no_grad():
                adapter.forward(hidden_states, kv_cache, position_ids)
            total_tokens += batch_size * seq_len

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start_time
        return total_tokens / elapsed if elapsed > 0 else 0.0

    async def _measure_decode_throughput(
        self,
        adapter: ModelAdapter,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        num_steps: int,
    ) -> float:
        """Measure decode throughput in tokens/sec.

        Args:
            adapter: Loaded model adapter.
            device: Device.
            dtype: Dtype.
            batch_size: Batch size.
            num_steps: Number of decode steps.

        Returns:
            Decode throughput (tokens/sec).
        """
        hidden_size = adapter.get_model_info()["hidden_size"]
        prefill_len = 32  # Short prefill context

        # Prefill first to build KV cache
        hidden_states = torch.randn(
            batch_size, prefill_len, hidden_size,
            device=device, dtype=dtype,
        )
        position_ids = torch.arange(prefill_len, device=device).unsqueeze(0).expand(batch_size, -1)

        kv_cache = adapter.init_kv_cache(batch_size, prefill_len + num_steps + 10, device)

        # Warmup
        warmup_cache = adapter.init_kv_cache(batch_size, prefill_len + num_steps + 10, device)
        with torch.no_grad():
            adapter.forward(hidden_states, warmup_cache, position_ids)

        # Now do actual measurement
        kv_cache = adapter.init_kv_cache(batch_size, prefill_len + num_steps + 10, device)
        with torch.no_grad():
            adapter.forward(hidden_states, kv_cache, position_ids)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # Decode steps
        total_tokens = 0
        start_time = time.perf_counter()

        for step in range(num_steps):
            decode_hidden = torch.randn(
                batch_size, 1, hidden_size,
                device=device, dtype=dtype,
            )
            decode_pos = torch.tensor(
                [[prefill_len + step]], device=device,
            )
            with torch.no_grad():
                adapter.forward(decode_hidden, kv_cache, decode_pos)
            total_tokens += batch_size

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start_time
        return total_tokens / elapsed if elapsed > 0 else 0.0

    def _estimate_model_memory(self, adapter: ModelAdapter) -> float:
        """Estimate total model memory in MB.

        Args:
            adapter: Loaded model adapter.

        Returns:
            Estimated model memory (MB).
        """
        total_bytes = 0

        # Sum parameter sizes
        for layer in adapter._layers:
            for param in layer.parameters():
                total_bytes += param.numel() * param.element_size()

        if adapter._embed_tokens is not None:
            for param in adapter._embed_tokens.parameters():
                total_bytes += param.numel() * param.element_size()

        if adapter._lm_head is not None:
            for param in adapter._lm_head.parameters():
                total_bytes += param.numel() * param.element_size()

        if adapter._norm is not None:
            for param in adapter._norm.parameters():
                total_bytes += param.numel() * param.element_size()

        return total_bytes / (1024 * 1024)
