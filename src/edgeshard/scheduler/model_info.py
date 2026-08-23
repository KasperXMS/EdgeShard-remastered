"""Model information for the scheduler.

The scheduler needs to know model architecture metadata (num_layers,
hidden_size, etc.) to make placement decisions, but it does NOT load
model weights. Instead, it derives this info from:

1. HuggingFace config.json (preferred — fast, no GPU needed)
2. Profiling data (fallback — approximate)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from edgeshard.common.logging import get_logger
from edgeshard.scheduler.snapshot import ProfileEntry

logger = get_logger(__name__)


# Mapping from torch dtype strings (as found in config.json torch_dtype)
# to bytes per parameter.
_TORCH_DTYPE_TO_BYTES: dict[str, int] = {
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float32": 4,
    "fp32": 4,
    "float64": 8,
    "fp64": 8,
    "int8": 1,
    "uint8": 1,
    "int4": 1,  # approximate (packed)
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.float64": 8,
}


def _torch_dtype_to_bytes(dtype_str: str) -> int:
    """Convert a torch dtype string (from config.json) to bytes per parameter."""
    return _TORCH_DTYPE_TO_BYTES.get(dtype_str, 2)  # default to 2 (float16)


@dataclass
class ModelSchedulingInfo:
    """Model metadata needed by the scheduler.

    Attributes:
        num_layers: Total number of transformer decoder layers.
        hidden_size: Hidden dimension size.
        num_attention_heads: Number of attention heads.
        num_kv_heads: Number of KV heads (for GQA). May equal num_attention_heads.
        intermediate_size: FFN intermediate dimension.
        vocab_size: Vocabulary size.
        per_layer_memory_mb: Estimated memory per layer in MB (model weights only).
        kv_cache_per_token_mb: KV cache memory per token in MB (from profiling).
        total_model_memory_mb: Total model weight memory in MB (from profiling).
    """

    num_layers: int
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    per_layer_memory_mb: float = 0.0
    kv_cache_per_token_mb: float = 0.0
    total_model_memory_mb: float = 0.0


# Well-known model configurations for offline planning.
# Used when config.json is not available and no profile data exists.
# Source: HuggingFace model cards.
_KNOWN_MODELS: dict[str, dict] = {
    # Qwen2.5 family
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "num_layers": 24, "hidden_size": 896, "num_attention_heads": 14,
        "num_kv_heads": 2, "intermediate_size": 4864, "vocab_size": 151936,
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "num_layers": 28, "hidden_size": 1536, "num_attention_heads": 12,
        "num_kv_heads": 2, "intermediate_size": 8960, "vocab_size": 151936,
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "num_layers": 36, "hidden_size": 2048, "num_attention_heads": 16,
        "num_kv_heads": 2, "intermediate_size": 11008, "vocab_size": 151936,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "num_layers": 28, "hidden_size": 3584, "num_attention_heads": 28,
        "num_kv_heads": 4, "intermediate_size": 18944, "vocab_size": 152064,
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "num_layers": 48, "hidden_size": 5120, "num_attention_heads": 40,
        "num_kv_heads": 8, "intermediate_size": 13824, "vocab_size": 152064,
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "num_layers": 80, "hidden_size": 8192, "num_attention_heads": 64,
        "num_kv_heads": 8, "intermediate_size": 29568, "vocab_size": 152064,
    },
    # Llama2 family
    "meta-llama/Llama-2-7b-hf": {
        "num_layers": 32, "hidden_size": 4096, "num_attention_heads": 32,
        "num_kv_heads": 32, "intermediate_size": 11008, "vocab_size": 32000,
    },
    "meta-llama/Llama-2-13b-hf": {
        "num_layers": 40, "hidden_size": 5120, "num_attention_heads": 40,
        "num_kv_heads": 40, "intermediate_size": 13824, "vocab_size": 32000,
    },
    "meta-llama/Llama-2-70b-hf": {
        "num_layers": 80, "hidden_size": 8192, "num_attention_heads": 64,
        "num_kv_heads": 8, "intermediate_size": 28672, "vocab_size": 32000,
    },
}


def load_model_info(
    model_path: str,
    profile: ProfileEntry | None = None,
    dtype_bytes: int = 2,
) -> ModelSchedulingInfo:
    """Load model scheduling info from config.json or profile data.

    Resolution order:
    1. config.json from local model directory (fast, no GPU needed)
    2. Well-known model registry (for popular HuggingFace models)
    3. Profile data (fallback — approximate)

    Args:
        model_path: HuggingFace model ID or local path.
        profile: Optional profiling data to fill in memory estimates.
        dtype_bytes: Bytes per parameter (2 for float16/bfloat16, 4 for float32).

    Returns:
        ModelSchedulingInfo with all available metadata.

    Raises:
        SchedulerError: If num_layers cannot be determined.
    """
    from edgeshard.common.errors import SchedulerError

    info = _try_load_from_config(model_path, dtype_bytes)

    if info is None:
        info = _try_known_model(model_path, dtype_bytes)

    if info is None:
        # Fall back to profile-only estimation
        if profile is not None:
            info = _estimate_from_profile(profile, dtype_bytes)
        else:
            raise SchedulerError(
                f"Cannot determine model info for '{model_path}': "
                "config.json not found, model not in known registry, "
                "and no profile data available."
            )

    # Supplement with profile data if available
    if profile is not None and info.kv_cache_per_token_mb == 0.0:
        info.kv_cache_per_token_mb = profile.kv_cache_per_token_mb

    if profile is not None and info.total_model_memory_mb == 0.0:
        # Estimate from profile's total model memory
        info.total_model_memory_mb = (
            profile.layer_forward_ms > 0
        ) and _estimate_total_memory(profile) or 0.0

    # Compute per-layer memory if we have totals
    if info.num_layers > 0 and info.total_model_memory_mb > 0.0:
        info.per_layer_memory_mb = info.total_model_memory_mb / info.num_layers
    elif info.num_layers > 0 and info.hidden_size > 0:
        # Estimate from architecture: GQA-aware attention + FFN
        if info.num_attention_heads > 0 and info.num_kv_heads > 0:
            head_dim = info.hidden_size / info.num_attention_heads
            kv_dim = info.num_kv_heads * head_dim
            attn_params = (
                2 * info.hidden_size * info.hidden_size  # Q + O
                + 2 * info.hidden_size * kv_dim           # K + V
            )
        else:
            attn_params = 4 * info.hidden_size * info.hidden_size
        if info.intermediate_size > 0:
            ffn_params = 3 * info.hidden_size * info.intermediate_size
        else:
            ffn_params = 4 * info.hidden_size * info.hidden_size
        layer_params = attn_params + ffn_params
        info.per_layer_memory_mb = (layer_params * dtype_bytes) / (1024 * 1024)

    # Compute KV cache per token if not set (from profile)
    if (
        info.kv_cache_per_token_mb == 0.0
        and info.num_layers > 0
        and info.num_kv_heads > 0
        and info.num_attention_heads > 0
        and info.hidden_size > 0
    ):
        head_dim = info.hidden_size / info.num_attention_heads
        kv_per_token_per_layer = 2 * info.num_kv_heads * head_dim * dtype_bytes
        info.kv_cache_per_token_mb = (
            info.num_layers * kv_per_token_per_layer / (1024 * 1024)
        )

    if info.num_layers <= 0:
        raise SchedulerError(
            f"Cannot determine num_layers for model '{model_path}'. "
            "Ensure config.json exists or provide profile data."
        )

    return info


def _try_load_from_config(
    model_path: str, dtype_bytes: int
) -> ModelSchedulingInfo | None:
    """Try to load model info from config.json.

    Supports both local paths and HuggingFace model IDs.
    Returns None if config.json cannot be found/loaded.

    Reads torch_dtype from config.json and logs if it differs from
    the target dtype — but memory estimation uses the TARGET dtype
    (from service spec) since that's what determines GPU memory.
    """
    config_path = _resolve_config_path(model_path)
    if config_path is None:
        return None

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load config.json from {model_path}: {e}")
        return None

    num_layers = config.get("num_hidden_layers", 0)
    if num_layers <= 0:
        return None

    # Log actual model weight dtype for debugging
    torch_dtype_str = config.get("torch_dtype", "")
    if torch_dtype_str:
        saved_dtype_bytes = _torch_dtype_to_bytes(torch_dtype_str)
        if saved_dtype_bytes != dtype_bytes:
            logger.info(
                f"Model saved as {torch_dtype_str} ({saved_dtype_bytes} bytes/param), "
                f"serving in target dtype ({dtype_bytes} bytes/param). "
                f"GPU memory estimated using target dtype."
            )

    # Return raw info — memory computation uses target dtype in load_model_info
    return ModelSchedulingInfo(
        num_layers=num_layers,
        hidden_size=config.get("hidden_size", 0),
        num_attention_heads=config.get("num_attention_heads", 0),
        num_kv_heads=config.get("num_key_value_heads", config.get("num_attention_heads", 0)),
        intermediate_size=config.get("intermediate_size", 0),
        vocab_size=config.get("vocab_size", 0),
    )


def _resolve_config_path(model_path: str) -> Path | None:
    """Resolve config.json path from a model path or HF ID.

    Checks:
    1. Direct path: model_path/config.json
    2. models/ directory: models/<model_name>/config.json
    """
    # Direct local path
    direct = Path(model_path) / "config.json"
    if direct.exists():
        return direct

    # Check under models/ directory
    # For HF IDs like "Qwen/Qwen2.5-0.5B-Instruct", try models/Qwen2.5-0.5B-Instruct
    model_name = model_path.split("/")[-1] if "/" in model_path else model_path
    under_models = Path("models") / model_name / "config.json"
    if under_models.exists():
        return under_models

    return None


def _try_known_model(model_path: str, dtype_bytes: int) -> ModelSchedulingInfo | None:
    """Look up model info from the built-in registry of well-known models.

    Matches by exact name or by partial match (e.g. "Qwen2.5-7B" matches
    "Qwen/Qwen2.5-7B-Instruct").
    """
    # Exact match
    if model_path in _KNOWN_MODELS:
        cfg = _KNOWN_MODELS[model_path]
        logger.info(f"Using known model config for '{model_path}'")
        return _build_info_from_config(cfg, dtype_bytes)

    # Partial match — check if any known model key is contained in model_path
    model_lower = model_path.lower()
    for known_name, cfg in _KNOWN_MODELS.items():
        if known_name.lower() in model_lower or model_lower in known_name.lower():
            logger.info(
                f"Using known model config for '{model_path}' "
                f"(matched '{known_name}')"
            )
            return _build_info_from_config(cfg, dtype_bytes)

    return None


def _build_info_from_config(cfg: dict, dtype_bytes: int) -> ModelSchedulingInfo:
    """Build ModelSchedulingInfo from a config dict."""
    info = ModelSchedulingInfo(
        num_layers=cfg.get("num_layers", 0),
        hidden_size=cfg.get("hidden_size", 0),
        num_attention_heads=cfg.get("num_attention_heads", 0),
        num_kv_heads=cfg.get("num_kv_heads", cfg.get("num_attention_heads", 0)),
        intermediate_size=cfg.get("intermediate_size", 0),
        vocab_size=cfg.get("vocab_size", 0),
    )

    # Estimate per-layer memory from architecture (GQA-aware)
    if info.hidden_size > 0:
        # Attention parameters (GQA-aware):
        #   Q projection: hidden_size * hidden_size
        #   K projection: hidden_size * (num_kv_heads * head_dim)
        #   V projection: hidden_size * (num_kv_heads * head_dim)
        #   O projection: hidden_size * hidden_size
        # For standard MHA (num_kv_heads == num_attention_heads): 4 * hidden^2
        # For GQA (num_kv_heads < num_attention_heads): less than 4 * hidden^2
        if info.num_attention_heads > 0 and info.num_kv_heads > 0:
            head_dim = info.hidden_size / info.num_attention_heads
            kv_dim = info.num_kv_heads * head_dim
            attn_params = (
                2 * info.hidden_size * info.hidden_size  # Q + O
                + 2 * info.hidden_size * kv_dim           # K + V
            )
        else:
            # Fallback: assume standard MHA
            attn_params = 4 * info.hidden_size * info.hidden_size

        # FFN parameters: gate + up + down projections
        if info.intermediate_size > 0:
            ffn_params = 3 * info.hidden_size * info.intermediate_size
        else:
            ffn_params = 4 * info.hidden_size * info.hidden_size

        layer_params = attn_params + ffn_params
        info.per_layer_memory_mb = (layer_params * dtype_bytes) / (1024 * 1024)
        info.total_model_memory_mb = info.per_layer_memory_mb * info.num_layers

    # Estimate KV cache per token from architecture (GQA-aware)
    if (
        info.num_layers > 0
        and info.num_kv_heads > 0
        and info.hidden_size > 0
        and info.num_attention_heads > 0
    ):
        head_dim = info.hidden_size / info.num_attention_heads
        # Per token per layer: K + V = 2 * num_kv_heads * head_dim * dtype_bytes
        kv_per_token_per_layer = (
            2 * info.num_kv_heads * head_dim * dtype_bytes
        )
        # Total across all layers
        info.kv_cache_per_token_mb = (
            info.num_layers * kv_per_token_per_layer / (1024 * 1024)
        )

    return info


def _estimate_from_profile(
    profile: ProfileEntry, dtype_bytes: int
) -> ModelSchedulingInfo:
    """Estimate model info purely from profiling data.

    This is a rough fallback when config.json is unavailable.
    """
    # We can't reliably determine num_layers from profile alone,
    # but we can use reasonable defaults based on common models.
    # The profile should have been run on this model, so the layer_forward_ms
    # reflects actual per-layer cost.
    #
    # For now, return a minimal info that the scheduler can work with.
    # The planner will need num_layers from another source.
    return ModelSchedulingInfo(
        num_layers=0,  # Must be filled in by caller
        kv_cache_per_token_mb=profile.kv_cache_per_token_mb,
    )


def _estimate_total_memory(profile: ProfileEntry) -> float:
    """Estimate total model memory from profile data.

    Uses the profile's aggregate measurements.
    """
    # If the profile has total_model_memory_mb, use it directly
    # (ProfileEntry doesn't have this field, but ProfileResult does)
    # For now, estimate from layer_forward_ms as a proxy
    return 0.0


def estimate_activation_size_mb(
    model_info: ModelSchedulingInfo,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> float:
    """Estimate the size of activations transferred between shards.

    When a shard boundary splits the model, the hidden states must be
    transferred between workers. This estimates that transfer size.

    Args:
        model_info: Model architecture info.
        seq_len: Sequence length.
        batch_size: Batch size.
        dtype_bytes: Bytes per element.

    Returns:
        Activation size in MB.
    """
    if model_info.hidden_size > 0:
        # hidden_states: [batch_size, seq_len, hidden_size]
        return (batch_size * seq_len * model_info.hidden_size * dtype_bytes) / (
            1024 * 1024
        )
    # Fallback: rough estimate based on typical LLM sizes
    return 10.0  # 10 MB default
