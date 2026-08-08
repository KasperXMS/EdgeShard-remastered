"""Example: Single-shard text generation (M2).

This example demonstrates:
1. Loading a complete model into a single ModelShard
2. Using GreedyDecoder for text generation
3. KV cache memory tracking

Usage:
    python examples/m2_single_shard_generation.py

Requirements:
    - CUDA GPU
    - A local model in models/Qwen2.5-0.5B-Instruct/
    - transformers library
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import torch

from edgeshard.common.identifiers import SessionId
from edgeshard.runtime.adapters.qwen2 import Qwen2Adapter
from edgeshard.runtime.decoder import GenerationConfig, GreedyDecoder
from edgeshard.runtime.kv_cache import get_kv_cache_manager
from edgeshard.runtime.shard import ModelShard


async def main() -> None:
    """Run single-shard text generation example."""
    # Check CUDA
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Model path
    model_path = Path("models/Qwen2.5-0.5B-Instruct")
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please download a model and place it in models/")
        return

    # Load tokenizer
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True
        )
    except ImportError:
        print("Error: transformers not installed")
        return

    # Load model adapter
    print(f"\nLoading model from {model_path}...")
    adapter = Qwen2Adapter()
    adapter.load(
        model_path=str(model_path),
        layer_start=0,
        layer_end=adapter._config.get("num_hidden_layers", 24),
        dtype=torch.float16,
        device=torch.device("cuda"),
    )

    model_info = adapter.get_model_info()
    print(f"Model type: {model_info['model_type']}")
    print(f"Loaded layers: {model_info['loaded_layers']}")
    print(f"Hidden size: {model_info['hidden_size']}")
    print(f"Vocab size: {model_info['vocab_size']}")

    # Create complete shard (first + last)
    shard = ModelShard(
        shard_id="example-shard",
        adapter=adapter,
        is_first_shard=True,
        is_last_shard=True,
    )

    # Create decoder
    decoder = GreedyDecoder(shard, tokenizer)

    # Generate text
    prompts = [
        "The capital of France is",
        "In a galaxy far, far away",
        "Once upon a time",
    ]

    for prompt in prompts:
        print(f"\n{'=' * 60}")
        print(f"Prompt: {prompt}")
        print("-" * 60)

        config = GenerationConfig(
            max_new_tokens=50,
            eos_token_id=tokenizer.eos_token_id,
        )

        result = await decoder.generate(prompt, config)

        print(f"Generated: {result.text}")
        print(f"Tokens: {result.num_tokens}")
        print(f"Finished: {result.finished}")

    # Print KV cache stats
    print(f"\n{'=' * 60}")
    print("KV Cache Statistics:")
    kv_manager = get_kv_cache_manager()
    kv_manager.print_debug_info()

    # Cleanup
    adapter.unload()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
