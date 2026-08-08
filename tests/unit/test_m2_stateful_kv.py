"""Test M2: stateful KV runtime correctness.

This test verifies that:
1. KV cache accumulates correctly across multiple decode steps
2. Greedy decoding produces the same output as Hugging Face reference
3. Session state is properly maintained and cleaned up
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch

from edgeshard.common.identifiers import SessionId
from edgeshard.runtime.adapters.qwen2 import Qwen2Adapter
from edgeshard.runtime.decoder import GenerationConfig, GreedyDecoder
from edgeshard.runtime.kv_cache import get_kv_cache_manager
from edgeshard.runtime.shard import ModelShard


@pytest.mark.asyncio
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
async def test_kv_cache_accumulation():
    """Test that KV cache grows correctly with each decode step."""
    adapter = Qwen2Adapter()

    model_path = Path("models/Qwen2.5-0.5B-Instruct")
    if not model_path.exists():
        pytest.skip(f"Model not found at {model_path}")

    # Load all layers for complete model
    adapter.load(
        model_path=str(model_path),
        layer_start=0,
        layer_end=adapter._config.get("num_hidden_layers", 24),
        dtype=torch.float16,
        device=torch.device("cuda"),
    )

    shard = ModelShard(
        shard_id="test-kv-accum",
        adapter=adapter,
        is_first_shard=True,
        is_last_shard=True,
    )

    session_id = SessionId.generate()
    shard.create_session(session_id, batch_size=1, max_seq_len=128)

    # Prefill with 3 tokens
    input_ids = torch.tensor([[1, 2, 3]], device=torch.device("cuda"))
    await shard.prefill(session_id, input_ids=input_ids)

    # Check session state
    session = shard._sessions[session_id]
    assert session.sequence_length == 3
    assert len(session.kv_cache) > 0

    # Decode 5 more tokens
    for i in range(5):
        await shard.decode(session_id, token_id=100 + i)

    # Verify sequence length grew
    assert session.sequence_length == 8  # 3 + 5

    shard.release_session(session_id)
    adapter.unload()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
async def test_greedy_decoding_correctness():
    """Test that greedy decoding matches Hugging Face reference."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    model_path = Path("models/Qwen2.5-0.5B-Instruct")
    if not model_path.exists():
        pytest.skip(f"Model not found at {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True
    )

    # Load model with our adapter
    adapter = Qwen2Adapter()
    adapter.load(
        model_path=str(model_path),
        layer_start=0,
        layer_end=adapter._config.get("num_hidden_layers", 24),
        dtype=torch.float16,
        device=torch.device("cuda"),
    )

    shard = ModelShard(
        shard_id="test-greedy",
        adapter=adapter,
        is_first_shard=True,
        is_last_shard=True,
    )

    # Create decoder
    decoder = GreedyDecoder(shard, tokenizer)

    # Generate with our decoder
    prompt = "The capital of France is"
    config = GenerationConfig(
        max_new_tokens=10,
        eos_token_id=tokenizer.eos_token_id,
    )
    our_result = await decoder.generate(prompt, config)

    # Generate with Hugging Face reference
    ref_model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    ref_model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(torch.device("cuda"))
    ref_output = ref_model.generate(
        input_ids,
        max_new_tokens=10,
        do_sample=False,  # greedy
        eos_token_id=tokenizer.eos_token_id,
    )
    ref_text = tokenizer.decode(ref_output[0], skip_special_tokens=True)

    # Compare outputs
    print(f"\nOur output:  {our_result.text}")
    print(f"Ref output:  {ref_text}")
    print(f"Our tokens:  {our_result.token_ids}")
    print(f"Ref tokens:  {ref_output[0].tolist()}")

    # For exact match, token IDs should be identical
    ref_tokens = ref_output[0].tolist()
    our_tokens = our_result.token_ids

    # The prompt tokens are the same, so compare generated tokens
    prompt_len = input_ids.shape[1]
    ref_generated = ref_tokens[prompt_len:]
    our_generated = our_tokens

    assert our_generated == ref_generated, (
        f"Token mismatch:\n"
        f"  Our: {our_generated}\n"
        f"  Ref: {ref_generated}"
    )

    # Cleanup
    adapter.unload()
    del ref_model
    torch.cuda.empty_cache()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
async def test_session_cleanup():
    """Test that session cleanup properly frees KV cache."""
    adapter = Qwen2Adapter()

    model_path = Path("models/Qwen2.5-0.5B-Instruct")
    if not model_path.exists():
        pytest.skip(f"Model not found at {model_path}")

    adapter.load(
        model_path=str(model_path),
        layer_start=0,
        layer_end=adapter._config.get("num_hidden_layers", 24),
        dtype=torch.float16,
        device=torch.device("cuda"),
    )

    shard = ModelShard(
        shard_id="test-cleanup",
        adapter=adapter,
        is_first_shard=True,
        is_last_shard=True,
    )

    # Create multiple sessions
    sessions = []
    for i in range(3):
        sid = SessionId.generate()
        shard.create_session(sid, batch_size=1, max_seq_len=64)
        sessions.append(sid)

    assert len(shard._sessions) == 3

    # Release all sessions
    for sid in sessions:
        shard.release_session(sid)

    assert len(shard._sessions) == 0

    adapter.unload()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
async def test_kv_cache_memory_tracking():
    """Test KV cache memory tracking."""
    from edgeshard.runtime.kv_cache import KVCacheManager

    manager = KVCacheManager()

    # Create dummy KV cache
    kv_cache = [
        (torch.randn(1, 4, 10, 64, device="cuda"), torch.randn(1, 4, 10, 64, device="cuda"))
        for _ in range(4)  # 4 layers
    ]

    session_id = SessionId.generate()
    manager.register_session(session_id, kv_cache, torch.device("cuda"))

    stats = manager.get_stats()
    assert stats.num_sessions == 1
    assert stats.total_memory_mb > 0

    manager.unregister_session(session_id)
    stats = manager.get_stats()
    assert stats.num_sessions == 0
    assert stats.total_memory_mb == 0


if __name__ == "__main__":
    print("Running M2 stateful KV runtime tests...")
    print("(These tests require a local model in models/ directory)")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU tests")
    else:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        asyncio.run(test_kv_cache_accumulation())
        print("[PASS] test_kv_cache_accumulation")
        asyncio.run(test_session_cleanup())
        print("[PASS] test_session_cleanup")
        asyncio.run(test_kv_cache_memory_tracking())
        print("[PASS] test_kv_cache_memory_tracking")
        asyncio.run(test_greedy_decoding_correctness())
        print("[PASS] test_greedy_decoding_correctness")
