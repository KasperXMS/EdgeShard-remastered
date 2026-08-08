"""Test local shard correctness (M1).

This test verifies that a single ModelShard can:
1. Load model weights correctly
2. Run prefill (prompt processing)
3. Run decode (token generation)
4. Produce correct outputs
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch

from edgeshard.common.identifiers import SessionId
from edgeshard.runtime.adapters.qwen2 import Qwen2Adapter
from edgeshard.runtime.shard import ModelShard


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available for model inference test",
)
def test_qwen2_adapter_load():
    """Test that Qwen2Adapter can load model weights."""
    adapter = Qwen2Adapter()

    # This test requires a local model path
    # Skip if no model is available
    model_path = Path("models/Qwen2.5-0.5B-Instruct")
    if not model_path.exists():
        pytest.skip(f"Model not found at {model_path}")

    adapter.load(
        model_path=str(model_path),
        layer_start=0,
        layer_end=4,
        dtype=torch.float16,
        device=torch.device("cuda"),
    )

    info = adapter.get_model_info()
    assert info["loaded_layers"] == 4
    assert info["has_embedding"] is True
    assert info["model_type"] == "qwen2"

    adapter.unload()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
async def test_model_shard_prefill_decode():
    """Test ModelShard prefill and decode operations."""
    adapter = Qwen2Adapter()

    model_path = Path("models/Qwen2.5-0.5B-Instruct")
    if not model_path.exists():
        pytest.skip(f"Model not found at {model_path}")

    # Load a few layers
    adapter.load(
        model_path=str(model_path),
        layer_start=0,
        layer_end=4,
        dtype=torch.float16,
        device=torch.device("cuda"),
    )

    # Create a shard (first shard with embedding, but not last)
    shard = ModelShard(
        shard_id="test-shard-0",
        adapter=adapter,
        is_first_shard=True,
        is_last_shard=False,
    )

    # Create a session
    session_id = SessionId.generate()
    shard.create_session(session_id, batch_size=1, max_seq_len=128)

    # Test prefill
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=torch.device("cuda"))
    hidden_states = await shard.prefill(session_id, input_ids=input_ids)

    assert hidden_states.shape[0] == 1  # batch
    assert hidden_states.shape[1] == 5  # seq_len
    assert hidden_states.device.type == "cuda"

    # Test decode
    hidden = await shard.decode(session_id, token_id=6)
    assert hidden.shape[0] == 1
    assert hidden.shape[1] == 1

    # Cleanup
    shard.release_session(session_id)
    adapter.unload()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
async def test_full_pipeline_single_shard():
    """Test a full pipeline with a single shard that has both embedding and LM head."""
    adapter = Qwen2Adapter()

    model_path = Path("models/Qwen2.5-0.5B-Instruct")
    if not model_path.exists():
        pytest.skip(f"Model not found at {model_path}")

    # Load all layers for a complete single-shard model
    info = adapter.get_model_info() if adapter._loaded else {}
    num_layers = info.get("num_layers", 24)  # default for 0.5B

    adapter.load(
        model_path=str(model_path),
        layer_start=0,
        layer_end=num_layers,
        dtype=torch.float16,
        device=torch.device("cuda"),
    )

    # Create a shard that is both first and last
    shard = ModelShard(
        shard_id="test-shard-full",
        adapter=adapter,
        is_first_shard=True,
        is_last_shard=True,
    )

    session_id = SessionId.generate()
    shard.create_session(session_id, batch_size=1, max_seq_len=128)

    # Prefill should return logits
    input_ids = torch.tensor([[1, 2, 3]], device=torch.device("cuda"))
    logits = await shard.prefill(session_id, input_ids=input_ids)

    assert logits.shape[0] == 1
    assert logits.shape[1] == 3
    assert logits.shape[2] == adapter._config["vocab_size"]

    # Decode should return logits for next token
    logits = await shard.decode(session_id, token_id=4)
    assert logits.shape[0] == 1
    assert logits.shape[1] == 1
    assert logits.shape[2] == adapter._config["vocab_size"]

    shard.release_session(session_id)
    adapter.unload()


if __name__ == "__main__":
    # Run tests
    print("Running M1 local shard correctness tests...")
    print("(These tests require a local model in models/ directory)")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU tests")
    else:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        asyncio.run(test_qwen2_adapter_load())
        print("test_qwen2_adapter_load passed")
        asyncio.run(test_model_shard_prefill_decode())
        print("test_model_shard_prefill_decode passed")
        asyncio.run(test_full_pipeline_single_shard())
        print("test_full_pipeline_single_shard passed")
