"""Test M4: distributed inference data plane.

This test verifies:
1. Tensor serialization/deserialization
2. gRPC tensor transport (local loopback)
3. Pipeline orchestration with multiple shards
4. Distributed greedy decoding correctness
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch

from edgeshard.common.identifiers import SessionId
from edgeshard.runtime.adapters.qwen2 import Qwen2Adapter
from edgeshard.runtime.pipeline import PipelineOrchestrator, ShardEndpoint
from edgeshard.runtime.pipeline_decoder import PipelineDecoder, GenerationConfig
from edgeshard.runtime.shard import ModelShard
from edgeshard.transport.grpc_transport import (
    GrpcTensorTransport,
    ShardServer,
    tensor_to_message,
    message_to_tensor,
)


def test_tensor_serialization():
    """Test tensor serialization and deserialization."""
    # Create a test tensor
    tensor = torch.randn(2, 3, 4, dtype=torch.float16)
    session_id = "test-session"

    # Serialize
    message = tensor_to_message(tensor, session_id)
    assert message.session_id == session_id
    assert list(message.shape) == [2, 3, 4]
    assert message.dtype == "float16"

    # Deserialize
    device = torch.device("cpu")
    recovered = message_to_tensor(message, device)

    assert recovered.shape == tensor.shape
    assert recovered.dtype == tensor.dtype
    assert torch.allclose(recovered, tensor, rtol=1e-3, atol=1e-3)


@pytest.mark.asyncio
async def test_grpc_tensor_transport_local():
    """Test gRPC tensor transport with local loopback."""
    # Start a shard server
    server = ShardServer(shard_id="test-server", port=50200)
    await server.start()

    try:
        # Create transport
        transport = GrpcTensorTransport(
            shard_id="test-client",
            shard_addresses={"test-server": "localhost:50200"},
        )

        # Send a tensor
        tensor = torch.randn(2, 4, 8, dtype=torch.float16)
        session_id = "test-session-123"

        await transport.send(tensor, "test-server", session_id)

        # Receive via server's buffer
        servicer = server.get_servicer()
        assert session_id in servicer._recv_buffers

        received = servicer._recv_buffers[session_id]
        assert received.shape == tensor.shape
        assert torch.allclose(received, tensor, rtol=1e-3, atol=1e-3)

        await transport.close()

    finally:
        await server.stop()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
async def test_pipeline_orchestrator_single_process():
    """Test pipeline orchestrator with multiple shards in single process."""
    model_path = Path("models/Qwen2.5-0.5B-Instruct")
    if not model_path.exists():
        pytest.skip(f"Model not found at {model_path}")

    # Load model info
    adapter_full = Qwen2Adapter()
    adapter_full.load(
        model_path=str(model_path),
        layer_start=0,
        layer_end=24,
        dtype=torch.float16,
        device=torch.device("cuda"),
    )
    num_layers = adapter_full.get_model_info()["loaded_layers"]
    adapter_full.unload()

    # Create two shards
    adapter0 = Qwen2Adapter()
    adapter0.load(
        model_path=str(model_path),
        layer_start=0,
        layer_end=num_layers // 2,
        dtype=torch.float16,
        device=torch.device("cuda"),
    )

    adapter1 = Qwen2Adapter()
    adapter1.load(
        model_path=str(model_path),
        layer_start=num_layers // 2,
        layer_end=num_layers,
        dtype=torch.float16,
        device=torch.device("cuda"),
    )

    shard0 = ModelShard(
        shard_id="shard-0",
        adapter=adapter0,
        is_first_shard=True,
        is_last_shard=False,
    )

    shard1 = ModelShard(
        shard_id="shard-1",
        adapter=adapter1,
        is_first_shard=False,
        is_last_shard=True,
    )

    # Create pipeline (local, no transport needed)
    pipeline = PipelineOrchestrator([
        ShardEndpoint(shard0, None, None, True),
        ShardEndpoint(shard1, None, None, True),
    ])

    # Create session
    session_id = SessionId.generate()
    pipeline.create_session(session_id, batch_size=1, max_seq_len=64)

    # Test prefill
    input_ids = torch.tensor([[1, 2, 3]], device=torch.device("cuda"))
    logits = await pipeline.prefill(session_id, input_ids=input_ids)

    assert logits.shape[0] == 1
    assert logits.shape[1] == 3
    assert logits.shape[2] > 0  # vocab size

    # Test decode
    logits = await pipeline.decode(session_id, token_id=100)
    assert logits.shape[0] == 1
    assert logits.shape[1] == 1

    # Cleanup
    pipeline.release_session(session_id)
    await pipeline.close()
    adapter0.unload()
    adapter1.unload()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
async def test_pipeline_decoder_correctness():
    """Test that pipeline decoder matches single-shard greedy decoding."""
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

    # Create two-shard pipeline
    adapter0 = Qwen2Adapter()
    adapter0.load(
        model_path=str(model_path),
        layer_start=0,
        layer_end=12,
        dtype=torch.float16,
        device=torch.device("cuda"),
    )

    adapter1 = Qwen2Adapter()
    adapter1.load(
        model_path=str(model_path),
        layer_start=12,
        layer_end=24,
        dtype=torch.float16,
        device=torch.device("cuda"),
    )

    shard0 = ModelShard(
        shard_id="shard-0",
        adapter=adapter0,
        is_first_shard=True,
        is_last_shard=False,
    )

    shard1 = ModelShard(
        shard_id="shard-1",
        adapter=adapter1,
        is_first_shard=False,
        is_last_shard=True,
    )

    pipeline = PipelineOrchestrator([
        ShardEndpoint(shard0, None, None, True),
        ShardEndpoint(shard1, None, None, True),
    ])

    # Create pipeline decoder
    decoder = PipelineDecoder(pipeline, tokenizer)

    # Generate with pipeline
    prompt = "The capital of France is"
    config = GenerationConfig(
        max_new_tokens=10,
        eos_token_id=tokenizer.eos_token_id,
    )
    pipeline_result = await decoder.generate(prompt, config)

    # Generate with reference (single model)
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
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    ref_text = tokenizer.decode(ref_output[0], skip_special_tokens=True)

    # Compare
    print(f"\nPipeline output: {pipeline_result.text}")
    print(f"Reference output: {ref_text}")

    # Token IDs should match
    prompt_len = input_ids.shape[1]
    ref_tokens = ref_output[0].tolist()[prompt_len:]
    pipeline_tokens = pipeline_result.token_ids

    assert pipeline_tokens == ref_tokens, (
        f"Token mismatch:\n"
        f"  Pipeline: {pipeline_tokens}\n"
        f"  Reference: {ref_tokens}"
    )

    # Cleanup
    await pipeline.close()
    adapter0.unload()
    adapter1.unload()
    del ref_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("Running M4 distributed inference tests...")
    print("(These tests require a local model in models/ directory)")

    test_tensor_serialization()
    print("[PASS] test_tensor_serialization")

    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        asyncio.run(test_grpc_tensor_transport_local())
        print("[PASS] test_grpc_tensor_transport_local")
        asyncio.run(test_pipeline_orchestrator_single_process())
        print("[PASS] test_pipeline_orchestrator_single_process")
        asyncio.run(test_pipeline_decoder_correctness())
        print("[PASS] test_pipeline_decoder_correctness")
    else:
        print("CUDA not available, skipping GPU tests")
