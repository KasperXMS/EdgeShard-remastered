"""Compare our distributed inference with native HuggingFace inference.

This test verifies that our model loading and forward pass produce
the same results as the reference HuggingFace implementation.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/Qwen2.5-0.5B-Instruct"
prompt = "The capital of France is"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load full model with HF
print("Loading reference model with HuggingFace...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda",
)
model.eval()

# Tokenize
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
print(f"Input: {prompt}")
print(f"Input tokens: {input_ids}")
print(f"Input shape: {input_ids.shape}")

# Generate with HF (greedy, no sampling)
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=10,
        do_sample=False,
        temperature=1.0,
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"\nHF Generated: {generated_text}")

# Now test: run a single forward pass and check logits
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

print(f"\nLogits shape: {logits.shape}")
print(f"Last position logits argmax: {torch.argmax(logits[0, -1, :]).item()}")
print(f"Last position logits top-5: {torch.topk(logits[0, -1, :], 5).indices.tolist()}")
print(f"Last position logits top-5 tokens: {tokenizer.batch_decode(torch.topk(logits[0, -1, :], 5).indices.tolist())}")

# Test decode step
next_token = torch.argmax(logits[0, -1, :], dim=-1).unsqueeze(0).unsqueeze(0)
print(f"\nFirst generated token: {next_token.item()} ({tokenizer.decode([next_token.item()])})")

# Run one decode step
with torch.no_grad():
    outputs2 = model(next_token, past_key_values=outputs.past_key_values, use_cache=True)
    logits2 = outputs2.logits

print(f"Decode logits shape: {logits2.shape}")
print(f"Decode argmax: {torch.argmax(logits2[0, -1, :]).item()} ({tokenizer.decode([torch.argmax(logits2[0, -1, :]).item()])})")

print("\n--- Now testing our implementation ---")

from edgeshard.runtime.adapters.qwen2 import Qwen2Adapter
from edgeshard.runtime.shard import ModelShard
from edgeshard.runtime.pipeline import PipelineOrchestrator, ShardEndpoint
from edgeshard.runtime.pipeline_decoder import PipelineDecoder, GenerationConfig
import asyncio

async def test_ours():
    # Single shard with all layers
    adapter = Qwen2Adapter()
    adapter.load(model_path, 0, 24, torch.float16, torch.device("cuda"))
    shard = ModelShard("shard-0", adapter, is_first_shard=True, is_last_shard=True)

    pipeline = PipelineOrchestrator([
        ShardEndpoint(shard, None, None, True),
    ])

    decoder = PipelineDecoder(pipeline, tokenizer)
    config = GenerationConfig(max_new_tokens=10, eos_token_id=tokenizer.eos_token_id)
    result = await decoder.generate(prompt, config)

    print(f"\nOur Generated: {result.text}")
    print(f"Tokens: {result.num_tokens}")

    # Also check prefill logits
    from edgeshard.common.identifiers import SessionId
    session_id = SessionId.generate()
    pipeline.create_session(session_id, batch_size=1, max_seq_len=100)
    our_logits = await pipeline.prefill(session_id, input_ids=input_ids)

    print(f"\nOur logits shape: {our_logits.shape}")
    print(f"Our last position argmax: {torch.argmax(our_logits[0, -1, :]).item()}")
    print(f"Our last position top-5: {torch.topk(our_logits[0, -1, :], 5).indices.tolist()}")
    print(f"Our last position top-5 tokens: {tokenizer.batch_decode(torch.topk(our_logits[0, -1, :], 5).indices.tolist())}")

    # Compare logits
    hf_logits = logits[0, -1, :]
    our_logits_last = our_logits[0, -1, :]
    diff = (hf_logits - our_logits_last).abs()
    print(f"\nLogits diff: mean={diff.mean().item():.6f}, max={diff.max().item():.6f}")

    pipeline.release_session(session_id)
    await pipeline.close()
    adapter.unload()

asyncio.run(test_ours())
