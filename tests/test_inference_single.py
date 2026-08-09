# test_inference.py
import asyncio
import logging
import torch
from pathlib import Path
from transformers import AutoTokenizer

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

from edgeshard.runtime.adapters.qwen2 import Qwen2Adapter
from edgeshard.runtime.shard import ModelShard
from edgeshard.runtime.pipeline import PipelineOrchestrator, ShardEndpoint
from edgeshard.runtime.pipeline_decoder import PipelineDecoder, GenerationConfig

async def main():
    model_path = "models/Qwen2.5-0.5B-Instruct"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Create shard 0 (local)
    adapter0 = Qwen2Adapter()
    adapter0.load(model_path, 0, 12, torch.float16, torch.device("cuda"))
    shard0 = ModelShard("shard-0", adapter0, is_first_shard=True, is_last_shard=False)
    
    # Create shard 1 (local)
    adapter1 = Qwen2Adapter()
    adapter1.load(model_path, 12, 24, torch.float16, torch.device("cuda"))
    shard1 = ModelShard("shard-1", adapter1, is_first_shard=False, is_last_shard=True)
    
    # Create pipeline
    pipeline = PipelineOrchestrator([
        ShardEndpoint(shard0, None, None, True),
        ShardEndpoint(shard1, None, None, True),
    ])
    
    # Create decoder
    decoder = PipelineDecoder(pipeline, tokenizer)
    
    # Generate
    config = GenerationConfig(max_new_tokens=50, eos_token_id=tokenizer.eos_token_id)
    result = await decoder.generate("The capital of France is", config)
    
    print(f"\nGenerated: {result.text}")
    print(f"Tokens: {result.num_tokens}")
    
    await pipeline.close()
    adapter0.unload()
    adapter1.unload()

if __name__ == "__main__":
    asyncio.run(main())
