"""Example: Distributed inference with multiple shards (M4).

This example demonstrates:
1. Running multiple shards in separate processes
2. Coordinating inference via PipelineOrchestrator
3. Distributed greedy decoding

For a real multi-machine setup:
- Machine A: runs shard-0 (layers 0-15)
- Machine B: runs shard-1 (layers 16-23)
- Machine C: runs the inference client

For local testing, we simulate this with multiple processes on one machine.

Usage:
    # Terminal 1: Start shard 0
    python examples/m4_distributed_inference.py shard0

    # Terminal 2: Start shard 1
    python examples/m4_distributed_inference.py shard1

    # Terminal 3: Run inference
    python examples/m4_distributed_inference.py infer "Hello world"
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import torch

# Configuration
MODEL_PATH = "models/Qwen2.5-0.5B-Instruct"
SHARD0_PORT = 50100
SHARD1_PORT = 50101


async def run_shard0() -> None:
    """Run shard 0 (first shard, layers 0-11)."""
    from edgeshard.common.logging import setup_logging
    from edgeshard.worker.shard_daemon import ShardDaemon

    setup_logging(level="INFO", component="shard-0")

    # For Qwen2.5-0.5B, it has 24 layers total
    # Shard 0: layers 0-11 (first 12 layers)
    daemon = ShardDaemon(
        shard_id="shard-0",
        model_path=MODEL_PATH,
        layer_start=0,
        layer_end=12,
        dtype="float16",
        data_plane_host="0.0.0.0",
        data_plane_port=SHARD0_PORT,
        is_first_shard=True,
        is_last_shard=False,
    )

    await daemon.start()
    print(f"Shard 0 running on port {SHARD0_PORT}")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await daemon.stop()


async def run_shard1() -> None:
    """Run shard 1 (last shard, layers 12-23 + LM head)."""
    from edgeshard.common.logging import setup_logging
    from edgeshard.worker.shard_daemon import ShardDaemon

    setup_logging(level="INFO", component="shard-1")

    # Shard 1: layers 12-23 (last 12 layers + LM head)
    daemon = ShardDaemon(
        shard_id="shard-1",
        model_path=MODEL_PATH,
        layer_start=12,
        layer_end=24,
        dtype="float16",
        data_plane_host="0.0.0.0",
        data_plane_port=SHARD1_PORT,
        is_first_shard=False,
        is_last_shard=True,
    )

    await daemon.start()
    print(f"Shard 1 running on port {SHARD1_PORT}")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await daemon.stop()


async def run_inference_client(prompt: str) -> None:
    """Run inference client that connects to both shards."""
    from transformers import AutoTokenizer

    from edgeshard.common.logging import setup_logging
    from edgeshard.runtime.adapters.qwen2 import Qwen2Adapter
    from edgeshard.runtime.pipeline import PipelineOrchestrator, ShardEndpoint
    from edgeshard.runtime.pipeline_decoder import PipelineDecoder
    from edgeshard.runtime.decoder import GenerationConfig
    from edgeshard.runtime.shard import ModelShard

    setup_logging(level="INFO", component="infer-client")

    print(f"Connecting to shards...")
    print(f"  Shard 0: localhost:{SHARD0_PORT}")
    print(f"  Shard 1: localhost:{SHARD1_PORT}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # For this example, we'll create local shards that connect to remote shards
    # In a real deployment, each shard would be a separate process

    # TODO: Implement remote shard proxy
    # For now, we'll demonstrate with a single-process pipeline

    print("\n[yellow]Note: This example requires manual shard setup.[/yellow]")
    print("[dim]For local testing, use m2_single_shard_generation.py[/dim]")

    # Placeholder for remote inference
    print(f"\nPrompt: {prompt}")
    print("Inference would run here if shards were connected...")


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python m4_distributed_inference.py shard0  # Start shard 0")
        print("  python m4_distributed_inference.py shard1  # Start shard 1")
        print('  python m4_distributed_inference.py infer "Hello"  # Run inference')
        sys.exit(1)

    command = sys.argv[1]

    if command == "shard0":
        print("Starting shard 0...")
        asyncio.run(run_shard0())
    elif command == "shard1":
        print("Starting shard 1...")
        asyncio.run(run_shard1())
    elif command == "infer":
        if len(sys.argv) < 3:
            print("Error: Please provide a prompt")
            sys.exit(1)
        prompt = sys.argv[2]
        asyncio.run(run_inference_client(prompt))
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
