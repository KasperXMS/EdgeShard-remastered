"""CLI command: edgeshard infer."""

from __future__ import annotations

import asyncio

import torch
from rich.console import Console

console = Console()


def run_inference(
    prompt: str,
    shard_addresses: list[str],
    max_tokens: int = 100,
) -> None:
    """Run distributed inference across multiple shards.

    Args:
        prompt: Input prompt text.
        shard_addresses: List of shard gRPC addresses.
        max_tokens: Maximum tokens to generate.
    """
    from edgeshard.common.logging import setup_logging
    from edgeshard.runtime.pipeline import PipelineOrchestrator, ShardEndpoint
    from edgeshard.runtime.pipeline_decoder import PipelineDecoder
    from edgeshard.runtime.decoder import GenerationConfig
    from edgeshard.transport.grpc_transport import GrpcTensorTransport

    setup_logging(level="INFO", component="infer")

    console.print(f"[bold]Running distributed inference[/bold]")
    console.print(f"  Prompt: {prompt}")
    console.print(f"  Shards: {shard_addresses}")
    console.print(f"  Max tokens: {max_tokens}")

    # For now, we assume all shards are local (in-process) for testing
    # In a real deployment, each shard would be a separate process
    # and we'd connect via gRPC

    # TODO: Implement remote shard connection
    # For now, this is a placeholder that shows the expected API

    console.print("[yellow]Remote inference not yet fully implemented.[/yellow]")
    console.print("[dim]Use the example scripts for local testing.[/dim]")

    # Example of how it would work:
    # 1. Connect to each shard via gRPC
    # 2. Create PipelineOrchestrator with ShardEndpoints
    # 3. Load tokenizer
    # 4. Create PipelineDecoder
    # 5. Call decoder.generate(prompt, config)
    # 6. Print result

    console.print("\n[dim]See examples/m4_distributed_inference.py for usage.[/dim]")
