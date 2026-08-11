"""CLI command: edgeshard infer."""

from __future__ import annotations

import asyncio

from rich.console import Console

console = Console()


def run_inference(
    prompt: str,
    shard_addresses: list[str] | None = None,
    master: str = "localhost:10500",
    service_name: str | None = None,
    max_tokens: int = 100,
) -> None:
    """Run distributed inference across multiple shards.

    Args:
        prompt: Input prompt text.
        shard_addresses: List of shard gRPC addresses. If None, auto-discover from Master.
        master: Master gRPC address for auto-discovery.
        service_name: Service name to query. If None, uses any deployed service.
        max_tokens: Maximum tokens to generate.
    """
    from edgeshard._grpc import edgeshard_pb2, edgeshard_pb2_grpc
    from edgeshard.common.logging import setup_logging

    setup_logging(level="INFO", component="infer")

    # Auto-discover shards from Master if not specified
    if shard_addresses is None:
        console.print(f"[cyan]Auto-discovering shards from Master at {master}...[/cyan]")
        try:
            import grpc

            channel = grpc.insecure_channel(master)
            stub = edgeshard_pb2_grpc.WorkerServiceStub(channel)

            response = stub.GetShardEndpoints(
                edgeshard_pb2.GetShardEndpointsRequest(
                    service_name=service_name or "",
                )
            )
            channel.close()

            if not response.endpoints:
                console.print("[red]Error: No deployed shards found.[/red]")
                console.print("[yellow]Did you run `edgeshard deploy` first?[/yellow]")
                raise SystemExit(1)

            shard_addresses = [ep.data_address for ep in response.endpoints]
            console.print(f"[green]Found {len(shard_addresses)} shard(s) in service '{response.service_name}'[/green]")
            for ep in response.endpoints:
                console.print(
                    f"  [dim]{ep.shard_id}: {ep.data_address} "
                    f"(layers {ep.layer_start}:{ep.layer_end}, {ep.device})[/dim]"
                )
            console.print()

        except Exception as e:
            console.print(f"[red]Error discovering shards from Master: {e}[/red]")
            console.print("[yellow]Is the Master running? Or specify --shards manually.[/yellow]")
            raise SystemExit(1)

    console.print(f"[bold]Running distributed inference[/bold]")
    console.print(f"  Prompt: {prompt}")
    console.print(f"  Shards: {shard_addresses}")
    console.print(f"  Max tokens: {max_tokens}")

    # For now, remote inference is a placeholder
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
