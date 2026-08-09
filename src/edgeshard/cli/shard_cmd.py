"""CLI command: edgeshard shard start."""

from __future__ import annotations

import asyncio

from rich.console import Console

console = Console()


def start_shard(
    shard_id: str,
    model_path: str,
    layer_start: int,
    layer_end: int,
    dtype: str,
    host: str,
    port: int,
    is_first_shard: bool,
    is_last_shard: bool,
) -> None:
    """Start a shard process for distributed inference.

    Args:
        shard_id: Unique shard identifier.
        model_path: Path to model or Hugging Face ID.
        layer_start: Start layer index (inclusive).
        layer_end: End layer index (exclusive).
        dtype: Model dtype.
        host: Data plane host.
        port: Data plane port.
        is_first_shard: True if this shard has embedding.
        is_last_shard: True if this shard has LM head.
    """
    from edgeshard.common.logging import setup_logging
    from edgeshard.worker.shard_daemon import ShardDaemon

    setup_logging(level="INFO", component=f"shard-{shard_id}")

    console.print(f"[bold green]Starting shard {shard_id}[/bold green]")
    console.print(f"  Model: {model_path}")
    console.print(f"  Layers: [{layer_start}, {layer_end})")
    console.print(f"  Data plane: {host}:{port}")
    console.print(f"  First shard: {is_first_shard}")
    console.print(f"  Last shard: {is_last_shard}")

    # Create and start shard daemon
    daemon = ShardDaemon(
        shard_id=shard_id,
        model_path=model_path,
        layer_start=layer_start,
        layer_end=layer_end,
        dtype=dtype,
        data_plane_host=host,
        data_plane_port=port,
        is_first_shard=is_first_shard,
        is_last_shard=is_last_shard,
    )

    async def run() -> None:
        await daemon.start()
        console.print("[dim]Shard is running. Press Ctrl+C to stop.[/dim]")
        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[yellow]Shutting down shard...[/yellow]")
            await daemon.stop()
            console.print("[yellow]Shard stopped.[/yellow]")

    asyncio.run(run())
