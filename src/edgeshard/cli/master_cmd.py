"""CLI command: edgeshard master start."""

from __future__ import annotations

import asyncio
from pathlib import Path

from rich.console import Console

console = Console()


def start_master(config_path: str) -> None:
    """Start the EdgeShard Master node.

    Args:
        config_path: Path to Master config YAML file.
    """
    from edgeshard.common.config import MasterConfig
    from edgeshard.common.logging import get_logger, setup_logging
    from edgeshard.master.server import MasterServer

    path = Path(config_path)
    if path.exists():
        config = MasterConfig.from_yaml(path)
    else:
        console.print(
            f"[yellow]Config {config_path} not found, using defaults.[/yellow]"
        )
        config = MasterConfig()

    setup_logging(level=config.log_level, json_output=config.log_json, component="master")
    logger = get_logger("cli.master")

    console.print(f"[bold green]Starting EdgeShard Master[/bold green]")
    console.print(f"  gRPC:   {config.grpc.host}:{config.grpc.port}")
    console.print(f"  API:    {config.api.host}:{config.api.port}")
    console.print(f"  State:  {config.state.path}")

    # Create and start server
    server = MasterServer(config)

    async def run() -> None:
        await server.start()
        console.print("[dim]Master is running. Press Ctrl+C to stop.[/dim]")
        await server.wait_for_termination()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down Master...[/yellow]")
        asyncio.run(server.stop())
        console.print("[yellow]Master stopped.[/yellow]")
