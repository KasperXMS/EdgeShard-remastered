"""CLI command: edgeshard worker start."""

from __future__ import annotations

import asyncio
from pathlib import Path

from rich.console import Console

console = Console()


def start_worker(master_address: str, config_path: str) -> None:
    """Start an EdgeShard Worker node and register with Master.

    Args:
        master_address: Master gRPC address (host:port).
        config_path: Path to Worker config YAML file.
    """
    from edgeshard.common.config import WorkerConfig
    from edgeshard.common.logging import get_logger, setup_logging
    from edgeshard.worker.daemon import WorkerDaemon

    path = Path(config_path)
    if path.exists():
        config = WorkerConfig.from_yaml(path)
    else:
        console.print(
            f"[yellow]Config {config_path} not found, using defaults.[/yellow]"
        )
        config = WorkerConfig()

    config.registration.master_address = master_address

    setup_logging(level=config.log_level, json_output=config.log_json, component="worker")
    logger = get_logger("cli.worker")

    console.print(f"[bold green]Starting EdgeShard Worker[/bold green]")
    console.print(f"  Master:       {config.registration.master_address}")
    console.print(f"  Model cache:  {config.model_cache.cache_dir}")

    # Create and start worker daemon
    worker = WorkerDaemon(config)

    async def run() -> None:
        await worker.start()
        console.print("[dim]Worker is running. Press Ctrl+C to stop.[/dim]")
        try:
            # Keep running
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[yellow]Shutting down Worker...[/yellow]")
            await worker.stop()
            console.print("[yellow]Worker stopped.[/yellow]")

    asyncio.run(run())
