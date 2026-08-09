"""CLI commands: edgeshard profile run / list."""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.table import Table

console = Console()


def run_profile(model_name: str, device: str, dtype: str) -> None:
    """Profile a model on local hardware to estimate performance.

    Args:
        model_name: Hugging Face model ID or local path.
        device: Target device for profiling (e.g., "cuda", "cuda:0", "cpu").
        dtype: Weight dtype (e.g., "float16", "bfloat16", "float32").
    """
    import torch

    from edgeshard.common.logging import setup_logging
    from edgeshard.profiler.executor import ProfileExecutor
    from edgeshard.profiler.store import ProfileStore

    setup_logging(level="INFO", component="profile")

    console.print(f"[bold]Profiling model:[/bold] {model_name}")
    console.print(f"[bold]Device:[/bold] {device}")
    console.print(f"[bold]Dtype:[/bold] {dtype}")
    console.print()

    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype.lower())
    if torch_dtype is None:
        console.print(f"[red]Unknown dtype: {dtype}[/red]")
        console.print(f"[dim]Supported: {', '.join(dtype_map.keys())}[/dim]")
        return

    # Parse device
    torch_device = torch.device(device)

    async def _run() -> None:
        executor = ProfileExecutor()

        console.print("[dim]Running profiling (this may take a minute)...[/dim]")
        console.print()

        result = await executor.profile(
            model_path=model_name,
            dtype=torch_dtype,
            device=torch_device,
        )

        # Save to store
        store = ProfileStore()
        store.save(result)

        # Display results
        console.print("[bold green]Profiling complete![/bold green]")
        console.print()

        # Results table
        table = Table(title="Profile Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        table.add_row("Model", result.model_name)
        table.add_row("Device", f"{result.device_name} ({result.device_type})")
        table.add_row("Device Memory", f"{result.device_memory_mb} MB")
        table.add_row("Dtype", result.dtype)
        table.add_row("Layers Profiled", str(result.num_layers_profiled))
        table.add_row("", "")
        table.add_row("Layer Forward Latency", f"{result.layer_forward_ms:.3f} ms")
        table.add_row("KV Cache / Token", f"{result.kv_cache_per_token_mb:.4f} MB")
        table.add_row("Prefill Throughput", f"{result.prefill_tokens_per_sec:.1f} tokens/s")
        table.add_row("Decode Throughput", f"{result.decode_tokens_per_sec:.1f} tokens/s")
        table.add_row("Total Model Memory", f"{result.total_model_memory_mb:.1f} MB")

        console.print(table)
        console.print()
        console.print(f"[dim]Profile saved to {store._db_path}[/dim]")

    asyncio.run(_run())


def list_profiles() -> None:
    """List all stored profiling results."""
    from edgeshard.profiler.store import ProfileStore

    store = ProfileStore()
    profiles = store.list_profiles()

    if not profiles:
        console.print("[yellow]No profiles stored.[/yellow]")
        console.print("[dim]Run: edgeshard profile run <model_path>[/dim]")
        return

    table = Table(title="Stored Profiles")
    table.add_column("Model", style="cyan")
    table.add_column("Device", style="magenta")
    table.add_column("Dtype", style="blue")
    table.add_column("Layer Fwd (ms)", justify="right", style="green")
    table.add_column("Prefill (tok/s)", justify="right", style="yellow")
    table.add_column("Decode (tok/s)", justify="right", style="yellow")
    table.add_column("KV/token (MB)", justify="right")
    table.add_column("Date", style="dim")

    import time

    for p in profiles:
        date_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.timestamp))
        table.add_row(
            p.model_name,
            f"{p.device_name}",
            p.dtype,
            f"{p.layer_forward_ms:.3f}",
            f"{p.prefill_tokens_per_sec:.1f}",
            f"{p.decode_tokens_per_sec:.1f}",
            f"{p.kv_cache_per_token_mb:.4f}",
            date_str,
        )

    console.print(table)
    console.print(f"\n[green]Total: {len(profiles)} profile(s)[/green]")
