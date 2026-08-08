"""CLI command: edgeshard profile run."""

from __future__ import annotations

from rich.console import Console

console = Console()


def run_profile(model_name: str, device: str) -> None:
    """Profile a model on local hardware to estimate performance.

    Args:
        model_name: Hugging Face model ID or local path.
        device: Target device for profiling.
    """
    from edgeshard.common.logging import get_logger, setup_logging

    setup_logging(level="INFO", component="profile")
    logger = get_logger("cli.profile")

    console.print(f"[bold]Profiling model:[/bold] {model_name}")
    console.print(f"[bold]Device:[/bold] {device}")

    # TODO (M6): Implement actual profiling logic.
    # This should:
    # 1. Detect hardware capabilities
    # 2. Load model (or layer subset)
    # 3. Measure forward pass latency
    # 4. Estimate KV cache memory per token
    # 5. Store results in ProfileStore

    logger.info("Profiling not yet implemented")
    console.print("\n[dim]Profiling is not yet active. See milestone M6.[/dim]")
    console.print("[dim]Results will be stored in SQLite once implemented.[/dim]")
