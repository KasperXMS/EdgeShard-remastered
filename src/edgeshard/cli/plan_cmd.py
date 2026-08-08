"""CLI command: edgeshard plan."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

console = Console()


def generate_plan(service_yaml: str, output_path: str) -> None:
    """Generate an immutable PlacementPlan from a service specification.

    Args:
        service_yaml: Path to service specification YAML.
        output_path: Output path for the generated plan.
    """
    from edgeshard.common.config import ServiceSpec
    from edgeshard.common.logging import get_logger, setup_logging

    setup_logging(level="INFO", component="plan")
    logger = get_logger("cli.plan")

    path = Path(service_yaml)
    if not path.exists():
        console.print(f"[red]Error: Service YAML not found: {service_yaml}[/red]")
        raise SystemExit(1)

    spec = ServiceSpec.from_yaml(path)
    console.print(f"[bold]Service:[/bold] {spec.name}")
    console.print(f"[bold]Model:[/bold] {spec.model.name}")
    console.print(f"[bold]Policy:[/bold] {spec.scheduling.policy.value}")

    # TODO (M7): Implement actual scheduling logic.
    # This should:
    # 1. Connect to Master or read cluster snapshot from cache
    # 2. Load profile data
    # 3. Call scheduler.schedule()
    # 4. Serialize PlacementPlan to YAML

    logger.info("Scheduling not yet implemented")
    console.print("\n[dim]Scheduler is not yet active. See milestone M7.[/dim]")
    console.print(f"[dim]Output would be written to: {output_path}[/dim]")
