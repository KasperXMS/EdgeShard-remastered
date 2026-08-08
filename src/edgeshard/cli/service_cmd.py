"""CLI commands: edgeshard service deploy/list/status/stop."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def deploy_service(service_yaml: str) -> None:
    """Deploy a service according to its specification.

    Args:
        service_yaml: Path to service specification YAML.
    """
    from edgeshard.common.config import ServiceSpec
    from edgeshard.common.logging import get_logger, setup_logging

    setup_logging(level="INFO", component="service")
    logger = get_logger("cli.service")

    path = Path(service_yaml)
    if not path.exists():
        console.print(f"[red]Error: Service YAML not found: {service_yaml}[/red]")
        raise SystemExit(1)

    spec = ServiceSpec.from_yaml(path)
    console.print(f"[bold green]Deploying service:[/bold green] {spec.name}")
    console.print(f"  Model: {spec.model.name}")

    # TODO (M8): Implement actual deployment logic.
    # This should:
    # 1. Validate spec
    # 2. Generate PlacementPlan (or use cached)
    # 3. Send deployment commands to Workers
    # 4. Wait for health checks
    # 5. Report service READY

    logger.info("Deployment not yet implemented")
    console.print("\n[dim]Deployment is not yet active. See milestone M8.[/dim]")


def list_services() -> None:
    """List all deployed services."""
    console.print("[bold]Deployed Services[/bold]")

    # TODO (M8): Connect to Master and fetch service list.
    table = Table(title="Services")
    table.add_column("Name", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Shards", justify="right")

    console.print(table)
    console.print("\n[dim]Service listing is not yet active.[/dim]")


def show_service_status(name: str) -> None:
    """Show status of a deployed service.

    Args:
        name: Service name.
    """
    console.print(f"[bold]Service:[/bold] {name}")

    # TODO (M8): Connect to Master and fetch service status.
    console.print("[dim]Service status is not yet active.[/dim]")


def stop_service(name: str) -> None:
    """Stop a deployed service.

    Args:
        name: Service name.
    """
    console.print(f"[bold yellow]Stopping service:[/bold yellow] {name}")

    # TODO (M8): Send stop commands to Workers.
    console.print("[dim]Service stop is not yet active.[/dim]")
