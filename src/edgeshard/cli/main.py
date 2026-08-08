"""EdgeShard CLI — main entry point.

Usage:
    edgeshard master start
    edgeshard worker start --master 192.168.1.10:10500
    edgeshard node list
    edgeshard profile run Qwen/Qwen2.5-7B-Instruct
    edgeshard plan service.yaml
    edgeshard service deploy service.yaml
"""

from __future__ import annotations

import typer
from rich.console import Console

from edgeshard._version import __version__

# Create sub-command groups
master_app = typer.Typer(help="Master node operations.")
worker_app = typer.Typer(help="Worker node operations.")
node_app = typer.Typer(help="Cluster node discovery.")
profile_app = typer.Typer(help="Model profiling operations.")
service_app = typer.Typer(help="Service lifecycle management.")
shard_app = typer.Typer(help="Shard process operations.")

# Main app
app = typer.Typer(
    name="edgeshard",
    help="Config-driven distributed inference for heterogeneous edge and GPU resources.",
)

# Register sub-command groups
app.add_typer(master_app, name="master")
app.add_typer(worker_app, name="worker")
app.add_typer(node_app, name="node")
app.add_typer(profile_app, name="profile")
app.add_typer(service_app, name="service")
app.add_typer(shard_app, name="shard")

console = Console()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        is_eager=True,
    ),
) -> None:
    """EdgeShard — distributed inference orchestration."""
    if version:
        console.print(f"EdgeShard v{__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


# ---------------------------------------------------------------------------
# Master commands
# ---------------------------------------------------------------------------

@master_app.command("start")
def master_start(
    config: str = typer.Option(
        "master.yaml",
        "--config",
        "-c",
        help="Path to Master config YAML.",
    ),
) -> None:
    """Start the EdgeShard Master node."""
    from edgeshard.cli.master_cmd import start_master

    start_master(config)


# ---------------------------------------------------------------------------
# Worker commands
# ---------------------------------------------------------------------------

@worker_app.command("start")
def worker_start(
    master: str = typer.Option(
        "localhost:10500",
        "--master",
        "-m",
        help="Master address (host:port).",
    ),
    config: str = typer.Option(
        "worker.yaml",
        "--config",
        "-c",
        help="Path to Worker config YAML.",
    ),
) -> None:
    """Start an EdgeShard Worker node and register with Master."""
    from edgeshard.cli.worker_cmd import start_worker

    start_worker(master_address=master, config_path=config)


# ---------------------------------------------------------------------------
# Node commands
# ---------------------------------------------------------------------------

@node_app.command("list")
def node_list(
    master: str = typer.Option(
        "localhost:10500",
        "--master",
        "-m",
        help="Master address (host:port).",
    ),
) -> None:
    """List all registered Worker nodes in the cluster."""
    from edgeshard.cli.node_cmd import list_nodes

    list_nodes(master_address=master)


# ---------------------------------------------------------------------------
# Profile commands
# ---------------------------------------------------------------------------

@profile_app.command("run")
def profile_run(
    model: str = typer.Argument(
        ...,
        help="Hugging Face model ID or local path (e.g. Qwen/Qwen2.5-7B-Instruct).",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        "-d",
        help="Target device for profiling (auto, cuda:0, cpu, etc.).",
    ),
) -> None:
    """Profile a model on local hardware to estimate performance."""
    from edgeshard.cli.profile_cmd import run_profile

    run_profile(model_name=model, device=device)


# ---------------------------------------------------------------------------
# Plan command (direct, not a group — matches §14: `edgeshard plan service.yaml`)
# ---------------------------------------------------------------------------

@app.command("plan")
def plan_generate(
    service_yaml: str = typer.Argument(
        ...,
        help="Path to service specification YAML.",
    ),
    output: str = typer.Option(
        "plan.yaml",
        "--output",
        "-o",
        help="Output path for the generated PlacementPlan.",
    ),
) -> None:
    """Generate an immutable PlacementPlan from a service specification."""
    from edgeshard.cli.plan_cmd import generate_plan

    generate_plan(service_yaml=service_yaml, output_path=output)


# ---------------------------------------------------------------------------
# Service commands
# ---------------------------------------------------------------------------

@service_app.command("deploy")
def service_deploy(
    service_yaml: str = typer.Argument(
        ...,
        help="Path to service specification YAML.",
    ),
) -> None:
    """Deploy a service according to its specification."""
    from edgeshard.cli.service_cmd import deploy_service

    deploy_service(service_yaml=service_yaml)


@service_app.command("list")
def service_list() -> None:
    """List all deployed services."""
    from edgeshard.cli.service_cmd import list_services

    list_services()


@service_app.command("status")
def service_status(
    name: str = typer.Argument(..., help="Service name."),
) -> None:
    """Show status of a deployed service."""
    from edgeshard.cli.service_cmd import show_service_status

    show_service_status(name)


@service_app.command("stop")
def service_stop(
    name: str = typer.Argument(..., help="Service name."),
) -> None:
    """Stop a deployed service."""
    from edgeshard.cli.service_cmd import stop_service

    stop_service(name)


# ---------------------------------------------------------------------------
# Shard commands
# ---------------------------------------------------------------------------

@shard_app.command("start")
def shard_start(
    model: str = typer.Argument(..., help="Model path or Hugging Face ID."),
    shard_id: str = typer.Option(..., "--shard-id", "-s", help="Unique shard ID."),
    layers: str = typer.Option(
        ...,
        "--layers",
        "-l",
        help="Layer range as START:END (e.g., 0:16).",
    ),
    dtype: str = typer.Option("float16", "--dtype", help="Model dtype."),
    host: str = typer.Option("0.0.0.0", "--host", help="Data plane host."),
    port: int = typer.Option(50100, "--port", "-p", help="Data plane port."),
    is_first: bool = typer.Option(
        False, "--first", help="This is the first shard (has embedding)."
    ),
    is_last: bool = typer.Option(
        False, "--last", help="This is the last shard (has LM head)."
    ),
) -> None:
    """Start a shard process for distributed inference."""
    from edgeshard.cli.shard_cmd import start_shard

    # Parse layer range
    try:
        layer_start, layer_end = map(int, layers.split(":"))
    except ValueError:
        console.print("[red]Error: --layers must be START:END format[/red]")
        raise typer.Exit(1)

    start_shard(
        shard_id=shard_id,
        model_path=model,
        layer_start=layer_start,
        layer_end=layer_end,
        dtype=dtype,
        host=host,
        port=port,
        is_first_shard=is_first,
        is_last_shard=is_last,
    )


# ---------------------------------------------------------------------------
# Inference command
# ---------------------------------------------------------------------------

@app.command("infer")
def infer(
    prompt: str = typer.Argument(..., help="Input prompt text."),
    shards: str = typer.Option(
        ...,
        "--shards",
        help="Comma-separated shard addresses (e.g., localhost:50100,localhost:50101).",
    ),
    max_tokens: int = typer.Option(100, "--max-tokens", help="Max tokens to generate."),
) -> None:
    """Run distributed inference across multiple shards."""
    from edgeshard.cli.infer_cmd import run_inference

    shard_addresses = [addr.strip() for addr in shards.split(",")]
    run_inference(prompt=prompt, shard_addresses=shard_addresses, max_tokens=max_tokens)
