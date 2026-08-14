"""EdgeShard CLI — main entry point.

Usage:
    edgeshard master start
    edgeshard worker start --master 192.168.1.10:10500
    edgeshard node list
    edgeshard profile run Qwen/Qwen2.5-7B-Instruct
    edgeshard service init Qwen/Qwen2.5-7B-Instruct
    edgeshard cluster snapshot -o cluster.yaml
    edgeshard plan service.yaml --cluster-yaml cluster.yaml
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
cluster_app = typer.Typer(help="Cluster topology and snapshot operations.")
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
app.add_typer(cluster_app, name="cluster")
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


@node_app.command("metrics")
def node_metrics(
    master: str = typer.Option(
        "localhost:10500",
        "--master",
        "-m",
        help="Master address (host:port).",
    ),
) -> None:
    """Show detailed metrics for all workers (GPU utilization, temperature, power, network)."""
    from edgeshard.cli.node_cmd import show_metrics

    show_metrics(master_address=master)


# ---------------------------------------------------------------------------
# Cluster commands
# ---------------------------------------------------------------------------

@cluster_app.command("snapshot")
def cluster_snapshot(
    output: str = typer.Option(
        ".edgeshard/cluster.yaml",
        "--output",
        "-o",
        help="Output path for the cluster YAML.",
    ),
    master: str = typer.Option(
        "localhost:10500",
        "--master",
        "-m",
        help="Master address (host:port).",
    ),
) -> None:
    """Export current cluster state to a YAML file for offline planning.

    Connects to a running Master, fetches worker information including
    devices, metrics, and network topology, then writes a cluster.yaml
    that can be used with `edgeshard plan`.
    """
    from edgeshard.cli.cluster_cmd import export_cluster_snapshot

    export_cluster_snapshot(master_address=master, output_path=output)


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
    dtype: str = typer.Option(
        "float16",
        "--dtype",
        help="Weight dtype (float16, bfloat16, float32).",
    ),
) -> None:
    """Profile a model on local hardware to estimate performance."""
    from edgeshard.cli.profile_cmd import run_profile

    # Handle "auto" device
    if device == "auto":
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    run_profile(model_name=model, device=device, dtype=dtype)


@profile_app.command("list")
def profile_list() -> None:
    """List all stored profiling results."""
    from edgeshard.cli.profile_cmd import list_profiles

    list_profiles()


# ---------------------------------------------------------------------------
# Plan command (direct, not a group — matches §14: `edgeshard plan service.yaml`)
# ---------------------------------------------------------------------------

@app.command("plan")
def plan_generate(
    service_yaml: str = typer.Argument(
        ".edgeshard/service.yaml",
        help="Path to service specification YAML.",
    ),
    output: str = typer.Option(
        ".edgeshard/plan.yaml",
        "--output",
        "-o",
        help="Output path for the generated PlacementPlan.",
    ),
    master: str = typer.Option(
        None,
        "--master",
        "-m",
        help="Master address (host:port) for live cluster data.",
    ),
    cluster_yaml: str = typer.Option(
        None,
        "--cluster-yaml",
        "-c",
        help="Path to cluster YAML for offline planning.",
    ),
) -> None:
    """Generate an immutable PlacementPlan from a service specification."""
    from edgeshard.cli.plan_cmd import generate_plan

    generate_plan(
        service_yaml=service_yaml,
        output_path=output,
        master=master,
        cluster_yaml=cluster_yaml,
    )


# ---------------------------------------------------------------------------
# Deploy command (M8)
# ---------------------------------------------------------------------------

@app.command("deploy")
def deploy(
    plan_yaml: str = typer.Argument(
        ".edgeshard/plan.yaml",
        help="Path to placement plan YAML.",
    ),
    master: str = typer.Option(
        "localhost:10500",
        "--master",
        "-m",
        help="Master address (host:port).",
    ),
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Wait for all shards to be ready.",
    ),
    timeout: float = typer.Option(
        120.0,
        "--timeout",
        "-t",
        help="Maximum wait time in seconds.",
    ),
) -> None:
    """Deploy a placement plan to the cluster.

    Reads the plan YAML and sends StartShard commands to the appropriate
    Workers via gRPC. Workers spawn shard subprocesses locally.

    Example:
        edgeshard deploy .edgeshard/plan.yaml
        edgeshard deploy .edgeshard/plan.yaml --master 192.168.1.10:10500
    """
    from edgeshard.cli.deploy_cmd import deploy_service

    deploy_service(
        plan_yaml=plan_yaml,
        master=master,
        wait=wait,
        timeout=timeout,
    )


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


@service_app.command("init")
def service_init(
    model: str = typer.Argument(
        ...,
        help="HuggingFace model ID or local path (e.g. Qwen/Qwen2.5-7B-Instruct).",
    ),
    output: str = typer.Option(
        ".edgeshard/service.yaml",
        "--output",
        "-o",
        help="Output path for the generated service YAML.",
    ),
    name: str = typer.Option(
        None,
        "--name",
        "-n",
        help="Service name (auto-generated from model if omitted).",
    ),
    dtype: str = typer.Option(
        "float16",
        "--dtype",
        "-d",
        help="Weight dtype (float16, bfloat16, float32).",
    ),
    max_seq_len: int = typer.Option(
        4096,
        "--max-seq-len",
        help="Maximum sequence length.",
    ),
    policy: str = typer.Option(
        "default",
        "--policy",
        "-p",
        help="Scheduling policy (default, latency-first, memory-balanced).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing file.",
    ),
) -> None:
    """Generate a service.yaml template from a model name.

    Auto-fills sensible defaults based on the model's known architecture.
    For unknown models, uses generic conservative defaults.
    """
    from edgeshard.cli.service_cmd import init_service

    init_service(
        model=model,
        output=output,
        name=name,
        dtype=dtype,
        max_seq_len=max_seq_len,
        policy=policy,
        force=force,
    )


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
        None,
        "--shards",
        "-s",
        help="Comma-separated shard addresses. If omitted, auto-discover from Master.",
    ),
    master: str = typer.Option(
        "localhost:10500",
        "--master",
        "-m",
        help="Master address for auto-discovery (used when --shards is not specified).",
    ),
    service: str = typer.Option(
        None,
        "--service",
        help="Service name to infer against. If omitted, uses any deployed service.",
    ),
    model: str = typer.Option(
        None,
        "--model",
        help="Model name/path for tokenizer. If omitted, auto-detect from deployed shards.",
    ),
    max_tokens: int = typer.Option(100, "--max-tokens", help="Max tokens to generate."),
) -> None:
    """Run distributed inference across multiple shards.

    If --shards is not specified, automatically discovers shard addresses from Master.
    """
    from edgeshard.cli.infer_cmd import run_inference

    shard_addresses = None
    if shards:
        shard_addresses = [addr.strip() for addr in shards.split(",")]

    run_inference(
        prompt=prompt,
        shard_addresses=shard_addresses,
        master=master,
        service_name=service,
        model_name=model,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Up command — one-command deployment
# ---------------------------------------------------------------------------

@app.command("up")
def up(
    model: str = typer.Argument(
        ...,
        help="HuggingFace model ID or local path (e.g. Qwen/Qwen2.5-7B-Instruct).",
    ),
    master: str = typer.Option(
        "localhost:10500",
        "--master",
        "-m",
        help="Master address (host:port).",
    ),
    policy: str = typer.Option(
        "default",
        "--policy",
        "-p",
        help="Scheduling policy (default, latency-first, memory-balanced).",
    ),
    dtype: str = typer.Option(
        "float16",
        "--dtype",
        "-d",
        help="Weight dtype.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Regenerate configs even if they exist.",
    ),
    deploy_shards: bool = typer.Option(
        False,
        "--deploy",
        help="Also deploy the plan (start shards on workers).",
    ),
) -> None:
    """One-command deployment: auto-generate configs, plan, and optionally deploy.

    This chains together: service init → cluster snapshot → plan → deploy.
    Generated files go to .edgeshard/ (gitignored).

    Example:
        edgeshard up Qwen/Qwen2.5-7B-Instruct
        edgeshard up Qwen/Qwen2.5-7B-Instruct --deploy
        edgeshard up Qwen/Qwen2.5-7B-Instruct --policy latency-first
    """
    from pathlib import Path

    from edgeshard.cli.cluster_cmd import export_cluster_snapshot
    from edgeshard.cli.plan_cmd import generate_plan
    from edgeshard.cli.service_cmd import init_service

    config_dir = Path(".edgeshard")
    service_yaml = config_dir / "service.yaml"
    cluster_yaml = config_dir / "cluster.yaml"
    plan_yaml = config_dir / "plan.yaml"

    console.print("[bold]EdgeShard Up[/bold]")
    console.print(f"[dim]Model: {model}[/dim]")
    console.print()

    total_steps = 4 if deploy_shards else 3
    step = 1

    # Step 1: Generate service.yaml if needed
    if not service_yaml.exists() or force:
        console.print(f"[cyan]Step {step}/{total_steps}:[/cyan] Generating service.yaml...")
        init_service(
            model=model,
            output=str(service_yaml),
            dtype=dtype,
            policy=policy,
            force=True,
        )
        console.print()
    else:
        console.print(f"[cyan]Step {step}/{total_steps}:[/cyan] Using existing {service_yaml}")
    step += 1

    # Step 2: Export cluster snapshot
    console.print(f"[cyan]Step {step}/{total_steps}:[/cyan] Exporting cluster snapshot...")
    try:
        export_cluster_snapshot(
            master_address=master,
            output_path=str(cluster_yaml),
        )
        console.print()
    except Exception as e:
        console.print(f"[red]Failed to export cluster: {e}[/red]")
        console.print("[yellow]Is the Master running? Try: edgeshard master start[/yellow]")
        raise SystemExit(1)
    step += 1

    # Step 3: Generate plan
    console.print(f"[cyan]Step {step}/{total_steps}:[/cyan] Generating placement plan...")
    generate_plan(
        service_yaml=str(service_yaml),
        output_path=str(plan_yaml),
        master=None,
        cluster_yaml=str(cluster_yaml),
    )
    step += 1

    # Step 4: Deploy (optional)
    if deploy_shards:
        console.print()
        console.print(f"[cyan]Step {step}/{total_steps}:[/cyan] Deploying shards to workers...")
        from edgeshard.cli.deploy_cmd import deploy_service

        try:
            deploy_service(
                plan_yaml=str(plan_yaml),
                master=master,
                wait=True,
                timeout=120.0,
            )
        except SystemExit:
            console.print("[red]Deployment failed[/red]")
            raise
    else:
        console.print()
        console.print("[bold green]Deployment plan ready![/bold green]")
        console.print()
        console.print("Generated files:")
        console.print(f"  [dim]{service_yaml}[/dim]")
        console.print(f"  [dim]{cluster_yaml}[/dim]")
        console.print(f"  [bold]{plan_yaml}[/bold]")
        console.print()
        console.print("[dim]Next steps:[/dim]")
        console.print(f"  [dim]# Deploy shards to workers[/dim]")
        console.print(f"  [dim]edgeshard deploy {plan_yaml}[/dim]")
        console.print()
        console.print(f"  [dim]# Or deploy directly with --deploy flag[/dim]")
        console.print(f"  [dim]edgeshard up {model} --deploy[/dim]")
