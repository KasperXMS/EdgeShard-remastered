"""CLI command: edgeshard deploy — deploy a placement plan."""

from __future__ import annotations

import asyncio
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def deploy_service(
    plan_yaml: str,
    master: str = "localhost:10500",
    wait: bool = True,
    timeout: float = 120.0,
) -> None:
    """Deploy a placement plan to the cluster.

    Reads the plan YAML and sends StartShard commands to the appropriate
    Workers via gRPC.

    Args:
        plan_yaml: Path to the placement plan YAML file.
        master: Master gRPC address.
        wait: Whether to wait for all shards to be ready.
        timeout: Maximum wait time in seconds.
    """
    from edgeshard._grpc import edgeshard_pb2, edgeshard_pb2_grpc
    from edgeshard.common.logging import get_logger, setup_logging
    from edgeshard.deployment.remote import RemoteDeploymentBackend
    from edgeshard.scheduler.placement import PlacementPlan

    setup_logging(level="INFO", component="deploy")
    logger = get_logger("cli.deploy")

    path = Path(plan_yaml)
    if not path.exists():
        console.print(f"[red]Error: Plan file not found: {plan_yaml}[/red]")
        raise SystemExit(1)

    # Load the plan
    plan = PlacementPlan.from_yaml(path)
    console.print(f"[bold]EdgeShard Deploy[/bold]")
    console.print(f"  Service: {plan.service_name.value}")
    console.print(f"  Model:   {plan.service_spec.model.name}")
    console.print(f"  Dtype:   [cyan]{plan.service_spec.model.dtype}[/cyan]")
    console.print(f"  Shards:  {len(plan.shards)}")
    console.print()

    # Show shard placements
    table = Table(title="Shard Placements")
    table.add_column("Shard", style="cyan")
    table.add_column("Worker", style="magenta")
    table.add_column("Layers", style="green")
    table.add_column("Device", style="yellow")

    for shard in plan.shards:
        table.add_row(
            f"shard-{shard.shard_index}",
            str(shard.worker_id),
            f"{shard.layer_start}:{shard.layer_end}",
            shard.device,
        )

    console.print(table)
    console.print()

    # Get worker addresses from Master
    console.print("[cyan]Fetching cluster info from Master...[/cyan]")
    try:
        import grpc

        channel = grpc.insecure_channel(master)
        stub = edgeshard_pb2_grpc.WorkerServiceStub(channel)
        response = stub.ListWorkers(edgeshard_pb2.ListWorkersRequest())

        # Build worker address map (worker_id → hostname:10600)
        # Workers listen on port 10600 for deployment commands
        worker_addresses = {}
        for worker in response.workers:
            # Worker gRPC server runs on port 10600
            worker_addresses[worker.worker_id] = f"{worker.hostname}:10600"

        channel.close()

        if not worker_addresses:
            console.print("[red]Error: No workers registered in cluster[/red]")
            raise SystemExit(1)

        console.print(f"  Found {len(worker_addresses)} worker(s)")

    except Exception as e:
        console.print(f"[red]Error connecting to Master: {e}[/red]")
        raise SystemExit(1)

    # Create deployment backend
    backend = RemoteDeploymentBackend(worker_addresses=worker_addresses)

    # Deploy the plan
    console.print()
    console.print("[cyan]Deploying shards...[/cyan]")

    async def do_deploy():
        from edgeshard.deployment.manager import DeploymentManager

        manager = DeploymentManager(backend)
        state = await manager.deploy(
            plan=plan,
            base_port=50100,
            wait_for_ready=wait,
            timeout_seconds=timeout,
        )

        return state

    try:
        state = asyncio.run(do_deploy())
    except Exception as e:
        console.print(f"[red]Deployment failed: {e}[/red]")
        raise SystemExit(1)

    # Show results
    console.print()
    result_table = Table(title="Deployment Result")
    result_table.add_column("Shard", style="cyan")
    result_table.add_column("Worker", style="magenta")
    result_table.add_column("Status", style="green")
    result_table.add_column("Address", style="yellow")

    for shard_id, handle in state.shards.items():
        status_style = "green" if handle.status.value == "ready" else "red"
        result_table.add_row(
            shard_id,
            handle.worker_id,
            f"[{status_style}]{handle.status.value}[/{status_style}]",
            handle.data_address,
        )

    console.print(result_table)
    console.print()

    if state.status == "ready":
        console.print("[bold green]Deployment successful![/bold green]")

        # Register shards with Master for auto-discovery
        try:
            import grpc as _grpc
            channel = _grpc.insecure_channel(master)
            stub = edgeshard_pb2_grpc.WorkerServiceStub(channel)

            # Determine first/last shard indices
            sorted_handles = sorted(
                state.shards.values(),
                key=lambda h: h.layer_start,
            )
            first_shard_id = sorted_handles[0].shard_id if sorted_handles else ""
            last_shard_id = sorted_handles[-1].shard_id if sorted_handles else ""

            shards_proto = []
            for handle in sorted_handles:
                shards_proto.append(
                    edgeshard_pb2.ShardEndpoint(
                        shard_id=handle.shard_id,
                        worker_id=handle.worker_id,
                        data_address=handle.data_address,
                        layer_start=handle.layer_start,
                        layer_end=handle.layer_end,
                        device=handle.device,
                        model_name=handle.model_name,
                        is_first_shard=(handle.shard_id == first_shard_id),
                        is_last_shard=(handle.shard_id == last_shard_id),
                    )
                )

            stub.RegisterShards(
                edgeshard_pb2.RegisterShardsRequest(
                    service_name=plan.service_name.value,
                    shards=shards_proto,
                )
            )
            channel.close()
            console.print("[dim]Shards registered with Master for auto-discovery.[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not register with Master: {e}[/yellow]")

        console.print()

        # Show shard addresses for inference
        addresses = []
        for handle in state.shards.values():
            if handle.data_address:
                addresses.append(handle.data_address)

        if addresses:
            console.print("[dim]To run inference:[/dim]")
            console.print(
                f'  [dim]edgeshard infer "Your prompt"[/dim]'
            )
            console.print(
                f'  [dim]edgeshard infer "Your prompt" --shards {",".join(addresses)}[/dim]'
            )
    else:
        console.print(f"[bold red]Deployment failed: {state.message}[/bold red]")
        raise SystemExit(1)

    # Cleanup
    backend.close()
