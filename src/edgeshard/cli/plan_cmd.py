"""CLI command: edgeshard plan."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def generate_plan(
    service_yaml: str,
    output_path: str,
    master: str | None = None,
    cluster_yaml: str | None = None,
) -> None:
    """Generate an immutable PlacementPlan from a service specification.

    Args:
        service_yaml: Path to service specification YAML.
        output_path: Output path for the generated plan.
        master: Optional Master address (host:port) for live cluster data.
        cluster_yaml: Optional path to cluster YAML for offline planning.
    """
    from edgeshard.common.config import ServiceSpec, SchedulingPolicyName
    from edgeshard.common.logging import get_logger, setup_logging
    from edgeshard.scheduler import (
        ClusterSnapshot,
        ProfileSnapshot,
        SchedulingPolicy,
        schedule,
    )
    from edgeshard.scheduler.bridge import profile_results_to_snapshot

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
    console.print()

    # --- Build ClusterSnapshot ---
    cluster = _build_cluster_snapshot(master, cluster_yaml)

    if not cluster.workers:
        console.print("[red]Error: No workers in cluster snapshot.[/red]")
        console.print(
            "[dim]Provide --master for live cluster or --cluster-yaml for offline.[/dim]"
        )
        raise SystemExit(1)

    console.print(f"[green]Cluster:[/green] {len(cluster.workers)} worker(s)")
    for w in cluster.workers:
        dev_str = ", ".join(f"{d.name} ({d.total_memory_mb} MB)" for d in w.devices)
        console.print(f"  - {w.worker_id}: {dev_str}")
    console.print()

    # --- Load Profiles ---
    profile_snapshot = _load_profiles(spec.model.name)
    console.print(f"[green]Profiles:[/green] {len(profile_snapshot.entries)} entry(ies)")
    console.print()

    # --- Build SchedulingPolicy ---
    policy = SchedulingPolicy(
        name=spec.scheduling.policy.value,
        params=dict(spec.scheduling.hints),
    )

    # --- Run Scheduler ---
    console.print("[bold]Running scheduler...[/bold]")
    try:
        plan = schedule(
            model=spec.model,
            cluster=cluster,
            profiles=profile_snapshot,
            policy=policy,
            service_name=spec.name,
            max_seq_len=spec.runtime.max_sequence_length,
        )
    except Exception as e:
        console.print(f"[red]Scheduling failed: {e}[/red]")
        raise SystemExit(1)

    # --- Display results ---
    console.print()
    console.print("[bold green]Placement Plan Generated![/bold green]")
    console.print()

    table = Table(title="Shard Placements")
    table.add_column("Shard", style="cyan", justify="right")
    table.add_column("Worker", style="magenta")
    table.add_column("Layers", justify="right", style="green")
    table.add_column("Device", style="yellow")

    for shard in plan.shards:
        table.add_row(
            str(shard.shard_index),
            str(shard.worker_id),
            f"{shard.layer_start}-{shard.layer_end - 1}",
            shard.device,
        )

    console.print(table)
    console.print()

    # --- Write YAML ---
    plan.to_yaml(output_path)
    console.print(f"[dim]Plan written to: {output_path}[/dim]")

    logger.info(f"Plan generated with {len(plan.shards)} shard(s)")


def _build_cluster_snapshot(
    master: str | None, cluster_yaml: str | None
) -> "ClusterSnapshot":
    """Build a ClusterSnapshot from live master or offline YAML."""
    from edgeshard.scheduler import ClusterSnapshot

    if master:
        return _cluster_from_master(master)
    elif cluster_yaml:
        from edgeshard.scheduler.bridge import cluster_snapshot_from_yaml
        return cluster_snapshot_from_yaml(cluster_yaml)
    else:
        # Try to connect to default master
        try:
            return _cluster_from_master("localhost:10500")
        except Exception:
            console.print(
                "[yellow]Warning: No master specified and default not available.[/yellow]"
            )
            console.print("[dim]Using empty cluster (will fail if no workers).[/dim]")
            return ClusterSnapshot()


def _cluster_from_master(master_address: str) -> "ClusterSnapshot":
    """Connect to Master via gRPC and build ClusterSnapshot."""
    import grpc

    from edgeshard._grpc import edgeshard_pb2, edgeshard_pb2_grpc
    from edgeshard.scheduler.bridge import worker_records_to_cluster_snapshot

    channel = grpc.insecure_channel(master_address)
    stub = edgeshard_pb2_grpc.WorkerServiceStub(channel)

    try:
        response = stub.ListWorkers(edgeshard_pb2.ListWorkersRequest())
    except grpc.RpcError as e:
        raise ConnectionError(
            f"Failed to connect to Master at {master_address}: {e.details()}"
        )
    finally:
        channel.close()

    # Convert proto WorkerStates to a mock WorkerRecord-like list
    # We need to build a lightweight adapter since bridge expects WorkerRecord
    records = []
    for ws in response.workers:
        record = _ProtoWorkerAdapter(ws)
        records.append(record)

    return worker_records_to_cluster_snapshot(records)


def _load_profiles(model_name: str) -> "ProfileSnapshot":
    """Load profiles from SQLite store."""
    from edgeshard.profiler.store import ProfileStore
    from edgeshard.scheduler import ProfileSnapshot
    from edgeshard.scheduler.bridge import profile_results_to_snapshot

    try:
        store = ProfileStore()
        all_profiles = store.list_profiles()
        return profile_results_to_snapshot(all_profiles)
    except Exception:
        console.print("[yellow]Warning: Could not load profiles.[/yellow]")
        return ProfileSnapshot()


class _ProtoWorkerAdapter:
    """Adapter to make proto WorkerState look like WorkerRecord for bridge."""

    def __init__(self, ws):
        self.worker_id = ws.worker_id
        self.hostname = ws.hostname
        self.devices = list(ws.devices)
        self.available_memory_mb = ws.available_memory_mb
        self.cpu_count = ws.cpu_count
        self.status = ws.status
        self.metadata = dict(ws.metadata)
        self.device_metrics = list(ws.device_metrics)
        self.network_metrics = (
            ws.network_metrics if ws.HasField("network_metrics") else None
        )

    def is_alive(self, timeout_seconds: float = 30.0) -> bool:
        return self.status == "online"
