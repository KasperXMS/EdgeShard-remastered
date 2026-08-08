"""CLI command: edgeshard node list."""

from __future__ import annotations

import grpc
from rich.console import Console
from rich.table import Table

from edgeshard._grpc import edgeshard_pb2, edgeshard_pb2_grpc

console = Console()


def list_nodes(master_address: str = "localhost:10500") -> None:
    """List all registered Worker nodes in the cluster.

    Args:
        master_address: Master gRPC address (host:port).
    """
    console.print(f"[bold]EdgeShard Cluster Nodes[/bold]")
    console.print(f"[dim]Connecting to Master at {master_address}...[/dim]")

    try:
        # Connect to Master
        channel = grpc.insecure_channel(master_address)
        stub = edgeshard_pb2_grpc.WorkerServiceStub(channel)

        # Call ListWorkers
        request = edgeshard_pb2.ListWorkersRequest()
        response = stub.ListWorkers(request)

        # Display results
        table = Table(title="Registered Workers")
        table.add_column("Worker ID", style="cyan")
        table.add_column("Hostname", style="magenta")
        table.add_column("GPUs", style="green")
        table.add_column("Memory (GB)", justify="right")
        table.add_column("Status", style="yellow")

        for worker in response.workers:
            # Count GPUs
            gpu_devices = [d for d in worker.devices if d.device_type == "cuda"]
            gpu_str = f"{len(gpu_devices)} GPU(s)"
            if gpu_devices:
                gpu_names = set(d.name for d in gpu_devices)
                gpu_str += f" ({', '.join(gpu_names)})"

            memory_gb = worker.available_memory_mb / 1024

            table.add_row(
                worker.worker_id,
                worker.hostname,
                gpu_str,
                f"{memory_gb:.1f}",
                worker.status,
            )

        console.print(table)
        console.print(f"\n[green]Total: {len(response.workers)} worker(s)[/green]")

        channel.close()

    except grpc.RpcError as e:
        console.print(f"[red]Error connecting to Master: {e.details()}[/red]")
        console.print("[yellow]Is the Master running? Try: edgeshard master start[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
