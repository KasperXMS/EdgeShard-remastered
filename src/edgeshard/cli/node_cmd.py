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

        # Display results with metrics
        table = Table(title="Registered Workers")
        table.add_column("Worker ID", style="cyan")
        table.add_column("Hostname", style="magenta")
        table.add_column("GPUs", style="green")
        table.add_column("GPU Util", justify="right", style="yellow")
        table.add_column("Temp", justify="right", style="red")
        table.add_column("Memory (GB)", justify="right")
        table.add_column("Status", style="blue")

        for worker in response.workers:
            # Count GPUs
            gpu_devices = [d for d in worker.devices if d.device_type == "cuda"]
            gpu_str = f"{len(gpu_devices)} GPU(s)"
            if gpu_devices:
                gpu_names = set(d.name for d in gpu_devices)
                gpu_str += f" ({', '.join(gpu_names)})"

            # Get GPU metrics summary
            gpu_util_str = "-"
            temp_str = "-"

            for dm in worker.device_metrics:
                if dm.HasField("gpu_metrics"):
                    gpu_metrics = dm.gpu_metrics
                    gpu_util_str = f"{gpu_metrics.utilization_percent}%"
                    temp_str = f"{gpu_metrics.temperature_c}°C"
                    break  # Show first GPU metrics

            memory_gb = worker.available_memory_mb / 1024

            table.add_row(
                worker.worker_id,
                worker.hostname,
                gpu_str,
                gpu_util_str,
                temp_str,
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


def show_metrics(master_address: str = "localhost:10500") -> None:
    """Show detailed metrics for all workers.

    Args:
        master_address: Master gRPC address (host:port).
    """
    console.print(f"[bold]EdgeShard Cluster Metrics[/bold]")
    console.print(f"[dim]Connecting to Master at {master_address}...[/dim]")

    try:
        # Connect to Master
        channel = grpc.insecure_channel(master_address)
        stub = edgeshard_pb2_grpc.WorkerServiceStub(channel)

        # Call ListWorkers
        request = edgeshard_pb2.ListWorkersRequest()
        response = stub.ListWorkers(request)

        for worker in response.workers:
            console.print(f"\n[bold cyan]Worker: {worker.worker_id}[/bold cyan] ({worker.hostname})")
            console.print(f"  Status: [blue]{worker.status}[/blue] | Memory: {worker.available_memory_mb} MB")

            # Show devices
            for device in worker.devices:
                if device.device_type == "cuda":
                    console.print(f"  [green]GPU {device.device_id}[/green]: {device.name}")
                    console.print(f"    Total Memory: {device.total_memory_mb} MB")

            # Show GPU metrics
            gpu_found = False
            for dm in worker.device_metrics:
                if dm.HasField("gpu_metrics"):
                    if not gpu_found:
                        console.print("  [yellow]GPU Metrics:[/yellow]")
                        gpu_found = True

                    gpu_metrics = dm.gpu_metrics
                    power_w = gpu_metrics.power_draw_mw / 1000
                    power_limit_w = gpu_metrics.power_limit_mw / 1000

                    console.print(f"    [green]{dm.device_id}[/green]:")
                    console.print(f"      Utilization: {gpu_metrics.utilization_percent}%")
                    console.print(f"      Memory Util: {gpu_metrics.memory_utilization_percent}%")
                    console.print(f"      Temperature: {gpu_metrics.temperature_c}°C")
                    console.print(f"      Power: {power_w:.0f}W / {power_limit_w:.0f}W")
                    console.print(f"      Free Memory: {gpu_metrics.free_memory_mb} MB")

            # Show CPU metrics
            for dm in worker.device_metrics:
                if dm.HasField("cpu_metrics"):
                    cpu_metrics = dm.cpu_metrics
                    console.print(f"  [yellow]CPU:[/yellow]")
                    console.print(f"    Utilization: {cpu_metrics.utilization_percent:.1f}%")
                    console.print(f"    Used Memory: {cpu_metrics.used_memory_mb} MB")

            # Show network metrics
            if worker.HasField("network_metrics") and worker.network_metrics.latency_ms_to_worker:
                console.print("  [yellow]Network Latency:[/yellow]")
                for target_id, latency_ms in worker.network_metrics.latency_ms_to_worker.items():
                    console.print(f"    → {target_id}: {latency_ms} ms")

                if worker.network_metrics.estimated_bandwidth_mbps:
                    console.print(f"    Bandwidth: ~{worker.network_metrics.estimated_bandwidth_mbps} Mbps")

        console.print()
        channel.close()

    except grpc.RpcError as e:
        console.print(f"[red]Error connecting to Master: {e.details()}[/red]")
        console.print("[yellow]Is the Master running? Try: edgeshard master start[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
