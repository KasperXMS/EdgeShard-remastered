"""CLI command: edgeshard cluster snapshot — export cluster state to YAML."""

from __future__ import annotations

from pathlib import Path

import grpc
import yaml
from rich.console import Console

from edgeshard._grpc import edgeshard_pb2, edgeshard_pb2_grpc

console = Console()


def export_cluster_snapshot(
    master_address: str = "localhost:10500",
    output_path: str = "cluster.yaml",
) -> None:
    """Export current cluster state to a YAML file for offline planning.

    Connects to a running Master, fetches all worker information including
    devices, metrics, and network topology, then writes it as a cluster
    YAML file that can be used with `edgeshard plan --cluster-yaml`.

    Args:
        master_address: Master gRPC address (host:port).
        output_path: Output YAML file path.
    """
    console.print(f"[bold]EdgeShard Cluster Snapshot[/bold]")
    console.print(f"[dim]Connecting to Master at {master_address}...[/dim]")

    try:
        channel = grpc.insecure_channel(master_address)
        stub = edgeshard_pb2_grpc.WorkerServiceStub(channel)
        response = stub.ListWorkers(edgeshard_pb2.ListWorkersRequest())

        if not response.workers:
            console.print("[yellow]No workers registered in the cluster.[/yellow]")
            console.print("[dim]Start workers with: edgeshard worker start --master ...[/dim]")
            channel.close()
            return

        # Build YAML data
        workers_data = []
        bandwidth_entries = []
        source_worker_id = None

        for worker in response.workers:
            if source_worker_id is None:
                source_worker_id = worker.worker_id

            # Build devices list
            devices_data = []
            for dev in worker.devices:
                dev_entry = {
                    "device_id": dev.device_id,
                    "device_type": dev.device_type,
                    "name": dev.name,
                    "total_memory_mb": dev.total_memory_mb,
                }

                # Per-device available memory from heartbeat metrics (live)
                # or registration-time gpu_metrics (static fallback)
                dev_available_mb = 0
                # Try heartbeat device_metrics first (most recent)
                for dm in worker.device_metrics:
                    if dm.device_id == dev.device_id and dm.HasField("gpu_metrics"):
                        dev_available_mb = dm.gpu_metrics.free_memory_mb
                        break
                # Fallback: registration-time gpu_metrics on DeviceInfo
                if dev_available_mb == 0 and dev.HasField("gpu_metrics"):
                    dev_available_mb = dev.gpu_metrics.free_memory_mb
                # For CPU/Jetson, use worker-level available_memory_mb
                if dev_available_mb == 0 and dev.device_type in ("cpu", "jetson"):
                    dev_available_mb = worker.available_memory_mb

                if dev_available_mb > 0:
                    dev_entry["available_memory_mb"] = dev_available_mb

                if dev.compute_capability:
                    dev_entry["compute_capability"] = dev.compute_capability
                devices_data.append(dev_entry)

            worker_entry = {
                "worker_id": worker.worker_id,
                "hostname": worker.hostname,
                "available_memory_mb": worker.available_memory_mb,
                "cpu_count": worker.cpu_count,
                "status": worker.status,
            }
            if devices_data:
                worker_entry["devices"] = devices_data
            if worker.metadata:
                worker_entry["metadata"] = dict(worker.metadata)

            workers_data.append(worker_entry)

            # Extract bandwidth from network metrics
            if worker.HasField("network_metrics"):
                nm = worker.network_metrics
                if nm.estimated_bandwidth_mbps > 0:
                    for target_id in nm.latency_ms_to_worker:
                        # Only add each pair once (use lexicographic ordering)
                        if worker.worker_id < target_id:
                            bandwidth_entries.append({
                                "from": worker.worker_id,
                                "to": target_id,
                                "bandwidth_mb_per_s": round(nm.estimated_bandwidth_mbps / 8.0, 1),
                            })

        cluster_data = {
            "source_worker_id": source_worker_id,
            "workers": workers_data,
        }
        if bandwidth_entries:
            cluster_data["bandwidth_matrix"] = bandwidth_entries

        # Write YAML
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(cluster_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        channel.close()

        # Summary
        console.print()
        console.print(f"[bold green]Cluster snapshot exported![/bold green]")
        console.print()
        console.print(f"  Workers:    [cyan]{len(workers_data)}[/cyan]")

        total_gpus = sum(
            sum(1 for d in w.get("devices", []) if d.get("device_type") == "cuda")
            for w in workers_data
        )
        total_jetsons = sum(
            sum(1 for d in w.get("devices", []) if d.get("device_type") == "jetson")
            for w in workers_data
        )
        console.print(f"  GPUs:       [green]{total_gpus}[/green]")
        if total_jetsons:
            console.print(f"  Jetsons:    [yellow]{total_jetsons}[/yellow]")
        console.print(f"  Bandwidth:  [dim]{len(bandwidth_entries)} link(s) measured[/dim]")

        # Show per-device free memory summary
        console.print()
        for w in workers_data:
            for d in w.get("devices", []):
                if d.get("device_type") == "cuda":
                    free = d.get("available_memory_mb", 0)
                    total = d.get("total_memory_mb", 0)
                    if free > 0:
                        console.print(
                            f"  [dim]{w['worker_id']}[/dim] {d['name']} "
                            f"[cyan]{d['device_id']}[/cyan]: "
                            f"free={free}/{total} MB"
                        )
        console.print()
        console.print(f"  Output:     [bold]{out}[/bold]")
        console.print()
        console.print(f"[dim]Use with: edgeshard plan service.yaml --cluster-yaml {out}[/dim]")

    except grpc.RpcError as e:
        console.print(f"[red]Error connecting to Master: {e.details()}[/red]")
        console.print("[yellow]Is the Master running? Try: edgeshard master start[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
