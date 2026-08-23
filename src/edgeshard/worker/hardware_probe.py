"""Hardware probe — discover and report device capabilities.

This module provides hardware discovery for Worker nodes:
- NVIDIA GPUs (via NVML)
- Jetson devices (via tegrastats)
- CPU and memory (via psutil)

Results are normalized into DeviceInfo messages for the Master.
"""

from __future__ import annotations

import platform
import subprocess
from typing import Any

import psutil

from edgeshard._grpc import edgeshard_pb2
from edgeshard.common.logging import get_logger

logger = get_logger(__name__)


# Global NVML state for metrics collection
_nvml_initialized = False


def _ensure_nvml_initialized() -> bool:
    """Ensure NVML is initialized. Returns True if available."""
    global _nvml_initialized
    if _nvml_initialized:
        return True

    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_initialized = True
        return True
    except ImportError:
        logger.debug("pynvml not available")
        return False
    except Exception as e:
        logger.debug(f"NVML init failed: {e}")
        return False


def probe_hardware() -> list[edgeshard_pb2.DeviceInfo]:
    """Discover all hardware devices on this Worker.

    Returns:
        List of DeviceInfo messages describing available devices.
    """
    devices = []

    # Probe NVIDIA GPUs
    try:
        gpu_devices = _probe_nvidia_gpus()
        devices.extend(gpu_devices)
    except Exception as e:
        logger.warning(f"Failed to probe NVIDIA GPUs: {e}")

    # Probe Jetson (if applicable)
    if _is_jetson():
        try:
            jetson_device = _probe_jetson()
            if jetson_device:
                devices.append(jetson_device)
        except Exception as e:
            logger.warning(f"Failed to probe Jetson: {e}")

    # Always add CPU
    cpu_device = _probe_cpu()
    devices.append(cpu_device)

    logger.info(f"Discovered {len(devices)} device(s)")
    return devices


def _probe_nvidia_gpus() -> list[edgeshard_pb2.DeviceInfo]:
    """Probe NVIDIA GPUs using NVML.

    Reports both total and available (free) memory. The free memory
    is embedded in the GpuMetrics field of DeviceInfo so the scheduler
    can make placement decisions based on actual available resources,
    not just total capacity.
    """
    try:
        import pynvml

        if not _ensure_nvml_initialized():
            return []

        device_count = pynvml.nvmlDeviceGetCount()

        devices = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory_mb = memory_info.total // (1024 * 1024)
            free_memory_mb = memory_info.free // (1024 * 1024)

            # Get compute capability
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            compute_capability = f"{major}.{minor}"

            # Include current free memory in gpu_metrics so the scheduler
            # can use it for placement decisions BEFORE the first heartbeat
            gpu_metrics = edgeshard_pb2.GpuMetrics(
                free_memory_mb=free_memory_mb,
            )

            device = edgeshard_pb2.DeviceInfo(
                device_id=f"cuda:{i}",
                device_type="cuda",
                name=name,
                total_memory_mb=total_memory_mb,
                compute_capability=compute_capability,
                gpu_metrics=gpu_metrics,
            )
            devices.append(device)

            logger.info(
                f"Found GPU {i}: {name} "
                f"(total={total_memory_mb} MB, free={free_memory_mb} MB)"
            )

        return devices

    except ImportError:
        logger.debug("pynvml not available, skipping NVIDIA GPU probe")
        return []
    except Exception as e:
        logger.debug(f"NVML probe failed: {e}")
        return []


def _is_jetson() -> bool:
    """Check if this is a Jetson device."""
    return platform.machine() in ("aarch64", "arm64") and platform.system() == "Linux"


def _get_jetson_model() -> str:
    """Get Jetson model name from device tree."""
    try:
        with open("/proc/device-tree/model", "rb") as f:
            model = f.read().decode("utf-8", errors="ignore").strip().rstrip("\x00")
            return model or "NVIDIA Jetson"
    except (FileNotFoundError, PermissionError):
        return "NVIDIA Jetson"


def _get_jetson_memory_mb() -> int:
    """Get total memory for Jetson device."""
    memory = psutil.virtual_memory()
    return memory.total // (1024 * 1024)


def _probe_jetson() -> edgeshard_pb2.DeviceInfo | None:
    """Probe Jetson device using tegrastats and device info."""
    import subprocess

    device_name = _get_jetson_model()
    total_memory_mb = _get_jetson_memory_mb()

    properties = {"platform": "jetson"}

    # Try to get tegrastats info
    try:
        result = subprocess.run(
            ["tegrastats", "--interval", "1000", "--count", "1"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            properties["tegrastats"] = "available"
            # Parse first line for initial info
            # Format: "RAM ... GR3D_FREQ ... CPU ... GPU ..."
            output = result.stdout.strip()
            if output:
                properties["tegrastats_sample"] = output[:200]  # Store sample
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"tegrastats not available: {e}")

    device = edgeshard_pb2.DeviceInfo(
        device_id="jetson:0",
        device_type="jetson",
        name=device_name,
        total_memory_mb=total_memory_mb,
        properties=properties,
    )

    logger.info(f"Found Jetson: {device_name} ({total_memory_mb} MB)")
    return device


def _probe_cpu() -> edgeshard_pb2.DeviceInfo:
    """Probe CPU and system memory."""
    cpu_count = psutil.cpu_count(logical=True)
    memory = psutil.virtual_memory()
    total_memory_mb = memory.total // (1024 * 1024)

    return edgeshard_pb2.DeviceInfo(
        device_id="cpu:0",
        device_type="cpu",
        name=platform.processor() or "Unknown CPU",
        total_memory_mb=total_memory_mb,
        properties={"cpu_count": str(cpu_count)},
    )


def get_available_memory_mb() -> int:
    """Get current available system memory in MB."""
    memory = psutil.virtual_memory()
    return memory.available // (1024 * 1024)


def get_used_memory_mb() -> int:
    """Get current used system memory in MB."""
    memory = psutil.virtual_memory()
    return memory.used // (1024 * 1024)


def get_cpu_count() -> int:
    """Get number of CPU cores."""
    return psutil.cpu_count(logical=True)


def get_cpu_utilization() -> float:
    """Get CPU utilization percentage."""
    return psutil.cpu_percent(interval=0.1)


def get_hostname() -> str:
    """Get hostname of this Worker."""
    return platform.node()


def collect_gpu_metrics(device_index: int) -> edgeshard_pb2.GpuMetrics | None:
    """Collect dynamic metrics for a single GPU.

    Args:
        device_index: GPU index (0-based).

    Returns:
        GpuMetrics message or None if collection fails.
    """
    try:
        import pynvml

        if not _ensure_nvml_initialized():
            return None

        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

        # Utilization rates
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

        # Temperature
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = 0

        # Power usage
        try:
            power_draw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        except Exception:
            power_draw = 0
            power_limit = 0

        # Memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory_mb = mem_info.free // (1024 * 1024)

        return edgeshard_pb2.GpuMetrics(
            utilization_percent=utilization.gpu,
            memory_utilization_percent=utilization.memory,
            temperature_c=temp,
            power_draw_mw=power_draw,
            power_limit_mw=power_limit,
            free_memory_mb=free_memory_mb,
        )

    except ImportError:
        return None
    except Exception as e:
        logger.debug(f"Failed to collect GPU {device_index} metrics: {e}")
        return None


def collect_all_device_metrics(
    devices: list[edgeshard_pb2.DeviceInfo],
) -> list[edgeshard_pb2.DeviceMetrics]:
    """Collect dynamic metrics for all devices.

    Args:
        devices: List of DeviceInfo from probe_hardware().

    Returns:
        List of DeviceMetrics for each device.
    """
    metrics_list = []

    for device in devices:
        device_metrics = edgeshard_pb2.DeviceMetrics(device_id=device.device_id)

        if device.device_type == "cuda":
            # Extract GPU index from device_id (format: "cuda:0")
            try:
                gpu_index = int(device.device_id.split(":")[1])
                gpu_metrics = collect_gpu_metrics(gpu_index)
                if gpu_metrics:
                    device_metrics.gpu_metrics.CopyFrom(gpu_metrics)
                    metrics_list.append(device_metrics)
            except (IndexError, ValueError):
                pass

        elif device.device_type == "cpu":
            # CPU metrics
            cpu_metrics = edgeshard_pb2.CpuMetrics(
                utilization_percent=get_cpu_utilization(),
                used_memory_mb=get_used_memory_mb(),
            )
            device_metrics.cpu_metrics.CopyFrom(cpu_metrics)
            metrics_list.append(device_metrics)

        elif device.device_type == "jetson":
            # Jetson metrics via tegrastats
            jetson_metrics = _collect_jetson_metrics()
            if jetson_metrics:
                device_metrics.cpu_metrics.CopyFrom(jetson_metrics)
                metrics_list.append(device_metrics)

    return metrics_list


def _collect_jetson_metrics() -> edgeshard_pb2.CpuMetrics | None:
    """Collect metrics from Jetson device via tegrastats.

    Returns:
        CpuMetrics with CPU utilization, or None if tegrastats unavailable.
    """
    try:
        result = subprocess.run(
            ["tegrastats", "--interval", "500", "--count", "1"],
            capture_output=True, text=True, timeout=3
        )

        if result.returncode != 0:
            return None

        output = result.stdout.strip()
        if not output:
            return None

        # Parse tegrastats output
        # Format: "RAM 1234/8192MB ... CPU [X%,Y%,Z%,W%] ... GR3D_FREQ X%"
        # Extract CPU utilization (average)
        cpu_percent = 0.0

        # Try to find CPU line
        import re
        cpu_match = re.search(r"CPU \[(\d+)%.*?\]", output)
        if cpu_match:
            # Get first CPU core as representative
            cpu_percent = float(cpu_match.group(1))

        # Use system memory as fallback
        memory = psutil.virtual_memory()
        used_memory_mb = memory.used // (1024 * 1024)

        return edgeshard_pb2.CpuMetrics(
            utilization_percent=cpu_percent,
            used_memory_mb=used_memory_mb,
        )

    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"tegrastats collection failed: {e}")
        return None
