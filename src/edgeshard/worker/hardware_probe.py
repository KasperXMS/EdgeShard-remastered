"""Hardware probe — discover and report device capabilities.

This module provides hardware discovery for Worker nodes:
- NVIDIA GPUs (via NVML)
- Jetson devices (via tegrastats)
- CPU and memory (via psutil)

Results are normalized into DeviceInfo messages for the Master.
"""

from __future__ import annotations

import platform
from typing import Any

import psutil

from edgeshard._grpc import edgeshard_pb2
from edgeshard.common.logging import get_logger

logger = get_logger(__name__)


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
    """Probe NVIDIA GPUs using NVML."""
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        devices = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory_mb = memory_info.total // (1024 * 1024)

            # Get compute capability
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            compute_capability = f"{major}.{minor}"

            device = edgeshard_pb2.DeviceInfo(
                device_id=f"cuda:{i}",
                device_type="cuda",
                name=name,
                total_memory_mb=total_memory_mb,
                compute_capability=compute_capability,
            )
            devices.append(device)

            logger.info(f"Found GPU {i}: {name} ({total_memory_mb} MB)")

        pynvml.nvmlShutdown()
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


def _probe_jetson() -> edgeshard_pb2.DeviceInfo | None:
    """Probe Jetson device (placeholder for tegrastats integration)."""
    # TODO: Integrate tegrastats for Jetson memory/GPU info
    # For now, return a basic CPU device
    return None


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


def get_cpu_count() -> int:
    """Get number of CPU cores."""
    return psutil.cpu_count(logical=True)


def get_hostname() -> str:
    """Get hostname of this Worker."""
    return platform.node()
