"""Local deployment backend — spawn shard as a local subprocess.

This backend is useful for development, testing, or single-node deployments.
It spawns the shard as a Python subprocess on the local machine.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from typing import TYPE_CHECKING

from edgeshard.common.logging import get_logger
from edgeshard.deployment.backend import DeploymentBackend, ShardHandle, ShardStatus

if TYPE_CHECKING:
    from edgeshard.scheduler.placement import ShardPlacement

logger = get_logger(__name__)


class LocalDeploymentBackend(DeploymentBackend):
    """Deploy shards as local subprocesses.

    Useful for development, testing, or single-node deployments.
    """

    def __init__(self) -> None:
        self._shards: dict[str, ShardHandle] = {}
        self._processes: dict[str, subprocess.Popen] = {}

    async def start_shard(
        self,
        shard_placement: "ShardPlacement",
        model_name: str,
        model_revision: str = "main",
        dtype: str = "float16",
        data_host: str = "0.0.0.0",
        data_port: int = 50100,
        master_address: str = "localhost:10500",
    ) -> ShardHandle:
        """Start a shard as a local subprocess."""
        shard_id = f"shard-{shard_placement.shard_index}"
        worker_id = "local"

        handle = ShardHandle(
            shard_id=shard_id,
            worker_id=worker_id,
            status=ShardStatus.STARTING,
            data_address=f"{data_host}:{data_port}",
            model_name=model_name,
            layer_start=shard_placement.layer_start,
            layer_end=shard_placement.layer_end,
            device=shard_placement.device,
        )

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "edgeshard",
            "shard",
            "start",
            model_name,
            "--shard-id",
            shard_id,
            "--layers",
            f"{shard_placement.layer_start}:{shard_placement.layer_end}",
            "--dtype",
            dtype,
            "--host",
            data_host,
            "--port",
            str(data_port),
        ]

        if shard_placement.shard_index == 0:
            cmd.append("--first")

        logger.info(f"Starting shard subprocess: {' '.join(cmd)}")

        try:
            # Start subprocess
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._processes[shard_id] = proc
            self._shards[shard_id] = handle

            # Wait a bit for the shard to start
            await asyncio.sleep(2.0)

            # Check if process is still running
            if proc.poll() is None:
                handle.status = ShardStatus.READY
                logger.info(f"Shard {shard_id} started (PID: {proc.pid})")
            else:
                _, stderr = proc.communicate()
                handle.status = ShardStatus.FAILED
                handle.error_message = stderr.decode()[:500]
                logger.error(f"Shard {shard_id} failed to start: {handle.error_message}")

        except Exception as e:
            handle.status = ShardStatus.FAILED
            handle.error_message = str(e)
            logger.error(f"Error starting shard {shard_id}: {e}")

        return handle

    async def stop_shard(self, shard_id: str) -> bool:
        """Stop a local shard subprocess."""
        handle = self._shards.get(shard_id)
        proc = self._processes.get(shard_id)

        if proc is None:
            logger.warning(f"Shard {shard_id} not found")
            return False

        try:
            proc.terminate()
            proc.wait(timeout=10.0)
            handle.status = ShardStatus.STOPPED
            logger.info(f"Shard {shard_id} stopped")
            return True
        except subprocess.TimeoutExpired:
            proc.kill()
            logger.warning(f"Shard {shard_id} killed (did not terminate gracefully)")
            return True
        except Exception as e:
            logger.error(f"Error stopping shard {shard_id}: {e}")
            return False

    async def list_shards(self) -> list[ShardHandle]:
        """List all local shards."""
        # Update status based on process state
        for shard_id, proc in self._processes.items():
            handle = self._shards[shard_id]
            if proc.poll() is not None and handle.status == ShardStatus.READY:
                handle.status = ShardStatus.STOPPED
        return list(self._shards.values())

    async def get_shard(self, shard_id: str) -> ShardHandle | None:
        """Get a specific shard handle."""
        handle = self._shards.get(shard_id)
        if handle is not None:
            proc = self._processes.get(shard_id)
            if proc is not None and proc.poll() is not None:
                handle.status = ShardStatus.STOPPED
        return handle

    async def stop_all(self) -> None:
        """Stop all running shards."""
        for shard_id in list(self._shards.keys()):
            await self.stop_shard(shard_id)
