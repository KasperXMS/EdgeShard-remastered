"""Shard daemon — standalone shard process for distributed inference.

A ShardDaemon:
1. Loads a model subset (layer range)
2. Starts a gRPC server for data-plane tensor transfers
3. Optionally registers with a Master
4. Waits for inference commands

Usage:
    edgeshard shard start \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --layers 0:16 \\
        --shard-id shard-0 \\
        --data-plane-port 50100
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import torch

from edgeshard.common.config import WorkerConfig
from edgeshard.common.logging import get_logger, setup_logging
from edgeshard.runtime.adapters.qwen2 import Qwen2Adapter
from edgeshard.runtime.shard import ModelShard
from edgeshard.transport.grpc_transport import ShardServer

logger = get_logger(__name__)


class ShardDaemon:
    """Standalone shard process."""

    def __init__(
        self,
        shard_id: str,
        model_path: str,
        layer_start: int,
        layer_end: int,
        dtype: str = "float16",
        data_plane_host: str = "0.0.0.0",
        data_plane_port: int = 50100,
        is_first_shard: bool = False,
        is_last_shard: bool = False,
    ) -> None:
        self._shard_id = shard_id
        self._model_path = model_path
        self._layer_start = layer_start
        self._layer_end = layer_end
        self._dtype = dtype
        self._data_plane_host = data_plane_host
        self._data_plane_port = data_plane_port
        self._is_first_shard = is_first_shard
        self._is_last_shard = is_last_shard

        self._adapter: Qwen2Adapter | None = None
        self._shard: ModelShard | None = None
        self._server: ShardServer | None = None

    async def start(self) -> None:
        """Start the shard daemon."""
        logger.info(f"Starting shard {self._shard_id}")
        logger.info(f"  Model: {self._model_path}")
        logger.info(f"  Layers: [{self._layer_start}, {self._layer_end})")
        logger.info(f"  First shard: {self._is_first_shard}")
        logger.info(f"  Last shard: {self._is_last_shard}")

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(self._dtype, torch.float16)

        logger.info(f"Loading model on {device} with dtype {dtype}")
        self._adapter = Qwen2Adapter()
        self._adapter.load(
            model_path=self._model_path,
            layer_start=self._layer_start,
            layer_end=self._layer_end,
            dtype=dtype,
            device=device,
        )

        # Create shard
        self._shard = ModelShard(
            shard_id=self._shard_id,
            adapter=self._adapter,
            is_first_shard=self._is_first_shard,
            is_last_shard=self._is_last_shard,
        )

        # Start data-plane server
        self._server = ShardServer(
            shard_id=self._shard_id,
            host=self._data_plane_host,
            port=self._data_plane_port,
        )
        await self._server.start()

        logger.info(f"Shard {self._shard_id} started successfully")

    async def stop(self) -> None:
        """Stop the shard daemon."""
        logger.info(f"Stopping shard {self._shard_id}")

        if self._server:
            await self._server.stop()

        if self._adapter:
            self._adapter.unload()

        logger.info(f"Shard {self._shard_id} stopped")

    def get_shard(self) -> ModelShard | None:
        """Get the ModelShard instance (for testing)."""
        return self._shard
